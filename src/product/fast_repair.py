"""FairGate Fast Repair: one command from a Hugging Face model name to a fairness repair.

    python -m src.product.fast_repair --model Qwen/Qwen2.5-3B-Instruct --mode fast --questions 1000 --out DIR

It runs the project's method on any supported model (Phi-3, Qwen2, Llama, Mistral; up to ~4B
parameters on a T4): compress to HQQ 3-bit, ask the original and the compressed model the same
seeded sample of BBQ questions, find the damage (the questions the original answered "unknown" and
the 3-bit model answers with the stereotype), scan the model's parts the way the greedy search does
(restore a part to 8-bit, re-ask, keep the best, with the hedging penalty), then check the repair on
questions the scan never saw. It writes, as it goes (sprint contract C4, docs/CONTRACTS.md):

    DIR/events.jsonl   one line per update, for the Studio's progress view
    DIR/report.json    before/after numbers, recovery with its interval, the kept parts, the verdict
    DIR/patch.json     the repair, in the project's patch format (with the model's part shapes)

The scan: if Person 2's Fast Scan (src/attribution/fast_scan.py, contract C2) is present it is used;
otherwise a built-in version of the same idea runs. It screens every part on a small subset of the
search questions, then runs the existing greedy search (src/attribution/greedy_search.py) on the best
`pool` parts with the full search questions. "exact" mode screens on the full search questions, so
it is the full greedy search restricted to round 1's top `pool`, which replaying our 7-hour Phi-3.5
search showed picks the identical parts from a pool of 10 upward.

The model-running part sits behind a small backend interface, so the whole pipeline is tested on a
CPU with a stand-in (tests/product/test_fast_repair.py); TorchBackend is the real one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator, Protocol

STAGES = ("download", "quantize", "score_original", "score_3bit", "find_damage", "scan", "verify", "report")
MODES = ("fast", "exact")

# Per-mode scan settings. "fast" screens on a subset to fit ~20 min on a T4; "exact" screens on the
# full search questions (the greedy search's own round 1) and keeps a wider pool.
SCAN_SETTINGS = {
    "fast": {"screen_sample": 60, "screen_probe": 40, "pool": 12, "search_probe": 120},
    "exact": {"screen_sample": None, "screen_probe": None, "pool": 16, "search_probe": 200},
}
SEARCH_CAP = 300          # flipped questions the scan uses, at most (like the Phi-3.5 search sample)
MIN_FLIPPED = 10          # fewer flipped search questions than this: nothing meaningful to repair
CHUNK = 128               # questions per progress update while scoring


# --- configuration and events ---------------------------------------------------------------------

@dataclass
class FastRepairConfig:
    model_id: str
    mode: str = "fast"
    questions: int = 1000
    out: Path = Path("results/fast_repair")
    seed: int = 0
    batch_size: int = 16
    penalty_weight: float = 1.0
    threshold: float = 0.01
    max_rounds: int = 5
    nbits: int = 8

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        if self.questions < 100:
            raise ValueError("use at least 100 questions per check")
        self.out = Path(self.out)


class Events:
    """Writes events.jsonl (contract C4): one complete JSON line per update, flushed immediately."""

    def __init__(self, out: Path):
        out.mkdir(parents=True, exist_ok=True)
        self.path = out / "events.jsonl"
        self.path.write_text("", encoding="utf-8")
        self.stage: str | None = None
        self._last_progress = 0.0

    def write(self, stage: str, status: str, done: int = 0, total: int = 0, message: str = "") -> None:
        record = {"time": datetime.now(timezone.utc).isoformat(timespec="milliseconds"), "stage": stage,
                  "status": status, "done": int(done), "total": int(total), "message": message}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    @contextmanager
    def stage_of(self, stage: str, total: int = 0, message: str = "") -> Iterator["Events"]:
        if stage not in STAGES:
            raise ValueError(f"unknown stage {stage!r}")
        self.stage = stage
        self.write(stage, "started", 0, total, message)
        try:
            yield self
        except BaseException as error:
            self.write(stage, "failed", 0, total, plain_error(error))
            raise
        self.stage = None

    def progress(self, done: int, total: int, message: str = "", force: bool = False) -> None:
        now = time.monotonic()
        if force or now - self._last_progress >= 0.5 or done >= total:
            self._last_progress = now
            self.write(self.stage, "progress", done, total, message)

    def done(self, done: int = 0, total: int = 0, message: str = "") -> None:
        self.write(self.stage, "done", done, total, message)


def plain_error(error: BaseException) -> str:
    """A one-line reason a person can act on."""
    text = f"{type(error).__name__}: {error}".strip()
    lowered = text.lower()
    if "out of memory" in lowered:
        return "The GPU ran out of memory: the model is probably too big for this GPU (about 4B parameters fit a T4)."
    if "gated" in lowered or "401" in lowered or "403" in lowered:
        return "This model needs a Hugging Face login or licence acceptance; it can't be downloaded anonymously."
    if "404" in lowered or ("not found" in lowered and "repository" in lowered):
        return "No model with this name exists on Hugging Face."
    if isinstance(error, KeyboardInterrupt):
        return "Stopped."
    return text[:400]


# --- the backend interface --------------------------------------------------------------------------

class Backend(Protocol):
    """What the pipeline needs from the models. TorchBackend runs them; tests use a stand-in."""

    registry: object  # active / trial(path, nbits) / commit(path, nbits), like SwapRegistry

    def download(self, model_id: str) -> str: ...
    def adapter(self, model_id: str): ...
    def load_compressed(self, model_id: str) -> str: ...
    def load_original(self, model_id: str) -> str: ...
    def score(self, view: str, items: list, on_progress: Callable[[int], None] | None = None) -> list[int]: ...
    def memory_mb(self) -> float: ...
    def gpu_name(self) -> str: ...


class TorchBackend:
    """The real models: the original at fp16 and the HQQ 3-bit copy whose parts the scan swaps."""

    def __init__(self, nbits: int = 8, batch_size: int = 16):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("No GPU found. Fast Repair needs a CUDA GPU (e.g. a Colab T4).")
        self.device, self.nbits, self.batch_size = "cuda", nbits, batch_size
        self.original = self.compressed = self.tokenizer = self.registry = None

    def download(self, model_id: str) -> str:
        from huggingface_hub import snapshot_download

        path = Path(snapshot_download(model_id, allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt",
                                                                 "tokenizer*", "*.tiktoken"]))
        size = sum(f.stat().st_size for f in path.rglob("*.safetensors")) / 1e9
        if size == 0:
            raise RuntimeError("The model has no .safetensors weights; only safetensors checkpoints are supported.")
        return f"{size:.1f} GB of weights"

    def adapter(self, model_id: str):
        from src.models.adapter import adapter_for_id

        return adapter_for_id(model_id)

    def load_compressed(self, model_id: str) -> str:
        import gc

        import torch

        from src.quant.hqq_quantizer import load_hqq_3bit_model
        from src.quant.module_swap import SwapRegistry
        from src.quant.weight_source import CheckpointWeightSource

        self.compressed, self.tokenizer = load_hqq_3bit_model(device=self.device, dtype="fp16", model_id=model_id)
        self.registry = SwapRegistry(self.compressed, CheckpointWeightSource(model_id), device=self.device)
        gc.collect()
        torch.cuda.empty_cache()
        return "HQQ 3-bit, group 64, fp16 compute"

    def load_original(self, model_id: str) -> str:
        from src.eval.model_loader import load_model

        self.original, _ = load_model("fp16", model_id=model_id)
        return "original model loaded at fp16"

    def score(self, view: str, items: list, on_progress=None) -> list[int]:
        from src.eval.option_scorer import score_items

        model = self.original if view == "original" else self.compressed
        picks = []
        for start in range(0, len(items), CHUNK):
            picks += score_items(model, self.tokenizer, items[start:start + CHUNK], self.device, batch_size=self.batch_size)
            if on_progress:
                on_progress(len(picks))
        return picks

    def memory_mb(self) -> float:
        return self.registry.deployed_overhead_bytes() / 2**20

    def gpu_name(self) -> str:
        import torch

        return torch.cuda.get_device_name(0)


# --- questions ---------------------------------------------------------------------------------------

def load_bbq_questions() -> tuple[list, list]:
    from src.data.bbq_loader import load_all_categories

    ambig = [it for items in load_all_categories("ambig").values() for it in items]
    disambig = [it for items in load_all_categories("disambig").values() for it in items]
    return ambig, disambig


def stratified_sample(items: list, n: int, seed: int) -> list:
    """`n` items with each category in proportion, seeded and independent of input order."""
    by_category: dict[str, list] = {}
    for item in sorted(items, key=lambda it: it.question_id):
        by_category.setdefault(item.category, []).append(item)
    total = sum(len(v) for v in by_category.values())
    n = min(n, total)
    quotas = {c: int(n * len(v) / total) for c, v in by_category.items()}
    for c in sorted(by_category, key=lambda c: -(n * len(by_category[c]) / total - quotas[c]))[: n - sum(quotas.values())]:
        quotas[c] += 1
    rng = random.Random(seed)
    picked = []
    for c in sorted(by_category):
        picked += rng.sample(by_category[c], quotas[c])
    return picked


def in_search_half(question_id: str, seed: int) -> bool:
    """A fixed half of the questions is for the scan, the other half only for checking the repair, so
    the repair is always judged on questions it was never chosen with."""
    digest = hashlib.sha256(f"{seed}:{question_id}".encode()).digest()
    return digest[0] % 2 == 0


# --- metrics -----------------------------------------------------------------------------------------

def rates(ambig: list, ambig_picks: list[int], disambig: list, disambig_picks: list[int]) -> dict:
    """The four numbers on each card: on ambiguous questions, the share answered "unknown" (usr) and
    with the stereotype (srs); on stated-answer questions, the share answered "unknown" anyway
    (false_unknown) and answered right (accuracy)."""
    def share(pairs, test):
        pairs = list(pairs)
        return sum(test(it, p) for it, p in pairs) / len(pairs) if pairs else None

    a, d = list(zip(ambig, ambig_picks)), list(zip(disambig, disambig_picks))
    return {"usr": share(a, lambda it, p: p == it.unknown_index),
            "srs": share(a, lambda it, p: p == it.stereotype_index),
            "false_unknown": share(d, lambda it, p: p == it.unknown_index),
            "accuracy": share(d, lambda it, p: p == it.label)}


def rows(items: list, picks: list[int], condition: str) -> list:
    from src.eval.scoring_run import build_results

    return build_results(items, picks, None, f"{condition}_fast_repair", condition)


# --- the scan ----------------------------------------------------------------------------------------

def make_scorer(backend: Backend, penalty_weight: float):
    """objective(sample, probe) = share of the flipped questions answered "unknown" again minus
    penalty_weight x share of the stated-answer probe answered "unknown" -- run_search's objective."""
    from src.attribution.run_search import correct_rate, false_unknown_rate, search_objective

    def objective(sample: list, probe: list) -> float:
        score = correct_rate(backend.score("compressed", sample), sample)
        hedging = false_unknown_rate(backend.score("compressed", probe), probe) if probe else None
        return search_objective(score, hedging, penalty_weight)

    return objective


def builtin_scan(backend: Backend, candidates: list[str], sample: list, probe: list, mode: str,
                 config: FastRepairConfig, events: Events, seed: int):
    """Screen every part on a subset, then the existing greedy search on the best `pool`. Returns the
    greedy SearchResult plus the screening gains."""
    from src.attribution.greedy_search import SearchConfig, greedy_search

    settings = SCAN_SETTINGS[mode]
    rng = random.Random(seed)
    screen_sample = sample if settings["screen_sample"] is None else rng.sample(sample, min(settings["screen_sample"], len(sample)))
    screen_probe = probe if settings["screen_probe"] is None else rng.sample(probe, min(settings["screen_probe"], len(probe)))
    search_probe = probe[: settings["search_probe"]]
    objective = make_scorer(backend, config.penalty_weight)
    nbits, pool_size = config.nbits, settings["pool"]
    total = len(candidates) + pool_size * config.max_rounds  # an upper bound for the progress bar
    done = 0

    base = objective(screen_sample, screen_probe)
    gains = {}
    for path in candidates:
        with backend.registry.trial(path, nbits):
            gains[path] = objective(screen_sample, screen_probe) - base
        done += 1
        events.progress(done, total, f"screening every part: {done} of {len(candidates)}")
    # Ties go to the earlier candidate, like the greedy search's own tie-break.
    pool = sorted(candidates, key=lambda p: (-gains[p], candidates.index(p)))[:pool_size]
    pool = [p for p in candidates if p in set(pool)]  # keep the search's candidate order

    def on_progress(result):
        events.progress(done + len(result.trials), total,
                        f"round {result.trials[-1].round if result.trials else 1} of {config.max_rounds}")

    result = greedy_search(pool, backend.registry, lambda: objective(sample, search_probe),
                           SearchConfig(threshold=config.threshold, nbits=nbits, max_rounds=config.max_rounds),
                           on_progress=on_progress)
    return result, gains, pool


def run_scan(backend, candidates, sample, probe, config, events):
    """Person 2's Fast Scan when it exists (contract C2), otherwise the built-in scan. Returns
    (kept module paths in keep order, a short name of the engine used)."""
    try:
        from src.attribution.fast_scan import FastScanConfig, fast_scan  # noqa: PLC0415 -- optional, Person 2
    except ImportError:
        result, _, _ = builtin_scan(backend, candidates, sample, probe, config.mode, config, events, config.seed)
        return list(result.kept), "built-in screen + greedy"
    scorer = make_scorer(backend, config.penalty_weight)
    sample_by_id = {it.question_id: it for it in sample}
    probe_by_id = {it.question_id: it for it in probe}

    def scan_scorer(sample_ids, probe_ids):
        return scorer([sample_by_id[q] for q in sample_ids], [probe_by_id[q] for q in probe_ids])

    result = fast_scan(candidates, backend.registry, scan_scorer, list(sample_by_id), list(probe_by_id),
                       FastScanConfig(mode=config.mode, max_rounds=config.max_rounds, threshold=config.threshold,
                                      nbits=config.nbits, penalty_weight=config.penalty_weight, seed=config.seed))
    return list(result.kept), "fast_scan"


# --- the pipeline ------------------------------------------------------------------------------------

def run(config: FastRepairConfig, backend: Backend | None = None,
        load_questions: Callable[[], tuple[list, list]] = load_bbq_questions) -> dict:
    """The whole pipeline. Returns the report it wrote; raises after writing a "failed" event."""
    started = time.monotonic()
    out = config.out
    events = Events(out)
    n = config.questions

    with events.stage_of("download", message="fetching the model"):
        if backend is None:  # inside the stage, so "no GPU" is reported like any other failure
            backend = TorchBackend(nbits=config.nbits, batch_size=config.batch_size)
        adapter = backend.adapter(config.model_id)  # fails fast on an unsupported architecture
        message = backend.download(config.model_id)
        events.done(message=f"{message} · {adapter.arch}, {adapter.n_layers} layers, {len(adapter.candidates())} parts")

    with events.stage_of("quantize", message="HQQ 3-bit"):
        events.done(message=backend.load_compressed(config.model_id))

    with events.stage_of("score_original", total=2 * n, message="loading the questions and the original model"):
        ambig_all, disambig_all = load_questions()
        ambig, disambig = stratified_sample(ambig_all, n, config.seed), stratified_sample(disambig_all, n, config.seed)
        backend.load_original(config.model_id)
        questions = ambig + disambig
        picks = backend.score("original", questions, lambda k: events.progress(k, len(questions)))
        # Answers are keyed by the question object, not its id: the two question sets are kept
        # apart even if an ambiguous and a stated-answer question ever shared an id.
        original = dict(zip(map(id, questions), picks))
        events.done(len(questions), len(questions))

    with events.stage_of("score_3bit", total=2 * n):
        picks = backend.score("compressed", questions, lambda k: events.progress(k, len(questions)))
        compressed = dict(zip(map(id, questions), picks))
        events.done(len(questions), len(questions))

    with events.stage_of("find_damage"):
        safe = [it for it in ambig if original[id(it)] == it.unknown_index]
        flipped = [it for it in safe if compressed[id(it)] == it.stereotype_index]
        knows = [it for it in disambig if original[id(it)] == it.label]
        search_flipped = [it for it in flipped if in_search_half(it.question_id, config.seed)][:SEARCH_CAP]
        probe = [it for it in knows if in_search_half(it.question_id, config.seed)]
        held_ambig = [it for it in ambig if not in_search_half(it.question_id, config.seed)]
        held_disambig = [it for it in disambig if not in_search_half(it.question_id, config.seed)]
        events.done(len(flipped), len(safe),
                    f"{len(flipped)} of {len(safe)} fair answers flipped to the stereotype; "
                    f"{len(search_flipped)} used to search, {len(held_ambig)} + {len(held_disambig)} kept back to check")

    candidates = adapter.candidates()
    damaged = len(search_flipped) >= MIN_FLIPPED
    with events.stage_of("scan", message=f"{len(candidates)} parts"):
        if damaged:
            kept, engine = run_scan(backend, candidates, search_flipped, probe, config, events)
            for path in kept:
                if path not in backend.registry.active:
                    backend.registry.commit(path, config.nbits)
            events.done(len(kept), len(kept), f"kept {len(kept)} parts ({engine})")
        else:
            kept, engine = [], "none"
            events.done(message="compression flipped too few answers here; nothing to repair")

    with events.stage_of("verify", total=len(held_ambig) + len(held_disambig)):
        held = held_ambig + held_disambig
        repaired_picks = (backend.score("repaired", held, lambda k: events.progress(k, len(held))) if kept
                          else [compressed[id(it)] for it in held])
        repaired = dict(zip(map(id, held), repaired_picks))
        report = build_report(config, adapter, backend, engine, ambig, disambig, safe, flipped, kept,
                              held_ambig, held_disambig, original, compressed, repaired, started)
        events.done(len(held), len(held))

    with events.stage_of("report"):
        if kept:
            patch = make_patch(config, adapter, kept, engine)
            (out / "patch.json").write_text(json.dumps(patch.to_dict(), indent=2) + "\n", encoding="utf-8")
        report["minutes"] = round((time.monotonic() - started) / 60, 1)
        tmp = out / "report.json.tmp"
        tmp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        tmp.replace(out / "report.json")
        events.done(message=f"verdict {report['gate']}")
    return report


def build_report(config, adapter, backend, engine, ambig, disambig, safe, flipped, kept, held_ambig,
                 held_disambig, original, compressed, repaired, started) -> dict:
    from src.product.fairgate import check
    from src.stats.metrics import is_unknown
    from src.stats.recovery import recovery_fraction

    def picks(table, items):
        return [table[id(it)] for it in items]

    cards = {name: rates(held_ambig, picks(table, held_ambig), held_disambig, picks(table, held_disambig))
             for name, table in (("original", original), ("compressed", compressed), ("repaired", repaired))}
    ref, deg, rep = (rows(held_ambig, picks(t, held_ambig), c)
                     for t, c in ((original, "fp16"), (compressed, "hqq_3bit"), (repaired, "hqq_repaired")))
    try:
        r = recovery_fraction(ref, deg, rep, is_unknown)
        recovery = {"point": r.point, "lower": r.lower, "upper": r.upper}
    except ValueError:
        recovery = {"point": None, "lower": None, "upper": None}  # no gap to recover
    gate = check(ref, rep)
    return {
        "model_id": config.model_id, "arch": adapter.arch, "mode": config.mode, "questions": config.questions,
        "gpu": backend.gpu_name(), "minutes": round((time.monotonic() - started) / 60, 1),
        "safe": len(safe), "flipped": len(flipped),
        "original": cards["original"], "compressed": cards["compressed"], "repaired": cards["repaired"],
        "recovery": recovery, "kept": list(kept), "memory_mb": round(backend.memory_mb(), 1) if kept else 0.0,
        "gate": gate["verdict"], "gate_failures": gate["failures"], "patch": "patch.json" if kept else None,
        "n_layers": adapter.n_layers, "part_types": list(adapter.part_types),
        "checked_on": {"ambiguous": len(held_ambig), "stated_answer": len(held_disambig)},
        "scan": {"engine": engine, "candidates": len(adapter.candidates()), **SCAN_SETTINGS[config.mode],
                 "penalty_weight": config.penalty_weight, "seed": config.seed},
    }


def make_patch(config, adapter, kept, engine):
    from src.attribution.patch import Patch

    is_phi = adapter.arch == "phi3" and adapter.n_layers == 32
    return Patch(name=f"fast_repair_{config.mode}", modules=tuple(kept), nbits=config.nbits,
                 source={"kind": "search", "engine": engine, "mode": config.mode, "questions": config.questions,
                         "seed": config.seed},
                 model_id=config.model_id,
                 part_shapes=None if is_phi else {k: tuple(v) for k, v in adapter.part_shapes.items()},
                 n_layers=None if is_phi else adapter.n_layers)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Hugging Face model id, e.g. Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--mode", choices=MODES, default="fast")
    parser.add_argument("--questions", type=int, default=1000, help="BBQ questions per check (each question set)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--penalty-weight", type=float, default=1.0)
    args = parser.parse_args(argv)
    config = FastRepairConfig(model_id=args.model, mode=args.mode, questions=args.questions, out=args.out,
                              seed=args.seed, batch_size=args.batch_size, penalty_weight=args.penalty_weight)
    print(f"Fast Repair: {config.model_id}, {config.mode} mode, {config.questions} questions -> {config.out}", flush=True)
    try:
        report = run(config)
    except BaseException:
        traceback.print_exc()
        return 1
    print(f"Done in {report['minutes']} min: kept {len(report['kept'])} parts, verdict {report['gate']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
