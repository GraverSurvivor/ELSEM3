"""Stage 4 driver: run the greedy attribution search on the real 3-bit model. Needs a GPU.

Loads Phi-3.5 at HQQ 3-bit (fp16 compute, matching the baseline), wraps it in a `SwapRegistry`
backed by `CheckpointWeightSource`, and runs `greedy_search` over the 128 candidates, scoring each
on the fixed search sample (`search_sample.py`). Every trial is checkpointed to
`results/<run_id>/trials.csv` as it happens -- the search is the most GPU-expensive step in the
project, and a dead session should cost the trials in flight, not the whole search.

Outputs, under results/<run_id>/:
    trials.csv    one row per candidate tried: round, path, usr, gain, kept. The recovery curve
                  and "which modules" results are drawn from this.
    summary.json  config, sample provenance, kept modules, USR per round, memory, wall clock.

Run after drawing the sample (see search_sample.py):
    python -m src.attribution.run_search --sample results/search_samples/hqq_3bit_n300_seed0.json
    python -m src.attribution.run_search --sample ... --threshold 0.02 --max-rounds 3
    python -m src.attribution.run_search --sample ... --candidates mlp            # MLP-only ablation
    python -m src.attribution.run_search --sample results/search_samples/capability_n300_seed0.json

What a trial measures is "share of the sample answered correctly". On the fairness sample that is
exactly USR, since every ambiguous item's correct answer is "unknown". On the general-ability
sample (disambiguated items fp16 gets right and 3-bit gets wrong) it is plain accuracy -- which is
what lets the same search ask "where does general damage live?" for the concentrated-vs-diffuse
comparison. The sample file says which question set its ids come from (`context`).

Optional hedging penalty (`--penalty-sample`, `--penalty-weight`): if the fairness controls show
the repair works by making the model answer "unknown" everywhere, the objective becomes
    correct on the sample  -  weight x (false-"unknown" rate on a disambiguated probe set),
so a module only scores well by recovering "unknown" where it is right. Off by default.

Long searches (the penalised one is ~7 GPU-hours) resume: `--resume <run_id>` re-reads trials.csv and
progress.json and continues where the search stopped. On Colab the machine's disk is lost with the
session, so pass `--mirror <folder on Google Drive>`: both files are copied there after every trial,
and a resume restores them from it.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import torch

from src.attribution.greedy_search import SearchConfig, SearchResult, Trial, greedy_search
from src.attribution.patch import patch_from_search_summary, save_patch
from src.quant.candidates import CANDIDATE_SETS, filter_candidates
from src.data.bbq_loader import load_all_categories
from src.eval.option_scorer import DEFAULT_BATCH_SIZE, score_items
from src.experiments.logger import RESULTS_DIR, _write_atomically, new_run_id

COMPUTE_DTYPE = "fp16"  # must match the baseline and the hqq_3bit reference run

# The sample is drawn from items the 3-bit model answers with the stereotype, so its USR on the
# sample should start at 0. Batch composition can nudge a near-tie or two; much more than that
# means the model loaded here isn't the one the flipped set was computed against.
START_USR_WARNING = 0.05


def correct_rate(predictions: list[int], items: list) -> float:
    """Share of items answered with their label (USR on ambiguous items, accuracy on disambiguated)."""
    return sum(p == item.label for p, item in zip(predictions, items)) / len(items)


def false_unknown_rate(predictions: list[int], items: list) -> float:
    """Share of items answered "unknown" although their label is a real answer (hedging)."""
    eligible = [(p, item) for p, item in zip(predictions, items) if item.label != item.unknown_index]
    if not eligible:
        raise ValueError("no items with a stated answer to measure hedging on")
    return sum(p == item.unknown_index for p, item in eligible) / len(eligible)


def search_objective(sample_score: float, hedging: float | None, weight: float) -> float:
    """The number a trial is ranked by: the sample score, minus the hedging penalty if enabled."""
    return sample_score if hedging is None else sample_score - weight * hedging


def _load_sample_items(sample: dict) -> list:
    context = sample.get("context", "ambig")
    items_by_id = {item.question_id: item for items in load_all_categories(context).values() for item in items}
    missing = [qid for qid in sample["question_ids"] if qid not in items_by_id]
    if missing:
        raise SystemExit(f"{len(missing)} sample ids not found in BBQ ({context}), e.g. {missing[:5]}")
    return [items_by_id[qid] for qid in sample["question_ids"]]


PROGRESS_FILE = "progress.json"


def read_trials_csv(path: Path) -> list[Trial]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return [Trial(round=int(r["round"]), path=r["path"], usr=float(r["usr"]), gain=float(r["gain"]),
                      kept=r["kept"].strip().lower() == "true") for r in csv.DictReader(handle)]


def resume_state(out_dir: Path, meta: dict) -> SearchResult:
    """The checkpoint of an interrupted search, refusing one started with different settings.

    The two files are written one after the other, and a mirror on Google Drive uploads them
    separately, so a session killed mid-upload can leave one a trial or a keep ahead of the other.
    trials.csv is the complete log of every measurement, so it decides which modules were kept.
    progress.json's per-round scores are used only as far as the two agree; any missing ones are
    re-measured by the search. Files that contradict each other (neither list of kept modules a
    prefix of the other) are refused.
    """
    progress = json.loads((Path(out_dir) / PROGRESS_FILE).read_text(encoding="utf-8"))
    if progress["meta"] != meta:
        raise SystemExit(f"Cannot resume: that search used {progress['meta']}, this one asks for {meta}.")
    trials_path = Path(out_dir) / "trials.csv"
    trials = read_trials_csv(trials_path) if trials_path.exists() else []
    kept = [t.path for t in trials if t.kept]
    saved_kept = list(progress["kept"])
    print(f"Checkpoint: trials.csv has {len(trials)} trials and {len(kept)} kept; "
          f"progress.json has {len(saved_kept)} kept.")
    if saved_kept and not trials:
        raise SystemExit(f"Cannot resume: progress.json says {len(saved_kept)} module(s) were kept but "
                         f"{trials_path} is {'empty' if trials_path.exists() else 'missing'}. Upload the "
                         "search's trials.csv under exactly that name; starting over would redo every trial.")
    if len(saved_kept) - len(kept) > 1:
        raise SystemExit(f"Cannot resume: progress.json is {len(saved_kept) - len(kept)} keeps ahead of trials.csv. "
                         "The two files can only drift one step apart; this trials.csv looks like an older copy.")
    shared = min(len(kept), len(saved_kept))
    if kept[:shared] != saved_kept[:shared]:
        raise SystemExit(f"Cannot resume: trials.csv kept {kept} but progress.json kept {saved_kept}.")
    if kept != saved_kept:
        print(f"Note: trials.csv kept {len(kept)} module(s), progress.json {len(saved_kept)}; "
              "following trials.csv, the complete log.")
    usr_after_round = list(progress["usr_after_round"])[: min(len(kept), len(saved_kept))]
    return SearchResult(start_usr=progress["start_usr"], kept=kept, usr_after_round=usr_after_round, trials=trials)


def _write_trials(trials: list[Trial], path: Path) -> None:
    fieldnames = list(Trial.__dataclass_fields__)

    def write(handle) -> None:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for trial in trials:
            writer.writerow(trial.to_dict())

    _write_atomically(path, write)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample", type=Path, required=True, help="search sample JSON from search_sample.py")
    parser.add_argument("--threshold", type=float, default=SearchConfig.threshold)
    parser.add_argument("--nbits", type=int, default=SearchConfig.nbits)
    parser.add_argument("--max-rounds", type=int, default=SearchConfig.max_rounds)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--candidates", default="all", choices=CANDIDATE_SETS, help="restrict to one module family")
    parser.add_argument("--penalty-sample", type=Path, help="disambiguated probe sample for the hedging penalty")
    parser.add_argument("--penalty-weight", type=float, default=1.0)
    parser.add_argument("--resume", metavar="RUN_ID", help="continue an interrupted search")
    parser.add_argument("--mirror", type=Path, help="folder (e.g. on Google Drive) to copy progress to after every trial")
    args = parser.parse_args()

    try:
        config = SearchConfig(threshold=args.threshold, nbits=args.nbits, max_rounds=args.max_rounds)
    except ValueError as exc:
        parser.error(str(exc))
    sample = json.loads(args.sample.read_text(encoding="utf-8"))
    penalty = json.loads(args.penalty_sample.read_text(encoding="utf-8")) if args.penalty_sample else None
    if penalty is not None and penalty.get("context") != "disambig":
        parser.error("--penalty-sample must be a disambiguated sample (its 'context' is 'disambig')")

    if not torch.cuda.is_available():
        raise SystemExit("No GPU found. The search needs CUDA -- on Kaggle, set Accelerator to GPU.")
    device = "cuda"

    # Imported here so --help and the tests work without hqq/GPU.
    from src.eval.model_loader import MODEL_ID
    from src.quant.candidates import candidate_paths_for_model
    from src.quant.hqq_quantizer import load_hqq_3bit_model
    from src.quant.module_swap import SwapRegistry
    from src.quant.weight_source import CheckpointWeightSource

    items = _load_sample_items(sample)
    penalty_items = _load_sample_items(penalty) if penalty is not None else None

    print(f"Loading Phi-3.5-mini-Instruct, HQQ 3-bit, {COMPUTE_DTYPE} compute...")
    model, tokenizer = load_hqq_3bit_model(device=device, dtype=COMPUTE_DTYPE)
    registry = SwapRegistry(model, CheckpointWeightSource(MODEL_ID), device=device)
    candidates = filter_candidates(candidate_paths_for_model(model), args.candidates)

    def score_fn() -> float:
        predictions = score_items(model, tokenizer, items, device, batch_size=args.batch_size)
        hedging = None
        if penalty_items is not None:
            penalty_predictions = score_items(model, tokenizer, penalty_items, device, batch_size=args.batch_size)
            hedging = false_unknown_rate(penalty_predictions, penalty_items)
        return search_objective(correct_rate(predictions, items), hedging, args.penalty_weight)

    meta = {"sample": str(args.sample), "penalty_sample": str(args.penalty_sample) if args.penalty_sample else None,
            "penalty_weight": args.penalty_weight if args.penalty_sample else None,
            "candidates": args.candidates, "config": asdict(config)}
    run_id = args.resume or new_run_id("search")
    out_dir = RESULTS_DIR / run_id
    mirror = args.mirror / run_id if args.mirror else None
    if args.resume and not (out_dir / PROGRESS_FILE).exists() and mirror and (mirror / PROGRESS_FILE).exists():
        out_dir.mkdir(parents=True, exist_ok=True)
        for name in (PROGRESS_FILE, "trials.csv"):
            if (mirror / name).exists():
                shutil.copy2(mirror / name, out_dir / name)
        print(f"Restored the checkpoint from {mirror}")
    if args.resume and not (out_dir / PROGRESS_FILE).exists():
        raise SystemExit(f"No checkpoint for {run_id} in {out_dir}" + (f" or {mirror}" if mirror else ""))
    resume = resume_state(out_dir, meta) if args.resume else None
    out_dir.mkdir(parents=True, exist_ok=True)
    trials_path = out_dir / "trials.csv"

    def checkpoint(result: SearchResult) -> None:
        _write_trials(result.trials, trials_path)
        progress = {"meta": meta, "start_usr": result.start_usr, "kept": list(result.kept),
                    "usr_after_round": list(result.usr_after_round)}
        _write_atomically(out_dir / PROGRESS_FILE, lambda handle: json.dump(progress, handle, indent=2))
        if mirror is not None:
            mirror.mkdir(parents=True, exist_ok=True)
            for name in (PROGRESS_FILE, "trials.csv"):
                shutil.copy2(out_dir / name, mirror / name)

    started = time.time()
    n_per_round = len(candidates)

    def on_progress(result: SearchResult) -> None:
        checkpoint(result)
        latest = result.trials[-1] if result.trials else None
        if latest is not None:
            done_in_round = sum(t.round == latest.round for t in result.trials)
            print(
                # Round r tries every candidate except the r - 1 kept before it. Not len(kept):
                # after a keep, kept already includes this round's winner, which read 124/123.
                f"\rround {latest.round}  {done_in_round}/{n_per_round - (latest.round - 1)}  "
                f"best gain so far {max(t.gain for t in result.trials if t.round == latest.round):+.3f}  "
                f"({(time.time() - started) / 60:.0f} min)",
                end="",
                flush=True,
            )

    print(f"\nRun ID: {run_id}  |  {len(items)} sample items x {len(candidates)} candidates  |  {config}")
    if resume is not None:
        print(f"Resuming {run_id}: {len(resume.kept)} kept, {len(resume.trials)} trials already done")
    result = greedy_search(candidates, registry, score_fn, config, on_progress=on_progress, resume=resume)
    checkpoint(result)
    print()
    if result.start_usr > START_USR_WARNING:
        print(
            f"WARNING: USR on the sample started at {result.start_usr:.3f}, not ~0. The model loaded "
            "here may not match the run the flipped set was computed from."
        )

    summary = {
        "run_id": run_id,
        "condition": "hqq_3bit",
        "compute_dtype": COMPUTE_DTYPE,
        "config": asdict(config),
        "sample": {key: sample[key] for key in ("baseline_run_id", "quantized_run_id", "n", "seed", "per_category")},
        "sample_path": str(args.sample),
        "context": sample.get("context", "ambig"),
        "metric": "usr" if sample.get("context", "ambig") == "ambig" else "accuracy",
        "candidate_set": args.candidates,
        "n_candidates": len(candidates),
        "penalty": (
            {"sample_path": str(args.penalty_sample), "weight": args.penalty_weight, "n": len(penalty_items)}
            if penalty_items is not None else None
        ),
        **result.to_dict(),
        "kept_nbits": registry.active,
        "deployed_overhead_bytes": registry.deployed_overhead_bytes(),
        "search_overhead_bytes": registry.search_overhead_bytes(),
        "wall_clock_seconds": round(time.time() - started, 1),
        # A resumed search's clock covers only the last session, so it undercounts the search.
        "resumed": bool(args.resume),
    }
    _write_trials(result.trials, trials_path)
    _write_atomically(out_dir / "summary.json", lambda handle: json.dump(summary, handle, indent=2))
    if result.kept:
        # The kept modules as a patch, so run_repair --patch, the prefix runs and the exporters all
        # read the same file instead of re-parsing this summary.
        save_patch(patch_from_search_summary(summary, name=run_id), out_dir / "patch.json")
    if mirror is not None:
        shutil.copytree(out_dir, mirror, dirs_exist_ok=True)

    print(f"\nStopped: {result.stop_reason}")
    print(f"USR on sample: {result.start_usr:.3f} -> {result.final_usr:.3f}")
    for i, (path, usr) in enumerate(zip(result.kept, result.usr_after_round), start=1):
        print(f"  {i}. {path:40} USR {usr:.3f}")
    print(f"Deployed overhead: {summary['deployed_overhead_bytes'] / 2**20:.1f} MB")
    print(f"Results: {out_dir.resolve()}")
    print("On Kaggle, download results/ before closing the session.")


if __name__ == "__main__":
    main()
