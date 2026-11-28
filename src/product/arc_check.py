"""General knowledge before and after the repair ("it did no harm"), on ARC science questions.

BBQ's stated-answer questions are the only "did we break anything" number in a Fast Repair report, and they
are still BBQ. This check asks the same three models -- the original, the compressed copy and the repaired
one -- a fixed sample of ARC (the AI2 Reasoning Challenge: grade-school science, four choices, nothing to do
with bias) and compares how often each is right, question by question.

It is report-only: no decision ever sees it, so it is an unseen check, and it doesn't change any verdict.

    python -m src.product.arc_check RUN_DIR [RUN_DIR ...]     # a finished fast_repair --out folder
        -> RUN_DIR/arc.json

The models are rebuilt from the run's own files, the way the repair was measured: the original at the
run's precision (or the 8-bit reference a big model's run used), the compressed copy at the run's method and
bits, and the repaired model from patch.json (`load_repaired`). A run without a patch (nothing restored, or
a rejected repair) gets original and compressed only.

Data: allenai/ai2_arc, test splits of ARC-Easy and ARC-Challenge (CC BY-SA 4.0), downloaded when the check
runs and never copied into the repository. Questions with other than four choices (0.5%) are left out, so
every question has the same answer letters. Scoring is the project's: the probability of each answer letter
at the end of the prompt, no free text.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATASET = "allenai/ai2_arc"
SETS = ("ARC-Easy", "ARC-Challenge")
LICENCE = "CC BY-SA 4.0"
LETTERS = ("A", "B", "C", "D")
N_QUESTIONS = 500          # half from each set
SEED = 0
N_RESAMPLES = 2000
VIEWS = ("original", "compressed", "repaired")
FILE = "arc.json"


@dataclass(frozen=True)
class ArcItem:
    qid: str
    set: str
    question: str
    choices: tuple[str, ...]
    answer: int             # index into choices


# --- the questions -----------------------------------------------------------------------------------

def to_item(row: dict, set_name: str) -> ArcItem | None:
    """One dataset row as a question, or None for one without exactly four choices. Answer keys are
    letters ("C") or, for 3% of questions, numbers ("3"): both become a position."""
    texts, labels = list(row["choices"]["text"]), list(row["choices"]["label"])
    if len(texts) != len(LETTERS) or row["answerKey"] not in labels:
        return None
    return ArcItem(qid=str(row["id"]), set=set_name, question=row["question"].strip(),
                   choices=tuple(t.strip() for t in texts), answer=labels.index(row["answerKey"]))


def sample(items_by_set: dict[str, list[ArcItem]], n: int = N_QUESTIONS, seed: int = SEED) -> list[ArcItem]:
    """A seeded sample of `n` questions, split evenly between the sets (a set with too few gives all it has).
    The same n and seed always pick the same questions, whatever order the dataset arrives in."""
    rng = random.Random(seed)
    out = []
    names = list(items_by_set)
    for i, name in enumerate(names):
        pool = sorted(items_by_set[name], key=lambda it: it.qid)
        want = n // len(names) + (1 if i < n % len(names) else 0)
        out += rng.sample(pool, min(want, len(pool)))
    return sorted(out, key=lambda it: (SETS.index(it.set) if it.set in SETS else len(SETS), it.qid))


def load_arc(n: int = N_QUESTIONS, seed: int = SEED) -> list[ArcItem]:
    from datasets import load_dataset

    by_set = {}
    for name in SETS:
        rows = load_dataset(DATASET, name, split="test")
        by_set[name] = [item for row in rows if (item := to_item(row, name)) is not None]
    return sample(by_set, n, seed)


# --- scoring -----------------------------------------------------------------------------------------

def format_prompt(item: ArcItem, tokenizer) -> str:
    """The same shape as the BBQ prompt: question, lettered choices, "answer with only the letter"; the
    model's chat template if it has one, else a plain completion ending in "Answer:"."""
    from src.eval.option_scorer import has_chat_template

    choices = "\n".join(f"{letter}) {text}" for letter, text in zip(LETTERS, item.choices))
    content = f"{item.question}\n\n{choices}\n\nAnswer with only the letter of the correct option."
    if not has_chat_template(tokenizer):
        return content + "\nAnswer:"
    return tokenizer.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                         add_generation_prompt=True)


def letter_ids(prompt: str, tokenizer) -> list[int]:
    """The token each answer letter would be at the end of this prompt (the letter's id depends on what
    precedes it, so it is read off `tokenize(prompt + letter)`, as the BBQ scorer does)."""
    from src.eval.option_scorer import has_chat_template

    base = tokenizer.encode(prompt, add_special_tokens=True)
    ids = []
    for letter in (LETTERS if has_chat_template(tokenizer) else tuple(" " + x for x in LETTERS)):
        new = tokenizer.encode(prompt + letter, add_special_tokens=True)[len(base):]
        if not new:
            raise ValueError(f"the answer letter {letter!r} adds no token after this prompt")
        ids.append(new[0])
    if len(set(ids)) != len(ids):
        raise ValueError("this tokenizer gives two answer letters the same token, so A-D can't be told apart")
    return ids


def score(model, tokenizer, items: list[ArcItem], device: str, batch_size: int = 8) -> list[tuple[int, float]]:
    """(pick, probability of the right answer) for each question, in order. Halves the batch when the GPU
    runs out of memory; stops with a plain reason if the numbers overflow (NaN)."""
    import gc

    import torch

    from src.eval.option_scorer import last_position_logits, prepare_tokenizer
    from src.product.torch_backend import PrecisionError

    if not items:
        return []
    prepare_tokenizer(tokenizer)
    prompts = [format_prompt(item, tokenizer) for item in items]
    ids = letter_ids(prompts[0], tokenizer)
    if letter_ids(prompts[-1], tokenizer) != ids:
        raise ValueError("the answer letters' tokens differ between prompts, so they can't be read once per run")
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    out: list = [None] * len(items)
    start = 0
    with torch.no_grad():
        while start < len(order):
            batch = order[start:start + batch_size]
            try:
                inputs = tokenizer([prompts[i] for i in batch], return_tensors="pt", padding=True,
                                   add_special_tokens=True).to(device)
                probs = torch.softmax(last_position_logits(model, inputs)[:, ids].float(), dim=-1)
            except Exception as error:  # noqa: BLE001 -- only out-of-memory is retried
                if "out of memory" not in str(error).lower() or batch_size == 1:
                    raise
                batch_size = max(1, batch_size // 2)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            if torch.isnan(probs).any():
                raise PrecisionError("The model's numbers overflow at 16-bit precision (fp16), so its ARC answers "
                                     "can't be trusted.")
            for i, row in zip(batch, probs.tolist()):
                out[i] = (int(np.argmax(row)), float(row[items[i].answer]))
            start += len(batch)
    return out


# --- rebuilding the run's models ---------------------------------------------------------------------

def run_settings(run: Path) -> dict:
    """What the run measured, from its report.json and (if a repair shipped) patch.json."""
    run = Path(run)
    report_path = run / "report.json"
    if not report_path.exists():
        raise FileNotFoundError(f"{run} has no report.json: is it a finished Fast Repair run?")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    patch_path = run / "patch.json"
    patch = json.loads(patch_path.read_text(encoding="utf-8")) if patch_path.exists() else None
    patch = patch if patch and patch.get("modules") else None
    base = (patch or {}).get("base") or {}
    return {
        "model_id": report["model_id"],
        "method": base.get("method", report.get("method", "hqq")),
        "nbits": int(base.get("nbits", report.get("base_nbits") or 3)),
        "dtype": base.get("compute_dtype", report.get("compute_dtype", "fp16")),
        "revision": (patch or {}).get("revision"),
        "reference_bits": int(report.get("reference_bits") or 16),
        "patch": patch,
        "outcome": report.get("outcome"),
    }


def model_loader(settings: dict, device: str = "cuda"):
    """view -> (model, tokenizer), built the way the run measured it (see the module's docstring)."""
    import torch

    from src.quant.repaired import load_repaired

    streaming = settings["reference_bits"] == 8  # the run built this model layer by layer
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[settings["dtype"]]

    def load(view: str):
        if view == "original":
            if streaming:
                from src.quant.streaming import load_streaming

                return load_streaming(settings["model_id"], 8, device=device, dtype=dtype)
            from src.eval.model_loader import load_model

            return load_model(settings["dtype"], model_id=settings["model_id"],
                              device_map="auto" if device == "cuda" else device)
        if view == "compressed":
            bare = {"model_id": settings["model_id"], "modules": [], "nbits": 8, "revision": settings["revision"],
                    "base": {"method": settings["method"], "nbits": settings["nbits"], "group_size": 64,
                             "compute_dtype": settings["dtype"]}}
            return load_repaired(bare, device, streaming=streaming)
        return load_repaired(settings["patch"], device, streaming=streaming)

    return load


# --- the comparison ----------------------------------------------------------------------------------

def paired_diff(first: np.ndarray, second: np.ndarray, n_resamples: int = N_RESAMPLES, seed: int = SEED) -> dict:
    """second minus first (share right), with a paired bootstrap 95% interval: the same questions resampled
    together, so the interval reflects only the questions the two models answer differently."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(first), size=(n_resamples, len(first)))
    diffs = second[idx].mean(axis=1) - first[idx].mean(axis=1)
    return {"point": float(second.mean() - first.mean()), "lower": float(np.percentile(diffs, 2.5)),
            "upper": float(np.percentile(diffs, 97.5))}


def judge(accuracy: dict, diffs: dict) -> tuple[str, str]:
    """(state, one plain sentence). Report-only: no verdict depends on it."""
    o, c, r = (accuracy.get(v) for v in VIEWS)
    if r is None:
        loss = diffs["compressed_minus_original"]
        return "no_repair", (f"No repair to check. Compression {'lowered' if loss['upper'] < 0 else 'kept'} "
                             f"general knowledge: {c:.0%} right against the original's {o:.0%}.")
    if diffs["repaired_minus_compressed"]["upper"] < 0:
        return "harm", (f"The repair lowered general knowledge: {r:.0%} right against the compressed model's "
                        f"{c:.0%} (original {o:.0%}).")
    if diffs["repaired_minus_original"]["upper"] >= 0:
        return "as_good_as_original", (f"No harm: the repaired model is right on {r:.0%}, no worse than the "
                                       f"original's {o:.0%} (compressed {c:.0%}).")
    return "no_harm_from_repair", (f"The repair did no harm ({r:.0%} right against the compressed model's {c:.0%}), "
                                   f"but part of compression's loss remains: the original is right on {o:.0%}.")


def summarize(items: list[ArcItem], picks: dict, settings: dict, minutes: float, n: int, seed: int) -> dict:
    """arc.json: accuracy per model and per set, the paired differences, the plain-words result, and every
    answer (so anyone can recount it)."""
    views = [v for v in VIEWS if picks.get(v) is not None]
    answer = np.array([it.answer for it in items])
    right = {v: (np.array([p for p, _ in picks[v]]) == answer).astype(float) for v in views}
    accuracy = {v: (float(right[v].mean()) if v in right else None) for v in VIEWS}
    by_set = {name: {v: float(right[v][[i for i, it in enumerate(items) if it.set == name]].mean())
                     for v in views}
              for name in dict.fromkeys(it.set for it in items)}
    diffs = {f"{b}_minus_{a}": paired_diff(right[a], right[b])
             for a, b in (("original", "compressed"), ("compressed", "repaired"), ("original", "repaired"))
             if a in right and b in right}
    state, message = judge(accuracy, diffs)
    return {
        "check": "ARC general knowledge (report only, never used by a decision)",
        "source": {"dataset": DATASET, "split": "test", "sets": list(SETS), "licence": LICENCE},
        "model_id": settings["model_id"], "method": settings["method"], "base_nbits": settings["nbits"],
        "compute_dtype": settings["dtype"], "reference_bits": settings["reference_bits"],
        "repaired": settings["patch"] is not None, "run_outcome": settings["outcome"],
        "sample": {"n": len(items), "requested": n, "seed": seed,
                   "per_set": {name: sum(it.set == name for it in items) for name in by_set}},
        "chance": 1 / len(LETTERS), "accuracy": accuracy, "by_set": by_set, "diff": diffs,
        "state": state, "message": message, "minutes": round(minutes, 1),
        "answers": {it.qid: {"set": it.set, "answer": it.answer,
                             **{v: (picks[v][i][0] if v in picks and picks[v] is not None else None) for v in VIEWS},
                             **{f"p_right_{v}": round(picks[v][i][1], 4) for v in views}}
                    for i, it in enumerate(items)},
    }


def run(run_dir: Path, n: int = N_QUESTIONS, seed: int = SEED, device: str = "cuda", batch_size: int = 8,
        items: list[ArcItem] | None = None, loader=None, on_progress=print) -> dict:
    """Asks the run's models the ARC sample and writes RUN_DIR/arc.json. `items` and `loader` replace the
    dataset and the model rebuilding in tests."""
    import gc

    started = time.monotonic()
    settings = run_settings(run_dir)
    items = items if items is not None else load_arc(n, seed)
    loader = loader or model_loader(settings, device)
    views = [v for v in VIEWS if v != "repaired" or settings["patch"] is not None]
    picks: dict = {}
    for view in views:
        on_progress(f"ARC: asking the {view} model {len(items)} questions...")
        model, tokenizer = loader(view)
        picks[view] = score(model, tokenizer, items, device, batch_size)
        del model
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
    result = summarize(items, picks, settings, (time.monotonic() - started) / 60, n, seed)
    Path(run_dir, FILE).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    on_progress(result["message"])
    return result


def summary_line(result: dict) -> str:
    """One line for a table or a card: right answers original / compressed / repaired."""
    acc = result["accuracy"]
    return " / ".join("-" if acc.get(v) is None else f"{acc[v]:.0%}" for v in VIEWS)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", type=Path, nargs="+", help="finished fast_repair --out folders")
    parser.add_argument("--questions", type=int, default=N_QUESTIONS, help="ARC questions, half from each set")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--redo", action="store_true", help="ask again even if the folder already has arc.json")
    args = parser.parse_args(argv)
    items = None
    failed = 0
    for folder in args.runs:
        if (folder / FILE).exists() and not args.redo:
            print(f"{folder}: arc.json already there (--redo asks again)")
            continue
        items = items if items is not None else load_arc(args.questions, args.seed)
        try:
            run(folder, args.questions, args.seed, args.device, args.batch_size, items=items)
        except Exception as error:  # noqa: BLE001 -- one bad folder shouldn't stop the others
            failed += 1
            print(f"{folder}: ARC check stopped: {error}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

