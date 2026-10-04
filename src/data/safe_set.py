"""Safe-set construction: the question IDs the uncompressed model answers correctly ("unknown").

This is the ground-truth reference Stage 2/3 diffs against after compression — an item that
flips off the safe set (was "unknown" uncompressed, isn't anymore post-quantization) is a
candidate for the flipped set.

The diff is only meaningful if both sides ran at the same precision: an item that flips because
the baseline was fp16 and the compressed run was bf16 is not quantization damage, but nothing
downstream can tell the difference. Both runs record their dtype as `condition` — check they
match before treating a flip as a symptom (see CLAUDE.md > Precision).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from ..experiments.logger import _write_atomically
from ..experiments.schema import QuestionResult
from ..stats.metrics import align_by_question, assert_comparable, is_stereotype


def build_safe_set(results: list[QuestionResult]) -> set[str]:
    """Returns the `question_id`s where the model predicted "unknown".

    Intended for the uncompressed baseline run's results, but deliberately doesn't enforce a
    particular `condition`: Stage 2 runs this same logic over the compressed model's results to
    find which safe-set items flipped.
    """
    return {r.question_id for r in results if r.is_unknown_predicted}


# --- Flipped set (Stage 3: Detect) -------------------------------------------------------------
#
# The flipped set is the symptom evidence Stage 4's search runs on: safe-set items the compressed
# model now answers with the stereotype. It is defined against the baseline, so both runs must
# share a compute dtype and cover exactly the same questions -- both guards live in
# src/stats/metrics.py and are reused here rather than written twice.

def build_flipped_set(
    baseline: list[QuestionResult], quantized: list[QuestionResult]
) -> list[dict]:
    """Safe-set items that the compressed model now answers with the stereotyped option.

    Flipping to the *other*, non-stereotyped person is deliberately not counted: the model got
    worse, but that is not a fairness symptom, and Stage 4 would go hunting for components
    responsible for bias that isn't there.

    Raises if the runs were taken at different dtypes, or if either run is missing questions the
    other has -- a truncated run would otherwise just look like a smaller flipped set.
    """
    assert_comparable(baseline, quantized)
    before, after = align_by_question(baseline, quantized)

    return [
        {
            "question_id": b.question_id,
            "category": b.category,
            "baseline_answer": b.predicted_index,
            "new_answer": a.predicted_index,
        }
        for b, a in zip(before, after)
        if b.is_unknown_predicted and is_stereotype(a)
    ]


def flipped_set_payload(
    baseline: list[QuestionResult], quantized: list[QuestionResult]
) -> dict:
    """The flipped set plus the header saying which two runs produced it.

    Without the header the file can't be traced back to its runs, and regenerating it costs a
    GPU session.
    """
    flipped = build_flipped_set(baseline, quantized)
    return {
        "baseline_run_id": baseline[0].run_id,
        "baseline_condition": baseline[0].condition,
        "quantized_run_id": quantized[0].run_id,
        "quantized_condition": quantized[0].condition,
        "safe_set_size": len(build_safe_set(baseline)),
        "total_flipped": len(flipped),
        "flipped_per_category": dict(sorted(Counter(f["category"] for f in flipped).items())),
        "items": flipped,
    }


# --- General-ability flipped set (revised plan: concentrated vs diffuse damage) ------------------
#
# The control counterpart of the flipped set. On the disambiguated questions the text states who
# did what, so the right answer is a real person; items the uncompressed model gets right and the
# 3-bit model gets wrong are general-ability damage, with no fairness content at all. Running the
# same greedy search on these asks where *general* damage lives -- and comparing that with where
# the fairness damage lives is the project's concentrated-vs-diffuse question.

def build_capability_flipped_set(
    baseline: list[QuestionResult], quantized: list[QuestionResult]
) -> list[dict]:
    """Disambiguated items the baseline answers correctly and the compressed model doesn't."""
    assert_comparable(baseline, quantized)
    before, after = align_by_question(baseline, quantized)
    if any(b.label == b.unknown_index for b in before):
        raise ValueError(
            "baseline contains items whose correct answer is 'unknown' -- the general-ability set "
            "must be built from disambiguated runs (--context disambig), not ambiguous ones"
        )
    return [
        {
            "question_id": b.question_id,
            "category": b.category,
            "baseline_answer": b.predicted_index,
            "new_answer": a.predicted_index,
        }
        for b, a in zip(before, after)
        if b.is_correct and not a.is_correct
    ]


def capability_flipped_payload(
    baseline: list[QuestionResult], quantized: list[QuestionResult]
) -> dict:
    """Same shape as `flipped_set_payload`, so search_sample.py draws from it unchanged, plus the
    `context`/`kind` markers run_search uses to load the ids from the disambiguated set."""
    flipped = build_capability_flipped_set(baseline, quantized)
    return {
        "kind": "capability",
        "context": "disambig",
        "baseline_run_id": baseline[0].run_id,
        "baseline_condition": baseline[0].condition,
        "quantized_run_id": quantized[0].run_id,
        "quantized_condition": quantized[0].condition,
        "safe_set_size": sum(r.is_correct for r in baseline),
        "total_flipped": len(flipped),
        "flipped_per_category": dict(sorted(Counter(f["category"] for f in flipped).items())),
        "items": flipped,
    }


def hedging_probe_payload(baseline: list[QuestionResult]) -> dict:
    """Disambiguated questions the original model answers correctly: the probe the search's hedging
    penalty scores. "Unknown" on any of these is a hedge, since the text states the answer and the
    original found it. Same shape as the flipped sets, so search_sample.py draws from it unchanged.

    The Phase 2 hedging check found the 5-module repair says "unknown" on 20.3% of disambiguated
    questions against the original's 7.3%; this is the set that lets the search see that happening.
    """
    if any(r.label == r.unknown_index for r in baseline):
        raise ValueError("the hedging probe must be built from a disambiguated run (--context disambig)")
    items = [{"question_id": r.question_id, "category": r.category, "baseline_answer": r.predicted_index}
             for r in baseline if r.is_correct]
    return {
        "kind": "hedging_probe",
        "context": "disambig",
        "baseline_run_id": baseline[0].run_id,
        "baseline_condition": baseline[0].condition,
        "quantized_run_id": None,
        "quantized_condition": None,
        "safe_set_size": len(baseline),
        "total_flipped": len(items),
        "flipped_per_category": dict(sorted(Counter(i["category"] for i in items).items())),
        "items": items,
    }


def write_flipped_set(
    baseline: list[QuestionResult], quantized: list[QuestionResult], out_path: Path
) -> Path:
    """Writes the flipped-set JSON (the Week 3 deliverable) atomically."""
    payload = flipped_set_payload(baseline, quantized)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return _write_atomically(out_path, lambda handle: json.dump(payload, handle, indent=2))


def read_results_file(path: Path) -> list[QuestionResult]:
    """Loads a question_results.jsonl by path -- works for results/reference/ runs, which
    logger.read_question_results can't reach because it resolves by run_id under results/."""
    with Path(path).open("r", encoding="utf-8") as f:
        return [QuestionResult(**json.loads(line)) for line in f if line.strip()]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build the flipped set from two runs.")
    parser.add_argument("--baseline", type=Path, required=True, help="baseline question_results.jsonl")
    parser.add_argument("--quantized", type=Path, help="compressed question_results.jsonl (not used by hedging-probe)")
    parser.add_argument("--out", type=Path, required=True, help="output JSON path")
    parser.add_argument(
        "--kind",
        default="fairness",
        choices=("fairness", "capability", "hedging-probe"),
        help="fairness: safe-set items that flip to the stereotype (ambiguous runs); "
        "capability: correct items that become wrong (disambiguated runs); "
        "hedging-probe: disambiguated items the baseline answers correctly",
    )
    args = parser.parse_args()
    if args.kind != "hedging-probe" and args.quantized is None:
        parser.error(f"--kind {args.kind} needs --quantized")

    baseline_rows = read_results_file(args.baseline)
    quantized_rows = read_results_file(args.quantized) if args.quantized else None
    if args.kind == "hedging-probe":
        payload = hedging_probe_payload(baseline_rows)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        path = _write_atomically(args.out, lambda handle: json.dump(payload, handle, indent=2))
    elif args.kind == "fairness":
        path = write_flipped_set(baseline_rows, quantized_rows, args.out)
    else:
        payload = capability_flipped_payload(baseline_rows, quantized_rows)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        path = _write_atomically(args.out, lambda handle: json.dump(payload, handle, indent=2))
    payload = json.loads(path.read_text(encoding="utf-8"))
    print(f"{payload['total_flipped']} flipped of {payload['safe_set_size']} safe -> {path}")
    for category, count in payload["flipped_per_category"].items():
        print(f"  {category:22} {count}")
