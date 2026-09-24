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
    parser.add_argument("--quantized", type=Path, required=True, help="compressed question_results.jsonl")
    parser.add_argument("--out", type=Path, required=True, help="output JSON path")
    args = parser.parse_args()

    path = write_flipped_set(read_results_file(args.baseline), read_results_file(args.quantized), args.out)
    payload = json.loads(path.read_text(encoding="utf-8"))
    print(f"{payload['total_flipped']} flipped of {payload['safe_set_size']} safe -> {path}")
    for category, count in payload["flipped_per_category"].items():
        print(f"  {category:22} {count}")
