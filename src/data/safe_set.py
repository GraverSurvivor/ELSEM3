"""Safe-set construction: the question IDs the model answers correctly ("unknown") at BF16.

This is the ground-truth reference Stage 2/3 diffs against after compression — an item that
flips off the safe set (was "unknown" at BF16, isn't anymore post-quantization) is a candidate
for the flipped set.
"""

from __future__ import annotations

from src.experiments.schema import QuestionResult


def build_safe_set(results: list[QuestionResult]) -> set[str]:
    """Returns the `question_id`s where the model predicted "unknown" — should be run on a
    BF16 baseline run's results, but doesn't enforce `condition == "bf16"` itself, since Stage 2
    also needs to run this same logic against the compressed model's results to find flips.
    """
    return {r.question_id for r in results if r.is_unknown_predicted}
