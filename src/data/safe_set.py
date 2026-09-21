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

from src.experiments.schema import QuestionResult


def build_safe_set(results: list[QuestionResult]) -> set[str]:
    """Returns the `question_id`s where the model predicted "unknown".

    Intended for the uncompressed baseline run's results, but deliberately doesn't enforce a
    particular `condition`: Stage 2 runs this same logic over the compressed model's results to
    find which safe-set items flipped.
    """
    return {r.question_id for r in results if r.is_unknown_predicted}
