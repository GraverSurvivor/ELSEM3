"""The scoring loop shared by every benchmark run: baseline, quantized and repaired.

`run_baseline`, `run_quantized` and `run_repair` all do the same thing once their model is loaded:
score each category, turn the answers into `QuestionResult` rows, and checkpoint after every
category so an interrupted run can resume. Keeping that loop in one place means the three scripts
cannot drift apart -- a field added to the results, like the answer confidence, lands in all three
at once instead of in whichever script someone remembered to edit.

It also owns the run naming for the two question sets. A run on the disambiguated control
questions is filed under the same condition plus "_disambig" (e.g. `fp16_disambig`), so a results
file always says which questions it scored, and a resume can never mix the two.
"""

from __future__ import annotations

from src.data.bbq_loader import BBQItem
from src.eval.option_scorer import score_items_with_confidence
from src.experiments.logger import write_question_results
from src.experiments.schema import QuestionResult

DISAMBIG_SUFFIX = "_disambig"


def condition_for(base_condition: str, context: str) -> str:
    """The condition label for a run of `base_condition` on the `context` question set."""
    if context == "ambig":
        return base_condition
    if context == "disambig":
        return base_condition + DISAMBIG_SUFFIX
    raise ValueError(f"Unknown context {context!r}; expected 'ambig' or 'disambig'")


def check_resume_id(run_id: str, condition: str) -> None:
    """Refuses to resume a run that wasn't started under exactly `condition`.

    A plain prefix test isn't enough: `fp16_disambig_2026...` starts with `fp16_` too, so resuming
    it as an ambiguous fp16 run would append ambiguous answers to a disambiguated file. Run ids are
    `<condition>_<timestamp>`, so what follows the condition must be the timestamp.
    """
    prefix = f"{condition}_"
    rest = run_id[len(prefix):] if run_id.startswith(prefix) else ""
    if not rest[:1].isdigit():
        raise SystemExit(
            f"Run {run_id!r} was not started as a {condition!r} run. Resume it with the options it "
            "was started with, or start a fresh run."
        )


def build_results(
    items: list[BBQItem],
    predictions: list[int],
    confidences: list[float] | None,
    run_id: str,
    condition: str,
) -> list[QuestionResult]:
    """One `QuestionResult` per item. `confidences` may be None for callers that don't have it."""
    confidences = confidences if confidences is not None else [None] * len(items)
    return [
        QuestionResult(
            run_id=run_id,
            condition=condition,
            question_id=item.question_id,
            category=item.category,
            predicted_index=predicted_index,
            label=item.label,
            unknown_index=item.unknown_index,
            is_unknown_predicted=(predicted_index == item.unknown_index),
            is_correct=(predicted_index == item.label),
            is_stereotyped_predicted=(predicted_index == item.stereotype_index),
            confidence=confidence,
        )
        for item, predicted_index, confidence in zip(items, predictions, confidences)
    ]


def score_categories(
    model,
    tokenizer,
    items_by_category: dict[str, list[BBQItem]],
    device: str,
    run_id: str,
    condition: str,
    batch_size: int,
    completed: list[QuestionResult] | None = None,
) -> list[QuestionResult]:
    """Scores every category not already in `completed`, checkpointing after each one.

    A category is only ever written once it is fully scored, so anything already in the file is
    complete -- which is what makes skipping it on resume safe rather than a source of
    half-scored categories.
    """
    all_results: list[QuestionResult] = list(completed or [])
    finished = {result.category for result in all_results}
    if finished:
        print(f"Resuming: {sorted(finished)} already scored, skipping.")

    for category, items in items_by_category.items():
        if category in finished:
            continue
        predictions, confidences = score_items_with_confidence(
            model, tokenizer, items, device, batch_size=batch_size, desc=category
        )
        all_results.extend(build_results(items, predictions, confidences, run_id, condition))
        write_question_results(all_results, run_id)
    return all_results
