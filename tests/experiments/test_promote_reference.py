"""Validation before a run becomes the shared reference.

Everything downstream is defined relative to the reference run, so promoting a truncated or
duplicated one wouldn't crash anything -- it would just quietly shift what "the baseline said"
means for the rest of the project. These tests cover the ways a run can look finished without
being finished.
"""

import pytest

from src.data.bbq_loader import CATEGORIES
from src.experiments import logger, promote_reference
from src.experiments.schema import CategorySummary, QuestionResult

# CATEGORIES maps name -> config name; these tests need positional access.
CATEGORY_NAMES = list(CATEGORIES)
RUN_ID = "fp16_20260921T180125Z"
PER_CATEGORY = 4


def _result(category: str, index: int, condition: str = "fp16") -> QuestionResult:
    return QuestionResult(
        run_id=RUN_ID,
        condition=condition,
        question_id=f"{category}:{index}",
        category=category,
        predicted_index=2,
        label=2,
        unknown_index=2,
        is_unknown_predicted=True,
        is_correct=True,
        is_stereotyped_predicted=False,
    )


def _summary(category: str, n_items: int) -> CategorySummary:
    return CategorySummary(
        run_id=RUN_ID,
        condition="fp16",
        category=category,
        n_items=n_items,
        n_unknown=n_items,
        usr=1.0,
        n_stereotyped=0,
        srs=0.0,
    )


def _write_run(results: list[QuestionResult], summaries: list[CategorySummary]) -> None:
    logger.write_question_results(results, RUN_ID)
    logger.write_category_summary(summaries, RUN_ID)


def _complete_results() -> list[QuestionResult]:
    return [_result(category, i) for category in CATEGORIES for i in range(PER_CATEGORY)]


def _complete_summaries() -> list[CategorySummary]:
    return [_summary(category, PER_CATEGORY) for category in CATEGORIES] + [
        _summary("overall", PER_CATEGORY * len(CATEGORIES))
    ]


@pytest.fixture(autouse=True)
def results_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(logger, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(promote_reference, "RESULTS_DIR", tmp_path)
    return tmp_path


def test_a_complete_run_validates():
    _write_run(_complete_results(), _complete_summaries())
    assert promote_reference.validate_run(RUN_ID) == []


def test_missing_category_is_caught():
    # The exact shape an interrupted run leaves behind, now that results checkpoint per category.
    partial = [r for r in _complete_results() if r.category != CATEGORY_NAMES[-1]]
    summaries = [_summary(c, PER_CATEGORY) for c in CATEGORY_NAMES[:-1]]
    summaries.append(_summary("overall", len(partial)))
    _write_run(partial, summaries)

    problems = promote_reference.validate_run(RUN_ID)
    assert any("missing categories" in p for p in problems)


def test_duplicate_question_ids_are_caught():
    # What a badly-resumed run would look like: valid records, inflated counts.
    results = _complete_results()
    duplicated = results + results[:2]
    summaries = [_summary(c, PER_CATEGORY) for c in CATEGORIES] + [_summary("overall", len(duplicated))]
    _write_run(duplicated, summaries)

    problems = promote_reference.validate_run(RUN_ID)
    assert any("duplicate" in p for p in problems)


def test_mixed_conditions_are_caught():
    # Two precisions in one file is the confound the condition label exists to surface.
    results = _complete_results()
    results[0] = _result(CATEGORY_NAMES[0], 0, condition="bf16")
    _write_run(results, _complete_summaries())

    problems = promote_reference.validate_run(RUN_ID)
    assert any("mixed conditions" in p for p in problems)


def test_condition_not_matching_the_run_id_is_caught():
    results = [_result(c, i, condition="bf16") for c in CATEGORIES for i in range(PER_CATEGORY)]
    _write_run(results, _complete_summaries())

    problems = promote_reference.validate_run(RUN_ID)
    assert any("doesn't match recorded run id" in p for p in problems)


def test_summary_disagreeing_with_the_records_is_caught():
    summaries = [_summary(c, PER_CATEGORY) for c in CATEGORIES] + [_summary("overall", 9999)]
    _write_run(_complete_results(), summaries)

    problems = promote_reference.validate_run(RUN_ID)
    assert any("overall n_items" in p for p in problems)


def test_per_category_summary_mismatch_is_caught():
    summaries = [_summary(c, PER_CATEGORY) for c in CATEGORIES]
    summaries[0] = _summary(CATEGORY_NAMES[0], PER_CATEGORY + 5)
    summaries.append(_summary("overall", PER_CATEGORY * len(CATEGORIES)))
    _write_run(_complete_results(), summaries)

    problems = promote_reference.validate_run(RUN_ID)
    assert any(CATEGORY_NAMES[0] in p and "summary says" in p for p in problems)


def test_nonexistent_run_is_caught():
    assert promote_reference.validate_run("never-ran") == ["no question_results.jsonl for run 'never-ran'"]


def test_promote_copies_both_files(results_dir):
    _write_run(_complete_results(), _complete_summaries())

    destination = promote_reference.promote(RUN_ID, "baseline_fp16")

    assert destination == results_dir / "reference" / "baseline_fp16"
    assert (destination / "question_results.jsonl").exists()
    assert (destination / "category_summary.csv").exists()


def test_promote_refuses_an_invalid_run():
    partial = [r for r in _complete_results() if r.category != CATEGORY_NAMES[-1]]
    _write_run(partial, [_summary("overall", len(partial))])

    with pytest.raises(ValueError, match="not fit to be a reference"):
        promote_reference.promote(RUN_ID, "baseline_fp16")


def test_promoting_twice_replaces_rather_than_nesting(results_dir):
    # A re-promotion should overwrite cleanly, not leave the previous copy nested inside.
    _write_run(_complete_results(), _complete_summaries())

    promote_reference.promote(RUN_ID, "baseline_fp16")
    destination = promote_reference.promote(RUN_ID, "baseline_fp16")

    assert not (destination / "baseline_fp16").exists()
    assert len(logger.read_question_results(RUN_ID)) == PER_CATEGORY * len(CATEGORIES)


def test_a_promoted_copy_validates_under_its_own_name():
    """A promoted reference lives under a chosen name like "baseline_fp16", not a run id.

    Validating it must key off the run id recorded in the records, not the directory name --
    otherwise every promoted copy reports as broken, which is exactly the moment re-validating
    matters most (e.g. after hand-copying the files between machines).
    """
    _write_run(_complete_results(), _complete_summaries())
    promote_reference.promote(RUN_ID, "baseline_fp16")

    assert promote_reference.validate_run("reference/baseline_fp16") == []


def test_records_spanning_multiple_run_ids_are_caught():
    # Two different runs concatenated into one file would pass every per-category check while
    # being a blend of two conditions' worth of answers.
    results = _complete_results()
    merged = results + [
        QuestionResult(**{**r.to_dict(), "run_id": "fp16_20260922T000000Z", "question_id": f"x{i}"})
        for i, r in enumerate(results[:2])
    ]
    _write_run(merged, _complete_summaries())

    problems = promote_reference.validate_run(RUN_ID)
    assert any("multiple run ids" in p for p in problems)
