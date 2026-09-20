import pytest

from src.experiments.schema import QuestionResult, summarize_category


def _result(question_id: str, category: str, is_unknown: bool, is_stereotyped: bool | None = None) -> QuestionResult:
    return QuestionResult(
        run_id="run1",
        condition="bf16",
        question_id=question_id,
        category=category,
        predicted_index=2 if is_unknown else 0,
        label=2,
        unknown_index=2,
        is_unknown_predicted=is_unknown,
        is_correct=is_unknown,
        is_stereotyped_predicted=is_stereotyped,
    )


def test_summarize_category_computes_usr():
    results = [
        _result("Age:1", "Age", is_unknown=True),
        _result("Age:2", "Age", is_unknown=True),
        _result("Age:3", "Age", is_unknown=False),
        _result("Age:4", "Age", is_unknown=False),
    ]
    summary = summarize_category(results, run_id="run1", condition="bf16", category="Age")
    assert summary.n_items == 4
    assert summary.n_unknown == 2
    assert summary.usr == 0.5


def test_summarize_category_filters_to_requested_category():
    results = [
        _result("Age:1", "Age", is_unknown=True),
        _result("Religion:1", "Religion", is_unknown=False),
    ]
    summary = summarize_category(results, run_id="run1", condition="bf16", category="Age")
    assert summary.n_items == 1


def test_summarize_category_srs_is_none_without_stereotype_data():
    results = [_result("Age:1", "Age", is_unknown=True, is_stereotyped=None)]
    summary = summarize_category(results, run_id="run1", condition="bf16", category="Age")
    assert summary.n_stereotyped is None
    assert summary.srs is None


def test_summarize_category_srs_computed_when_stereotype_data_present():
    results = [
        _result("Age:1", "Age", is_unknown=True, is_stereotyped=False),
        _result("Age:2", "Age", is_unknown=False, is_stereotyped=True),
    ]
    summary = summarize_category(results, run_id="run1", condition="bf16", category="Age")
    assert summary.n_stereotyped == 1
    assert summary.srs == 0.5


def test_summarize_category_raises_for_missing_category():
    results = [_result("Age:1", "Age", is_unknown=True)]
    with pytest.raises(ValueError):
        summarize_category(results, run_id="run1", condition="bf16", category="Religion")
