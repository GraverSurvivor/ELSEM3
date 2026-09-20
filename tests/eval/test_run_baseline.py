import pytest

from src.eval.run_baseline import _aggregate_overall
from src.experiments.schema import QuestionResult


def _result(category: str, is_unknown: bool, is_stereotyped: bool | None) -> QuestionResult:
    return QuestionResult(
        run_id="run1",
        condition="bf16",
        question_id=f"{category}:1",
        category=category,
        predicted_index=2 if is_unknown else 0,
        label=2,
        unknown_index=2,
        is_unknown_predicted=is_unknown,
        is_correct=is_unknown,
        is_stereotyped_predicted=is_stereotyped,
    )


def test_aggregate_overall_spans_every_category():
    results = [
        _result("Age", is_unknown=True, is_stereotyped=False),
        _result("Religion", is_unknown=False, is_stereotyped=True),
        _result("SES", is_unknown=True, is_stereotyped=False),
    ]
    overall = _aggregate_overall(results, run_id="run1", condition="bf16")

    assert overall.category == "overall"
    assert overall.n_items == 3
    assert overall.n_unknown == 2
    assert overall.usr == pytest.approx(2 / 3)
    assert overall.n_stereotyped == 1
    assert overall.srs == pytest.approx(1 / 3)


def test_aggregate_overall_srs_none_without_full_stereotype_data():
    results = [
        _result("Age", is_unknown=True, is_stereotyped=None),
        _result("Religion", is_unknown=False, is_stereotyped=True),
    ]
    overall = _aggregate_overall(results, run_id="run1", condition="bf16")

    assert overall.n_stereotyped is None
    assert overall.srs is None
