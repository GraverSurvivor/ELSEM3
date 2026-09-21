from src.data.safe_set import build_safe_set
from src.experiments.schema import QuestionResult


def _result(question_id: str, is_unknown: bool) -> QuestionResult:
    return QuestionResult(
        run_id="run1",
        condition="bf16",
        question_id=question_id,
        category="Age",
        predicted_index=2 if is_unknown else 0,
        label=2,
        unknown_index=2,
        is_unknown_predicted=is_unknown,
        is_correct=is_unknown,
    )


def test_build_safe_set_keeps_only_unknown_predictions():
    results = [
        _result("Age:1", is_unknown=True),
        _result("Age:2", is_unknown=False),
        _result("Age:3", is_unknown=True),
    ]
    assert build_safe_set(results) == {"Age:1", "Age:3"}


def test_build_safe_set_empty_when_nothing_predicted_unknown():
    results = [_result("Age:1", is_unknown=False)]
    assert build_safe_set(results) == set()
