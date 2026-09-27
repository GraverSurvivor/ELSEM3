"""run_detection: the flipped set and its statistics from one call, matching the pieces it wraps."""

import json

import pytest

from src.data.safe_set import build_flipped_set
from src.experiments.schema import QuestionResult
from src.stats import stereotype_chisq, usr_delta
from src.stats.detect import run_detection
from src.stats.metrics import ConditionMismatchError

STEREOTYPE, OTHER, UNKNOWN = 0, 1, 2
FAST = dict(n_resamples=200)


def _run(answers, *, condition, run_id, category="Age"):
    return [
        QuestionResult(
            run_id=run_id,
            condition=condition,
            question_id=f"{category}:{i}",
            category=category,
            predicted_index=answer,
            label=UNKNOWN,
            unknown_index=UNKNOWN,
            is_unknown_predicted=answer == UNKNOWN,
            is_correct=answer == UNKNOWN,
            is_stereotyped_predicted=answer == STEREOTYPE,
        )
        for i, answer in enumerate(answers)
    ]


def _pair():
    # 10 questions, 8 safe. Compression flips 3 safe ones to the stereotype and 1 to the other
    # person (not a symptom); the 2 unsafe ones stay stereotyped.
    base = [UNKNOWN] * 8 + [STEREOTYPE] * 2
    comp = [STEREOTYPE] * 3 + [OTHER] + [UNKNOWN] * 4 + [STEREOTYPE] * 2
    return (
        _run(base, condition="fp16", run_id="base") + _run(base, condition="fp16", run_id="base", category="SES"),
        _run(comp, condition="hqq_3bit", run_id="q3") + _run(comp, condition="hqq_3bit", run_id="q3", category="SES"),
    )


def test_flipped_set_counts_only_flips_to_the_stereotype():
    result = run_detection(*_pair(), **FAST)
    assert result.safe_set_size == 16
    assert result.total_flipped == 6
    assert result.flipped_per_category == {"Age": 3, "SES": 3}
    assert (result.baseline_run_id, result.compressed_run_id) == ("base", "q3")


def test_it_returns_exactly_what_the_wrapped_functions_return():
    base, comp = _pair()
    result = run_detection(base, comp, seed=4, **FAST)
    assert result.flipped == build_flipped_set(base, comp)
    assert result.usr == usr_delta(base, comp, seed=4, **FAST)
    assert result.chisq.p_value == stereotype_chisq(base, comp).p_value


def test_usr_drop_and_effect_size_point_the_right_way():
    result = run_detection(*_pair(), **FAST)
    assert result.usr.delta == pytest.approx(0.4 - 0.8)
    assert result.srs.delta == pytest.approx(0.5 - 0.2)
    assert 0 < result.cramers_v <= 1


def test_it_serialises_for_results():
    payload = json.loads(json.dumps(run_detection(*_pair(), **FAST).to_dict()))
    assert payload["total_flipped"] == 6
    assert set(payload) >= {"usr", "srs", "chisq", "cramers_v", "flipped", "flipped_per_category"}


def test_runs_at_different_dtypes_raise():
    base, _ = _pair()
    bf16 = _run([UNKNOWN] * 10, condition="bf16", run_id="b")
    with pytest.raises(ConditionMismatchError):
        run_detection(base[:10], bf16, **FAST)


def test_a_truncated_compressed_run_raises():
    base, comp = _pair()
    with pytest.raises(ValueError, match="missing from the"):
        run_detection(base, comp[:15], **FAST)
