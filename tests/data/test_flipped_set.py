"""Flipped set: safe-set items that the compressed model now answers with the stereotype."""

import json

import pytest

from src.data.safe_set import (
    build_flipped_set,
    flipped_set_payload,
    read_results_file,
    write_flipped_set,
)
from src.experiments.schema import QuestionResult
from src.stats.metrics import ConditionMismatchError

# Answer indices for every synthetic question: 0 = stereotype, 1 = other person, 2 = unknown.
STEREOTYPE, OTHER, UNKNOWN = 0, 1, 2


def _result(question_id, answer, *, condition="fp16", run_id="base", category="Age"):
    return QuestionResult(
        run_id=run_id,
        condition=condition,
        question_id=question_id,
        category=category,
        predicted_index=answer,
        label=UNKNOWN,
        unknown_index=UNKNOWN,
        is_unknown_predicted=answer == UNKNOWN,
        is_correct=answer == UNKNOWN,
        is_stereotyped_predicted=answer == STEREOTYPE,
    )


def _quantized(question_id, answer, **kwargs):
    return _result(question_id, answer, condition="hqq_3bit", run_id="q3", **kwargs)


def _ids(flipped):
    return [f["question_id"] for f in flipped]


# --- the four row cases ----------------------------------------------------------------------


def test_unknown_to_stereotype_is_a_flip():
    flipped = build_flipped_set([_result("Age:1", UNKNOWN)], [_quantized("Age:1", STEREOTYPE)])
    assert flipped == [
        {"question_id": "Age:1", "category": "Age", "baseline_answer": UNKNOWN, "new_answer": STEREOTYPE}
    ]


def test_flip_to_other_person_is_not_counted():
    # The model got worse, but not in a biased direction -- not a fairness symptom.
    assert build_flipped_set([_result("Age:1", UNKNOWN)], [_quantized("Age:1", OTHER)]) == []


def test_still_unknown_is_not_counted():
    assert build_flipped_set([_result("Age:1", UNKNOWN)], [_quantized("Age:1", UNKNOWN)]) == []


def test_item_outside_the_safe_set_is_not_counted():
    # Already stereotyped before compression: not damage compression caused.
    assert build_flipped_set([_result("Age:1", STEREOTYPE)], [_quantized("Age:1", STEREOTYPE)]) == []


def test_mixed_run_keeps_only_the_real_flips_in_baseline_order():
    baseline = [
        _result("Age:1", UNKNOWN),
        _result("Age:2", UNKNOWN),
        _result("Age:3", UNKNOWN),
        _result("Age:4", STEREOTYPE),
        _result("Age:5", UNKNOWN),
    ]
    quantized = [
        _quantized("Age:5", STEREOTYPE),  # deliberately out of order
        _quantized("Age:1", STEREOTYPE),
        _quantized("Age:2", OTHER),
        _quantized("Age:3", UNKNOWN),
        _quantized("Age:4", STEREOTYPE),
    ]
    assert _ids(build_flipped_set(baseline, quantized)) == ["Age:1", "Age:5"]


def test_a_run_against_itself_flips_nothing():
    run = [_result(f"Age:{i}", a) for i, a in enumerate([UNKNOWN, STEREOTYPE, OTHER, UNKNOWN])]
    assert build_flipped_set(run, run) == []


# --- the guards ------------------------------------------------------------------------------


def test_truncated_quantized_run_raises_instead_of_shrinking_the_set():
    baseline = [_result("Age:1", UNKNOWN), _result("Age:2", UNKNOWN)]
    with pytest.raises(ValueError, match="missing from the"):
        build_flipped_set(baseline, [_quantized("Age:1", STEREOTYPE)])


def test_dtype_mismatch_raises():
    # fp16 vs bf16: ~1.7% of ambiguous answers differ from precision alone, not compression.
    baseline = [_result("Age:1", UNKNOWN, condition="fp16")]
    other_dtype = [_result("Age:1", STEREOTYPE, condition="bf16", run_id="b")]
    with pytest.raises(ConditionMismatchError):
        build_flipped_set(baseline, other_dtype)


def test_missing_stereotype_label_raises():
    unlabelled = QuestionResult(
        run_id="q3", condition="hqq_3bit", question_id="Age:1", category="Age",
        predicted_index=STEREOTYPE, label=UNKNOWN, unknown_index=UNKNOWN,
        is_unknown_predicted=False, is_correct=False, is_stereotyped_predicted=None,
    )
    with pytest.raises(ValueError, match="no stereotype label"):
        build_flipped_set([_result("Age:1", UNKNOWN)], [unlabelled])


# --- the JSON deliverable --------------------------------------------------------------------


def _two_category_runs():
    baseline = [
        _result("Age:1", UNKNOWN),
        _result("Age:2", UNKNOWN),
        _result("Age:3", STEREOTYPE),
        _result("SES:1", UNKNOWN, category="SES"),
    ]
    quantized = [
        _quantized("Age:1", STEREOTYPE),
        _quantized("Age:2", STEREOTYPE),
        _quantized("Age:3", STEREOTYPE),
        _quantized("SES:1", STEREOTYPE, category="SES"),
    ]
    return baseline, quantized


def test_payload_header_traces_both_runs_and_counts_by_category():
    payload = flipped_set_payload(*_two_category_runs())
    assert payload["baseline_run_id"] == "base"
    assert payload["baseline_condition"] == "fp16"
    assert payload["quantized_run_id"] == "q3"
    assert payload["quantized_condition"] == "hqq_3bit"
    assert payload["safe_set_size"] == 3
    assert payload["total_flipped"] == 3
    assert payload["flipped_per_category"] == {"Age": 2, "SES": 1}
    assert len(payload["items"]) == 3


def test_write_flipped_set_round_trips_and_leaves_no_temp_file(tmp_path):
    out = tmp_path / "flipped" / "hqq_3bit.json"
    write_flipped_set(*_two_category_runs(), out)
    assert json.loads(out.read_text(encoding="utf-8")) == flipped_set_payload(*_two_category_runs())
    assert list(out.parent.iterdir()) == [out]


def test_read_results_file_loads_a_jsonl_by_path(tmp_path):
    rows = [_result("Age:1", UNKNOWN), _result("Age:2", STEREOTYPE)]
    path = tmp_path / "question_results.jsonl"
    path.write_text("".join(json.dumps(r.to_dict()) + "\n" for r in rows), encoding="utf-8")
    assert read_results_file(path) == rows
