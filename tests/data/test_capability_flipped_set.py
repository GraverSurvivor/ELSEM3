"""General-ability flipped set: disambiguated items the baseline gets right and 3-bit gets wrong."""

import pytest

from src.attribution.search_sample import draw_search_sample
from src.data.safe_set import build_capability_flipped_set, capability_flipped_payload
from src.experiments.schema import QuestionResult

UNKNOWN, PERSON_A, PERSON_B = 2, 0, 1


def _row(qid, answer, *, label=PERSON_A, condition="fp16_disambig", run_id="base", category="Age"):
    return QuestionResult(
        run_id=run_id, condition=condition, question_id=qid, category=category,
        predicted_index=answer, label=label, unknown_index=UNKNOWN,
        is_unknown_predicted=answer == UNKNOWN, is_correct=answer == label,
        is_stereotyped_predicted=answer == PERSON_A,
    )


def _q(qid, answer, **kw):
    return _row(qid, answer, condition="hqq_3bit_disambig", run_id="q3", **kw)


def test_keeps_only_correct_then_wrong():
    base = [_row("Age:1", PERSON_A), _row("Age:2", PERSON_A), _row("Age:3", PERSON_B), _row("Age:4", PERSON_A)]
    quant = [_q("Age:1", PERSON_B), _q("Age:2", UNKNOWN), _q("Age:3", PERSON_B), _q("Age:4", PERSON_A)]
    flipped = build_capability_flipped_set(base, quant)
    # 1 and 2 broke; 3 was already wrong at baseline; 4 is still right.
    assert [f["question_id"] for f in flipped] == ["Age:1", "Age:2"]


def test_refuses_ambiguous_runs():
    base = [_row("Age:1", UNKNOWN, label=UNKNOWN, condition="fp16")]
    quant = [_row("Age:1", PERSON_A, label=UNKNOWN, condition="hqq_3bit", run_id="q3")]
    with pytest.raises(ValueError, match="disambiguated"):
        build_capability_flipped_set(base, quant)


def test_payload_marks_its_question_set_and_feeds_the_sampler():
    base = [_row(f"{c}:{i}", PERSON_A, category=c) for c in ("Age", "SES") for i in range(5)]
    quant = [_q(f"{c}:{i}", PERSON_B, category=c) for c in ("Age", "SES") for i in range(5)]
    payload = capability_flipped_payload(base, quant)
    assert payload["context"] == "disambig" and payload["kind"] == "capability"
    assert payload["total_flipped"] == 10 and payload["safe_set_size"] == 10

    sample = draw_search_sample(payload, n=4, seed=0)
    assert sample["context"] == "disambig" and sample["kind"] == "capability"
    assert len(sample["question_ids"]) == 4


def test_fairness_samples_are_unchanged_by_the_new_markers():
    # headline.py re-draws committed fairness samples and compares them byte-for-byte, so a
    # fairness flipped set must not gain a context key.
    payload = {"baseline_run_id": "b", "quantized_run_id": "q", "total_flipped": 2,
               "items": [{"question_id": "Age:1", "category": "Age"}, {"question_id": "Age:2", "category": "Age"}]}
    assert "context" not in draw_search_sample(payload, n=1, seed=0)
