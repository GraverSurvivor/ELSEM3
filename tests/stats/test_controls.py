"""Improvement #1 statistics: BBQ bias scores, hedging, and the search-vs-baseline difference."""

import pytest

from src.experiments.schema import QuestionResult
from src.stats.bbq_bias import (
    accuracy,
    bias_score_ambig,
    bias_score_disambig,
    false_unknown_rate,
    raw_bias_score,
)
from src.stats.controls import baselines_block, hedging_block, verdict
from src.stats.metrics import is_unknown
from src.stats.recovery import recovery_difference

UNKNOWN, STEREO, ANTI = 2, 0, 1


def _row(qid, answer, condition, label=UNKNOWN, category="Age"):
    return QuestionResult(
        run_id=condition, condition=condition, question_id=qid, category=category,
        predicted_index=answer, label=label, unknown_index=UNKNOWN,
        is_unknown_predicted=answer == UNKNOWN, is_correct=answer == label,
        is_stereotyped_predicted=answer == STEREO,
    )


def _run(answers, condition, label=UNKNOWN):
    return [_row(f"Age:{i}", a, condition, label=label) for i, a in enumerate(answers)]


# --- BBQ bias scores ------------------------------------------------------------------------------

def test_raw_bias_score_ignores_unknown_answers():
    rows = _run([STEREO, STEREO, ANTI, UNKNOWN], "fp16")
    assert raw_bias_score(rows) == pytest.approx(2 * 2 / 3 - 1)


def test_ambiguous_score_is_scaled_by_error_rate():
    rows = _run([STEREO, UNKNOWN, UNKNOWN, UNKNOWN], "fp16")  # 25% wrong, all wrong answers biased
    assert accuracy(rows) == 0.75
    assert bias_score_ambig(rows) == pytest.approx(0.25 * 1.0)


def test_all_unknown_is_zero_bias_not_nan():
    assert bias_score_ambig(_run([UNKNOWN] * 3, "fp16")) == 0.0


def test_disambiguated_score_and_hedging_rate():
    rows = _run([STEREO, ANTI, UNKNOWN, UNKNOWN], "fp16_disambig", label=STEREO)
    assert bias_score_disambig(rows) == pytest.approx(0.0)  # one biased, one not, among answered
    assert false_unknown_rate(rows) == 0.5


def test_hedging_rate_needs_stated_answers():
    with pytest.raises(ValueError):
        false_unknown_rate(_run([UNKNOWN], "fp16"))


# --- search vs baseline ---------------------------------------------------------------------------

N = 200
BASE = _run([UNKNOWN] * N, "fp16")
Q3 = _run([STEREO] * N, "hqq_3bit")
SEARCH = _run([UNKNOWN] * 180 + [STEREO] * 20, "hqq_repaired")    # recovers 90%
WEAK = _run([UNKNOWN] * 60 + [STEREO] * 140, "hqq_repaired")      # recovers 30%
EQUAL = _run([UNKNOWN] * 180 + [STEREO] * 20, "hqq_repaired")     # recovers 90%


def test_recovery_difference_detects_a_better_patch():
    diff = recovery_difference(BASE, Q3, SEARCH, WEAK, is_unknown, n_resamples=300)
    assert diff.point == pytest.approx(0.6)
    assert diff.lower > 0


def test_recovery_difference_of_identical_patches_straddles_zero():
    diff = recovery_difference(BASE, Q3, SEARCH, EQUAL, is_unknown, n_resamples=300)
    assert diff.point == pytest.approx(0.0) and diff.lower <= 0 <= diff.upper


def test_baselines_block_and_verdict():
    block = baselines_block(BASE, Q3, {"search_top5": SEARCH, "random_s0": WEAK})
    assert block["random_s0"]["search_beats_this"] is True
    assert verdict({"baselines": block})["outcome"] == "claim_holds"

    block = baselines_block(BASE, Q3, {"search_top5": SEARCH, "positional": EQUAL})
    assert verdict({"baselines": block})["outcome"] == "baseline_matches"


def test_hedging_block_flags_a_repair_that_answers_unknown_where_it_is_wrong():
    fp16 = _run([STEREO] * 95 + [UNKNOWN] * 5, "fp16_disambig", label=STEREO)
    q3 = _run([STEREO] * 80 + [ANTI] * 20, "hqq_3bit_disambig", label=STEREO)
    hedger = _run([STEREO] * 60 + [UNKNOWN] * 40, "hqq_repaired_disambig", label=STEREO)
    block = hedging_block(fp16, q3, {"search_top5": hedger})
    assert block["repaired"]["search_top5"]["hedging_detected"] is True
    assert verdict({"hedging": block, "baselines": {}})["outcome"] == "hedging"
