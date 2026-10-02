"""Flip predictability, sensitivity, per-group breakdown, and the concentrated-vs-diffuse measures."""

import pytest

from src.experiments.schema import QuestionResult
from src.stats.dissociation import concentration, overlap, trajectory
from src.stats.flip_predictability import analyze, auroc, coverage_curve, quartile_rates
from src.stats.groups import breakdown, group_key
from src.stats.sensitivity import sample_representativeness, subset_stability

UNKNOWN, STEREO, ANTI = 2, 0, 1


def _row(qid, answer, condition, confidence=None, category="Age"):
    return QuestionResult(
        run_id=condition, condition=condition, question_id=qid, category=category,
        predicted_index=answer, label=UNKNOWN, unknown_index=UNKNOWN,
        is_unknown_predicted=answer == UNKNOWN, is_correct=answer == UNKNOWN,
        is_stereotyped_predicted=answer == STEREO, confidence=confidence,
    )


# --- flip predictability --------------------------------------------------------------------------

def test_auroc_perfect_random_and_ties():
    assert auroc([0.9, 0.8, 0.2, 0.1], [True, True, False, False]) == 1.0
    assert auroc([0.1, 0.2, 0.8, 0.9], [True, True, False, False]) == 0.0
    assert auroc([0.5, 0.5], [True, False]) == 0.5


def test_coverage_probes_least_confident_first():
    confidence = [0.4, 0.5, 0.9, 0.95]
    flipped = [True, True, False, False]
    curve = {p["probed"]: p["flips_caught"] for p in coverage_curve(confidence, flipped, points=(0.5, 1.0))}
    assert curve == {0.5: 1.0, 1.0: 1.0}


def test_quartiles_report_the_low_vs_high_ratio():
    confidence = list(range(8))
    flipped = [True, True, False, False, False, False, False, True]
    q = quartile_rates(confidence, flipped)
    assert q["least_to_most_confident"] == [1.0, 0.0, 0.0, 0.5]
    assert q["low_vs_high_ratio"] == 2.0


def test_analyze_refuses_confidence_from_a_different_model():
    committed = [_row(f"Age:{i}", UNKNOWN, "fp16") for i in range(4)]
    other = [_row(f"Age:{i}", STEREO if i == 0 else UNKNOWN, "fp16", confidence=0.9) for i in range(4)]
    quant = [_row(f"Age:{i}", STEREO, "hqq_3bit") for i in range(4)]
    with pytest.raises(ValueError, match="differ"):
        analyze(other, committed, quant)


def test_analyze_end_to_end():
    committed = [_row(f"Age:{i}", UNKNOWN, "fp16") for i in range(8)]
    conf = [_row(f"Age:{i}", UNKNOWN, "fp16", confidence=0.4 + 0.07 * i) for i in range(8)]
    # the four least-confident items flip
    quant = [_row(f"Age:{i}", STEREO if i < 4 else UNKNOWN, "hqq_3bit") for i in range(8)]
    result = analyze(conf, committed, quant)
    assert result["flipped"] == 4 and result["auroc"] == 1.0


# --- sensitivity -------------------------------------------------------------------------------------

def test_subset_recovery_matches_full_when_uniform():
    base = [_row(f"Age:{i}", UNKNOWN, "fp16") for i in range(100)]
    q3 = [_row(f"Age:{i}", STEREO, "hqq_3bit") for i in range(100)]
    rep = [_row(f"Age:{i}", UNKNOWN if i % 2 else STEREO, "hqq_repaired") for i in range(100)]
    rows = subset_stability(base, q3, rep, fractions=(0.5,), draws=5)
    assert 0.3 < rows[0]["mean"] < 0.7


def test_sample_representativeness_gap():
    repaired = {f"Age:{i}": _row(f"Age:{i}", UNKNOWN if i < 6 else STEREO, "hqq_repaired") for i in range(10)}
    out = sample_representativeness([f"Age:{i}" for i in range(10)], {"Age:0", "Age:1"}, repaired)
    assert out["recovered_in_sample"] == 1.0 and out["recovered_outside_sample"] == 0.5
    assert out["gap"] == 0.5


# --- groups ---------------------------------------------------------------------------------------------

def test_group_key():
    assert group_key(("Black", "African American")) == "Black / African American"
    assert group_key(()) == "(unlabelled)"


def test_breakdown_flags_a_group_the_repair_left_worse_than_the_original():
    n = 60
    ids = [f"Age:{i}" for i in range(2 * n)]
    groups = {qid: ("Age", "old" if i < n else "young") for i, qid in enumerate(ids)}
    base = [_row(q, UNKNOWN, "fp16") for q in ids]
    quant = [_row(q, STEREO, "hqq_3bit") for q in ids]
    # repair fixes "young" fully but leaves half of "old" stereotyped
    rep = [_row(q, STEREO if (i < n and i % 2) else UNKNOWN, "hqq_repaired") for i, q in enumerate(ids)]
    rows = {r["group"]: r for r in breakdown(groups, {"baseline": base, "hqq_3bit": quant, "hqq_repaired": rep})}
    assert rows["old"]["worse_than_baseline"] is True
    assert rows["young"]["worse_than_baseline"] is False


# --- concentrated vs diffuse ------------------------------------------------------------------------

def test_concentration_shares():
    trials = [{"round": 1, "gain": g} for g in (0.5, 0.3, 0.1, 0.1, -0.2)] + [{"round": 2, "gain": 9.0}]
    c = concentration(trials, top_k=(1, 2))
    assert c["top1_share"] == pytest.approx(0.5) and c["top2_share"] == pytest.approx(0.8)
    assert c["n_helpful"] == 4 and c["n_candidates"] == 5


def test_trajectory_rounds_to_target():
    t = trajectory({"start_usr": 0.0, "usr_after_round": [0.5, 0.7, 0.8], "kept": ["a", "b", "c"]})
    assert t["rounds_to_75pct"] == 3


def test_overlap():
    assert overlap(["a", "b"], ["b", "c"]) == {"shared": ["b"], "jaccard": pytest.approx(1 / 3)}
