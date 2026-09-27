"""Recovery fraction and the Stage 5 report, on synthetic runs with known answers."""

import pytest

from src.stats.metrics import ConditionMismatchError, is_stereotype, is_unknown
from src.stats.recovery import recovery_fraction
from src.stats.repair_report import build_bundle, build_comparison

from helpers import make_rows, pattern

FAST = dict(n_resamples=200)


def _runs(n_base_unknown, n_3bit_unknown, n_rep_unknown, n=100):
    """Three runs over the same n questions; the rest of each run answers the stereotype."""
    return (
        make_rows(pattern(n_base_unknown, n - n_base_unknown), condition="fp16"),
        make_rows(pattern(n_3bit_unknown, n - n_3bit_unknown), condition="hqq_3bit"),
        make_rows(pattern(n_rep_unknown, n - n_rep_unknown), condition="hqq_repaired"),
    )


# --- recovery_fraction ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "repaired, expected",
    [(40, 0.0), (60, 0.5), (80, 1.0), (84, 1.1)],
)
def test_recovery_is_the_share_of_the_gap_closed(repaired, expected):
    # Baseline 80% unknown, 3-bit 40%: the gap is 40 points.
    result = recovery_fraction(*_runs(80, 40, repaired), is_unknown, **FAST)
    assert result.point == pytest.approx(expected)


def test_the_same_formula_works_for_a_metric_that_rose():
    # SRS: baseline 20%, 3-bit 60%, repaired 30% -> 30 of the 40-point rise undone.
    result = recovery_fraction(*_runs(80, 40, 70), is_stereotype, **FAST)
    assert result.point == pytest.approx(0.75)


def test_the_interval_brackets_the_point_and_is_reproducible():
    runs = _runs(80, 40, 70)
    a = recovery_fraction(*runs, is_unknown, seed=3, **FAST)
    b = recovery_fraction(*runs, is_unknown, seed=3, **FAST)
    assert a.lower <= a.point <= a.upper
    assert (a.lower, a.upper) == (b.lower, b.upper)


def test_no_gap_to_recover_raises():
    with pytest.raises(ValueError, match="nothing to recover"):
        recovery_fraction(*_runs(50, 50, 60), is_unknown, **FAST)


def test_runs_at_different_dtypes_raise():
    base, deg, rep = _runs(80, 40, 70)
    bf16 = make_rows(pattern(70, 30), condition="bf16")
    with pytest.raises(ConditionMismatchError):
        recovery_fraction(base, deg, bf16, is_unknown, **FAST)


def test_a_repaired_run_missing_questions_raises():
    base, deg, rep = _runs(80, 40, 70)
    with pytest.raises(ValueError, match="missing from the"):
        recovery_fraction(base, deg, rep[:50], is_unknown, **FAST)


# --- the report -------------------------------------------------------------------------------


def _two_category_runs():
    runs = []
    for condition, (age, ses) in {"fp16": (80, 90), "hqq_3bit": (40, 50), "hqq_repaired": (80, 70)}.items():
        runs.append(
            make_rows(pattern(age, 100 - age), condition=condition, category="Age", prefix="a")
            + make_rows(pattern(ses, 100 - ses), condition=condition, category="SES", prefix="s")
        )
    return runs


def test_comparison_covers_overall_and_every_category():
    comparison = build_comparison(*_two_category_runs())
    assert set(comparison) == {"overall", "Age", "SES"}
    assert comparison["Age"]["recovery"]["usr"]["point"] == pytest.approx(1.0)
    assert comparison["SES"]["recovery"]["usr"]["point"] == pytest.approx(0.5)
    assert comparison["overall"]["n_items"] == 200
    assert comparison["overall"]["repaired"]["usr"] == pytest.approx(0.75)
    assert comparison["Age"]["repair_effect"]["usr"]["delta"] == pytest.approx(0.40)
    assert comparison["SES"]["gap_left"]["usr"]["delta"] == pytest.approx(-0.20)


def test_bundle_carries_the_four_reported_numbers():
    comparison = build_comparison(*_two_category_runs())
    summary = {
        "run_id": "hqq_repaired_X",
        "search_run_id": "search_Y",
        "modules": ["m1", "m2"],
        "restored": {"pct_params_restored": 3.4, "pct_modules_restored": 3.9,
                     "modules_restored": 5, "modules_total": 128},
        "memory": {"deployed_overhead_bytes": 70 * 2**20, "gpu_allocated_before_bytes": 2000 * 2**20,
                   "gpu_allocated_after_bytes": 2074 * 2**20, "gpu_delta_bytes": 74 * 2**20},
        "wall_clock_seconds": {"repair": 3.0, "scoring": 1197.0, "cycle": 1200.0},
        "resumed": False,
    }
    bundle = build_bundle(comparison, summary)
    assert bundle["fairness_recovered"]["usr"]["recovery"]["point"] == pytest.approx(0.75)
    assert bundle["model_restored"] == {"pct_params": 3.4, "pct_modules": 3.9, "modules": "5/128"}
    assert bundle["memory_overhead"]["deployed_mb"] == pytest.approx(70.0)
    assert bundle["memory_overhead"]["pct_of_3bit_gpu_memory"] == pytest.approx(3.5)
    assert bundle["wall_clock"]["cycle_minutes"] == pytest.approx(20.0)
    assert bundle["wall_clock"]["partial_because_resumed"] is False
