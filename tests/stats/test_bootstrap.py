"""Bootstrap intervals: determinism, coverage, and the paired difference."""

from __future__ import annotations

import numpy as np
import pytest

from src.stats.bootstrap import (
    DEFAULT_N_RESAMPLES,
    bootstrap_ci,
    bootstrap_paired_diff_ci,
    bootstrap_proportion_ci,
    bootstrap_values,
)
from src.stats.metrics import ConditionMismatchError, is_stereotype, is_unknown, usr

from helpers import make_rows, pattern

FAST = dict(n_resamples=200)


class TestDegenerateData:
    def test_an_all_unknown_run_has_usr_one_and_a_zero_width_interval(self):
        # Every resample is identical, so there is nothing for the interval to
        # be uncertain about.
        result = bootstrap_proportion_ci(make_rows("u" * 100), is_unknown, **FAST)
        assert result.point == 1.0
        assert result.lower == 1.0
        assert result.upper == 1.0
        assert result.width == 0.0

    def test_an_all_stereotype_run_has_usr_zero(self):
        result = bootstrap_proportion_ci(make_rows("s" * 50), is_unknown, **FAST)
        assert result.point == 0.0
        assert result.width == 0.0


class TestDeterminism:
    def test_the_same_seed_gives_identical_replicates(self):
        rows = make_rows(pattern(70, 20, 10))
        a = bootstrap_proportion_ci(rows, is_unknown, seed=7, **FAST)
        b = bootstrap_proportion_ci(rows, is_unknown, seed=7, **FAST)
        assert np.array_equal(a.replicates, b.replicates)
        assert (a.lower, a.upper) == (b.lower, b.upper)

    def test_a_different_seed_gives_a_different_interval(self):
        rows = make_rows(pattern(70, 20, 10))
        a = bootstrap_proportion_ci(rows, is_unknown, seed=1, **FAST)
        b = bootstrap_proportion_ci(rows, is_unknown, seed=2, **FAST)
        assert not np.array_equal(a.replicates, b.replicates)

    def test_the_seed_travels_in_the_result(self):
        result = bootstrap_proportion_ci(make_rows("us" * 20), is_unknown, seed=99, **FAST)
        assert result.seed == 99
        assert result.as_dict()["seed"] == 99

    def test_replicates_are_left_out_of_the_serialised_form(self):
        result = bootstrap_proportion_ci(make_rows("us" * 20), is_unknown, **FAST)
        assert "replicates" not in result.as_dict()


class TestIntervalBehaviour:
    def test_the_plan_default_is_one_thousand_resamples(self):
        result = bootstrap_proportion_ci(make_rows(pattern(7, 2, 1)), is_unknown)
        assert result.n_resamples == DEFAULT_N_RESAMPLES == 1000
        assert result.replicates.size == 1000

    def test_the_interval_brackets_the_point_estimate(self):
        rows = make_rows(pattern(700, 200, 100))
        result = bootstrap_proportion_ci(rows, is_unknown, **FAST)
        assert result.point == pytest.approx(0.7)
        assert result.lower <= result.point <= result.upper

    def test_interval_width_is_close_to_the_normal_approximation(self):
        # p=0.7, n=1000 -> standard error 0.0145, so a 95% interval is ~0.057 wide.
        rows = make_rows(pattern(700, 200, 100))
        result = bootstrap_proportion_ci(rows, is_unknown)
        expected_width = 2 * 1.96 * np.sqrt(0.7 * 0.3 / 1000)
        assert result.width == pytest.approx(expected_width, rel=0.15)

    def test_more_data_gives_a_narrower_interval(self):
        small = bootstrap_proportion_ci(make_rows(pattern(70, 30)), is_unknown, **FAST)
        large = bootstrap_proportion_ci(make_rows(pattern(700, 300)), is_unknown, **FAST)
        assert large.width < small.width

    def test_a_ninety_percent_interval_is_narrower_than_a_ninety_five(self):
        rows = make_rows(pattern(700, 300))
        wide = bootstrap_proportion_ci(rows, is_unknown, alpha=0.05, **FAST)
        narrow = bootstrap_proportion_ci(rows, is_unknown, alpha=0.10, **FAST)
        assert narrow.width < wide.width
        assert narrow.confidence == pytest.approx(0.90)


class TestValidation:
    def test_no_rows_raises(self):
        with pytest.raises(ValueError, match="no rows"):
            bootstrap_values(np.array([]))

    def test_zero_resamples_raises(self):
        with pytest.raises(ValueError, match="n_resamples"):
            bootstrap_values(np.ones(10), n_resamples=0)

    @pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 2.0])
    def test_alpha_outside_the_open_unit_interval_raises(self, alpha):
        with pytest.raises(ValueError, match="alpha"):
            bootstrap_values(np.ones(10), alpha=alpha)


class TestGenericBootstrap:
    def test_it_agrees_with_the_fast_proportion_path(self):
        # Different RNG draw order, so not identical -- but they estimate the
        # same quantity and must land in the same place.
        rows = make_rows(pattern(700, 200, 100))
        generic = bootstrap_ci(rows, usr, n_resamples=300, seed=3)
        fast = bootstrap_proportion_ci(rows, is_unknown, n_resamples=300, seed=3)
        assert generic.point == pytest.approx(fast.point)
        assert generic.lower == pytest.approx(fast.lower, abs=0.02)
        assert generic.upper == pytest.approx(fast.upper, abs=0.02)

    def test_it_handles_a_metric_that_is_not_a_simple_mean(self):
        ratio = lambda rows: usr(rows) / max(len(rows), 1)
        result = bootstrap_ci(make_rows(pattern(70, 30)), ratio, n_resamples=100)
        assert result.point == pytest.approx(0.7 / 100)


class TestPairedDifference:
    def test_two_identical_runs_show_no_change(self):
        before = make_rows(pattern(70, 20, 10))
        after = make_rows(pattern(70, 20, 10), condition="hqq_3bit")
        result = bootstrap_paired_diff_ci(before, after, is_unknown, **FAST)
        assert result.point == 0.0
        assert (result.lower, result.upper) == (0.0, 0.0)

    def test_a_drop_in_usr_comes_out_negative(self):
        # 90% unknown before, 50% after: half the safe items flipped.
        before = make_rows(pattern(90, 5, 5))
        after = make_rows(
            pattern(50, 45, 5), condition="hqq_3bit"
        )
        result = bootstrap_paired_diff_ci(before, after, is_unknown, **FAST)
        assert result.point == pytest.approx(-0.40)
        assert result.upper < 0.0

    def test_the_matching_rise_in_srs_comes_out_positive(self):
        before = make_rows(pattern(90, 5, 5))
        after = make_rows(pattern(50, 45, 5), condition="hqq_3bit")
        result = bootstrap_paired_diff_ci(before, after, is_stereotype, **FAST)
        assert result.point == pytest.approx(0.40)
        assert result.lower > 0.0

    def test_pairing_gives_a_tighter_interval_than_treating_runs_separately(self):
        before = make_rows(pattern(70, 20, 10))
        after = make_rows(pattern(60, 30, 10), condition="hqq_3bit")
        paired = bootstrap_paired_diff_ci(before, after, is_unknown, n_resamples=500)
        unpaired_width = np.hypot(
            bootstrap_proportion_ci(before, is_unknown, n_resamples=500).width,
            bootstrap_proportion_ci(after, is_unknown, n_resamples=500).width,
        )
        assert paired.width < unpaired_width

    def test_comparing_across_dtypes_raises(self):
        with pytest.raises(ConditionMismatchError):
            bootstrap_paired_diff_ci(
                make_rows("uuuu", condition="fp16"),
                make_rows("usuu", condition="bf16"),
                is_unknown,
                **FAST,
            )

    def test_a_truncated_second_run_raises(self):
        with pytest.raises(ValueError, match="missing from the"):
            bootstrap_paired_diff_ci(
                make_rows("uuuu"),
                make_rows("uuuu", condition="hqq_3bit")[:2],
                is_unknown,
                **FAST,
            )
