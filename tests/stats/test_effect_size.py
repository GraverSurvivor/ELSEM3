"""Cramer's V, Cohen's h, and the before/after rate change."""

from __future__ import annotations

import pytest

from src.stats.chisq import chisq_from_table
from src.stats.effect_size import (
    cohens_h,
    cramers_v,
    cramers_v_from_result,
    srs_delta,
    usr_delta,
)
from src.stats.metrics import ConditionMismatchError

from helpers import make_rows, pattern

FAST = dict(n_resamples=200)


class TestCramersV:
    def test_no_association_is_zero(self):
        assert cramers_v([[50, 50], [50, 50]]) == pytest.approx(0.0)

    def test_perfect_association_is_one(self):
        assert cramers_v([[10, 0], [0, 10]]) == pytest.approx(1.0)

    def test_it_sits_between_zero_and_one_for_a_partial_association(self):
        value = cramers_v([[20, 80], [40, 60]])
        assert 0.0 < value < 1.0

    def test_a_stronger_shift_gives_a_larger_value(self):
        mild = cramers_v([[20, 80], [25, 75]])
        severe = cramers_v([[20, 80], [70, 30]])
        assert severe > mild

    def test_sample_size_alone_does_not_move_it(self):
        # This is the whole point of an effect size: the p-value collapses when
        # you scale the counts up, the effect size does not.
        small = cramers_v([[20, 80], [40, 60]])
        large = cramers_v([[2000, 8000], [4000, 6000]])
        assert small == pytest.approx(large)
        assert chisq_from_table([[2000, 8000], [4000, 6000]]).p_value < chisq_from_table(
            [[20, 80], [40, 60]]
        ).p_value

    def test_it_reuses_an_existing_result_without_refitting(self):
        result = chisq_from_table([[20, 80], [40, 60]])
        assert cramers_v_from_result(result) == pytest.approx(cramers_v(result.table))

    def test_a_corrected_statistic_is_rejected(self):
        corrected = chisq_from_table([[20, 80], [40, 60]], correction=True)
        with pytest.raises(ValueError, match="correction=False"):
            cramers_v_from_result(corrected)


class TestCohensH:
    def test_no_change_is_zero(self):
        assert cohens_h(0.7, 0.7) == pytest.approx(0.0)

    def test_a_fall_is_negative_and_a_rise_is_positive(self):
        assert cohens_h(0.8, 0.6) < 0
        assert cohens_h(0.6, 0.8) > 0

    def test_it_is_antisymmetric(self):
        assert cohens_h(0.8, 0.6) == pytest.approx(-cohens_h(0.6, 0.8))

    def test_the_same_raw_delta_counts_for_more_near_the_extremes(self):
        middle = abs(cohens_h(0.50, 0.40))
        edge = abs(cohens_h(0.10, 0.00))
        assert edge > middle

    @pytest.mark.parametrize("bad", [-0.1, 1.1])
    def test_values_outside_zero_to_one_raise(self, bad):
        with pytest.raises(ValueError, match="proportion"):
            cohens_h(0.5, bad)


class TestProportionDelta:
    def test_identical_runs_report_no_change_and_do_not_exclude_zero(self):
        before = make_rows(pattern(70, 20, 10))
        after = make_rows(pattern(70, 20, 10), condition="hqq_3bit")
        delta = usr_delta(before, after, **FAST)
        assert delta.delta == 0.0
        assert delta.cohens_h == pytest.approx(0.0)
        assert not delta.excludes_zero

    def test_a_usr_drop_is_negative_with_an_interval_clear_of_zero(self):
        before = make_rows(pattern(800, 100, 100))
        after = make_rows(pattern(600, 300, 100), condition="hqq_3bit")
        delta = usr_delta(before, after, **FAST)
        assert delta.before == pytest.approx(0.8)
        assert delta.after == pytest.approx(0.6)
        assert delta.delta == pytest.approx(-0.2)
        assert delta.lower <= delta.delta <= delta.upper
        assert delta.upper < 0.0
        assert delta.excludes_zero

    def test_the_matching_srs_rise_is_positive(self):
        before = make_rows(pattern(800, 100, 100))
        after = make_rows(pattern(600, 300, 100), condition="hqq_3bit")
        delta = srs_delta(before, after, **FAST)
        assert delta.delta == pytest.approx(0.2)
        assert delta.lower > 0.0

    def test_metadata_needed_to_reproduce_it_travels_with_the_result(self):
        before = make_rows(pattern(80, 10, 10))
        after = make_rows(pattern(60, 30, 10), condition="hqq_3bit")
        payload = usr_delta(before, after, seed=11, **FAST).as_dict()
        assert payload["seed"] == 11
        assert payload["n_resamples"] == 200
        assert payload["n_pairs"] == 100
        assert payload["name"] == "USR"

    def test_it_refuses_to_compare_across_dtypes(self):
        with pytest.raises(ConditionMismatchError, match="different dtypes"):
            usr_delta(
                make_rows(pattern(70, 30), condition="fp16"),
                make_rows(pattern(50, 50), condition="bf16"),
                **FAST,
            )

    def test_a_truncated_run_raises(self):
        with pytest.raises(ValueError, match="missing from the"):
            usr_delta(
                make_rows("uuuu"),
                make_rows("uuuu", condition="hqq_3bit")[:2],
                **FAST,
            )
