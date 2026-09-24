"""Chi-squared on the stereotype table."""

from __future__ import annotations

import numpy as np
import pytest

from src.stats.chisq import chisq_from_table, stereotype_chisq, stereotype_table
from src.stats.metrics import ConditionMismatchError

from helpers import make_rows, pattern


class TestTableConstruction:
    def test_counts_stereotype_against_everything_else(self):
        before = make_rows(pattern(n_unknown=70, n_stereotype=20, n_other=10))
        after = make_rows(
            pattern(n_unknown=50, n_stereotype=40, n_other=10), condition="hqq_3bit"
        )
        assert np.array_equal(
            stereotype_table(before, after), np.array([[20, 80], [40, 60]])
        )

    def test_the_other_person_counts_as_not_stereotyped(self):
        # 10 wrong answers, none of them stereotyped -- SRS is 0, not 1.
        table = stereotype_table(make_rows("o" * 10), make_rows("u" * 10, condition="hqq_3bit"))
        assert np.array_equal(table, np.array([[0, 10], [0, 10]]))


class TestTestBehaviour:
    def test_identical_distributions_give_a_large_p_value(self):
        result = chisq_from_table([[20, 80], [20, 80]])
        assert result.statistic == pytest.approx(0.0)
        assert result.p_value == pytest.approx(1.0)

    def test_a_lopsided_table_gives_a_tiny_p_value(self):
        result = chisq_from_table([[100, 900], [400, 600]])
        assert result.p_value < 1e-10

    def test_a_two_by_two_table_has_one_degree_of_freedom(self):
        assert chisq_from_table([[20, 80], [30, 70]]).dof == 1

    def test_the_table_comes_back_with_the_result(self):
        # When a number looks strange, the first thing you want is the counts
        # that produced it.
        result = chisq_from_table([[20, 80], [30, 70]])
        assert np.array_equal(result.table, np.array([[20, 80], [30, 70]]))
        assert result.expected.shape == (2, 2)
        assert result.n == 200

    def test_continuity_correction_is_off_by_default_and_shrinks_the_statistic(self):
        uncorrected = chisq_from_table([[20, 80], [30, 70]])
        corrected = chisq_from_table([[20, 80], [30, 70]], correction=True)
        assert uncorrected.correction is False
        assert corrected.statistic < uncorrected.statistic

    def test_it_serialises_to_json_safe_types(self):
        payload = chisq_from_table([[20, 80], [30, 70]]).as_dict()
        assert payload["table"] == [[20, 80], [30, 70]]
        assert isinstance(payload["p_value"], float)


class TestValidation:
    def test_an_all_zero_row_raises(self):
        with pytest.raises(ValueError, match="all-zero row or column"):
            chisq_from_table([[0, 0], [30, 70]])

    def test_an_all_zero_column_raises(self):
        with pytest.raises(ValueError, match="all-zero row or column"):
            chisq_from_table([[0, 80], [0, 70]])

    def test_an_empty_table_raises(self):
        with pytest.raises(ValueError, match="empty"):
            chisq_from_table([[0, 0], [0, 0]])

    def test_negative_counts_raise(self):
        with pytest.raises(ValueError, match="negative counts"):
            chisq_from_table([[-1, 80], [30, 70]])

    def test_a_one_dimensional_table_raises(self):
        with pytest.raises(ValueError, match="2-D table"):
            chisq_from_table([20, 80])

    def test_small_expected_counts_warn(self):
        with pytest.warns(UserWarning, match="expected cell count"):
            chisq_from_table([[1, 9], [2, 8]])


class TestRunLevelEntryPoint:
    def test_it_runs_end_to_end_on_rows(self):
        before = make_rows(pattern(700, 200, 100))
        after = make_rows(pattern(500, 400, 100), condition="hqq_3bit")
        result = stereotype_chisq(before, after)
        assert result.p_value < 1e-10
        assert np.array_equal(result.table, np.array([[200, 800], [400, 600]]))

    def test_it_refuses_to_compare_across_dtypes(self):
        with pytest.raises(ConditionMismatchError, match="different dtypes"):
            stereotype_chisq(
                make_rows(pattern(70, 30), condition="fp16"),
                make_rows(pattern(50, 50), condition="bf16"),
            )

    def test_fp16_against_hqq_3bit_is_allowed(self):
        result = stereotype_chisq(
            make_rows(pattern(70, 30), condition="fp16"),
            make_rows(pattern(50, 50), condition="hqq_3bit"),
        )
        assert result.dof == 1
