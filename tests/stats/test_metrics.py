"""Row access, the two metrics, condition guards, and run alignment."""

from __future__ import annotations

import pytest

from src.stats import metrics
from src.stats.metrics import (
    ConditionMismatchError,
    align_by_question,
    assert_comparable,
    by_category,
    get_field,
    indicators,
    is_stereotype,
    is_unknown,
    run_condition,
    srs,
    usr,
)

from helpers import as_dicts, make_rows, pattern


class TestFieldAccess:
    def test_reads_objects_and_dicts_alike(self):
        rows = make_rows("usou")
        assert usr(rows) == usr(as_dicts(rows))
        assert srs(rows) == srs(as_dicts(rows))

    def test_missing_field_on_dict_raises_with_the_available_keys(self):
        row = as_dicts(make_rows("u"))[0]
        del row["predicted_index"]
        with pytest.raises(KeyError, match="predicted_index"):
            is_unknown(row)

    def test_missing_field_on_object_raises(self):
        class Bare:
            pass

        with pytest.raises(KeyError, match="predicted_index"):
            get_field(Bare(), "predicted_index")


class TestMetrics:
    def test_usr_and_srs_on_a_known_pattern(self):
        rows = make_rows(pattern(n_unknown=7, n_stereotype=2, n_other=1))
        assert usr(rows) == pytest.approx(0.7)
        assert srs(rows) == pytest.approx(0.2)

    def test_picking_the_other_person_is_not_a_stereotype(self):
        rows = make_rows("oooo")
        assert usr(rows) == 0.0
        assert srs(rows) == 0.0

    def test_rates_do_not_sum_to_one_when_other_answers_exist(self):
        rows = make_rows(pattern(n_unknown=5, n_stereotype=3, n_other=2))
        assert usr(rows) + srs(rows) == pytest.approx(0.8)

    def test_indicators_preserve_order(self):
        rows = make_rows("usu")
        assert list(indicators(rows, is_unknown)) == [True, False, True]

    def test_empty_rows_raise(self):
        with pytest.raises(ValueError, match="no rows"):
            usr([])

    def test_by_category_groups_and_preserves_order(self):
        rows = make_rows("uu", category="Age") + make_rows(
            "s", category="Religion", prefix="r"
        )
        grouped = by_category(rows)
        assert set(grouped) == {"Age", "Religion"}
        assert [r.question_id for r in grouped["Age"]] == ["q0", "q1"]


class TestConditionGuards:
    def test_hqq_3bit_is_comparable_with_the_fp16_baseline(self):
        # Different experiment, same compute dtype -- this is the comparison the
        # whole project is built on, so it must be allowed.
        assert_comparable(
            make_rows("uu", condition="fp16"),
            make_rows("us", condition="hqq_3bit"),
        )

    def test_fp16_against_bf16_raises(self):
        with pytest.raises(ConditionMismatchError, match="different dtypes"):
            assert_comparable(
                make_rows("uu", condition="fp16"),
                make_rows("uu", condition="bf16"),
            )

    def test_unknown_condition_raises_rather_than_defaulting(self):
        with pytest.raises(ConditionMismatchError, match="unknown condition"):
            assert_comparable(
                make_rows("uu", condition="fp16"),
                make_rows("uu", condition="hqq_2bit"),
            )

    def test_a_run_mixing_conditions_is_not_a_run(self):
        rows = make_rows("uu", condition="fp16") + make_rows(
            "uu", condition="bf16", prefix="b"
        )
        with pytest.raises(ConditionMismatchError, match="mixes conditions"):
            run_condition(rows)

    def test_every_condition_in_the_table_maps_to_a_known_dtype(self):
        assert set(metrics.CONDITION_COMPUTE_DTYPE.values()) <= {"fp16", "bf16"}


class TestAlignByQuestion:
    def test_pairs_in_the_order_of_the_first_run(self):
        before = make_rows("uuu")
        after = list(reversed(make_rows("sus", condition="hqq_3bit")))
        paired_before, paired_after = align_by_question(before, after)
        assert [r.question_id for r in paired_before] == ["q0", "q1", "q2"]
        assert [r.question_id for r in paired_after] == ["q0", "q1", "q2"]

    def test_a_truncated_second_run_raises_instead_of_shrinking_the_comparison(self):
        before = make_rows("uuuu")
        after = make_rows("uuuu", condition="hqq_3bit")[:2]
        with pytest.raises(ValueError, match="missing from the"):
            align_by_question(before, after)

    def test_extra_questions_in_the_second_run_raise(self):
        with pytest.raises(ValueError, match="only in the second run"):
            align_by_question(make_rows("uu"), make_rows("uuu", condition="hqq_3bit"))

    def test_duplicate_ids_raise_in_either_run(self):
        dupes = make_rows("uu")
        dupes[1].question_id = dupes[0].question_id
        with pytest.raises(ValueError, match="duplicate question id"):
            align_by_question(dupes, make_rows("uu", condition="hqq_3bit"))
