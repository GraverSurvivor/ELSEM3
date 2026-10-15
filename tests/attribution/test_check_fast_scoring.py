"""The comparison logic of the GPU equivalence check (the GPU run itself is Person 4's)."""

import pytest

from src.attribution.check_fast_scoring import check_layer, compare_scores


def test_identical_answers_and_close_confidences_pass():
    verdict = compare_scores(([0, 1, 2], [0.5, 0.6, 0.7]), ([0, 1, 2], [0.5004, 0.6, 0.6995]))
    assert verdict["ok"] and verdict["different_answers"] == 0


def test_a_different_answer_fails_even_with_equal_confidence():
    verdict = compare_scores(([0, 1, 2], [0.5] * 3), ([0, 2, 2], [0.5] * 3))
    assert not verdict["ok"] and verdict["first_different"] == [1]


def test_a_confidence_drift_over_the_tolerance_fails():
    assert not compare_scores(([0], [0.50]), ([0], [0.502]))["ok"]


def test_different_lengths_are_an_error():
    with pytest.raises(ValueError):
        compare_scores(([0], [0.5]), ([0, 1], [0.5, 0.5]))


def test_check_layer_reports_timings():
    row = check_layer(lambda: (([1], [0.9]), ([1], [0.9]), 2.0, 1.0), "layer 3")
    assert row["ok"] and row["label"] == "layer 3" and row["seconds_normal"] == 2.0 and row["seconds_fast"] == 1.0
