"""The fixed search sample: exact size, reproducible, stratified, and traceable to its runs."""

import json

import pytest

from src.attribution.search_sample import draw_search_sample, write_search_sample

COUNTS = {"Age": 472, "Gender_identity": 802, "Race_ethnicity": 944, "Religion": 131, "SES": 1043}


def _flipped(counts=COUNTS):
    items = [
        {"question_id": f"{cat}:{i}", "category": cat, "baseline_answer": 2, "new_answer": 0}
        for cat, n in counts.items()
        for i in range(n)
    ]
    return {
        "baseline_run_id": "fp16_20260921T180125Z",
        "quantized_run_id": "hqq_3bit_20260924T175317Z",
        "total_flipped": len(items),
        "items": items,
    }


def test_draws_exactly_n_distinct_flipped_items():
    sample = draw_search_sample(_flipped(), n=300)
    ids = sample["question_ids"]
    assert len(ids) == len(set(ids)) == 300
    assert set(ids) <= {item["question_id"] for item in _flipped()["items"]}


def test_same_seed_gives_the_same_sample_regardless_of_input_order():
    flipped = _flipped()
    shuffled = {**flipped, "items": list(reversed(flipped["items"]))}
    assert draw_search_sample(flipped, seed=7) == draw_search_sample(shuffled, seed=7)


def test_a_different_seed_gives_a_different_sample():
    a = draw_search_sample(_flipped(), seed=1)["question_ids"]
    b = draw_search_sample(_flipped(), seed=2)["question_ids"]
    assert a != b


def test_default_split_follows_each_categorys_share_of_the_flipped_set():
    per_category = draw_search_sample(_flipped(), n=300)["per_category"]
    total = sum(COUNTS.values())
    for cat, n in COUNTS.items():
        assert abs(per_category[cat] - 300 * n / total) < 1
    assert sum(per_category.values()) == 300


def test_weights_skew_the_split_for_the_failure_profiles():
    weights = {"Age": 0.6, "Gender_identity": 0.1, "Race_ethnicity": 0.1, "Religion": 0.1, "SES": 0.1}
    per_category = draw_search_sample(_flipped(), n=300, category_weights=weights)["per_category"]
    assert per_category == {"Age": 180, "Gender_identity": 30, "Race_ethnicity": 30, "Religion": 30, "SES": 30}


def test_weights_asking_for_more_than_a_category_has_raise_rather_than_rebalance():
    weights = {"Religion": 1.0}  # 300 requested, only 131 exist
    with pytest.raises(ValueError, match="not enough flipped items"):
        draw_search_sample(_flipped(), n=300, category_weights=weights)


def test_weights_naming_an_unknown_category_raise():
    with pytest.raises(ValueError, match="no flipped items"):
        draw_search_sample(_flipped(), category_weights={"Nationality": 1.0})


def test_asking_for_more_than_the_flipped_set_raises():
    with pytest.raises(ValueError, match="only has"):
        draw_search_sample(_flipped({"Age": 10}), n=11)


def test_header_traces_the_sample_back_to_its_runs(tmp_path):
    sample = draw_search_sample(_flipped(), n=300, seed=3)
    assert sample["baseline_run_id"] == "fp16_20260921T180125Z"
    assert sample["quantized_run_id"] == "hqq_3bit_20260924T175317Z"
    assert (sample["n"], sample["seed"], sample["flipped_total"]) == (300, 3, sum(COUNTS.values()))

    path = write_search_sample(sample, tmp_path / "s" / "sample.json")
    assert json.loads(path.read_text(encoding="utf-8")) == sample
