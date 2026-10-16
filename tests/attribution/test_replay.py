"""Replay of Fast Scan on a finished search's logs."""

import csv
import json

import pytest

from src.attribution.fast_scan import FastScanConfig
from src.attribution.replay import (ItemData, kept_in_order, load_run, monte_carlo, read_trial_rows, replay_pool,
                                    replay_trials, round_one_ranking, simulate_fast_scan)

PATHS = [f"m{i}" for i in range(8)]


def _rows():
    """Round 1 ranks m5 > m2 > m7 > the rest; the full search kept m5, then m7 (rank 3 in round 1)."""
    usr1 = {"m5": 0.9, "m2": 0.8, "m7": 0.7, "m0": 0.1, "m1": 0.1, "m3": 0.1, "m4": 0.1, "m6": 0.1}
    rows = [{"round": 1, "path": p, "usr": usr1[p], "gain": usr1[p], "kept": p == "m5"} for p in PATHS]
    usr2 = {"m2": 0.92, "m7": 0.95, "m0": 0.9, "m1": 0.9, "m3": 0.9, "m4": 0.9, "m6": 0.9}
    rows += [{"round": 2, "path": p, "usr": u, "gain": u - 0.9, "kept": p == "m7"} for p, u in usr2.items()]
    return rows


def test_kept_and_ranking_follow_the_log():
    rows = _rows()
    assert kept_in_order(rows) == ["m5", "m7"]
    assert round_one_ranking(rows)[:3] == ["m5", "m2", "m7"]
    assert round_one_ranking(rows)[3:] == ["m0", "m1", "m3", "m4", "m6"]  # ties keep the candidate order


def test_pool_must_reach_the_lowest_ranked_winner():
    rows = _rows()
    assert not replay_pool(rows, 2)["matches"] and replay_pool(rows, 2)["diverged_at_round"] == 2
    assert replay_pool(rows, 3)["matches"] and replay_pool(rows, 8)["matches"]
    report = replay_trials(rows, kmin=1, kmax=8)
    assert report["smallest_k_that_matches"] == 3 and report["all_match_from_k"] == 3
    assert report["round1_rank_of_kept"] == {"m5": 1, "m7": 3}


def test_reads_the_real_trial_log_format(tmp_path):
    path = tmp_path / "trials.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["round", "path", "usr", "gain", "kept"])
        writer.writeheader()
        writer.writerows(_rows())
    assert read_trial_rows(path) == _rows()


def _items_data(n_sample=40, n_probe=0, noise=0.0):
    """m5 wins round 1 clearly and m7 is round 1's runner-up; m7 wins round 2 clearly."""
    import random

    rng = random.Random(0)

    def bits(good, n):
        return "".join("1" if (i < good) != (rng.random() < noise) else "0" for i in range(n))

    records = [{"round": 0, "path": None, "sample": "0" * n_sample, "probe": "0" * n_probe}]
    for p in PATHS:
        records.append({"round": 1, "path": p, "sample": bits({"m5": 30, "m7": 15}.get(p, 0), n_sample), "probe": "0" * n_probe})
    for p in PATHS:
        if p != "m5":
            records.append({"round": 2, "path": p, "sample": bits(38 if p == "m7" else 25, n_sample), "probe": "0" * n_probe})
    return ItemData(records), ["m5", "m7"]


def test_simulation_agrees_when_the_winners_stand_out():
    data, kept = _items_data()
    config = FastScanConfig(mode="fast", screen_sample=20, screen_probe=0, pool=4)
    assert simulate_fast_scan(data, kept, config, seed=0)["same"]
    result = monte_carlo(data, kept, config, seeds=50)
    assert result["agreement"] == 1.0 and result["diverged_at_round"] == {}


def test_simulation_is_deterministic_and_reports_where_it_diverges():
    data, kept = _items_data()
    config = FastScanConfig(mode="fast", screen_sample=20, screen_probe=0, pool=1)
    # a pool of one holds only round 1's winner, so round 2 has nothing to pick m7 from
    first = monte_carlo(data, kept, config, seeds=10)
    assert first == monte_carlo(data, kept, config, seeds=10)
    assert first["agreement"] == 0.0 and first["diverged_at_round"] == {2: 10}
    assert first["true_module_missing_from_pool"] == 10 or first["agreement"] == 0


def test_exact_mode_rescored_on_the_full_sample():
    data, kept = _items_data()
    config = FastScanConfig(mode="exact", screen_sample=10, screen_probe=0, survivors=4, pool=3)
    assert monte_carlo(data, kept, config, seeds=30)["agreement"] == 1.0


def test_load_run_reads_the_weight_from_the_summary(tmp_path):
    data, kept = _items_data(n_probe=2)
    (tmp_path / "items.jsonl").write_text("".join(json.dumps(r) + "\n" for r in data.by_key.values()), encoding="utf-8")
    with (tmp_path / "trials.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["round", "path", "usr", "gain", "kept"])
        writer.writeheader()
        writer.writerows(_rows())
    (tmp_path / "summary.json").write_text(json.dumps({"penalty": {"weight": 2.5}}), encoding="utf-8")
    loaded, loaded_kept = load_run(tmp_path)
    assert loaded.weight == 2.5 and loaded_kept == ["m5", "m7"] and loaded.n_probe == 2


def test_needs_a_baseline_line():
    with pytest.raises(ValueError, match="baseline"):
        ItemData([{"round": 1, "path": "m0", "sample": "1", "probe": ""}])
