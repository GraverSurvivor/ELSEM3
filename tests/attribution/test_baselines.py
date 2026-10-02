"""Baseline patches for the fairness controls: same weight budget as the search, different choice."""

import pytest

from src.attribution.baselines import (
    BUDGET_TOLERANCE,
    build_all,
    largest_error_patch,
    positional_patch,
    random_patch,
    within_budget,
)
from src.attribution.patch import Patch
from src.quant.candidates import candidate_paths

QKV = "model.layers.{}.self_attn.qkv_proj"
SEARCH = Patch(
    name="search_top5",
    modules=(QKV.format(1), QKV.format(14), QKV.format(15), QKV.format(13), "model.layers.10.self_attn.o_proj"),
    nbits=8,
    source={"kind": "search", "search_run_id": "s"},
)


@pytest.mark.parametrize("seed", range(5))
def test_random_patches_match_the_budget(seed):
    patch = random_patch(SEARCH, seed)
    assert within_budget(patch.params, SEARCH.params)
    assert patch.nbits == SEARCH.nbits and patch.source["baseline"] == "random"


def test_random_patches_are_reproducible_and_differ_by_seed():
    assert random_patch(SEARCH, 3) == random_patch(SEARCH, 3)
    assert len({random_patch(SEARCH, s).modules for s in range(5)}) == 5


def test_positional_keeps_composition_and_moves_to_early_layers():
    patch = positional_patch(SEARCH)
    assert set(patch.modules) == {QKV.format(i) for i in range(4)} | {"model.layers.0.self_attn.o_proj"}
    assert patch.params == SEARCH.params  # identical composition -> identical weight count


def test_largest_error_takes_the_most_damaged_modules_within_budget():
    # Pretend the four MLP gate_up modules of layers 0-3 are the most damaged, then everything else.
    worst = [f"model.layers.{i}.mlp.gate_up_proj" for i in range(4)]
    rest = [p for p in candidate_paths() if p not in worst]
    ranked = [{"path": p, "rel_error": 1.0 - i / 1000} for i, p in enumerate(worst + rest)]
    patch = largest_error_patch(SEARCH, ranked)
    assert within_budget(patch.params, SEARCH.params)
    assert patch.modules[:2] == tuple(worst[:2])  # 2 x 50.3M fits the 122.7M budget, a third doesn't


def test_build_all_includes_prefixes_and_skips_error_baseline_without_data():
    names = [p.name for p in build_all(SEARCH, quant_error=None)]
    assert names[:4] == [f"search_top5_top{k}" for k in range(1, 5)]
    assert "positional" in names and "largest_error" not in names
    assert sum(n.startswith("random_s") for n in names) == 5


def test_tolerance_is_five_percent():
    assert BUDGET_TOLERANCE == 0.05
    assert within_budget(105, 100) and not within_budget(106, 100)
