"""Patch files: the shared format for 'which modules to restore', across search, baselines and prefixes."""

import json

import pytest

from src.attribution.patch import (
    Patch,
    load_patch,
    module_params,
    patch_from_dict,
    patch_from_search_summary,
    save_patch,
    total_candidate_params,
)

QKV = "model.layers.{}.self_attn.qkv_proj"
SEARCH_TOP5 = [QKV.format(1), QKV.format(14), QKV.format(15), QKV.format(13), "model.layers.10.self_attn.o_proj"]


def test_sizes_match_the_measured_repair():
    # run_repair measured these off the live model for the committed 5-module repair.
    patch = Patch(name="search_top5", modules=tuple(SEARCH_TOP5), nbits=8)
    assert patch.params == 122_683_392
    assert total_candidate_params() == 3_623_878_656
    assert patch.pct_params == pytest.approx(3.385, abs=1e-3)


def test_module_params_per_type():
    assert module_params(QKV.format(0)) == 9216 * 3072
    assert module_params("model.layers.31.mlp.down_proj") == 3072 * 8192


@pytest.mark.parametrize("bad", ["model.layers.32.self_attn.qkv_proj", "model.layers.0.self_attn.q_proj", "lm_head"])
def test_rejects_paths_that_are_not_candidates(bad):
    with pytest.raises(ValueError):
        Patch(name="x", modules=(bad,), nbits=8)


def test_rejects_empty_duplicate_and_non_restoring_patches():
    with pytest.raises(ValueError):
        Patch(name="x", modules=(), nbits=8)
    with pytest.raises(ValueError):
        Patch(name="x", modules=(QKV.format(1), QKV.format(1)), nbits=8)
    with pytest.raises(ValueError, match="restore anything"):
        Patch(name="x", modules=(QKV.format(1),), nbits=3)  # same precision as the 3-bit base


def test_round_trips_through_a_file(tmp_path):
    patch = Patch(name="positional", modules=(QKV.format(0),), nbits=8, source={"kind": "baseline"})
    loaded = load_patch(save_patch(patch, tmp_path / "p.json"))
    assert loaded == patch


def test_detects_a_hand_edited_size(tmp_path):
    data = Patch(name="p", modules=(QKV.format(0),), nbits=8).to_dict()
    data["params"] = 1
    with pytest.raises(ValueError, match="edited by hand"):
        patch_from_dict(data)


def test_prefixes_keep_restore_order():
    patch = Patch(name="search_top5", modules=tuple(SEARCH_TOP5), nbits=8, source={"kind": "search", "search_run_id": "s"})
    top2 = patch.prefix(2)
    assert top2.modules == tuple(SEARCH_TOP5[:2])
    assert top2.source == {"kind": "prefix", "search_run_id": "s", "prefix_of": "search_top5", "k": 2}
    with pytest.raises(ValueError):
        patch.prefix(6)


def test_from_a_run_repair_summary():
    # The committed hqq_repaired summary lists the kept modules under "modules", in keep order.
    summary = json.load(open("results/reference/hqq_repaired/summary.json"))
    patch = patch_from_search_summary(summary, "search_top5")
    assert list(patch.modules) == SEARCH_TOP5
    assert patch.nbits == 8 and patch.source["search_run_id"] == "search_20260926T155230Z"
