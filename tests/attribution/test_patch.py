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


def test_every_committed_patch_rewrites_identically(tmp_path):
    """Version 2's keys are optional: not one byte of an existing patch file changes."""
    from pathlib import Path

    paths = sorted(Path("results/patches").glob("*.json"))
    assert len(paths) >= 10
    for path in paths:
        again = save_patch(load_patch(path), tmp_path / path.name)
        assert again.read_text(encoding="utf-8") == path.read_text(encoding="utf-8"), path.name


def test_mixed_bits_round_trip_and_follow_the_prefix(tmp_path):
    patch = Patch(name="mixed", modules=("model.layers.1.self_attn.qkv_proj", "model.layers.2.mlp.down_proj",
                                         "model.layers.3.self_attn.o_proj"), nbits=8,
                  module_bits={"model.layers.2.mlp.down_proj": 4}, revision="abc123", prompt_style="examples")
    assert [patch.bits(p) for p in patch.modules] == [8, 4, 8]
    again = load_patch(save_patch(patch, tmp_path / "mixed.json"))
    assert again == patch and again.to_dict()["module_bits"] == {"model.layers.2.mlp.down_proj": 4}
    assert patch.prefix(1).module_bits is None and patch.prefix(2).bits("model.layers.2.mlp.down_proj") == 4
    assert patch.prefix(2).revision == "abc123" and patch.prefix(2).prompt_style == "examples"
    with pytest.raises(KeyError, match="doesn't restore"):
        patch.bits("model.layers.9.self_attn.o_proj")


@pytest.mark.parametrize("bits", [{"model.layers.1.self_attn.qkv_proj": 3}, {"model.layers.1.self_attn.qkv_proj": 32},
                                  {"model.layers.1.self_attn.qkv_proj": 4.0}, {"model.layers.5.self_attn.qkv_proj": 4}])
def test_rejects_bits_that_restore_nothing(bits):
    with pytest.raises(ValueError):
        Patch(name="bad", modules=("model.layers.1.self_attn.qkv_proj",), nbits=8, module_bits=bits)
    with pytest.raises(ValueError, match="prompt style"):
        Patch(name="bad", modules=("model.layers.1.self_attn.qkv_proj",), nbits=8, prompt_style="poem")


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
