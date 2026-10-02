"""Stage 5 repair: reading the search's plan, and applying it with its memory measured.

The real promotion is covered in tests/quant/; here a fake registry records what it was asked to do,
so the repair's own logic -- what it restores, in what order, and what it reports -- runs on CPU.
"""

import json
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from hqq.core.quantize import BaseQuantizeConfig, HQQLinear

from src.attribution import run_repair
from src.attribution.run_repair import (
    RepairPlan,
    apply_repair,
    linear_params,
    load_repair_plan,
    restored_share,
)
from src.data.bbq_loader import CATEGORIES

KEPT = [
    "model.layers.1.self_attn.qkv_proj",
    "model.layers.14.self_attn.qkv_proj",
    "model.layers.10.self_attn.o_proj",
]


def _summary(**overrides):
    summary = {
        "run_id": "search_20260926T155230Z",
        "compute_dtype": "fp16",
        "config": {"threshold": 0.01, "nbits": 8, "max_rounds": 5},
        "kept": list(KEPT),
    }
    summary.update(overrides)
    return summary


class FakeRegistry:
    def __init__(self, overhead=70 * 2**20):
        self._committed = {}
        self.calls = []
        self.overhead = overhead

    @property
    def active(self):
        return dict(self._committed)

    def commit(self, path, nbits):
        self.calls.append(("commit", path, nbits))
        self._committed[path] = nbits

    def release_baselines(self):
        self.calls.append(("release",))

    def deployed_overhead_bytes(self):
        return self.overhead


class FakeMemory:
    """Returns successive readings, so 'before' and 'after' are distinguishable."""

    def __init__(self, *readings):
        self.readings = list(readings)

    def __call__(self):
        return self.readings.pop(0)


# --- load_repair_plan -------------------------------------------------------------------------


def test_plan_takes_the_kept_modules_in_order_at_the_searched_precision():
    plan = load_repair_plan(_summary())
    assert plan == RepairPlan(modules=tuple(KEPT), nbits=8, search_run_id="search_20260926T155230Z")


def test_plan_refuses_a_search_run_at_another_dtype():
    with pytest.raises(ValueError, match="different dtypes"):
        load_repair_plan(_summary(compute_dtype="bf16"))


def test_plan_refuses_an_empty_search_result():
    with pytest.raises(ValueError, match="nothing to repair"):
        load_repair_plan(_summary(kept=[]))


def test_plan_refuses_a_module_listed_twice():
    with pytest.raises(ValueError, match="twice"):
        load_repair_plan(_summary(kept=[KEPT[0], KEPT[0]]))


@pytest.mark.parametrize("missing", ["kept", "config", "compute_dtype", "run_id"])
def test_plan_refuses_a_file_that_is_not_a_search_summary(missing):
    summary = _summary()
    del summary[missing]
    with pytest.raises(ValueError, match=missing):
        load_repair_plan(summary)


# --- apply_repair -----------------------------------------------------------------------------


def test_repair_commits_every_module_in_order_then_releases_the_originals():
    registry = FakeRegistry()
    apply_repair(registry, load_repair_plan(_summary()), FakeMemory(0, 0))
    assert registry.calls == [("commit", path, 8) for path in KEPT] + [("release",)]


def test_repair_reports_deployed_overhead_as_the_headline_and_the_gpu_delta_as_a_cross_check():
    registry = FakeRegistry(overhead=70 * 2**20)
    memory = apply_repair(registry, load_repair_plan(_summary()), FakeMemory(1_000, 1_000 + 75 * 2**20))
    assert memory.deployed_overhead_bytes == 70 * 2**20
    assert memory.gpu_allocated_before_bytes == 1_000
    assert memory.gpu_delta_bytes == 75 * 2**20


def test_memory_after_is_read_only_once_the_originals_are_released():
    # If "after" were read before release_baselines, the retained 3-bit modules would be counted as
    # part of the repair's cost.
    order = []
    registry = FakeRegistry()
    registry.release_baselines = lambda: order.append("release")

    def memory():
        order.append("read")
        return 0

    apply_repair(registry, load_repair_plan(_summary()), memory)
    assert order == ["read", "release", "read"]


def test_repair_raises_if_the_registry_does_not_end_up_holding_the_plan():
    class DroppingRegistry(FakeRegistry):
        def commit(self, path, nbits):
            if path != KEPT[-1]:
                super().commit(path, nbits)

    with pytest.raises(RuntimeError, match="registry reports"):
        apply_repair(DroppingRegistry(), load_repair_plan(_summary()), FakeMemory(0, 0))


# --- restored_share -------------------------------------------------------------------------


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.big = nn.Linear(300, 100, bias=False)    # 30,000 weights
        self.small = nn.Linear(100, 100, bias=False)  # 10,000 weights


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_Layer(), _Layer()])


_CANDIDATES = [f"layers.{i}.{name}" for name in ("big", "small") for i in (0, 1)]


def test_linear_params_reads_the_represented_weights_even_when_packed():
    linear = nn.Linear(256, 128, bias=False)
    packed = HQQLinear(
        nn.Linear(256, 128, bias=False),
        quant_config=BaseQuantizeConfig(nbits=3, group_size=64),
        compute_dtype=torch.float32,
        device="cpu",
    )
    assert linear_params(linear) == linear_params(packed) == 256 * 128


def test_share_is_by_parameters_not_by_module_count():
    share = restored_share(_Model(), _CANDIDATES, ("layers.0.big",))
    assert share["modules_restored"] == 1 and share["modules_total"] == 4
    assert share["pct_modules_restored"] == pytest.approx(25.0)
    # 30,000 of 80,000 weights: a count-based 25% would understate a big module's share.
    assert share["params_restored"] == 30_000 and share["params_total"] == 80_000
    assert share["pct_params_restored"] == pytest.approx(37.5)


def test_share_refuses_a_restored_module_that_was_never_a_candidate():
    with pytest.raises(ValueError, match="not search candidates"):
        restored_share(_Model(), _CANDIDATES, ("layers.0.missing",))


# --- main(), end to end on CPU ---------------------------------------------------------------
#
# main() only runs for real on a GPU, and a bug in its bookkeeping would surface only after a
# 30-45 minute Kaggle run. So it runs here with the model, registry and scorer swapped for tiny
# fakes, and the test checks the summary.json it writes.

N_LAYERS = 16
UNKNOWN, STEREOTYPE = 2, 0


def _tiny_phi():
    def block():
        layer = nn.Module()
        layer.self_attn = nn.Module()
        layer.self_attn.qkv_proj = nn.Linear(8, 24, bias=False)
        layer.self_attn.o_proj = nn.Linear(8, 8, bias=False)
        layer.mlp = nn.Module()
        layer.mlp.gate_up_proj = nn.Linear(8, 32, bias=False)
        layer.mlp.down_proj = nn.Linear(16, 8, bias=False)
        return layer

    model = nn.Module()
    model.model = nn.Module()
    model.model.layers = nn.ModuleList([block() for _ in range(N_LAYERS)])
    return model


def _items():
    # Two items per category; the fake scorer answers "unknown" on the first, stereotype on the second.
    return {
        cat: [
            SimpleNamespace(question_id=f"{cat}:{i}", category=cat, label=UNKNOWN,
                            unknown_index=UNKNOWN, stereotype_index=STEREOTYPE)
            for i in range(2)
        ]
        for cat in CATEGORIES
    }


def test_main_writes_the_full_reporting_bundle(tmp_path, monkeypatch):
    import src.experiments.logger as logger
    import src.quant.hqq_quantizer as hqq_quantizer
    import src.quant.module_swap as module_swap
    import src.quant.weight_source as weight_source

    monkeypatch.setattr(logger, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(run_repair, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(run_repair.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(run_repair, "_cuda_memory_allocated", FakeMemory(1_000, 1_000 + 5 * 2**20))
    import src.eval.scoring_run as scoring_run

    monkeypatch.setattr(hqq_quantizer, "load_hqq_3bit_model", lambda device, dtype, nbits=3: (_tiny_phi(), None))
    monkeypatch.setattr(weight_source, "CheckpointWeightSource", lambda model_id: None)
    monkeypatch.setattr(module_swap, "SwapRegistry", lambda model, source, device: FakeRegistry(overhead=4 * 2**20))
    monkeypatch.setattr(run_repair, "load_all_categories", lambda context="ambig": _items())
    monkeypatch.setattr(
        scoring_run, "score_items_with_confidence",
        lambda model, tok, items, device, batch_size, desc: (
            [UNKNOWN, STEREOTYPE][: len(items)], [0.9, 0.6][: len(items)]
        ),
    )

    kept = ["model.layers.1.self_attn.qkv_proj", "model.layers.10.self_attn.o_proj"]
    search_summary = tmp_path / "search_summary.json"
    search_summary.write_text(json.dumps(_summary(kept=kept)), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["run_repair", "--search-summary", str(search_summary)])

    run_repair.main()

    (run_dir,) = [p for p in tmp_path.iterdir() if p.name.startswith("hqq_repaired_")]
    assert {p.name for p in run_dir.iterdir()} >= {"question_results.jsonl", "category_summary.csv", "summary.json"}
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))

    assert summary["condition"] == "hqq_repaired" and summary["compute_dtype"] == "fp16"
    assert summary["modules"] == kept and summary["nbits"] == 8
    assert summary["search_run_id"] == "search_20260926T155230Z"

    # Fairness: half of every category answered "unknown", the other half stereotyped.
    assert summary["fairness"]["overall"] == {"usr": 0.5, "srs": 0.5, "n_items": 10}
    assert set(summary["fairness"]["per_category"]) == set(CATEGORIES)

    # Share: one qkv_proj (8x24=192) + one o_proj (8x8=64) of 16 layers x (192+64+256+128) weights.
    restored = summary["restored"]
    assert (restored["modules_restored"], restored["modules_total"]) == (2, 4 * N_LAYERS)
    assert restored["params_restored"] == 192 + 64
    assert restored["params_total"] == N_LAYERS * (192 + 64 + 256 + 128)

    assert summary["memory"]["deployed_overhead_bytes"] == 4 * 2**20
    assert summary["memory"]["gpu_delta_bytes"] == 5 * 2**20

    clock = summary["wall_clock_seconds"]
    assert clock["cycle"] == pytest.approx(clock["repair"] + clock["scoring"], abs=0.2)
    assert summary["resumed"] is False
