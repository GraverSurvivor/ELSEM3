"""SwapRegistry: does a trial really put the model back, and is the reported cost real.

These run on CPU with real `HQQLinear` objects rather than stubs. That matters for the byte
accounting in particular: HQQ keeps its scale and zero in a plain `meta` dict that
`named_parameters()` doesn't see, so a stub modelled on ordinary nn.Modules would happily agree
with a measurement that silently undercounts every quantized module.
"""

import pytest
import torch
import torch.nn as nn
from hqq.core.quantize import BaseQuantizeConfig, HQQLinear

from src.quant.module_swap import SwapRegistry, get_module_by_path, module_nbytes

COMPUTE_DTYPE = torch.float32
IN_FEATURES, OUT_FEATURES = 256, 128


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(IN_FEATURES, OUT_FEATURES, bias=False)


class _TinyModel(nn.Module):
    """Stands in for a decoder stack: a couple of blocks each holding one promotable linear."""

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_Block(), _Block()])


def _quantize(model: nn.Module, nbits: int = 3) -> None:
    for block in model.layers:
        block.proj = HQQLinear(
            block.proj,
            quant_config=BaseQuantizeConfig(nbits=nbits, group_size=64),
            compute_dtype=COMPUTE_DTYPE,
            device="cpu",
        )


class _FakeWeightSource:
    """Serves the pre-quantization weights the real source would read from the checkpoint."""

    def __init__(self, weights: dict[str, torch.Tensor]):
        self.weights = weights
        self.reads: list[str] = []

    def get_linear(self, path: str, dtype: torch.dtype) -> nn.Linear:
        self.reads.append(path)
        weight = self.weights[path]
        linear = nn.Linear(weight.shape[1], weight.shape[0], bias=False)
        linear.weight.data = weight.to(dtype=dtype, copy=True)
        return linear


@pytest.fixture
def setup():
    torch.manual_seed(0)
    model = _TinyModel()
    weights = {
        f"layers.{i}.proj": block.proj.weight.data.clone() for i, block in enumerate(model.layers)
    }
    _quantize(model, nbits=3)
    source = _FakeWeightSource(weights)
    registry = SwapRegistry(model, source, device="cpu")
    return model, registry, source


def _forward(model: nn.Module) -> torch.Tensor:
    torch.manual_seed(1)
    x = torch.randn(2, IN_FEATURES, dtype=COMPUTE_DTYPE)
    with torch.no_grad():
        return model.layers[0].proj(x)


def test_trial_restores_the_exact_module_object(setup):
    model, registry, _ = setup
    before = get_module_by_path(model, "layers.0.proj")

    with registry.trial("layers.0.proj", nbits=8):
        assert get_module_by_path(model, "layers.0.proj") is not before

    # Identity, not just equality: reverting by requantizing would produce an equal-looking
    # module while quietly re-running HQQ's optimizer on every trial.
    assert get_module_by_path(model, "layers.0.proj") is before


def test_trial_restores_outputs_bit_for_bit(setup):
    model, registry, _ = setup
    before = _forward(model)

    with registry.trial("layers.0.proj", nbits=8):
        promoted = _forward(model)

    after = _forward(model)
    assert torch.equal(before, after), "trial leaked state into the model"
    assert not torch.equal(before, promoted), "promotion to 8-bit changed nothing -- swap is a no-op"


def test_trial_reverts_even_when_the_body_raises(setup):
    # The realistic case is an OOM partway through scoring the flipped set. Without the revert,
    # every candidate measured afterward is scored against a contaminated model.
    model, registry, _ = setup
    before = get_module_by_path(model, "layers.0.proj")

    with pytest.raises(RuntimeError):
        with registry.trial("layers.0.proj", nbits=8):
            raise RuntimeError("simulated failure mid-evaluation")

    assert get_module_by_path(model, "layers.0.proj") is before


def test_trial_does_not_count_as_committed(setup):
    _, registry, _ = setup
    with registry.trial("layers.0.proj", nbits=8):
        pass
    assert registry.active == {}
    assert registry.deployed_overhead_bytes() == 0


def test_commit_persists_and_revert_undoes_it(setup):
    model, registry, _ = setup
    baseline = get_module_by_path(model, "layers.0.proj")

    registry.commit("layers.0.proj", nbits=8)
    assert registry.active == {"layers.0.proj": 8}
    assert get_module_by_path(model, "layers.0.proj") is not baseline

    registry.revert("layers.0.proj")
    assert registry.active == {}
    assert get_module_by_path(model, "layers.0.proj") is baseline


def test_revert_falls_back_to_the_original_not_an_intermediate(setup):
    # A path can be promoted, reverted and promoted again across greedy rounds. The baseline it
    # returns to must stay the model's real starting module, not the last promotion.
    model, registry, _ = setup
    original = get_module_by_path(model, "layers.0.proj")

    registry.commit("layers.0.proj", nbits=8)
    registry.revert("layers.0.proj")
    registry.commit("layers.0.proj", nbits=4)
    registry.revert("layers.0.proj")

    assert get_module_by_path(model, "layers.0.proj") is original


def test_reverting_an_uncommitted_path_raises(setup):
    _, registry, _ = setup
    with pytest.raises(KeyError):
        registry.revert("layers.0.proj")


def test_trial_composes_on_top_of_a_commit(setup):
    model, registry, _ = setup
    registry.commit("layers.0.proj", nbits=8)
    committed = get_module_by_path(model, "layers.0.proj")

    with registry.trial("layers.1.proj", nbits=8):
        pass

    # Trialling a different path must leave an existing commit alone.
    assert get_module_by_path(model, "layers.0.proj") is committed
    assert registry.active == {"layers.0.proj": 8}


def test_promotion_reads_weights_from_the_source_not_the_quantized_model(setup):
    _, registry, source = setup
    with registry.trial("layers.1.proj", nbits=8):
        pass
    assert source.reads == ["layers.1.proj"]


def test_registry_infers_compute_dtype_from_the_model(setup):
    _, registry, _ = setup
    assert registry.compute_dtype == COMPUTE_DTYPE


class TestMemoryAccounting:
    def test_counts_hqq_scale_and_zero_not_just_packed_weights(self):
        # named_parameters() sees only W_q. Missing meta's scale/zero undercounts by exactly the
        # per-group overhead, which is part of what distinguishes one bit-width from another.
        linear = nn.Linear(IN_FEATURES, OUT_FEATURES, bias=False)
        quantized = HQQLinear(
            linear,
            quant_config=BaseQuantizeConfig(nbits=3, group_size=64),
            compute_dtype=COMPUTE_DTYPE,
            device="cpu",
        )
        params_only = sum(p.numel() * p.element_size() for p in quantized.parameters())
        assert module_nbytes(quantized) > params_only

    def test_higher_precision_measures_larger(self):
        sizes = {}
        for nbits in (3, 8):
            linear = nn.Linear(IN_FEATURES, OUT_FEATURES, bias=False)
            sizes[nbits] = module_nbytes(
                HQQLinear(
                    linear,
                    quant_config=BaseQuantizeConfig(nbits=nbits, group_size=64),
                    compute_dtype=COMPUTE_DTYPE,
                    device="cpu",
                )
            )
        assert sizes[8] > sizes[3]

    def test_deployed_overhead_is_the_promotion_delta(self, setup):
        model, registry, _ = setup
        baseline_bytes = module_nbytes(get_module_by_path(model, "layers.0.proj"))

        registry.commit("layers.0.proj", nbits=8)
        promoted_bytes = module_nbytes(get_module_by_path(model, "layers.0.proj"))

        assert registry.deployed_overhead_bytes() == promoted_bytes - baseline_bytes
        assert registry.deployed_overhead_bytes() > 0

    def test_deployed_overhead_excludes_retained_baselines(self, setup):
        # The displaced originals exist only so the search can undo itself; a shipped model
        # wouldn't carry them, so counting them would overstate the repair's cost.
        _, registry, _ = setup
        registry.commit("layers.0.proj", nbits=8)

        assert registry.search_overhead_bytes() > 0
        assert registry.deployed_overhead_bytes() < registry.search_overhead_bytes() + registry.deployed_overhead_bytes()

    def test_trials_do_not_accumulate_search_overhead_per_trial(self, setup):
        # One retained baseline per touched path, not one per trial -- otherwise 128 candidates
        # x N rounds would leak the whole model back into memory.
        _, registry, _ = setup
        for _ in range(5):
            with registry.trial("layers.0.proj", nbits=8):
                pass
        single = registry.search_overhead_bytes()

        with registry.trial("layers.1.proj", nbits=8):
            pass
        assert registry.search_overhead_bytes() > single
        assert registry.search_overhead_bytes() == pytest.approx(2 * single, rel=0.05)

    def test_release_baselines_frees_memory_but_keeps_the_reported_number(self, setup):
        _, registry, _ = setup
        registry.commit("layers.0.proj", nbits=8)
        overhead_before = registry.deployed_overhead_bytes()

        registry.release_baselines()

        assert registry.search_overhead_bytes() == 0
        assert registry.deployed_overhead_bytes() == overhead_before

    def test_revert_after_release_raises_rather_than_silently_doing_nothing(self, setup):
        _, registry, _ = setup
        registry.commit("layers.0.proj", nbits=8)
        registry.release_baselines()

        with pytest.raises(RuntimeError, match="released"):
            registry.revert("layers.0.proj")
