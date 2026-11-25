"""SwapRegistry.without: a committed module taken out for a moment, the exact object put back."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("hqq")


@pytest.fixture
def registry():
    import torch.nn as nn
    from hqq.core.quantize import HQQLinear

    from src.quant.hqq_quantizer import build_quant_config
    from src.quant.module_swap import SwapRegistry

    torch.manual_seed(0)
    weight = torch.randn(64, 128)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            linear = nn.Linear(128, 64, bias=False)
            linear.weight.data = weight.clone()
            self.part = HQQLinear(linear, build_quant_config(3), compute_dtype=torch.float32, device="cpu")

    class Weights:
        def get_linear(self, path, dtype):
            linear = nn.Linear(128, 64, bias=False)
            linear.weight.data = weight.clone().to(dtype)
            return linear

    return SwapRegistry(Model(), Weights(), device="cpu")


def test_without_shows_the_starting_module_and_puts_the_promoted_one_back(registry):
    start = registry.model.part
    registry.commit("part", 4)
    promoted = registry.model.part
    assert promoted is not start
    with registry.without("part"):
        assert registry.model.part is start and registry.active == {"part": 4}
    assert registry.model.part is promoted
    with pytest.raises(RuntimeError), registry.without("part"):
        raise RuntimeError("out of memory")
    assert registry.model.part is promoted  # an error inside changes nothing
    registry.revert("part")
    assert registry.model.part is start


def test_trials_nest_on_top_of_a_committed_part(registry):
    registry.commit("part", 4)
    promoted = registry.model.part
    with registry.without("part"), registry.trial("part", 8):
        assert registry.model.part.meta["nbits"] == 8
    with registry.trial("part", 8):
        assert registry.model.part.meta["nbits"] == 8
    assert registry.model.part is promoted and registry.active == {"part": 4}


def test_only_a_committed_part_can_be_taken_out(registry):
    with pytest.raises(KeyError), registry.without("part"):
        pass
    registry.commit("part", 8)
    registry.release_baselines()
    with pytest.raises(KeyError), registry.without("part"):
        pass
