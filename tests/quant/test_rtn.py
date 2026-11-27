"""RTN: HQQ's container with plain min-max rounding. The check that matters is that it equals the
hand-written rounding, written out below, bit for bit -- and that HQQ's default did not move."""

import pytest
import torch
import torch.nn as nn
from hqq.core.quantize import HQQLinear

from src.quant.hqq_quantizer import build_quant_config, quantize_in_place
from src.quant.methods import get_method
from src.quant.methods.rtn import RTNMethod
from src.quant.module_swap import SwapRegistry, get_module_by_path

DTYPE = torch.float32


def min_max_rounding(weight: torch.Tensor, nbits: int, group_size: int = 64) -> torch.Tensor:
    """Round-to-nearest by hand: groups of `group_size` consecutive input weights, each rounded onto
    the grid between its own min and max (unrounded zero-point), then dequantized."""
    groups = weight.float().reshape(-1, group_size)
    low, high = groups.min(dim=1, keepdim=True)[0], groups.max(dim=1, keepdim=True)[0]
    max_code = 2 ** nbits - 1
    scale = max_code / (high - low)
    zero = -low * scale
    codes = (groups * scale + zero).round().clamp(0, max_code)
    return ((codes - zero) * (1.0 / scale)).reshape(weight.shape)


def compress(linear: nn.Linear, nbits: int, method) -> HQQLinear:
    """How the whole model is compressed (3-bit isn't a restore width, so not via `restore`)."""
    return HQQLinear(linear, build_quant_config(nbits, 64, optimize=method.optimize),
                     compute_dtype=DTYPE, device="cpu")


def _linear(seed=0, out_features=128, in_features=256):
    torch.manual_seed(seed)
    return nn.Linear(in_features, out_features, bias=False)


@pytest.mark.parametrize("nbits", [3, 4, 8])
def test_rtn_equals_hand_written_min_max_rounding(nbits):
    original = _linear()
    by_hand = min_max_rounding(original.weight.data, nbits)
    assert torch.equal(compress(original, nbits, RTNMethod()).dequantize(), by_hand)
    if nbits in RTNMethod.restore_bits:  # a restored part is the same RTN
        assert torch.equal(RTNMethod().restore(_linear(), nbits, "cpu", DTYPE).dequantize(), by_hand)


def test_rtn_is_not_hqq_but_hqq_is_still_hqq():
    weight = _linear().weight.data
    rtn = compress(_linear(), 3, get_method("rtn")).dequantize()
    hqq = compress(_linear(), 3, get_method("hqq")).dequantize()
    assert not torch.equal(rtn, hqq)
    # HQQ's optimiser is there to lower the rounding error, so on the same weights it must.
    assert (hqq - weight).pow(2).mean() < (rtn - weight).pow(2).mean()


def test_the_default_config_is_still_hqqs_and_rtn_only_switches_off_the_optimiser():
    default = build_quant_config(4, 64)["weight_quant_params"]
    assert default["optimize"] is True and default["round_zero"] is True  # unchanged HQQ
    plain = build_quant_config(4, 64, optimize=False)["weight_quant_params"]
    assert plain["optimize"] is False and plain["round_zero"] is False
    assert {k: v for k, v in plain.items() if k not in ("optimize", "round_zero")} == \
           {k: v for k, v in default.items() if k not in ("optimize", "round_zero")}


def test_rtn_offers_the_bit_widths_of_the_table_and_refuses_others():
    method = get_method("rtn")
    assert (method.name, method.bits, method.restore_bits) == ("rtn", (3, 4), (4, 8))
    with pytest.raises(ValueError, match="RTN compresses to"):
        method.load("any/model", 2, "cpu", "fp16")
    with pytest.raises(ValueError, match="RTN restores parts"):
        method.restore(_linear(), 6, "cpu", DTYPE)


def _tiny():
    from transformers import AutoModelForCausalLM, Qwen2Config

    torch.manual_seed(0)
    config = Qwen2Config(vocab_size=100, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64)
    return AutoModelForCausalLM.from_config(config).eval()


class _Weights:
    def __init__(self, model):
        self.weights = {n: m.weight.data.clone() for n, m in model.named_modules() if isinstance(m, nn.Linear)}

    def get_linear(self, path, dtype):
        weight = self.weights[path]
        linear = nn.Linear(weight.shape[1], weight.shape[0], bias=False)
        linear.weight.data = weight.to(dtype=dtype, copy=True)
        return linear


def _logits(model):
    ids = torch.randint(0, 100, (2, 12), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        return model(input_ids=ids).logits


def test_a_whole_model_compressed_with_rtn_repairs_and_reverts_exactly():
    model = _tiny()
    weights = _Weights(model)
    quantize_in_place(model, nbits=3, group_size=64, compute_dtype=DTYPE, device="cpu", optimize=False)
    path = "model.layers.0.self_attn.q_proj"

    # Every compressed layer is plain rounding of the original weights.
    layer = get_module_by_path(model, path)
    assert torch.equal(layer.dequantize(), min_max_rounding(weights.weights[path], 3))

    before = _logits(model)
    registry = SwapRegistry(model, weights, device="cpu", method="rtn")
    assert registry.method.name == "rtn"
    registry.commit(path, 8)
    restored = get_module_by_path(model, path)
    assert isinstance(restored, HQQLinear)
    assert torch.equal(restored.dequantize(), min_max_rounding(weights.weights[path], 8))  # RTN at 8-bit
    assert not torch.equal(before, _logits(model))
    assert registry.deployed_overhead_bytes() > 0
    registry.revert_all()
    assert torch.equal(before, _logits(model))


def test_a_registry_defaults_to_hqq_even_next_to_rtn():
    model = _tiny()
    weights = _Weights(model)
    quantize_in_place(model, nbits=3, group_size=64, compute_dtype=DTYPE, device="cpu")
    registry = SwapRegistry(model, weights, device="cpu")
    path = "model.layers.0.self_attn.q_proj"
    registry.commit(path, 8)
    hqq_part = get_module_by_path(model, path)
    assert not torch.equal(hqq_part.dequantize(), min_max_rounding(weights.weights[path], 8))
