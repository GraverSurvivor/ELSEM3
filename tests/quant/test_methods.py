"""The compression-method interface (contract C8) and HQQ behind it.

The point of this PR is that nothing changes for HQQ: a model compressed and repaired through the
`Method` interface must give exactly the logits the pre-interface recipe gave. The reference here is
that old recipe written out by hand (`HQQLinear(linear, build_quant_config(...), compute_dtype, device)`),
so these tests fail if the wrapper ever drifts from it. Tiny Qwen2 and GPT-2 (Conv1D) models on CPU."""

import pytest
import torch
import torch.nn as nn
from hqq.core.quantize import HQQLinear

from src.quant.hqq_quantizer import build_quant_config, linearize_conv1d, quantize_in_place
from src.quant.methods import DEFAULT_METHOD, get_method, method_names
from src.quant.methods.base import Method
from src.quant.methods.hqq import HQQMethod, dtype_name, torch_dtype
from src.quant.module_swap import (SwapRegistry, get_module_by_path, infer_compute_dtype,
                                   module_nbytes, set_module_by_path)

DTYPE = torch.float32


def _tiny(model_type: str):
    from transformers import AutoModelForCausalLM, GPT2Config, Qwen2Config

    torch.manual_seed(0)
    if model_type == "qwen2":
        config = Qwen2Config(vocab_size=100, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                             num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64)
    else:
        config = GPT2Config(vocab_size=100, n_embd=64, n_layer=2, n_head=4, n_positions=64)
    model = AutoModelForCausalLM.from_config(config).eval()
    linearize_conv1d(model)
    return model


class _Weights:
    """The original weights of every linear layer, as the checkpoint reader would serve them."""

    def __init__(self, model):
        self.weights = {name: module.weight.data.clone() for name, module in model.named_modules()
                        if isinstance(module, nn.Linear)}

    def get_linear(self, path, dtype):
        weight = self.weights[path]
        linear = nn.Linear(weight.shape[1], weight.shape[0], bias=False)
        linear.weight.data = weight.to(dtype=dtype, copy=True)
        return linear


def _logits(model):
    ids = torch.randint(0, 100, (2, 12), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        return model(input_ids=ids).logits


def _parts(model, count=2):
    return [name for name, module in model.named_modules()
            if isinstance(module, (nn.Linear, HQQLinear)) and not name.endswith("lm_head")][:count]


@pytest.fixture(params=["qwen2", "gpt2"])
def compressed(request):
    model = _tiny(request.param)
    weights = _Weights(model)
    quantize_in_place(model, nbits=3, group_size=64, compute_dtype=DTYPE, device="cpu")
    return model, weights


# --- the registry of methods -------------------------------------------------------------------

def test_hqq_is_the_default_method_and_implements_the_interface():
    method = get_method()
    assert DEFAULT_METHOD == "hqq" and method.name == "hqq" and isinstance(method, HQQMethod)
    assert isinstance(method, Method)
    assert method_names() == ["hqq"]
    assert method.bits == (2, 3, 4) and method.restore_bits == (4, 8)


def test_an_unknown_method_is_refused_in_plain_words():
    with pytest.raises(ValueError, match="Unknown compression method 'awq'.*hqq"):
        get_method("awq")


def test_an_instance_passes_through():
    method = HQQMethod(group_size=32)
    assert get_method(method) is method


def test_hqq_refuses_bit_widths_it_does_not_store():
    # HQQ stores 5- and 6-bit as 8-bit, so they are not offered.
    method = get_method("hqq")
    for bits in (5, 6):
        with pytest.raises(ValueError):
            method.restore(nn.Linear(64, 64, bias=False), bits, "cpu", DTYPE)
    with pytest.raises(ValueError):
        method.load("any/model", 8, "cpu", "fp16")


def test_dtype_names_and_torch_dtypes_convert_both_ways():
    assert dtype_name("fp16") == "fp16" and dtype_name(torch.float16) == "fp16"
    assert torch_dtype("fp16") is torch.float16 and torch_dtype(torch.bfloat16) is torch.bfloat16
    with pytest.raises(ValueError):
        dtype_name("int4")


# --- HQQ through the interface == HQQ the old way ----------------------------------------------

def test_restore_equals_the_hand_built_hqqlinear():
    torch.manual_seed(0)
    original = nn.Linear(256, 128, bias=False)
    reference = nn.Linear(256, 128, bias=False)
    reference.weight.data = original.weight.data.clone()
    by_hand = HQQLinear(reference, build_quant_config(8, 64), compute_dtype=DTYPE, device="cpu")
    by_method = get_method("hqq").restore(original, 8, "cpu", DTYPE)

    assert isinstance(by_method, HQQLinear)
    assert torch.equal(by_method.W_q, by_hand.W_q)
    for key in ("scale", "zero"):
        assert torch.equal(by_method.meta[key], by_hand.meta[key])
    x = torch.randn(3, 256)
    assert torch.equal(by_method(x), by_hand(x))


def test_nbytes_is_the_measured_module_size(compressed):
    model, _ = compressed
    module = get_module_by_path(model, _parts(model, 1)[0])
    assert get_method("hqq").nbytes(module) == module_nbytes(module) > 0


def test_registry_promotions_are_bit_identical_to_the_old_recipe(compressed):
    """Repair two parts through SwapRegistry (now via Method.restore); the logits must equal those of a
    model whose two parts were rebuilt with the pre-interface recipe, and reverting must restore the
    compressed logits exactly."""
    model, weights = compressed
    before = _logits(model)
    paths = _parts(model)

    registry = SwapRegistry(model, weights, device="cpu")
    assert registry.method.name == "hqq"
    for path in paths:
        registry.commit(path, 8)
    repaired = _logits(model)
    assert not torch.equal(before, repaired), "the repair changed nothing"

    # The old recipe, by hand, on a second identical model.
    reference = _tiny("qwen2" if hasattr(model, "model") else "gpt2")
    quantize_in_place(reference, nbits=3, group_size=64, compute_dtype=DTYPE, device="cpu")
    for path in paths:
        set_module_by_path(reference, path, HQQLinear(
            weights.get_linear(path, DTYPE), build_quant_config(8, 64), compute_dtype=DTYPE, device="cpu"))
    assert torch.equal(repaired, _logits(reference))

    registry.revert_all()
    assert torch.equal(before, _logits(model))


def test_registry_takes_the_method_by_name_or_instance(compressed):
    model, weights = compressed
    assert SwapRegistry(model, weights, device="cpu", method="hqq").method.name == "hqq"
    custom = HQQMethod(group_size=64)
    assert SwapRegistry(model, weights, device="cpu", method=custom).method is custom
    assert SwapRegistry(model, weights, device="cpu", group_size=32).method.group_size == 32


def test_overhead_numbers_are_unchanged_by_the_interface(compressed):
    model, weights = compressed
    registry = SwapRegistry(model, weights, device="cpu")
    path = _parts(model, 1)[0]
    before = module_nbytes(get_module_by_path(model, path))
    registry.commit(path, 8)
    assert registry.deployed_overhead_bytes() == module_nbytes(get_module_by_path(model, path)) - before > 0
    assert registry.search_overhead_bytes() == before


# --- infer_compute_dtype no longer assumes HQQ -------------------------------------------------

def test_compute_dtype_is_read_off_an_hqq_layer(compressed):
    model, _ = compressed
    assert infer_compute_dtype(model) == DTYPE


def test_compute_dtype_of_a_model_without_hqq_layers_is_its_float_dtype():
    model = nn.Sequential(nn.Linear(8, 8)).to(torch.float16)
    assert infer_compute_dtype(model) == torch.float16


def test_compute_dtype_with_nothing_to_read_is_an_error():
    with pytest.raises(ValueError, match="compute dtype"):
        infer_compute_dtype(nn.Module())


# --- publishing --------------------------------------------------------------------------------

def test_can_publish_is_empty_for_qwen2_and_explains_gpt2(compressed):
    model, _ = compressed
    reason = get_method("hqq").can_publish(model)
    if hasattr(model, "transformer"):  # GPT-2: HQQ's loader can't rebuild Conv1D models
        assert "GPT-2" in reason
    else:
        assert reason == ""
