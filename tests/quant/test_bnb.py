"""bitsandbytes NF4 / int8 behind the Method interface.

bitsandbytes is built for GPUs. Recent versions also run on CPU, which these tests use when it works
(`bnb_runs`); where it doesn't (some laptops), they skip, and the Colab cell in RUNBOOK.md runs them on
the T4. The skip-list tests need no bitsandbytes at all."""

import pytest
import torch
import torch.nn as nn

from src.quant.methods import get_method
from src.quant.methods.bnb import BnbInt8Method, BnbNF4Method, skip_modules
from src.quant.module_swap import SwapRegistry, get_module_by_path, module_nbytes

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TINY = dict(vocab_size=100, hidden_size=64, intermediate_size=128, num_hidden_layers=12,
            num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64)


def bnb_runs() -> bool:
    try:
        import bitsandbytes  # noqa: F401
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig, Qwen2Config

        config = Qwen2Config(**{**TINY, "num_hidden_layers": 1})
        folder = __import__("tempfile").mkdtemp()
        AutoModelForCausalLM.from_config(config).save_pretrained(folder)
        model = AutoModelForCausalLM.from_pretrained(
            folder, quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                           bnb_4bit_compute_dtype=torch.float16),
            dtype=torch.float16, device_map={"": DEVICE})
        with torch.no_grad():
            model(input_ids=torch.zeros(1, 4, dtype=torch.long, device=DEVICE))
        return True
    except Exception:  # noqa: BLE001 -- any failure means "not usable here"
        return False


needs_bnb = pytest.mark.skipif(not bnb_runs(), reason="bitsandbytes doesn't run on this machine (needs a GPU)")


# --- the skip list (no bitsandbytes needed) ----------------------------------------------------

def test_skip_list_is_escaped_full_paths_plus_lm_head():
    patterns = skip_modules(["model.layers.1.self_attn.qkv_proj"])
    assert patterns == [r"model\.layers\.1\.self_attn\.qkv_proj", "lm_head"]
    assert skip_modules() == ["lm_head"]


def test_layer_1_does_not_also_match_layer_11_in_transformers_own_matcher():
    # transformers matches the list as regular expressions and suffixes; this is its real function.
    try:
        from transformers.quantizers.quantizers_utils import should_convert_module
    except ImportError:
        pytest.skip("this transformers has no should_convert_module")
    patterns = skip_modules(["model.layers.1.self_attn.qkv_proj", "model.layers.2.mlp.down_proj"])
    assert not should_convert_module("model.layers.1.self_attn.qkv_proj", patterns)  # kept: not converted
    assert not should_convert_module("model.layers.2.mlp.down_proj", patterns)
    assert not should_convert_module("lm_head", patterns)
    assert should_convert_module("model.layers.11.self_attn.qkv_proj", patterns)  # layer 11 IS converted
    assert should_convert_module("model.layers.12.mlp.down_proj", patterns)
    assert should_convert_module("model.layers.1.self_attn.o_proj", patterns)
    assert should_convert_module("model.layers.22.mlp.down_proj", patterns)


def test_the_table_and_the_refusals():
    nf4, int8 = get_method("bnb_nf4"), get_method("bnb_int8")
    assert (nf4.bits, nf4.restore_bits, int8.bits, int8.restore_bits) == ((4,), (16,), (8,), (16,))
    with pytest.raises(ValueError, match="compresses to \\[4\\]"):
        nf4.load("any/model", 3, "cpu", "fp16")
    with pytest.raises(ValueError, match="layer by layer"):
        int8.load("any/model", 8, "cpu", "fp16", streaming=True)
    with pytest.raises(ValueError, match="plain 16-bit"):
        nf4.restore(nn.Linear(8, 8), 8, "cpu", torch.float16)
    assert nf4.can_publish(object()) == ""


# --- real bitsandbytes -------------------------------------------------------------------------

@pytest.fixture(scope="module")
def folder(tmp_path_factory):
    from transformers import AutoModelForCausalLM, Qwen2Config

    torch.manual_seed(0)
    path = tmp_path_factory.mktemp("tiny")
    AutoModelForCausalLM.from_config(Qwen2Config(**TINY)).save_pretrained(path)
    return path


@pytest.fixture(autouse=True)
def no_tokenizer(monkeypatch):
    monkeypatch.setattr("src.quant.methods.bnb.load_tokenizer", lambda model_id: None)


def _logits(model):
    ids = torch.randint(0, 100, (2, 12), generator=torch.Generator().manual_seed(1)).to(DEVICE)
    with torch.no_grad():
        return model(input_ids=ids).logits.float().cpu()


@needs_bnb
@pytest.mark.parametrize("method_name, bits, kind", [("bnb_nf4", 4, "Linear4bit"), ("bnb_int8", 8, "Linear8bitLt")])
def test_load_quantizes_everything_but_the_head_and_the_kept_parts(folder, method_name, bits, kind):
    method = get_method(method_name)
    kept = ["model.layers.1.self_attn.q_proj", "model.layers.2.mlp.down_proj"]
    model, _ = method.load(str(folder), bits, DEVICE, "fp16", restored=kept)
    assert type(get_module_by_path(model, "model.layers.0.self_attn.q_proj")).__name__ == kind
    assert type(get_module_by_path(model, "model.layers.11.self_attn.q_proj")).__name__ == kind  # not layer 1
    for path in kept + ["lm_head"]:
        part = get_module_by_path(model, path)
        assert type(part) is nn.Linear and part.weight.dtype == torch.float16, path


@needs_bnb
@pytest.mark.parametrize("method_name, bits", [("bnb_nf4", 4), ("bnb_int8", 8)])
def test_nbytes_counts_the_extra_state_not_only_the_packed_weight(folder, method_name, bits):
    method = get_method(method_name)
    model, _ = method.load(str(folder), bits, DEVICE, "fp16")
    layer = get_module_by_path(model, "model.layers.0.mlp.down_proj")  # 64 x 128 weights
    weights = 64 * 128
    packed = weights // 2 if bits == 4 else weights
    measured = method.nbytes(layer)
    assert measured == module_nbytes(layer)
    assert packed < measured < weights * 2, "measured size should be the packed weight plus scales, below fp16"
    plain = method.restore(nn.Linear(128, 64, bias=False), 16, DEVICE, torch.float16)
    assert method.nbytes(plain) == weights * 2 > measured  # restoring costs real memory


@needs_bnb
@pytest.mark.parametrize("method_name, bits", [("bnb_nf4", 4), ("bnb_int8", 8)])
def test_restoring_parts_through_the_registry_and_reverting_is_exact(folder, method_name, bits):
    from transformers import AutoModelForCausalLM

    method = get_method(method_name)
    model, _ = method.load(str(folder), bits, DEVICE, "fp16")
    original = AutoModelForCausalLM.from_pretrained(str(folder), dtype=torch.float16)

    class Source:  # the original 16-bit weights, as the checkpoint reader serves them
        def get_linear(self, path, dtype):
            return get_module_by_path(original, path).to(dtype)

    registry = SwapRegistry(model, Source(), device=DEVICE, method=method)
    path = "model.layers.3.mlp.down_proj"
    before_module, before = get_module_by_path(model, path), _logits(model)
    registry.commit(path, 16)
    part = get_module_by_path(model, path)
    assert type(part) is nn.Linear and torch.equal(part.weight.cpu(), get_module_by_path(original, path).weight)
    assert not torch.equal(before, _logits(model))
    assert registry.deployed_overhead_bytes() == method.nbytes(part) - method.nbytes(before_module) > 0
    registry.revert_all()
    assert get_module_by_path(model, path) is before_module
    assert torch.equal(before, _logits(model))


@needs_bnb
def test_a_repaired_model_saves_and_loads_with_the_normal_from_pretrained(folder, tmp_path):
    from transformers import AutoModelForCausalLM

    method = get_method("bnb_nf4")
    kept = ["model.layers.1.self_attn.q_proj"]
    model, _ = method.load(str(folder), 4, DEVICE, "fp16", restored=kept)
    model.save_pretrained(tmp_path / "saved")
    loaded = AutoModelForCausalLM.from_pretrained(tmp_path / "saved", dtype=torch.float16, device_map={"": DEVICE})
    assert type(get_module_by_path(loaded, kept[0])) is nn.Linear
    assert type(get_module_by_path(loaded, "model.layers.11.self_attn.q_proj")).__name__ == "Linear4bit"
    assert torch.equal(_logits(model), _logits(loaded))
