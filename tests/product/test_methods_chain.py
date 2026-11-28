"""The whole chain for any compression method: the kit rebuilds the repaired model, it publishes, the
backend compresses with the method, and llama.cpp gets the right lines.

Tiny random Qwen2 and GPT-2 models and a tiny tokenizer are made on the spot (tests/quant/tiny_models.py), so
none of this waits on a download. bitsandbytes cases run where bitsandbytes does (tests/quant/tiny_models.py)."""

import importlib.util
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("hqq")
import torch.nn as nn  # noqa: E402
from hqq.core.quantize import HQQLinear  # noqa: E402

from src.product.hub_publish import PublishError, load_code, model_card, publish  # noqa: E402
from src.product.repair_kit import build_kit, readme  # noqa: E402
from src.quant.module_swap import get_module_by_path  # noqa: E402
from src.quant.repaired import load_repaired  # noqa: E402
from tests.product.test_hub_publish import TOKEN, FakeHub  # noqa: E402
from tests.quant.tiny_models import needs_bnb, save_tiny  # noqa: E402

KIT = Path("src/product/kit_files/fairgate.py")
PARTS = ["model.layers.0.self_attn.q_proj", "model.layers.1.mlp.down_proj"]


def kit_module():
    spec = importlib.util.spec_from_file_location("kit_fairgate", KIT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    folder = tmp_path_factory.mktemp("qwen2")
    return folder, save_tiny(folder, "qwen2")


def patch_for(folder, method="hqq", base_bits=3, nbits=8, **extra):
    return {"name": "t", "model_id": str(folder), "modules": PARTS, "nbits": nbits,
            "base": {"method": method, "nbits": base_bits, "group_size": 64, "compute_dtype": "fp16"}, **extra}


def logits(model, tok):
    ids = tok(["A nurse and a mechanic met. Who was late?"], return_tensors="pt")["input_ids"]
    with torch.no_grad():
        return model(input_ids=ids).logits.float()


# --- the kit -----------------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["hqq", "rtn"])
def test_the_kit_builds_the_same_quantized_layers_as_load_repaired(tiny, method):
    """Mixed 4/8-bit parts included. (The kit leaves the rotary buffer in fp32 while HQQ's loader casts it to
    fp16, so the logits differ in the last digits; every compressed layer is identical.)"""
    folder, _ = tiny
    patch = patch_for(folder, method, module_bits={PARTS[0]: 4})
    kit_model, _, _ = kit_module().load_repaired(patch, "cpu")
    src_model, _ = load_repaired(patch, "cpu")
    count = 0
    for (name, a), (_, b) in zip(kit_model.named_modules(), src_model.named_modules()):
        if isinstance(a, HQQLinear):
            count += 1
            assert isinstance(b, HQQLinear) and a.meta["nbits"] == b.meta["nbits"], name
            assert torch.equal(a.dequantize(), b.dequantize()), name
    assert count > 4
    assert get_module_by_path(kit_model, PARTS[0]).meta["nbits"] == 4
    assert get_module_by_path(kit_model, PARTS[1]).meta["nbits"] == 8


def test_an_old_patch_with_no_method_still_builds_hqq_in_the_kit(tiny):
    folder, _ = tiny
    patch = patch_for(folder)
    del patch["base"]["method"], patch["base"]["compute_dtype"]
    model, _, _ = kit_module().load_repaired(patch, "cpu")
    assert get_module_by_path(model, PARTS[0]).meta["nbits"] == 8
    assert get_module_by_path(model, "model.layers.0.self_attn.k_proj").meta["nbits"] == 3


def test_the_kit_refuses_a_method_it_does_not_know_in_plain_words(tiny):
    folder, _ = tiny
    with pytest.raises(SystemExit, match="can't rebuild a 'awq' model yet"):
        kit_module().load_repaired(patch_for(folder, "awq"), "cpu")


@needs_bnb
def test_the_kit_rebuilds_a_bitsandbytes_repair_with_its_parts_plain(tiny):
    folder, _ = tiny
    patch = patch_for(folder, "bnb_nf4", base_bits=4, nbits=16)
    model, tokenizer, _ = kit_module().load_repaired(patch, "cpu")
    for path in PARTS:
        assert type(get_module_by_path(model, path)) is nn.Linear
    assert type(get_module_by_path(model, "model.layers.1.self_attn.q_proj")).__name__ == "Linear4bit"
    reference, _ = load_repaired(patch, "cpu")
    assert torch.equal(logits(model, tokenizer), logits(reference, tokenizer))  # the kit == the pipeline


def test_the_kit_readme_installs_what_the_method_needs():
    hqq = patch_for("x")
    bnb = patch_for("x", "bnb_nf4", 4, 16)
    assert "pip install transformers hqq accelerate" in readme(hqq, {"memory_mb": 1})
    text = readme(bnb, {"memory_mb": 1})
    assert "pip install transformers accelerate bitsandbytes" in text and "(bitsandbytes NF4)" in text
    assert "(bitsandbytes" not in readme(hqq, {"memory_mb": 1})  # an HQQ readme is as it always was
    from src.product.byom import llamacpp_lines

    assert llamacpp_lines(bnb) == ['--tensor-type "blk\\.0\\.attn_q=f16"', '--tensor-type "blk\\.1\\.ffn_down=f16"']


def test_the_kit_zip_carries_the_method_in_its_patch(tiny, tmp_path):
    import zipfile

    folder, _ = tiny
    path = build_kit(patch_for(folder, "rtn"), {"memory_mb": 2}, [], tmp_path / "kit.zip")
    with zipfile.ZipFile(path) as kit:
        assert json.loads(kit.read("patch.json"))["base"]["method"] == "rtn"


# --- publishing --------------------------------------------------------------------------------

def test_an_rtn_repair_publishes_in_hqqs_format_and_loads_back_bit_identical(tiny, tmp_path):
    from hqq.models.hf.base import AutoHQQHFModel

    folder, tok = tiny
    patch = patch_for(folder, "rtn", module_bits={PARTS[0]: 4})
    model, tokenizer = load_repaired(patch, "cpu")
    hub = FakeHub(tmp_path)
    publish(model, tokenizer, patch, {"memory_mb": 1.0}, TOKEN, api=hub)
    uploaded = tmp_path / "uploaded"
    assert {"qmodel.pt", "config.json", "patch.json", "README.md"} <= {p.name for p in uploaded.iterdir()}
    card = (uploaded / "README.md").read_text(encoding="utf-8")
    assert "compressed with RTN" in card and "AutoHQQHFModel.from_quantized" in card
    loaded = AutoHQQHFModel.from_quantized(str(uploaded), device="cpu", compute_dtype=torch.float16)
    assert loaded.model.layers[0].self_attn.q_proj.meta["nbits"] == 4
    assert torch.equal(logits(model, tokenizer), logits(loaded, tokenizer))


@needs_bnb
def test_a_bitsandbytes_repair_publishes_with_save_pretrained_and_loads_with_plain_from_pretrained(tiny, tmp_path):
    from transformers import AutoModelForCausalLM

    folder, _ = tiny
    patch = patch_for(folder, "bnb_nf4", base_bits=4, nbits=16)
    model, tokenizer = load_repaired(patch, "cpu")
    hub = FakeHub(tmp_path)
    publish(model, tokenizer, patch, {"memory_mb": 1.0}, TOKEN, api=hub)
    uploaded = tmp_path / "uploaded"
    names = {p.name for p in uploaded.iterdir()}
    assert "qmodel.pt" not in names and "config.json" in names and {"README.md", "patch.json"} <= names
    for file in uploaded.iterdir():
        assert TOKEN.encode() not in file.read_bytes()
    card = (uploaded / "README.md").read_text(encoding="utf-8")
    assert "library_name: transformers" in card and "AutoModelForCausalLM.from_pretrained" in card
    assert "AutoHQQHFModel" not in card and "compressed with bitsandbytes NF4" in card

    loaded = AutoModelForCausalLM.from_pretrained(str(uploaded), device_map={"": "cpu"})
    assert torch.equal(logits(model, tokenizer), logits(loaded, tokenizer))
    # The model card's code, run as it is (on the local copy and on CPU).
    exec(compile(load_code(str(uploaded).replace("\\", "/"), device="cpu", method="bnb_nf4"), "card", "exec"), {})


def test_load_code_names_the_loader_that_fits_the_format():
    assert "AutoHQQHFModel.from_quantized" in load_code("a/b")
    assert "AutoHQQHFModel.from_quantized" in load_code("a/b", method="rtn")
    bnb = load_code("a/b", method="bnb_int8")
    assert "AutoModelForCausalLM.from_pretrained" in bnb and "bitsandbytes" in bnb and "AutoHQQHFModel" not in bnb


def test_the_model_card_of_an_old_patch_is_unchanged_hqq():
    card = model_card({"name": "t", "model_id": "m/x", "modules": PARTS, "nbits": 8,
                       "base": {"nbits": 3}}, {"memory_mb": 30}, "student/x-fairgate")
    assert "library_name: hqq" in card and "compressed with HQQ" in card and "HQQ-quantized to 3-bit" in card


def test_gpt2_style_models_are_refused_for_every_method_before_uploading(tmp_path):
    from transformers import GPT2Config, GPT2LMHeadModel

    model = GPT2LMHeadModel(GPT2Config(n_embd=32, n_layer=1, n_head=2, vocab_size=50))
    for method in ("hqq", "rtn", "bnb_nf4", "bnb_int8"):
        hub = FakeHub(tmp_path)
        with pytest.raises(PublishError, match="GPT-2 style"):
            publish(model, None, {**patch_for("gpt2", method), "model_id": "gpt2"}, {}, TOKEN, api=hub)
        assert hub.calls == []


# --- the backend -------------------------------------------------------------------------------

def test_the_backend_compresses_with_the_method_and_restores_through_it(tiny):
    from src.product.torch_backend import TorchBackend

    folder, _ = tiny
    backend = TorchBackend(device="cpu", method="rtn", streaming=False)
    assert (backend.method.name, backend.base_nbits, backend.restore_bits) == ("rtn", 3, (4, 8))
    note = backend.load_compressed(str(folder))
    assert note == "RTN 3-bit, fp16 compute" and backend.registry.method.name == "rtn"
    layer = backend.compressed.model.layers[0].self_attn.q_proj
    assert isinstance(layer, HQQLinear) and layer.meta["nbits"] == 3
    weight = backend.registry.weight_source.get_linear("model.layers.0.self_attn.q_proj", torch.float32).weight.data
    groups = weight.reshape(-1, 64)
    low, high = groups.min(1, keepdim=True)[0], groups.max(1, keepdim=True)[0]
    scale = 7 / (high - low)
    codes = (groups * scale + (-low * scale)).round().clamp(0, 7)
    by_hand = ((codes - (-low * scale)) * (1.0 / scale)).reshape(weight.shape)
    step = ((high - low) / 7).expand_as(groups).reshape(weight.shape)
    off = ((layer.dequantize().float() - by_hand).abs() > 0.05 * step).float().mean()
    assert off < 0.01  # RTN (HQQ's optimiser would move a large share of the weights): fp16 storage of the scale only
    backend.registry.commit("model.layers.0.self_attn.q_proj", 8)
    assert backend.memory_mb() > 0


def test_the_default_backend_is_hqq_exactly_as_before(tiny):
    from src.product.torch_backend import TorchBackend
    from src.quant.hqq_quantizer import load_hqq_3bit_model

    folder, tok = tiny
    backend = TorchBackend(device="cpu", streaming=False)
    assert backend.method.name == "hqq" and backend.restore_bits == (4, 8)
    assert backend.load_compressed(str(folder)) == "HQQ 3-bit, group 64, fp16 compute"
    direct, _ = load_hqq_3bit_model(device="cpu", dtype="fp16", model_id=str(folder), nbits=3)
    assert torch.equal(logits(backend.compressed, tok), logits(direct, tok))


def test_a_bit_width_the_method_does_not_have_is_refused_up_front():
    from src.product.torch_backend import TorchBackend

    with pytest.raises(ValueError, match=r"bitsandbytes NF4 compresses to \[4\] bits, not 3"):
        TorchBackend(device="cpu", method="bnb_nf4", base_nbits=3)
    nf4, int8 = TorchBackend(device="cpu", method="bnb_nf4"), TorchBackend(device="cpu", method="bnb_int8", base_nbits=0)
    assert (nf4.base_nbits, nf4.restore_bits, int8.base_nbits, int8.restore_bits) == (4, (16,), 8, (16,))


def test_a_big_model_with_a_method_that_cannot_stream_is_refused_before_downloading(tiny):
    from src.product.torch_backend import TooBigError, TorchBackend

    folder, _ = tiny
    backend = TorchBackend(device="cpu", method="bnb_nf4", streaming=True)
    with pytest.raises(TooBigError, match="can't be built layer by layer.*Choose HQQ or RTN"):
        backend.preflight(str(folder))


@needs_bnb
def test_the_backend_compresses_with_bitsandbytes_and_restores_a_part_to_16_bit(tiny):
    from src.product.torch_backend import TorchBackend

    folder, _ = tiny
    backend = TorchBackend(device="cpu", method="bnb_nf4", streaming=False)
    assert backend.load_compressed(str(folder)) == "bitsandbytes NF4 4-bit, fp16 compute"
    backend.registry.commit("model.layers.0.self_attn.q_proj", 16)
    part = backend.compressed.model.layers[0].self_attn.q_proj
    assert type(part) is nn.Linear and backend.memory_mb() > 0
    backend.registry.revert_all()
    assert type(backend.compressed.model.layers[0].self_attn.q_proj).__name__ == "Linear4bit"
