"""load_repaired: a patch (any method, any mix of restored bits) -> the repaired model.

The reference for HQQ is the rebuild that existed before methods did (the Studio's layer-by-layer loader with
`keep=`, which is bit-identical to the model the repairs were measured on). A patch with no `method` /
`module_bits` must give exactly those logits."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("hqq")
import torch.nn as nn  # noqa: E402
from hqq.core.quantize import HQQLinear  # noqa: E402

from src.quant.module_swap import get_module_by_path  # noqa: E402
from src.quant.repaired import PatchError, load_repaired, method_of, part_bits  # noqa: E402
from tests.quant.tiny_models import needs_bnb, save_tiny  # noqa: E402

PARTS = {"qwen2": ["model.layers.0.self_attn.q_proj", "model.layers.1.mlp.down_proj"],
         "gpt2": ["transformer.h.0.attn.c_attn", "transformer.h.1.mlp.c_proj"]}


@pytest.fixture(scope="module", params=["qwen2", "gpt2"])
def tiny(request, tmp_path_factory):
    folder = tmp_path_factory.mktemp(request.param)
    tok = save_tiny(folder, request.param)
    return request.param, folder, tok


def patch_for(kind, folder, **extra):
    return {"name": "t", "model_id": str(folder), "modules": PARTS[kind], "nbits": 8,
            "base": {"method": "hqq", "nbits": 3, "group_size": 64, "compute_dtype": "fp16"}, **extra}


def logits(model, tok):
    inputs = tok(["A nurse and a mechanic met. Who was late?"], return_tensors="pt")
    with torch.no_grad():
        return model(input_ids=inputs["input_ids"]).logits.float()


def existing_rebuild(folder, keep, base_bits=3, optimize=True):
    """The rebuild that existed before methods: the Studio's layer-by-layer loader with the kept parts at
    their own bits (bit-identical to the model the repair was measured on)."""
    from src.quant.streaming import load_streaming

    return load_streaming(str(folder), base_bits, device="cpu", keep=keep, optimize=optimize)[0].eval()


def test_an_old_patch_gives_exactly_the_old_models_logits(tiny):
    kind, folder, tok = tiny
    patch = patch_for(kind, folder)  # no module_bits, no revision: an old patch
    model, tokenizer = load_repaired(patch, device="cpu")
    reference = existing_rebuild(folder, {path: 8 for path in PARTS[kind]})
    assert torch.equal(logits(model, tokenizer), logits(reference, tok))


def test_a_patch_with_no_base_at_all_reads_as_hqq_3bit(tiny):
    kind, folder, tok = tiny
    patch = patch_for(kind, folder)
    del patch["base"]
    assert method_of(patch).name == "hqq"
    model, tokenizer = load_repaired(patch, device="cpu")
    assert torch.equal(logits(model, tokenizer), logits(existing_rebuild(folder, {p: 8 for p in PARTS[kind]}), tok))


def test_module_bits_give_each_part_its_own_precision(tiny):
    kind, folder, tok = tiny
    first, second = PARTS[kind]
    patch = patch_for(kind, folder, module_bits={first: 4})  # first at 4-bit, second at the patch's 8
    assert part_bits(patch) == {first: 4, second: 8}
    model, tokenizer = load_repaired(patch, device="cpu")
    assert get_module_by_path(model, first).meta["nbits"] == 4
    assert get_module_by_path(model, second).meta["nbits"] == 8
    other = next(n for n, m in model.named_modules() if isinstance(m, HQQLinear) and n not in PARTS[kind])
    assert get_module_by_path(model, other).meta["nbits"] == 3
    assert torch.equal(logits(model, tokenizer), logits(existing_rebuild(folder, {first: 4, second: 8}), tok))


def test_rtn_patch_rebuilds_a_plain_rounded_model_with_plain_rounded_parts(tiny):
    kind, folder, tok = tiny
    patch = patch_for(kind, folder)
    patch["base"] = {**patch["base"], "method": "rtn"}
    model, tokenizer = load_repaired(patch, device="cpu")
    reference = existing_rebuild(folder, {path: 8 for path in PARTS[kind]}, optimize=False)
    assert torch.equal(logits(model, tokenizer), logits(reference, tok))
    hqq_logits = logits(load_repaired(patch_for(kind, folder), device="cpu")[0], tokenizer)
    assert not torch.equal(logits(model, tokenizer), hqq_logits)  # and it is not the HQQ model


def test_the_patchs_commit_and_dtype_reach_the_loader(tiny, monkeypatch):
    kind, folder, _ = tiny
    seen = {}
    from src.quant.methods.hqq import HQQMethod

    real = HQQMethod.load

    def spy(self, model_id, nbits, device, dtype, streaming=False, revision=None):
        seen.update(model_id=model_id, nbits=nbits, dtype=dtype, revision=revision)
        return real(self, model_id, nbits, device, dtype, streaming, revision)

    monkeypatch.setattr(HQQMethod, "load", spy)
    patch = patch_for(kind, folder, revision="abc123")
    patch["base"]["nbits"] = 4
    load_repaired(patch, device="cpu")
    assert seen == {"model_id": str(folder), "nbits": 4, "dtype": "fp16", "revision": "abc123"}


def test_a_part_the_model_does_not_have_is_named(tiny):
    kind, folder, _ = tiny
    patch = patch_for(kind, folder)
    patch["modules"] = ["model.layers.9.self_attn.q_proj"]
    with pytest.raises(PatchError, match="aren't in .*layers.9"):
        load_repaired(patch, device="cpu")


def test_bits_a_method_does_not_store_are_refused_in_plain_words(tiny):
    kind, folder, _ = tiny
    with pytest.raises(PatchError, match=r"HQQ can restore parts only to \[4, 8\] bits.*\[6\]"):
        load_repaired(patch_for(kind, folder, module_bits={PARTS[kind][0]: 6}), device="cpu")
    wrong_base = patch_for(kind, folder)
    wrong_base["base"]["nbits"] = 5
    with pytest.raises(PatchError, match=r"compresses to \[2, 3, 4\] bits"):
        load_repaired(wrong_base, device="cpu")
    nf4 = patch_for(kind, folder)
    nf4["base"] = {**nf4["base"], "method": "bnb_nf4", "nbits": 4}
    with pytest.raises(PatchError, match=r"restore parts only to \[16\]"):  # nbits 8 isn't a bitsandbytes width
        load_repaired(nf4, device="cpu")


def test_a_patch_object_is_taken_like_its_dict():
    class Stub:
        def to_dict(self):
            return {"modules": ["a"], "nbits": 8, "base": {"method": "rtn", "nbits": 3, "group_size": 32}}

    assert method_of(Stub()).name == "rtn" and method_of(Stub()).group_size == 32
    assert part_bits(Stub()) == {"a": 8}


@needs_bnb
def test_bitsandbytes_patch_keeps_exactly_its_parts_in_16_bit(tmp_path):
    folder = tmp_path / "q"
    tok = save_tiny(folder, "qwen2")
    parts = PARTS["qwen2"]
    patch = patch_for("qwen2", folder, nbits=16)
    patch["base"] = {"method": "bnb_nf4", "nbits": 4, "compute_dtype": "fp16"}
    model, tokenizer = load_repaired(patch, device="cpu")
    for path in parts:
        part = get_module_by_path(model, path)
        assert type(part) is nn.Linear and part.weight.dtype == torch.float16
    assert type(get_module_by_path(model, "model.layers.1.self_attn.q_proj")).__name__ == "Linear4bit"
    assert torch.isfinite(logits(model, tokenizer)).all()
