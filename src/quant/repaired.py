"""Rebuild a repaired model from its patch: any compression method, any mix of restored bits.

    model, tokenizer = load_repaired(patch, device)

`patch` is a `Patch` or a patch.json dict. The model is built the way the repair was measured: the
patch's base method and bit-width, at the patch's compute dtype (default fp16), from the patch's commit of
the model (`revision`, if it has one), with each kept part restored to its own bits (`module_bits`, else
the patch's `nbits`). A patch without `method`, `module_bits`, `revision` or a dtype reads as it always
did: HQQ, every part at `nbits`, fp16, the latest model.

Methods that swap parts in (HQQ, RTN, GPTQ) load the compressed model and restore each kept part with the
same `SwapRegistry` the search used, from the checkpoint's original weights. bitsandbytes lists the kept
parts when it loads instead (`method.restores_on_load`), so they simply stay plain 16-bit.
"""

from __future__ import annotations

from src.quant.methods import GROUPED, Method, get_method

DEFAULT_BASE = {"method": "hqq", "nbits": 3, "group_size": 64, "compute_dtype": "fp16"}


class PatchError(ValueError):
    """A patch that can't be rebuilt, said in words a person can act on."""


def as_dict(patch) -> dict:
    return patch.to_dict() if hasattr(patch, "to_dict") else dict(patch)


def method_of(patch) -> Method:
    """The compression method a patch's base model was built with (HQQ for an old patch)."""
    base = {**DEFAULT_BASE, **as_dict(patch).get("base", {})}
    name = base["method"]
    return get_method(name, group_size=base["group_size"]) if name in GROUPED else get_method(name)


def part_bits(patch) -> dict:
    """part -> the bits it is restored to, in the patch's order: `module_bits`, else `nbits`."""
    data = as_dict(patch)
    own = data.get("module_bits") or {}
    return {path: int(own.get(path, data["nbits"])) for path in data["modules"]}


def load_repaired(patch, device: str = "cuda", streaming: bool = False, model_id: str | None = None):
    """(model, tokenizer): the compressed model with the patch's parts restored. `model_id` overrides
    where the weights come from (a local copy); `streaming` builds HQQ-family models layer by layer."""
    data = as_dict(patch)
    base = {**DEFAULT_BASE, **data.get("base", {})}
    method = method_of(data)
    keep = part_bits(data)
    source_id, revision = model_id or data["model_id"], data.get("revision")

    if base["nbits"] not in method.bits:
        raise PatchError(f"{method.label} compresses to {list(method.bits)} bits, but this patch's base is "
                         f"{base['nbits']}-bit.")
    wrong = {path: bits for path, bits in keep.items() if bits not in method.restore_bits}
    if wrong:
        raise PatchError(f"{method.label} can restore parts only to {list(method.restore_bits)} bits; this patch "
                         f"asks for {sorted(set(wrong.values()))} ({len(wrong)} parts).")

    if method.restores_on_load:
        model, tokenizer = method.load(source_id, base["nbits"], device, base["compute_dtype"],
                                       streaming=streaming, restored=list(keep), revision=revision)
        _check_parts(model, keep, data["model_id"])
        return model, tokenizer

    from src.models.adapter import adapter_for_id
    from src.quant.module_swap import SwapRegistry
    from src.quant.weight_source import CheckpointWeightSource

    model, tokenizer = method.load(source_id, base["nbits"], device, base["compute_dtype"],
                                   streaming=streaming, revision=revision)
    _check_parts(model, keep, data["model_id"])
    adapter = adapter_for_id(source_id)
    transposed = {path for path in keep if adapter.is_transposed(path)}  # GPT-2 style Conv1D weights
    registry = SwapRegistry(model, CheckpointWeightSource(source_id, revision=revision,
                                                          transposed=lambda path: path in transposed),
                            device=device, method=method)
    for path, bits in keep.items():
        registry.commit(path, bits)
    registry.release_baselines()  # nothing will be reverted: a deployed model carries only the repair
    return model, tokenizer


def _check_parts(model, keep: dict, model_id: str) -> None:
    names = {name for name, _ in model.named_modules()}
    missing = sorted(set(keep) - names)
    if missing:
        raise PatchError(f"These parts aren't in {model_id}: {missing[:5]}. Is this the right model?")
