"""Runtime module swap: promote one HQQ-quantized linear layer to a different bit-width without
reloading the whole model.

Week 2 spike (highest-risk piece of the pipeline): promote one attention projection from 3-bit
back to 8-bit at runtime and confirm the model's outputs actually change. This validates the
mechanism Stage 5 (selective repair) depends on — keep a higher-precision copy of a chosen
module in memory, swap it into the live quantized model in place.

IMPORTANT — Phi-3.5 doesn't have separate q_proj/k_proj/v_proj like Llama-family models. Its
attention is `qkv_proj` (fused Q+K+V into one Linear) + `o_proj`; its MLP is `gate_up_proj`
(fused) + `down_proj` (checked directly against transformers' own modeling_phi3.py). CLAUDE.md's
"q_proj/k_proj/v_proj/o_proj" candidate list describes a Llama-style architecture, not this
model — the real per-layer candidate set is {qkv_proj, o_proj, gate_up_proj, down_proj}, 4 total
(not 4 attention + N MLP), 128 across 32 layers. Worth fixing in CLAUDE.md before Week 4's
attribution search is built around the wrong candidate names.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from hqq.core.quantize import HQQLinear

from src.quant.hqq_quantizer import DEFAULT_GROUP_SIZE, build_quant_config


def infer_compute_dtype(model: nn.Module) -> torch.dtype:
    """Reads the compute dtype off an already-quantized layer in `model`.

    A promoted module that dequantizes into a different dtype than the rest of the model is a
    silent correctness bug, not a crash: torch will happily upcast across the boundary and the
    model keeps answering, just at a precision nothing recorded. Since the project's baseline
    dtype is now a measured choice rather than a constant (see CLAUDE.md > Precision), taking
    the dtype from the live model is safer than any default this module could hardcode.
    """
    for module in model.modules():
        if isinstance(module, HQQLinear):
            return module.compute_dtype
    raise ValueError(
        "No HQQLinear found in model - promote_module expects an already-quantized model. "
        "Pass compute_dtype explicitly if swapping into an unquantized one."
    )


def get_module_by_path(model: nn.Module, path: str) -> nn.Module:
    """Resolves a dotted module path (e.g. "model.layers.0.self_attn.qkv_proj") to the module."""
    module = model
    for part in path.split("."):
        module = getattr(module, part)
    return module


def set_module_by_path(model: nn.Module, path: str, new_module: nn.Module) -> None:
    """Replaces the module at `path` with `new_module`, in place — no reload."""
    *parent_parts, leaf = path.split(".")
    parent = model
    for part in parent_parts:
        parent = getattr(parent, part)
    setattr(parent, leaf, new_module)


def snapshot_linear(model: nn.Module, path: str) -> nn.Linear:
    """Clones a not-yet-quantized nn.Linear's weight/bias, to requantize at a different
    bit-width later without needing the original checkpoint again.

    Must be called BEFORE the module at `path` is quantized: `HQQLinear` deletes the original
    linear layer's weights in place by default (`del_orig=True`, to free memory as it packs the
    quantized weights), so there's no way to recover them from the model after quantization.
    """
    original = get_module_by_path(model, path)
    if not isinstance(original, nn.Linear):
        raise TypeError(f"Expected nn.Linear at {path!r}, got {type(original).__name__}")

    clone = nn.Linear(original.in_features, original.out_features, bias=original.bias is not None)
    clone.weight.data = original.weight.data.clone()
    if original.bias is not None:
        clone.bias.data = original.bias.data.clone()
    return clone


def promote_module(
    model: nn.Module,
    path: str,
    snapshot: nn.Linear,
    nbits: int,
    group_size: int = DEFAULT_GROUP_SIZE,
    compute_dtype: torch.dtype | None = None,
    device: str = "cuda",
) -> None:
    """Swaps the quantized module at `path` for a fresh HQQLinear at `nbits`, requantized from
    `snapshot` (the pre-quantization weights — see `snapshot_linear`). Modifies `model` in
    place; no reload of the model, no re-running `from_pretrained`.

    `snapshot` itself is never mutated: a fresh nn.Linear is built from its weights each call,
    since `HQQLinear(..., del_orig=True)` deletes whatever linear layer it's handed, and the
    snapshot needs to survive being promoted more than once (e.g. demoted back to 3-bit later).

    `compute_dtype=None` (the default) reads the dtype off the model's existing quantized
    layers, so the promoted module can't end up at a different precision than everything around
    it. Pass it explicitly only to override that deliberately.
    """
    if compute_dtype is None:
        compute_dtype = infer_compute_dtype(model)

    fresh = nn.Linear(snapshot.in_features, snapshot.out_features, bias=snapshot.bias is not None)
    fresh.weight.data = snapshot.weight.data.clone()
    if snapshot.bias is not None:
        fresh.bias.data = snapshot.bias.data.clone()

    quant_config = build_quant_config(nbits=nbits, group_size=group_size)
    promoted = HQQLinear(fresh, quant_config=quant_config, compute_dtype=compute_dtype, device=device)
    set_module_by_path(model, path, promoted)
