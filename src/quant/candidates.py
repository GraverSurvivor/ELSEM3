"""The candidate modules Stage 4's attribution search is allowed to restore precision on.

Phi-3.5 fuses projections that Llama-family models keep separate: attention is `qkv_proj`
(Q+K+V in one Linear) + `o_proj`, and the MLP is `gate_up_proj` (gate+up fused) + `down_proj`.
So the per-layer candidate set is 4 modules, not the 4 attention projections + N MLP that a
Llama-shaped plan would assume — 128 total across 32 layers. Checked against transformers' own
`modeling_phi3.py`.

Ordering is attention-before-MLP by design (CLAUDE.md, Stage 4). Attention modules are also the
cheaper half to promote — `qkv_proj` and `o_proj` are 28.3M and 9.4M parameters against
`gate_up_proj`'s 50.3M — so a search that stops early on an attention hit buys its fairness
recovery at a lower memory cost than one that finds the same recovery in the MLP.
"""

from __future__ import annotations

import torch.nn as nn

ATTENTION_MODULES = ("self_attn.qkv_proj", "self_attn.o_proj")
MLP_MODULES = ("mlp.gate_up_proj", "mlp.down_proj")

LAYER_PREFIX = "model.layers"


def candidate_paths(n_layers: int = 32) -> list[str]:
    """All promotable module paths, attention across every layer first, then MLP.

    Grouped by module type rather than by layer so that a search truncated by a time budget
    still covers all 32 layers of the cheaper attention modules, instead of exhaustively
    searching layer 0 and never reaching layer 31.
    """
    return [
        f"{LAYER_PREFIX}.{layer}.{suffix}"
        for suffix in ATTENTION_MODULES + MLP_MODULES
        for layer in range(n_layers)
    ]


def candidate_paths_for_model(model: nn.Module) -> list[str]:
    """Same list, with the layer count read off the model instead of assumed."""
    return candidate_paths(n_layers=len(model.model.layers))
