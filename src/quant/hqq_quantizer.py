"""HQQ 3-bit quantization of Phi-3.5-mini-Instruct.

Week 1 scope: uniform 3-bit quantization end-to-end, no mixed precision yet (that's Week 2 —
runtime module swap via `restore_modules`, to live in this module once built).

Uses transformers' native HQQ integration (`HqqConfig` passed to `from_pretrained`) rather than
the `hqq` library's own `AutoHQQHFModel` wrapper: it's the path HQQ's own README lists as
preferred, and it reuses the same native `Phi3ForCausalLM` path `model_loader.py` already relies
on (no `trust_remote_code`), instead of a separate model-patching code path with its own quirks.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, HqqConfig, PreTrainedModel, PreTrainedTokenizerBase

from src.eval.model_loader import MODEL_ID

DEFAULT_NBITS = 3
DEFAULT_GROUP_SIZE = 64


def build_3bit_quant_config(nbits: int = DEFAULT_NBITS, group_size: int = DEFAULT_GROUP_SIZE) -> HqqConfig:
    """Builds a uniform (non-mixed-precision) HQQ quant config applied to every linear layer."""
    return HqqConfig(nbits=nbits, group_size=group_size)


def load_hqq_3bit_model(
    device_map: str = "cuda", nbits: int = DEFAULT_NBITS, group_size: int = DEFAULT_GROUP_SIZE
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Loads Phi-3.5-mini-Instruct quantized to `nbits` (default 3-bit) via HQQ.

    Quantization happens during `from_pretrained` itself (HQQ quantizes each linear layer as its
    weights are loaded), so there's no separate "load then quantize" step. `device_map="cuda"`
    (not "auto") because HQQ's transformers integration quantizes on-device and does not support
    CPU offloading the way plain BF16 loading does — this requires a GPU.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    quant_config = build_3bit_quant_config(nbits=nbits, group_size=group_size)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        quantization_config=quant_config,
    )
    model.eval()
    return model, tokenizer
