"""HQQ 3-bit quantization of Phi-3.5-mini-Instruct.

Week 1 scope: uniform 3-bit quantization end-to-end, no mixed precision yet (that's Week 2 —
runtime module swap via `restore_modules`, to live in this module once built).

Uses the `hqq` library's own `AutoHQQHFModel.quantize_model` rather than transformers' native
`HqqConfig` integration in `from_pretrained`: the native path was tried first (it's what HQQ's
own README lists as preferred), but on the transformers version this project's floor pins
resolve to, it fails with `NotImplementedError: QuantizationMethod.HQQ is not available yet and
will be supported soon` — a stub inside transformers' own internal model-loading refactor
(`core_model_loading.py`), not a bug in this code. `AutoHQQHFModel` is architecture-generic (it
resolves the model class from `config.architectures`, no Phi-3-specific file needed in `hqq`
itself) and doesn't go through that broken code path.
"""

from __future__ import annotations

import torch
from hqq.core.quantize import BaseQuantizeConfig
from hqq.models.hf.base import AutoHQQHFModel
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from src.eval.model_loader import MODEL_ID

DEFAULT_NBITS = 3
DEFAULT_GROUP_SIZE = 64


def build_quant_config(nbits: int, group_size: int = DEFAULT_GROUP_SIZE) -> BaseQuantizeConfig:
    """Builds an HQQ quant config for one bit-width, applied uniformly to whatever it's given.

    Not 3-bit-specific despite this module's name/original purpose: `src/quant/module_swap.py`
    reuses this at a different `nbits` (e.g. 8) to build the promoted-precision replacement for
    a single module, not the whole model.
    """
    return BaseQuantizeConfig(nbits=nbits, group_size=group_size)


def load_hqq_3bit_model(
    device: str = "cuda", nbits: int = DEFAULT_NBITS, group_size: int = DEFAULT_GROUP_SIZE
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Loads Phi-3.5-mini-Instruct quantized to `nbits` (default 3-bit) via HQQ.

    Loads the raw BF16 model on CPU first, then `AutoHQQHFModel.quantize_model` patches each
    linear layer in place and moves it onto `device` as it goes — this keeps peak memory lower
    than materializing the full BF16 model on GPU before quantizing.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)

    quant_config = build_quant_config(nbits=nbits, group_size=group_size)
    AutoHQQHFModel.quantize_model(model, quant_config=quant_config, compute_dtype=torch.bfloat16, device=device)

    model.eval()
    return model, tokenizer
