"""Loads Phi-3.5-mini-Instruct at BF16 — the reference model all quantized variants compare to."""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

MODEL_ID = "microsoft/Phi-3.5-mini-instruct"


def load_bf16_model(device_map: str = "auto") -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Loads the BF16 baseline model + tokenizer.

    Requires a CUDA device for `device_map="auto"` to place the model on GPU; falls back to
    whatever `accelerate` resolves (CPU) if no GPU is visible, which will be slow but correct.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer
