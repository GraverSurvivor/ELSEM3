"""Loads Phi-3.5-mini-Instruct at BF16 — the reference model all quantized variants compare to."""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

MODEL_ID = "microsoft/Phi-3.5-mini-instruct"


def load_bf16_model(device_map: str = "auto") -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Loads the BF16 baseline model + tokenizer.

    Requires a CUDA device for `device_map="auto"` to place the model on GPU; falls back to
    whatever `accelerate` resolves (CPU) if no GPU is visible, which will be slow but correct.

    Deliberately does NOT pass `trust_remote_code=True`: `transformers` has shipped a native,
    maintained `Phi3ForCausalLM` implementation since ~4.41 (this model's config.json declares
    `model_type: "phi3"`), so omitting the flag makes `from_pretrained` use that instead of the
    Hub's own frozen `modeling_phi3.py`. The Hub copy calls APIs (e.g. `DynamicCache.seen_tokens`)
    that newer `transformers` versions have since removed, causing an AttributeError in
    `generate()` that the native implementation doesn't have, since it's kept in sync with the
    library's own Cache API.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )
    model.eval()
    return model, tokenizer
