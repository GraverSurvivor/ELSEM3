"""Spike: promote one attention projection from 3-bit to 8-bit at runtime, no reload,
and confirm the model's outputs actually change.

This is the highest-risk technical piece of the whole pipeline (Stage 5, selective repair,
depends entirely on this mechanism working) — de-risked here on a single module before Week 4
builds the full greedy search around it.

Run in Colab after `pip install -r requirements.txt`:
    python -m src.quant.verify_module_swap
    python -m src.quant.verify_module_swap --dtype bf16
"""

from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from hqq.models.hf.base import AutoHQQHFModel

from src.eval.model_loader import DEFAULT_DTYPE, DTYPES, MODEL_ID
from src.quant.hqq_quantizer import DEFAULT_GROUP_SIZE, DEFAULT_NBITS, build_quant_config
from src.quant.module_swap import get_module_by_path, promote_module, snapshot_linear

# Phi-3.5 has no separate q_proj — attention is a fused qkv_proj + o_proj (see module_swap.py
# docstring). This is the fused projection for decoder layer 0.
TARGET_MODULE_PATH = "model.layers.0.self_attn.qkv_proj"
PROMOTED_NBITS = 8

PROMPT = "The capital of France is"


def _logits_for_prompt(model, tokenizer, device: str) -> torch.Tensor:
    inputs = tokenizer(PROMPT, return_tensors="pt").to(device)
    with torch.no_grad():
        return model(**inputs).logits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", default=DEFAULT_DTYPE, choices=list(DTYPES))
    args = parser.parse_args()
    torch_dtype = DTYPES[args.dtype]

    print(f"CUDA available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise RuntimeError("HQQ quantizes onto a GPU device — needs a GPU.")
    device = "cuda"

    print(f"\nLoading raw Phi-3.5-mini-Instruct in {args.dtype} (pre-quantization)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch_dtype)

    print(f"Snapshotting {TARGET_MODULE_PATH!r} before it gets quantized away...")
    snapshot = snapshot_linear(model, TARGET_MODULE_PATH)

    print(f"Quantizing whole model to {DEFAULT_NBITS}-bit (group_size={DEFAULT_GROUP_SIZE})...")
    quant_config = build_quant_config(nbits=DEFAULT_NBITS, group_size=DEFAULT_GROUP_SIZE)
    AutoHQQHFModel.quantize_model(model, quant_config=quant_config, compute_dtype=torch_dtype, device=device)
    model.eval()

    target_module = get_module_by_path(model, TARGET_MODULE_PATH)
    print(f"Target module class before promotion: {type(target_module).__name__}")

    print(f"\nRunning fixed prompt through the all-{DEFAULT_NBITS}-bit model...")
    logits_before = _logits_for_prompt(model, tokenizer, device)

    print(f"Promoting {TARGET_MODULE_PATH!r} from {DEFAULT_NBITS}-bit to {PROMOTED_NBITS}-bit, in place...")
    # compute_dtype is left to promote_module, which reads it off the model's own quantized
    # layers -- restating it here is exactly how the two sides could drift apart.
    promote_module(
        model,
        TARGET_MODULE_PATH,
        snapshot,
        nbits=PROMOTED_NBITS,
        device=device,
    )

    target_module_after = get_module_by_path(model, TARGET_MODULE_PATH)
    print(f"Target module class after promotion: {type(target_module_after).__name__}")

    print(f"\nRunning the SAME fixed prompt again, post-swap...")
    logits_after = _logits_for_prompt(model, tokenizer, device)

    max_abs_diff = (logits_after - logits_before).abs().max().item()
    outputs_changed = not torch.allclose(logits_before, logits_after)
    print(f"\nMax abs logit difference: {max_abs_diff:.6f}")
    print(f"Outputs changed after swap: {outputs_changed}")

    if not outputs_changed:
        raise RuntimeError(
            "Logits are identical before/after promotion -- the swap did not take effect. "
            "The module-swap mechanism this project's repair step depends on is broken."
        )

    print("\nWeek 2 module-swap spike passed: hot-swap works without a full reload.")


if __name__ == "__main__":
    main()
