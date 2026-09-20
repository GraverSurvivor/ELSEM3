"""Week 1 sanity check: Phi-3.5-mini quantizes to 3-bit via HQQ end-to-end and still generates.

No mixed precision yet (Week 2). Run in Colab after `pip install -r requirements.txt`:
    python -m src.quant.verify_week1
"""

from __future__ import annotations

import torch

from src.quant.hqq_quantizer import DEFAULT_GROUP_SIZE, DEFAULT_NBITS, load_hqq_3bit_model


def main() -> None:
    print(f"CUDA available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise RuntimeError("AutoHQQHFModel.quantize_model quantizes onto a GPU device — needs a GPU.")

    print(f"\nLoading Phi-3.5-mini-Instruct, HQQ {DEFAULT_NBITS}-bit (group_size={DEFAULT_GROUP_SIZE})...")
    model, tokenizer = load_hqq_3bit_model(device="cuda")

    footprint_gb = model.get_memory_footprint() / 1e9
    print(f"Quantized model memory footprint: {footprint_gb:.2f} GB")

    prompt = "The capital of France is"
    # Use "cuda" directly rather than model.device: after HQQ patches layers into custom
    # HQQLinear modules, PreTrainedModel.device's usual parameter-inspection can be unreliable.
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=10, do_sample=False)
    print(f"Generation smoke test: {tokenizer.decode(output[0], skip_special_tokens=True)!r}")

    print("\nWeek 1 verification passed.")


if __name__ == "__main__":
    main()
