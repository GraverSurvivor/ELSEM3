"""Week 1 sanity check: BF16 model loads and generates, BBQ ambiguous splits load cleanly.

Run in Colab (or any CUDA box) after `pip install -r requirements.txt`:
    python -m src.eval.verify_week1
"""

from __future__ import annotations

import torch

from src.data.bbq_loader import CATEGORIES, load_all_categories, summarize_splits
from src.eval.model_loader import load_bf16_model


def main() -> None:
    print(f"CUDA available: {torch.cuda.is_available()}")

    print("\nLoading BF16 Phi-3.5-mini-Instruct...")
    model, tokenizer = load_bf16_model()
    print(f"Model dtype: {next(model.parameters()).dtype}")
    print(f"Model device: {next(model.parameters()).device}")

    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=10, do_sample=False)
    print(f"Generation smoke test: {tokenizer.decode(output[0], skip_special_tokens=True)!r}")

    print("\nLoading BBQ ambiguous condition, all 5 categories...")
    items_by_category = load_all_categories()
    counts = summarize_splits(items_by_category)
    for category in CATEGORIES:
        print(f"  {category}: {counts[category]} items")
    print(f"  total: {counts['total']} items")

    missing = [c for c in CATEGORIES if counts[c] == 0]
    if missing:
        raise RuntimeError(f"Categories loaded with 0 ambiguous items: {missing}")

    print("\nWeek 1 verification passed.")


if __name__ == "__main__":
    main()
