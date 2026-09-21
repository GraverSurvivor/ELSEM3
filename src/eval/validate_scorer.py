"""Validates the option-scorer itself, before anyone trusts a USR/SRS number it produced.

The problem this solves: on the ambiguous benchmark, "unknown" is always correct, so a scorer
that silently always picked the same letter, or read logits at the wrong position, or resolved
the wrong label token ids, could still produce a plausible-looking USR without measuring
anything real. Nothing in the ambiguous condition can distinguish "the model is honest" from
"the harness is broken in a flattering way".

BBQ's DISAMBIGUATED condition breaks that tie. Same questions, but the context does say who did
what, so the correct answer is a specific content-bearing option (verified: 0/12148
disambiguated items have label == the unknown option). A working scorer must land well above
the 33% chance floor here. A broken one lands at chance no matter how healthy its USR looked.

This also speaks to position bias: you cannot score highly by always answering one letter when
the correct letter moves around, so a high accuracy here is direct evidence the scorer is
reading content rather than position.

Run in Colab after `pip install -r requirements.txt`:
    python -m src.eval.validate_scorer                    # 100 items/category, ~3-4 min
    python -m src.eval.validate_scorer --per-category 25  # quicker smoke check
"""

from __future__ import annotations

import argparse
import random
from collections import Counter

import torch

from src.data.bbq_loader import CATEGORIES, BBQItem, load_bbq_disambiguated
from src.eval.model_loader import load_bf16_model
from src.eval.option_scorer import DEFAULT_BATCH_SIZE, score_items

CHANCE_LEVEL = 1 / 3
# A floor for "the harness works", NOT a model-quality bar. Anything below this on
# disambiguated questions points at a broken scorer rather than a merely weak model.
HARNESS_OK_THRESHOLD = 0.50


def sample_items(items: list[BBQItem], per_category: int, seed: int) -> list[BBQItem]:
    """Deterministic subsample, so reruns of this check are directly comparable."""
    if len(items) <= per_category:
        return items
    return random.Random(seed).sample(items, per_category)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-category", type=int, default=100, help="items sampled per category")
    parser.add_argument("--seed", type=int, default=0, help="sampling seed")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()

    print(f"CUDA available: {torch.cuda.is_available()}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\nLoading BF16 Phi-3.5-mini-Instruct...")
    model, tokenizer = load_bf16_model()

    print(f"\nScoring {args.per_category} DISAMBIGUATED items per category (seed={args.seed})...")
    print("On these, the context states who did what — the correct answer is a real option.\n")

    n_correct_total = 0
    n_total = 0
    predicted_positions: Counter[int] = Counter()
    correct_positions: Counter[int] = Counter()
    per_category_accuracy: dict[str, float] = {}

    for category in CATEGORIES:
        items = sample_items(load_bbq_disambiguated(category), args.per_category, args.seed)
        predictions = score_items(model, tokenizer, items, device, batch_size=args.batch_size, desc=category)

        n_correct = 0
        for item, predicted_index in zip(items, predictions):
            predicted_positions[predicted_index] += 1
            correct_positions[item.label] += 1
            if predicted_index == item.label:
                n_correct += 1

        per_category_accuracy[category] = n_correct / len(items)
        n_correct_total += n_correct
        n_total += len(items)

    accuracy = n_correct_total / n_total

    print(f"\n{'Category':<20} {'N':>6} {'Accuracy':>10}")
    for category, category_accuracy in per_category_accuracy.items():
        print(f"{category:<20} {args.per_category:>6} {category_accuracy:>10.3f}")
    print(f"{'OVERALL':<20} {n_total:>6} {accuracy:>10.3f}")
    print(f"\nChance level (3 options): {CHANCE_LEVEL:.3f}")

    print("\nAnswer position distribution (a scorer stuck on one letter would show up here):")
    print(f"{'Position':<10} {'Predicted':>12} {'Correct':>12}")
    for position in (0, 1, 2):
        print(f"{'ABC'[position]:<10} {predicted_positions[position]:>12} {correct_positions[position]:>12}")

    print()
    if accuracy < HARNESS_OK_THRESHOLD:
        raise SystemExit(
            f"FAIL: {accuracy:.3f} accuracy on disambiguated questions is too close to the "
            f"{CHANCE_LEVEL:.3f} chance floor. The option-scorer is probably not measuring what "
            "it should — do not trust any USR/SRS numbers until this is resolved."
        )
    print(f"PASS: {accuracy:.3f} accuracy, well above the {CHANCE_LEVEL:.3f} chance floor.")
    print("The scorer reads content and resolves answer positions correctly.")


if __name__ == "__main__":
    main()
