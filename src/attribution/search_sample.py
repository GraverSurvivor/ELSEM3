"""The fixed flipped-set sample the greedy search scores every candidate on.

The full flipped set is 3,392 questions. Scoring all of them for 128 candidates per round is
~4.8 GPU-hours a round at ~25 items/sec -- about 24 hours for a 5-round search, and three times
that for the headline experiment's three failure profiles. A fixed ~300-question sample brings a
round to ~25 minutes.

This is not a corner cut, as long as the sample is FIXED: the search only ranks candidates
against each other, and every candidate is scored on the identical questions, so "A beat B" is
an exact comparison. Sampling only affects how well the 300 stand in for the whole flipped set --
which the Stage 5 verification settles anyway, by re-running the full 12,140-question benchmark
with the kept modules. No reported number comes from the sample.

So the sample is drawn once, seeded, written to disk, and read back by every run. A fresh
`random.sample` per candidate, or per run, would destroy the argument above.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

DEFAULT_SIZE = 300
DEFAULT_SEED = 0


def _allocate(available: dict[str, int], n: int, weights: dict[str, float]) -> dict[str, int]:
    """Splits `n` across categories in proportion to `weights`, summing to exactly `n`.

    Largest-remainder rounding, so the split is as close to the requested shares as integers
    allow. A category can't be given more items than it has; if the weights ask for that, raise
    rather than quietly shifting the sample toward other categories.
    """
    total_weight = sum(weights.values())
    exact = {cat: n * w / total_weight for cat, w in weights.items()}
    counts = {cat: int(value) for cat, value in exact.items()}
    by_remainder = sorted(exact, key=lambda cat: (exact[cat] - counts[cat], cat), reverse=True)
    for cat in by_remainder[: n - sum(counts.values())]:
        counts[cat] += 1

    short = {cat: (counts[cat], available.get(cat, 0)) for cat in counts if counts[cat] > available.get(cat, 0)}
    if short:
        raise ValueError(f"not enough flipped items for the requested split (wanted, have): {short}")
    return counts


def draw_search_sample(
    flipped: dict,
    n: int = DEFAULT_SIZE,
    seed: int = DEFAULT_SEED,
    category_weights: dict[str, float] | None = None,
) -> dict:
    """Draws a seeded, category-stratified sample from a flipped-set payload.

    `flipped` is the JSON `src.data.safe_set.write_flipped_set` produces. By default each
    category's share of the sample matches its share of the flipped set; pass `category_weights`
    to skew it -- that's how the headline experiment's Age-heavy and SES-heavy profiles are made.

    Deterministic for a given (flipped set, n, seed, weights): ids are sorted before sampling, so
    the order items happen to appear in the input file doesn't change the draw.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")

    by_category: dict[str, list[str]] = {}
    for item in flipped["items"]:
        by_category.setdefault(item["category"], []).append(item["question_id"])
    available = {cat: len(ids) for cat, ids in by_category.items()}
    if n > sum(available.values()):
        raise ValueError(f"asked for {n} items but the flipped set only has {sum(available.values())}")

    weights = category_weights if category_weights is not None else available
    unknown = set(weights) - set(by_category)
    if unknown:
        raise ValueError(f"weights name categories with no flipped items: {sorted(unknown)}")
    counts = _allocate(available, n, weights)

    rng = random.Random(seed)
    question_ids: list[str] = []
    for category in sorted(counts):
        question_ids.extend(rng.sample(sorted(by_category[category]), counts[category]))

    return {
        "baseline_run_id": flipped["baseline_run_id"],
        "quantized_run_id": flipped["quantized_run_id"],
        "flipped_total": flipped["total_flipped"],
        "n": n,
        "seed": seed,
        "category_weights": category_weights,
        "per_category": dict(sorted(counts.items())),
        "question_ids": sorted(question_ids),
    }


def write_search_sample(sample: dict, out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(sample, indent=2), encoding="utf-8")
    return out_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Draw the fixed search sample from a flipped set.")
    parser.add_argument("--flipped", type=Path, default=Path("results/flipped/hqq_3bit.json"))
    parser.add_argument("--out", type=Path, default=Path("results/search_samples/hqq_3bit_n300_seed0.json"))
    parser.add_argument("--n", type=int, default=DEFAULT_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    payload = json.loads(args.flipped.read_text(encoding="utf-8"))
    sample = draw_search_sample(payload, n=args.n, seed=args.seed)
    path = write_search_sample(sample, args.out)
    print(f"{sample['n']} of {sample['flipped_total']} flipped items, seed {sample['seed']} -> {path}")
    for category, count in sample["per_category"].items():
        print(f"  {category:22} {count}")
