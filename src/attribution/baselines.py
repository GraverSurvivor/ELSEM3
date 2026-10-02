"""Builds the patches the fairness controls compare the search against, plus the search's prefixes.

The claim "the search finds the modules responsible for fairness damage" only holds if simpler
choices with the same memory budget do worse. Each baseline here is a patch restoring about the
same number of weights as the search's patch (within +-5%), verified on the same full benchmark
by `run_repair --patch`:

    random_s{0..4}   random modules, five different draws -- does choosing carefully matter?
    positional       the search patch's exact module types, moved to the earliest layers -- is it
                     just "fix the early layers"? (Early layers hold known quantization-fragile
                     weights, so this is the most dangerous baseline.)
    largest_error    the modules quantization changed most (src/quant/quant_error.py), no data --
                     is it just "fix what was damaged most"?

The fourth control, the general-ability search, is a real search run, not built here.

It also writes the search patch's prefixes (top1..top4): verifying each on the full benchmark gives
recovery as a function of modules restored, which is also what the original plan's threshold
sweep asked for -- a threshold only decides where the search stops, not the order it picks
modules, so each threshold corresponds to one prefix.

    python -m src.attribution.baselines
    python -m src.attribution.baselines --quant-error results/quant_error/hqq_3bit_g64.json
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from src.attribution.patch import (
    Patch,
    load_patch,
    module_params,
    module_suffix,
    patch_from_search_summary,
    save_patch,
)
from src.quant.candidates import candidate_paths

PATCH_DIR = Path("results/patches")
REPAIRED_SUMMARY = Path("results/reference/hqq_repaired/summary.json")
BUDGET_TOLERANCE = 0.05
N_RANDOM = 5
MAX_ATTEMPTS = 10_000


def within_budget(params: int, budget: int, tolerance: float = BUDGET_TOLERANCE) -> bool:
    return abs(params - budget) <= tolerance * budget


def random_patch(reference: Patch, seed: int, tolerance: float = BUDGET_TOLERANCE) -> Patch:
    """Random candidates adding up to the reference's weight budget (+-tolerance).

    Shuffles the candidates and adds each one that still fits under the upper bound; if the total
    misses the band, reshuffles. Deterministic for a given seed.
    """
    budget = reference.params
    rng = random.Random(seed)
    paths = candidate_paths()
    for _ in range(MAX_ATTEMPTS):
        rng.shuffle(paths)
        chosen, total = [], 0
        for path in paths:
            size = module_params(path)
            if total + size <= budget * (1 + tolerance):
                chosen.append(path)
                total += size
            if within_budget(total, budget, tolerance):
                break
        if within_budget(total, budget, tolerance):
            return Patch(
                name=f"random_s{seed}",
                modules=tuple(chosen),
                nbits=reference.nbits,
                source={"kind": "baseline", "baseline": "random", "seed": seed, "budget_of": reference.name},
            )
    raise RuntimeError(f"no random draw landed within {tolerance:.0%} of {budget} weights")


def positional_patch(reference: Patch) -> Patch:
    """The reference's module types, each at the earliest layer not already used for that type.

    For the committed search patch (qkv x4 + o_proj x1) this is qkv in layers 0-3 plus o_proj in
    layer 0: the same composition and the same weight count, differing only in which layers.
    """
    counts = Counter(module_suffix(path) for path in reference.modules)
    chosen = [f"model.layers.{layer}.{suffix}" for suffix, n in counts.items() for layer in range(n)]
    return Patch(
        name="positional",
        modules=tuple(chosen),
        nbits=reference.nbits,
        source={"kind": "baseline", "baseline": "positional", "budget_of": reference.name},
    )


def largest_error_patch(reference: Patch, ranked: list[dict], tolerance: float = BUDGET_TOLERANCE) -> Patch:
    """Modules in descending quantization-error order, each added if it still fits the budget."""
    budget = reference.params
    chosen, total = [], 0
    for entry in ranked:
        size = module_params(entry["path"])
        if total + size <= budget * (1 + tolerance):
            chosen.append(entry["path"])
            total += size
        if within_budget(total, budget, tolerance):
            break
    if not within_budget(total, budget, tolerance):
        raise RuntimeError(f"largest-error fill reached {total} weights, outside the {budget} budget")
    return Patch(
        name="largest_error",
        modules=tuple(chosen),
        nbits=reference.nbits,
        source={"kind": "baseline", "baseline": "largest_error", "budget_of": reference.name},
    )


def build_all(reference: Patch, quant_error: dict | None) -> list[Patch]:
    patches = [reference.prefix(k) for k in range(1, len(reference.modules))]
    patches += [random_patch(reference, seed) for seed in range(N_RANDOM)]
    patches.append(positional_patch(reference))
    if quant_error is not None:
        patches.append(largest_error_patch(reference, quant_error["ranked"]))
    return patches


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", type=Path, help="the search's patch.json (default: built from the committed repair)")
    parser.add_argument("--quant-error", type=Path, help="output of src.quant.quant_error, for the largest-error baseline")
    parser.add_argument("--out-dir", type=Path, default=PATCH_DIR)
    args = parser.parse_args()

    if args.reference:
        reference = load_patch(args.reference)
    else:
        summary = json.loads(REPAIRED_SUMMARY.read_text(encoding="utf-8"))
        reference = patch_from_search_summary(summary, name="search_top5")
    save_patch(reference, args.out_dir / f"{reference.name}.json")

    quant_error = json.loads(args.quant_error.read_text(encoding="utf-8")) if args.quant_error else None
    if quant_error is None:
        print("No --quant-error file: skipping the largest-error baseline (run src.quant.quant_error on a GPU first).")

    print(f"Reference: {reference.name}, {len(reference.modules)} modules, {reference.params:,} weights")
    for patch in build_all(reference, quant_error):
        save_patch(patch, args.out_dir / f"{patch.name}.json")
        print(f"  {patch.name:16} {len(patch.modules):>2} modules  {patch.params:>12,} weights "
              f"({patch.params / reference.params:6.1%} of budget)")
    print(f"Written to {args.out_dir}")


if __name__ == "__main__":
    main()
