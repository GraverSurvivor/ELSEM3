"""Sensitivity analysis: are the headline numbers stable, or artifacts of choices we made?

The original plan's Week 6 Person A task. Everything here reuses the committed runs (CPU only):

1. Bootstrap resamples. The recovery CI at 200 / 500 / 1,000 / 2,000 / 5,000 resamples. If the
   interval moves materially past 1,000 (the plan's default), 1,000 was too few.
2. Bootstrap seed. The same CI under ten different seeds -- how much of its edge is noise.
3. Evaluation size. Recovery recomputed on random 25% / 50% / 75% subsets of the 12,140 questions
   (20 draws each). A number that only holds on the full set is fragile.
4. Search-sample representativeness. The search chose its modules from 300 flipped items. Does the
   repair recover those 300 more than the other 3,092 flipped items it never saw? A large gap would
   mean the modules were fitted to the sample rather than to the damage.

    python -m src.stats.sensitivity
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

from src.data.safe_set import read_results_file

from .metrics import align_by_question, is_unknown
from .recovery import recovery_fraction

REFERENCE_DIR = Path("results/reference")
SAMPLE = Path("results/search_samples/hqq_3bit_n300_seed0.json")
FLIPPED = Path("results/flipped/hqq_3bit.json")
RESAMPLES = (200, 500, 1000, 2000, 5000)
FRACTIONS = (0.25, 0.5, 0.75)
DRAWS = 20


def resample_stability(base, q3, rep) -> list[dict]:
    out = []
    for n in RESAMPLES:
        r = recovery_fraction(base, q3, rep, is_unknown, n_resamples=n)
        out.append({"n_resamples": n, "point": r.point, "lower": r.lower, "upper": r.upper, "width": r.upper - r.lower})
    return out


def seed_stability(base, q3, rep, seeds=range(10)) -> dict:
    lowers, uppers = [], []
    for seed in seeds:
        r = recovery_fraction(base, q3, rep, is_unknown, seed=seed)
        lowers.append(r.lower)
        uppers.append(r.upper)
    return {"lower_range": [min(lowers), max(lowers)], "upper_range": [min(uppers), max(uppers)]}


def subset_stability(base, q3, rep, fractions=FRACTIONS, draws=DRAWS, seed=0) -> list[dict]:
    a, b = align_by_question(base, q3)
    _, c = align_by_question(base, rep)
    rng = random.Random(seed)
    out = []
    for fraction in fractions:
        points = []
        for _ in range(draws):
            idx = rng.sample(range(len(a)), round(fraction * len(a)))
            ref = sum(is_unknown(a[i]) for i in idx)
            deg = sum(is_unknown(b[i]) for i in idx)
            fix = sum(is_unknown(c[i]) for i in idx)
            points.append((fix - deg) / (ref - deg))
        out.append({"fraction": fraction, "mean": statistics.mean(points), "sd": statistics.stdev(points),
                    "min": min(points), "max": max(points)})
    return out


def sample_representativeness(flipped_ids: list[str], sample_ids: set[str], repaired: dict) -> dict:
    """Share of flipped items the repair moved back to "unknown": inside vs outside the search sample."""
    def recovered(ids):
        ids = [i for i in ids if i in repaired]
        return sum(is_unknown(repaired[i]) for i in ids) / len(ids), len(ids)

    inside, n_in = recovered([i for i in flipped_ids if i in sample_ids])
    outside, n_out = recovered([i for i in flipped_ids if i not in sample_ids])
    return {"recovered_in_sample": inside, "n_in_sample": n_in,
            "recovered_outside_sample": outside, "n_outside_sample": n_out,
            "gap": inside - outside}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repaired", type=Path, default=REFERENCE_DIR / "hqq_repaired")
    parser.add_argument("--out", type=Path, default=Path("results/stats/sensitivity.json"))
    args = parser.parse_args()

    load = lambda d: read_results_file(Path(d) / "question_results.jsonl")
    base, q3, rep = load(REFERENCE_DIR / "baseline_fp16"), load(REFERENCE_DIR / "hqq_3bit"), load(args.repaired)
    flipped_ids = [item["question_id"] for item in json.loads(FLIPPED.read_text(encoding="utf-8"))["items"]]
    sample_ids = set(json.loads(SAMPLE.read_text(encoding="utf-8"))["question_ids"])

    result = {
        "resamples": resample_stability(base, q3, rep),
        "seeds": seed_stability(base, q3, rep),
        "subsets": subset_stability(base, q3, rep),
        "search_sample": sample_representativeness(flipped_ids, sample_ids, {r.question_id: r for r in rep}),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print("Recovery CI by bootstrap resamples:")
    for r in result["resamples"]:
        print(f"  {r['n_resamples']:>5}: {r['point']:.1%} [{r['lower']:.1%}, {r['upper']:.1%}]  width {r['width']:.2%}")
    s = result["seeds"]
    print(f"Across 10 seeds: lower bound {s['lower_range'][0]:.1%}-{s['lower_range'][1]:.1%}, "
          f"upper {s['upper_range'][0]:.1%}-{s['upper_range'][1]:.1%}")
    print("Recovery on random subsets of the benchmark:")
    for r in result["subsets"]:
        print(f"  {r['fraction']:.0%}: mean {r['mean']:.1%}  sd {r['sd']:.2%}  range {r['min']:.1%}-{r['max']:.1%}")
    m = result["search_sample"]
    print(f"Flipped items moved back to 'unknown': {m['recovered_in_sample']:.1%} of the {m['n_in_sample']} the search saw, "
          f"{m['recovered_outside_sample']:.1%} of the {m['n_outside_sample']} it didn't (gap {m['gap']:+.1%})")
    print(f"Written to {args.out}")


if __name__ == "__main__":
    main()
