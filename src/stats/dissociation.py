"""Concentrated vs diffuse: is fairness damage localized while general-ability damage is spread out?

The revised plan's scientific claim. Hu & Ramachandran (2026) find quantization's damage to general
ability is diffuse -- recovering 75% of it takes about half the layers. The fairness repair here
recovers 92% with 5 of 128 modules. If both are measured on the SAME model with the SAME search,
and fairness damage is markedly more concentrated, that is the finding; if not, it is an honest
negative result.

Three measures, all from committed files (no GPU):

1. Concentration of single-module effects (round 1 of each search, all candidates tried alone):
   the share of the summed positive gain captured by the top 1 / 5 / 10 modules. Concentrated
   damage puts most of the gain in a few modules; diffuse damage spreads it.
2. Search trajectory: kept modules, the score after each round, and how many rounds it took to
   reach 75% of the gap -- for the fairness search vs the general-ability search.
3. Cross-test (2x2), from full-benchmark repair runs: the fairness patch and the general-ability
   patch, each measured on both USR recovery (ambiguous set) and accuracy recovery
   (disambiguated set). A dissociation shows up as each patch recovering its own target far more
   than the other's.

    python -m src.stats.dissociation \
        --fairness-trials results/search_<fair>/trials.csv --capability-trials results/search_<cap>/trials.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from src.data.safe_set import read_results_file

from .metrics import is_unknown
from .recovery import recovery_fraction

REFERENCE_DIR = Path("results/reference")
TOP_K = (1, 5, 10)


def read_trials(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["round"], row["gain"], row["usr"] = int(row["round"]), float(row["gain"]), float(row["usr"])
    return rows


def concentration(trials: list[dict], top_k=TOP_K) -> dict:
    """Share of round-1 positive gain captured by the top-k single modules."""
    gains = sorted((max(t["gain"], 0.0) for t in trials if t["round"] == 1), reverse=True)
    total = sum(gains)
    if total <= 0:
        raise ValueError("no candidate improved the score in round 1")
    return {
        "n_candidates": len(gains),
        "n_helpful": sum(g > 0 for g in gains),
        **{f"top{k}_share": sum(gains[:k]) / total for k in top_k},
    }


def trajectory(summary: dict, target: float = 0.75) -> dict:
    """Rounds needed to close `target` of the gap between the start score and 1.0, if reached."""
    start, scores = summary["start_usr"], summary.get("usr_after_round", [])
    rounds_needed = next((i for i, s in enumerate(scores, start=1) if (s - start) / (1 - start) >= target), None)
    return {"kept": summary.get("kept", []), "start": start, "after_round": scores,
            f"rounds_to_{int(target * 100)}pct": rounds_needed}


def overlap(kept_a: list[str], kept_b: list[str]) -> dict:
    a, b = set(kept_a), set(kept_b)
    return {"shared": sorted(a & b), "jaccard": len(a & b) / len(a | b) if a | b else 0.0}


def accuracy_recovery(reference, degraded, repaired):
    """Recovery of plain accuracy (for the disambiguated set)."""
    from .metrics import get_field

    return recovery_fraction(reference, degraded, repaired, lambda r: bool(get_field(r, "is_correct")))


def cross_test(runs: dict) -> dict:
    """runs: baseline/3-bit for both question sets, plus each patch's repair on both sets."""
    table = {}
    for patch in ("fairness", "capability"):
        row = {}
        if f"{patch}_ambig" in runs:
            r = recovery_fraction(runs["fp16"], runs["hqq_3bit"], runs[f"{patch}_ambig"], is_unknown)
            row["usr_recovery"] = {"point": r.point, "lower": r.lower, "upper": r.upper}
        if f"{patch}_disambig" in runs:
            r = accuracy_recovery(runs["fp16_disambig"], runs["hqq_3bit_disambig"], runs[f"{patch}_disambig"])
            row["accuracy_recovery"] = {"point": r.point, "lower": r.lower, "upper": r.upper}
        table[patch] = row
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fairness-trials", type=Path, required=True)
    parser.add_argument("--capability-trials", type=Path, required=True)
    parser.add_argument("--fairness-summary", type=Path, help="defaults to summary.json next to the trials")
    parser.add_argument("--capability-summary", type=Path)
    parser.add_argument("--run", nargs="*", default=[],
                        help="NAME=results_dir for cross-test runs: fairness_ambig, fairness_disambig, "
                             "capability_ambig, capability_disambig")
    parser.add_argument("--out", type=Path, default=Path("results/stats/dissociation.json"))
    args = parser.parse_args()

    fair_summary = json.loads((args.fairness_summary or args.fairness_trials.with_name("summary.json")).read_text(encoding="utf-8"))
    cap_summary = json.loads((args.capability_summary or args.capability_trials.with_name("summary.json")).read_text(encoding="utf-8"))
    result = {
        "concentration": {
            "fairness": concentration(read_trials(args.fairness_trials)),
            "capability": concentration(read_trials(args.capability_trials)),
        },
        "trajectory": {"fairness": trajectory(fair_summary), "capability": trajectory(cap_summary)},
        "overlap": overlap(fair_summary.get("kept", []), cap_summary.get("kept", [])),
    }

    load = lambda d: read_results_file(Path(d) / "question_results.jsonl")
    runs = {}
    for name, ref in (("fp16", "baseline_fp16"), ("hqq_3bit", "hqq_3bit"),
                      ("fp16_disambig", "baseline_fp16_disambig"), ("hqq_3bit_disambig", "hqq_3bit_disambig")):
        if (REFERENCE_DIR / ref).exists():
            runs[name] = load(REFERENCE_DIR / ref)
    for pair in args.run:
        name, _, path = pair.partition("=")
        runs[name] = load(path)
    result["cross_test"] = cross_test(runs)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    for kind in ("fairness", "capability"):
        c = result["concentration"][kind]
        print(f"{kind:<11} top-1 {c['top1_share']:.0%}  top-5 {c['top5_share']:.0%}  top-10 {c['top10_share']:.0%} "
              f"of round-1 gain  ({c['n_helpful']}/{c['n_candidates']} modules help)")
    print(f"Kept modules shared by both searches: {result['overlap']['shared']} (Jaccard {result['overlap']['jaccard']:.2f})")
    for patch, row in result["cross_test"].items():
        parts = [f"{k} {v['point']:.0%} [{v['lower']:.0%}, {v['upper']:.0%}]" for k, v in row.items()]
        print(f"{patch} patch: " + ("; ".join(parts) if parts else "no cross-test runs yet"))
    print(f"Written to {args.out}")


if __name__ == "__main__":
    main()
