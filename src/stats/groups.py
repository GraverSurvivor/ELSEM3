"""Per-group breakdown: which targeted groups did compression hurt, and did the repair help them all?

Category averages can hide opposite movements underneath: Hua et al. (COLM 2026) found bias can
worsen by up to 18.6% for some groups while improving 14.1% for others, netting to "no change".
BBQ marks each question's targeted group (e.g. "old", "F", "low SES"), so this reports USR and SRS
per group for the baseline, 3-bit and repaired runs, and flags groups where the repaired model is
worse than the original -- the cases an aggregate number would hide.

Group labels come from the dataset (src/data/bbq_loader.py), so this downloads BBQ's metadata on
first run; no GPU or model is needed.

    python -m src.stats.groups
    python -m src.stats.groups --min-n 50
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from src.data.safe_set import read_results_file

from .bootstrap import bootstrap_paired_diff_ci
from .metrics import align_by_question, is_stereotype, is_unknown

REFERENCE_DIR = Path("results/reference")
DEFAULT_MIN_N = 30


def group_key(groups: tuple[str, ...]) -> str:
    return " / ".join(groups) if groups else "(unlabelled)"


def breakdown(groups_by_id: dict[str, tuple[str, str]], runs: dict[str, list], min_n: int = DEFAULT_MIN_N) -> list[dict]:
    """USR/SRS per (category, group) for each run, plus the repaired-vs-baseline SRS change."""
    base, quant, rep = runs["baseline"], runs["hqq_3bit"], runs["hqq_repaired"]
    a, b = align_by_question(base, quant)
    _, c = align_by_question(base, rep)

    buckets: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, row in enumerate(a):
        if row.question_id in groups_by_id:
            buckets[groups_by_id[row.question_id]].append(i)

    rows = []
    for (category, group), idx in sorted(buckets.items()):
        if len(idx) < min_n:
            continue

        def rates(run):
            picked = [run[i] for i in idx]
            return (sum(is_unknown(r) for r in picked) / len(picked), sum(is_stereotype(r) for r in picked) / len(picked))

        (bu, bs), (qu, qs), (ru, rs) = rates(a), rates(b), rates(c)
        change = bootstrap_paired_diff_ci([a[i] for i in idx], [c[i] for i in idx], is_stereotype,
                                          check_conditions=False, n_resamples=500)
        rows.append({
            "category": category, "group": group, "n": len(idx),
            "usr": {"baseline": bu, "hqq_3bit": qu, "repaired": ru},
            "srs": {"baseline": bs, "hqq_3bit": qs, "repaired": rs},
            "repaired_minus_baseline_srs": {"point": change.point, "lower": change.lower, "upper": change.upper},
            "worse_than_baseline": change.lower > 0,
        })
    return rows


def load_group_labels() -> dict[str, tuple[str, str]]:
    from src.data.bbq_loader import load_all_categories

    return {
        item.question_id: (item.category, group_key(item.stereotyped_groups))
        for items in load_all_categories("ambig").values() for item in items
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--min-n", type=int, default=DEFAULT_MIN_N, help="skip groups with fewer questions")
    parser.add_argument("--out", type=Path, default=Path("results/stats/groups.json"))
    parser.add_argument("--repaired", help="folder under results/reference/ (default: the main repair)")
    args = parser.parse_args()

    from src.experiments.main_repair import main_repair

    repaired = args.repaired or main_repair().ambig
    print(f"Repair: {repaired}")
    load = lambda name: read_results_file(REFERENCE_DIR / name / "question_results.jsonl")
    runs = {"baseline": load("baseline_fp16"), "hqq_3bit": load("hqq_3bit"), "hqq_repaired": load(repaired)}
    rows = breakdown(load_group_labels(), runs, args.min_n)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")

    print(f"{'category':<16}{'group':<34}{'n':>6}{'SRS base':>10}{'3-bit':>8}{'repaired':>10}")
    for r in rows:
        flag = "  <- worse than original" if r["worse_than_baseline"] else ""
        print(f"{r['category']:<16}{r['group'][:33]:<34}{r['n']:>6}{r['srs']['baseline']:>10.3f}"
              f"{r['srs']['hqq_3bit']:>8.3f}{r['srs']['repaired']:>10.3f}{flag}")
    worse = [r for r in rows if r["worse_than_baseline"]]
    print(f"\n{len(worse)} of {len(rows)} groups have a significantly higher stereotype rate after repair than originally.")
    print(f"Written to {args.out}")


if __name__ == "__main__":
    main()
