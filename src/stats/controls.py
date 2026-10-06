"""Improvement #1: is the repair really about fairness? Hedging check + matched-budget baselines.

Two questions decide whether "the search found the modules responsible for fairness damage" holds:

1. Hedging. On the disambiguated control questions the text states the answer, so "unknown" is
   wrong there. If the repaired model says "unknown" there more than the original did, part of its
   recovered USR is the model hedging, not becoming fairer. Compared with paired bootstrap CIs on
   the false-"unknown" rate and on accuracy, fp16 vs 3-bit vs every repaired run.

2. Baselines. Each baseline patch (random, positional, largest-error, the general-ability search)
   restores the same weight budget as the search's patch. The search's patch must recover more
   USR than the best of them, with the paired CI on the difference above zero.

Reads only committed result files (no GPU), and writes results/stats/controls.json plus a table.

    python -m src.stats.controls
    python -m src.stats.controls --repaired search_top5=results/reference/hqq_repaired \
        --repaired positional=results/reference/repair_positional ...

By default it uses every promoted repair under results/reference/ whose summary names a patch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.safe_set import read_results_file

from .bbq_bias import bias_score_ambig, bias_score_disambig, false_unknown_rate, is_false_unknown
from .bootstrap import bootstrap_paired_diff_ci, bootstrap_proportion_ci
from .metrics import align_by_question, is_unknown
from .recovery import recovery_difference, recovery_fraction

REFERENCE_DIR = Path("results/reference")
OUT_DIR = Path("results/stats")
SEARCH_NAME = "search_top5"


def _load(run_dir: Path) -> list:
    return read_results_file(Path(run_dir) / "question_results.jsonl")


def _ci(result) -> dict:
    return {"point": result.point, "lower": result.lower, "upper": result.upper}


def hedging_block(fp16_dis: list, q3_dis: list, repaired_dis: dict[str, list]) -> dict:
    """False-"unknown" rate and accuracy per condition on the disambiguated set, plus the change
    each repair makes relative to fp16 (paired), and a verdict."""
    from .bbq_bias import accuracy

    def describe(rows):
        return {
            "n": len(rows),
            "accuracy": accuracy(rows),
            "false_unknown_rate": _ci(bootstrap_proportion_ci(rows, is_false_unknown)),
            "bias_score_disambig": bias_score_disambig(rows),
        }

    block = {"fp16": describe(fp16_dis), "hqq_3bit": describe(q3_dis), "repaired": {}}
    for name, rows in repaired_dis.items():
        a, b = align_by_question(fp16_dis, rows)
        change = bootstrap_paired_diff_ci(a, b, is_false_unknown, check_conditions=False)
        hedging = change.lower > 0  # CI entirely above zero: the repair hedges more than fp16 did
        block["repaired"][name] = {
            **describe(rows),
            "false_unknown_change_vs_fp16": _ci(change),
            "hedging_detected": hedging,
        }
    return block


def baselines_block(base: list, q3: list, repaired: dict[str, list], search_name: str = SEARCH_NAME) -> dict:
    """USR recovery per patch, and the paired difference search - baseline for each baseline."""
    rows = {}
    for name, runs in repaired.items():
        rows[name] = {
            "usr_recovery": _ci(recovery_fraction(base, q3, runs, is_unknown)),
            "bias_score_ambig": bias_score_ambig(runs),
        }
    if search_name in repaired:
        for name, runs in repaired.items():
            if name == search_name:
                continue
            diff = recovery_difference(base, q3, repaired[search_name], runs, is_unknown)
            rows[name]["search_minus_this"] = _ci(diff)
            rows[name]["search_beats_this"] = diff.lower > 0
    return rows


def verdict(controls: dict, search_name: str = SEARCH_NAME) -> dict:
    """The decision-point-1 outcome, in the three cases the plan names."""
    hedging = controls.get("hedging", {}).get("repaired", {}).get(search_name, {}).get("hedging_detected")
    others = {k: v for k, v in controls.get("baselines", {}).items() if k != search_name}
    beaten = [v.get("search_beats_this") for v in others.values() if "search_beats_this" in v]
    if hedging:
        outcome = "hedging"
        message = ("The repaired model answers 'unknown' significantly more often where the answer is stated. "
                   "Re-run the search with the hedging penalty (run_search --penalty-sample).")
    elif beaten and not all(beaten):
        outcome = "baseline_matches"
        message = ("At least one simple baseline is not significantly worse than the search. Reframe the claim "
                   "around cheap restoration verified by the pipeline, rather than search-found modules.")
    elif beaten:
        outcome = "claim_holds"
        message = "The search beats every baseline at the same budget, with no hedging detected."
    else:
        outcome = "incomplete"
        message = "Missing disambiguated runs or baseline repairs -- see the runbook."
    return {"outcome": outcome, "message": message}


def discover_repairs(reference_dir: Path = REFERENCE_DIR) -> dict[str, Path]:
    """Promoted ambiguous 3-bit repair runs that belong in the controls comparison.

    The committed search repair (`hqq_repaired/`) is the search patch. Baseline patches keep their
    patch name. Another search's patch -- the general-ability search, the fourth control -- is
    labelled by its folder (e.g. `repair_capability`) rather than its timestamped run id.

    Prefix runs (the search's own top 1-4 modules) are left out: they are the recovery-by-modules
    curve, not alternatives at the same budget, and the full patch can't be expected to beat its own
    first four modules significantly -- including them would wrongly read as "a baseline matches".
    """
    found = {}
    for summary_path in sorted(reference_dir.glob("*/summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("condition") != "hqq_repaired":
            continue
        folder = summary_path.parent.name
        kind = (summary.get("patch_source") or {}).get("kind")
        if folder == "hqq_repaired":
            found[SEARCH_NAME] = summary_path.parent
        elif kind == "baseline":
            found[summary.get("patch_name") or folder] = summary_path.parent
        elif kind == "search":
            found[folder] = summary_path.parent
    return found


def _named_paths(pairs: list[str] | None) -> dict[str, Path]:
    out = {}
    for pair in pairs or []:
        name, _, path = pair.partition("=")
        if not path:
            raise SystemExit(f"expected NAME=PATH, got {pair!r}")
        out[name] = Path(path)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repaired", nargs="*", help="NAME=results_dir for ambiguous repair runs (default: discover)")
    parser.add_argument("--repaired-disambig", nargs="*", help="NAME=results_dir for disambiguated repair runs")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    base = _load(REFERENCE_DIR / "baseline_fp16")
    q3 = _load(REFERENCE_DIR / "hqq_3bit")
    repaired_paths = _named_paths(args.repaired) or discover_repairs()
    repaired = {name: _load(path) for name, path in repaired_paths.items()}
    controls: dict = {"baselines": baselines_block(base, q3, repaired)}

    dis_paths = _named_paths(args.repaired_disambig)
    if not dis_paths and (REFERENCE_DIR / "hqq_repaired_disambig").exists():
        dis_paths = {SEARCH_NAME: REFERENCE_DIR / "hqq_repaired_disambig"}
        from src.experiments.main_repair import main_repair

        main = main_repair()
        if main.disambig_dir().exists() and main.name != SEARCH_NAME:
            dis_paths[main.name] = main.disambig_dir()  # the main repair's hedging, shown alongside
    fp16_dis_dir, q3_dis_dir = REFERENCE_DIR / "baseline_fp16_disambig", REFERENCE_DIR / "hqq_3bit_disambig"
    if fp16_dis_dir.exists() and q3_dis_dir.exists() and dis_paths:
        controls["hedging"] = hedging_block(
            _load(fp16_dis_dir), _load(q3_dis_dir), {name: _load(path) for name, path in dis_paths.items()}
        )
    else:
        print("Disambiguated reference runs not found yet -- hedging check skipped (see the runbook).")

    controls["verdict"] = verdict(controls)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "controls.json").write_text(json.dumps(controls, indent=2) + "\n", encoding="utf-8")

    print(f"\n{'patch':<20}{'USR recovered':>22}{'search minus this':>26}")
    for name, row in controls["baselines"].items():
        rec = row["usr_recovery"]
        diff = row.get("search_minus_this")
        diff_s = f"{diff['point']:+.0%} [{diff['lower']:+.0%}, {diff['upper']:+.0%}]" if diff else "-"
        print(f"{name:<20}{rec['point']:>9.0%} [{rec['lower']:.0%}, {rec['upper']:.0%}]{diff_s:>26}")
    if "hedging" in controls:
        h = controls["hedging"]
        print(f"\nFalse-'unknown' rate (disambiguated): fp16 {h['fp16']['false_unknown_rate']['point']:.3f}, "
              f"3-bit {h['hqq_3bit']['false_unknown_rate']['point']:.3f}")
        for name, row in h["repaired"].items():
            print(f"  {name}: {row['false_unknown_rate']['point']:.3f}  hedging detected: {row['hedging_detected']}")
    print(f"\nVerdict: {controls['verdict']['outcome']} -- {controls['verdict']['message']}")
    print(f"Written to {args.out_dir / 'controls.json'}")


if __name__ == "__main__":
    main()
