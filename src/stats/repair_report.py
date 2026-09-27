"""Stage 5 report: baseline vs 3-bit vs repaired, and the per-run reporting bundle.

Reads three promoted reference runs and writes, under results/stats/:

    repair_comparison.json  overall and per category: the three USR/SRS rates, the repair's own
                            effect (3-bit -> repaired), the gap left (baseline -> repaired), and
                            the recovery fraction, each with a 95% paired-bootstrap CI.
    repair_bundle.json      the reporting bundle CLAUDE.md asks for per run: fairness recovered,
                            share of the model restored, memory overhead vs plain 3-bit, and
                            wall-clock time of the cycle. The last three come from the repaired
                            run's own summary.json (written by src/attribution/run_repair.py).

CPU only -- everything is read off committed files:
    python -m src.stats.repair_report
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.safe_set import read_results_file

from . import by_category, cramers_v_from_result, srs_delta, stereotype_chisq, usr_delta
from .metrics import indicators, is_stereotype, is_unknown
from .recovery import recovery_fraction

REFERENCE_DIR = Path("results/reference")


def _rates(rows) -> dict:
    return {"usr": float(indicators(rows, is_unknown).mean()), "srs": float(indicators(rows, is_stereotype).mean())}


def _compare(before, after) -> dict:
    chi = stereotype_chisq(before, after)
    return {
        "usr": usr_delta(before, after).as_dict(),
        "srs": srs_delta(before, after).as_dict(),
        "chisq_p": chi.p_value,
        "cramers_v": cramers_v_from_result(chi),
    }


def compare_block(baseline, degraded, repaired) -> dict:
    """Everything about one slice (overall or one category) of the three runs."""
    return {
        "n_items": len(baseline),
        "baseline": _rates(baseline),
        "hqq_3bit": _rates(degraded),
        "repaired": _rates(repaired),
        "repair_effect": _compare(degraded, repaired),
        "gap_left": _compare(baseline, repaired),
        "recovery": {
            "usr": recovery_fraction(baseline, degraded, repaired, is_unknown).as_dict(),
            "srs": recovery_fraction(baseline, degraded, repaired, is_stereotype).as_dict(),
        },
    }


def build_comparison(baseline, degraded, repaired) -> dict:
    out = {"overall": compare_block(baseline, degraded, repaired)}
    degraded_by, repaired_by = by_category(degraded), by_category(repaired)
    for category, rows in by_category(baseline).items():
        out[category] = compare_block(rows, degraded_by[category], repaired_by[category])
    return out


def build_bundle(comparison: dict, repair_summary: dict) -> dict:
    """The four-number reporting bundle, from the comparison and the repair run's summary."""
    overall = comparison["overall"]
    memory = repair_summary["memory"]
    return {
        "repair_run_id": repair_summary["run_id"],
        "search_run_id": repair_summary["search_run_id"],
        "modules": repair_summary["modules"],
        "fairness_recovered": {
            "usr": {"baseline": overall["baseline"]["usr"], "hqq_3bit": overall["hqq_3bit"]["usr"],
                    "repaired": overall["repaired"]["usr"], "recovery": overall["recovery"]["usr"]},
            "srs": {"baseline": overall["baseline"]["srs"], "hqq_3bit": overall["hqq_3bit"]["srs"],
                    "repaired": overall["repaired"]["srs"], "recovery": overall["recovery"]["srs"]},
        },
        "model_restored": {
            "pct_params": repair_summary["restored"]["pct_params_restored"],
            "pct_modules": repair_summary["restored"]["pct_modules_restored"],
            "modules": f"{repair_summary['restored']['modules_restored']}/{repair_summary['restored']['modules_total']}",
        },
        "memory_overhead": {
            "deployed_mb": memory["deployed_overhead_bytes"] / 2**20,
            # Relative to what the 3-bit model occupied on the GPU just before the repair.
            "pct_of_3bit_gpu_memory": 100 * memory["deployed_overhead_bytes"] / memory["gpu_allocated_before_bytes"],
            "gpu_delta_mb_cross_check": memory["gpu_delta_bytes"] / 2**20,
        },
        "wall_clock": {
            "cycle_minutes": repair_summary["wall_clock_seconds"]["cycle"] / 60,
            "repair_seconds": repair_summary["wall_clock_seconds"]["repair"],
            "partial_because_resumed": repair_summary["resumed"],
        },
    }


def _ci(d: dict) -> str:
    return f"{d['delta']:+.3f} [{d['lower']:+.3f}, {d['upper']:+.3f}]"


def _pct(d: dict) -> str:
    return f"{d['point']:.0%} [{d['lower']:.0%}, {d['upper']:.0%}]"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", default="baseline_fp16")
    parser.add_argument("--degraded", default="hqq_3bit")
    parser.add_argument("--repaired", default="hqq_repaired")
    parser.add_argument("--out-dir", type=Path, default=Path("results/stats"))
    args = parser.parse_args()

    def load(name):
        return read_results_file(REFERENCE_DIR / name / "question_results.jsonl")

    comparison = build_comparison(load(args.baseline), load(args.degraded), load(args.repaired))
    repair_summary = json.loads((REFERENCE_DIR / args.repaired / "summary.json").read_text(encoding="utf-8"))
    bundle = build_bundle(comparison, repair_summary)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "repair_comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    (args.out_dir / "repair_bundle.json").write_text(json.dumps(bundle, indent=2), encoding="utf-8")

    print(f"{'':16} {'USR base/3bit/rep':>19}  {'recovered USR':>16}  {'gap left USR':>22}  {'gap left SRS':>22}")
    for name, block in comparison.items():
        rates = f"{block['baseline']['usr']:.3f}/{block['hqq_3bit']['usr']:.3f}/{block['repaired']['usr']:.3f}"
        print(f"{name:16} {rates:>19}  {_pct(block['recovery']['usr']):>16}  "
              f"{_ci(block['gap_left']['usr']):>22}  {_ci(block['gap_left']['srs']):>22}")

    f, m, mem, clock = bundle["fairness_recovered"], bundle["model_restored"], bundle["memory_overhead"], bundle["wall_clock"]
    print("\nReporting bundle")
    print(f"  Fairness   USR {f['usr']['hqq_3bit']:.3f} -> {f['usr']['repaired']:.3f} (baseline {f['usr']['baseline']:.3f}), "
          f"{_pct(f['usr']['recovery'])} recovered")
    print(f"             SRS {f['srs']['hqq_3bit']:.3f} -> {f['srs']['repaired']:.3f} (baseline {f['srs']['baseline']:.3f}), "
          f"{_pct(f['srs']['recovery'])} recovered")
    print(f"  Restored   {m['pct_params']:.2f}% of candidate parameters ({m['modules']} modules)")
    print(f"  Memory     +{mem['deployed_mb']:.1f} MB (+{mem['pct_of_3bit_gpu_memory']:.1f}% of the 3-bit model on GPU)")
    print(f"  Time       {clock['cycle_minutes']:.1f} min for repair + full 12,140-item verification"
          + (" (resumed -- partial)" if clock["partial_because_resumed"] else ""))
    print(f"\nWritten to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
