"""Headline experiment report: which modules each failure window restored, and what that bought.
CPU only -- reads what `src.experiments.headline` left in results/.

Writes under results/<headline_id>/report/:

    headline_table.{csv,md}        one row per window: sample mix, modules kept, fairness on the
                                   full benchmark with recovery CIs, % restored, memory, wall clock
    per_category_recovery.{csv,md} USR recovered per category per window -- does the Age-heavy
                                   window repair Age best?
    overlap.json                   pairwise Jaccard of the kept sets, modules/layers common to all,
                                   and the verdict (category-invariant / window-specific / partial)
    headline.json                  everything above, machine-readable
    restored_blocks.png            the money shot: kept modules on the 4 x 32 layout, per window
    recovery_curves.png            USR on each window's sample as modules are restored
    conditions_usr.png             baseline vs 3-bit vs each window's repair, per category

    python -m src.experiments.headline_report headline_<ts>

When the windows ran as separate experiments (one per account, in parallel), name them all and say
where their folders are; their windows are combined into one report under `headline_combined/`:

    python -m src.experiments.headline_report headline_a headline_b headline_c --results-dir results/reference/headline_runs
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.data.bbq_loader import CATEGORIES  # noqa: E402
from src.data.safe_set import read_results_file  # noqa: E402
from src.experiments.logger import read_category_summary  # noqa: E402
from src.plotting.condition_chart import plot_conditions_per_category  # noqa: E402
from src.plotting.recovery_curve import plot_recovery_curves  # noqa: E402
from src.plotting.restored_blocks import parse_module_path, plot_restored_blocks, short_module_name  # noqa: E402
from src.plotting.style import WINDOW_PALETTE, set_house_style  # noqa: E402
from src.plotting.tables import _ci_pct, _pct, write_table  # noqa: E402

RESULTS_DIR = Path("results")
REFERENCE_DIR = RESULTS_DIR / "reference"

VERDICT_IDENTICAL = "identical in every window: localization is category-invariant"
VERDICT_DISJOINT = "no module shared between any two windows: localization is symptom-specific"
VERDICT_PARTIAL = "partial overlap: a shared core plus window-specific modules"


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else 1.0


def localization_overlap(kept_by_window: dict[str, list[str]]) -> dict:
    """How much the windows agree on what to restore -- the headline question, as numbers.

    Compared as sets: the order modules were kept in is a property of the greedy path, not of
    which components are damaged. Layer-level overlap is reported too, since two windows that
    restore different projections of the same layer are closer than the module sets suggest.
    """
    if len(kept_by_window) < 2:
        raise ValueError("need at least two windows to compare")
    sets = {w: set(kept) for w, kept in kept_by_window.items()}
    layers = {w: {parse_module_path(p)[0] for p in kept} for w, kept in sets.items()}
    pairs = list(itertools.combinations(sets, 2))

    module_jaccard = {f"{a} vs {b}": jaccard(sets[a], sets[b]) for a, b in pairs}
    if all(s == next(iter(sets.values())) for s in sets.values()):
        verdict = VERDICT_IDENTICAL
    elif all(not (sets[a] & sets[b]) for a, b in pairs):
        verdict = VERDICT_DISJOINT
    else:
        verdict = VERDICT_PARTIAL
    return {
        "verdict": verdict,
        "module_jaccard": module_jaccard,
        "layer_jaccard": {f"{a} vs {b}": jaccard(layers[a], layers[b]) for a, b in pairs},
        "common_modules": sorted(set.intersection(*sets.values())),
        "common_layers": sorted(set.intersection(*layers.values())),
        "union_modules": sorted(set.union(*sets.values())),
    }


def window_row(name: str, sample: dict, search: dict, bundle: dict, repair_summary: dict) -> dict:
    """One headline-table row. Cycle time is search + repair: the whole localize -> repair loop."""
    f, restored, memory = bundle["fairness_recovered"], bundle["model_restored"], bundle["memory_overhead"]
    cycle_minutes = (search["wall_clock_seconds"] + repair_summary["wall_clock_seconds"]["cycle"]) / 60
    return {
        "Window": name,
        "Sample mix": ", ".join(f"{c[:3]} {k}" for c, k in sample["per_category"].items()),
        "Modules kept": ", ".join(short_module_name(p) for p in search["kept"]),
        "Sample USR": f"{search['start_usr']:.0%} -> {search['final_usr']:.0%}",
        "USR repaired": _pct(f["usr"]["repaired"]),
        "USR recovered": _ci_pct(f["usr"]["recovery"]),
        "SRS repaired": _pct(f["srs"]["repaired"]),
        "SRS recovered": _ci_pct(f["srs"]["recovery"]),
        "% params restored": f"{restored['pct_params']:.2f}%",
        "Memory": f"+{memory['deployed_mb']:.0f} MB (+{memory['pct_of_3bit_gpu_memory']:.1f}%)",
        "Cycle (min)": f"{cycle_minutes:.0f}" + (" (partial)" if bundle["wall_clock"]["partial_because_resumed"] else ""),
    }


def per_category_row(name: str, comparison: dict) -> dict:
    row = {"Window": name}
    for category in [c for c in CATEGORIES if c in comparison] + ["overall"]:
        row["Overall" if category == "overall" else category] = _ci_pct(comparison[category]["recovery"]["usr"])
    return row


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_report(headline_id: str, results_dir: Path = RESULTS_DIR) -> dict:
    from src.stats.repair_report import build_bundle, build_comparison

    manifest = _load_json(results_dir / headline_id / "manifest.json")
    baseline = read_results_file(REFERENCE_DIR / "baseline_fp16" / "question_results.jsonl")
    degraded = read_results_file(REFERENCE_DIR / "hqq_3bit" / "question_results.jsonl")

    windows, skipped = {}, {}
    for name, state in manifest["windows"].items():
        if state.get("repair_skipped"):
            skipped[name] = state["repair_skipped"]
            continue
        if not state.get("repair_run_id") or not (results_dir / state["repair_run_id"] / "summary.json").exists():
            skipped[name] = "not finished -- run `python -m src.experiments.headline --resume` first"
            continue
        search = _load_json(results_dir / state["search_run_id"] / "summary.json")
        repair_summary = _load_json(results_dir / state["repair_run_id"] / "summary.json")
        repaired = read_results_file(results_dir / state["repair_run_id"] / "question_results.jsonl")
        print(f"[{name}] bootstrapping CIs...")
        comparison = build_comparison(baseline, degraded, repaired)
        windows[name] = {
            "state": state,
            "sample": _load_json(Path(state["sample_path"])),
            "search": search,
            "repair_summary": repair_summary,
            "comparison": comparison,
            "bundle": build_bundle(comparison, repair_summary),
        }

    report = {"headline_id": headline_id, "config": manifest["config"], "skipped": skipped, "windows": {}}
    for name, w in windows.items():
        report["windows"][name] = {
            "search_run_id": w["state"]["search_run_id"],
            "repair_run_id": w["state"]["repair_run_id"],
            "per_category": w["sample"]["per_category"],
            "kept": w["search"]["kept"],
            "stop_reason": w["search"]["stop_reason"],
            "bundle": w["bundle"],
            "comparison": w["comparison"],
        }
    kept = {name: w["search"]["kept"] for name, w in windows.items()}
    report["overlap"] = localization_overlap(kept) if len(kept) >= 2 else None
    report["_windows"] = windows  # in-memory only, for the writers below
    return report


def combine_manifests(headline_ids: list[str], results_dir: Path, combined_id: str = "headline_combined") -> str:
    """One manifest holding every window of several experiments, so one report covers them. Refuses
    experiments run with different search settings, or the same window twice."""
    combined = None
    for headline_id in headline_ids:
        manifest = _load_json(results_dir / headline_id / "manifest.json")
        config = {k: v for k, v in manifest["config"].items() if k != "mirror"}
        if combined is None:
            combined = {**manifest, "headline_id": combined_id, "config": config, "windows": {}, "combined_from": []}
        elif config != combined["config"]:
            raise SystemExit(f"{headline_id} used different settings ({config}) from the others ({combined['config']})")
        for name, state in manifest["windows"].items():
            if name in combined["windows"] and state.get("repair_run_id"):
                raise SystemExit(f"window {name} appears in more than one experiment")
            if state.get("repair_run_id") or name not in combined["windows"]:
                combined["windows"][name] = state
        combined["combined_from"].append(headline_id)
    out = results_dir / combined_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(combined, indent=2), encoding="utf-8")
    return combined_id


def _run_ref(results_dir: Path, run_id: str) -> str:
    """A run id as read_category_summary resolves it: relative to results/."""
    return Path(results_dir, run_id).resolve().relative_to(RESULTS_DIR.resolve()).as_posix()


def write_report(report: dict, out_dir: Path, results_dir: Path = RESULTS_DIR) -> None:
    windows = report.pop("_windows")
    out_dir.mkdir(parents=True, exist_ok=True)
    if not windows:
        raise SystemExit(f"No finished windows to report. Skipped: {report['skipped']}")

    rows = [window_row(n, w["sample"], w["search"], w["bundle"], w["repair_summary"]) for n, w in windows.items()]
    write_table(rows, out_dir / "headline_table")
    write_table([per_category_row(n, w["comparison"]) for n, w in windows.items()], out_dir / "per_category_recovery")
    if report["overlap"] is not None:
        (out_dir / "overlap.json").write_text(json.dumps(report["overlap"], indent=2), encoding="utf-8")
    (out_dir / "headline.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    set_house_style()
    colors = {n: WINDOW_PALETTE.get(n, "#1F3A68") for n in windows}
    figures = {
        "restored_blocks": plot_restored_blocks({n: w["search"]["kept"] for n, w in windows.items()}, colors=colors),
        "recovery_curves": plot_recovery_curves({n: w["search"] for n, w in windows.items()}, colors=colors),
    }
    runs = {"baseline": read_category_summary("reference/baseline_fp16"),
            "hqq_3bit": read_category_summary("reference/hqq_3bit")}
    for n, w in windows.items():
        runs[f"repaired ({n})"] = read_category_summary(_run_ref(results_dir, w["state"]["repair_run_id"]))
    figures["conditions_usr"] = plot_conditions_per_category(
        runs, "usr", title="USR per category: baseline vs 3-bit vs each window's repair",
        colors={f"repaired ({n})": c for n, c in colors.items()},
    )
    for name, fig in figures.items():
        fig.savefig(out_dir / f"{name}.png")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("headline_ids", nargs="+", help="e.g. headline_20261020T101500Z (several are combined)")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR,
                        help="where the experiment and run folders are (default: results/)")
    args = parser.parse_args()

    headline_id = (combine_manifests(args.headline_ids, args.results_dir) if len(args.headline_ids) > 1
                   else args.headline_ids[0])
    report = build_report(headline_id, args.results_dir)
    out_dir = args.results_dir / headline_id / "report"
    write_report(report, out_dir, args.results_dir)

    print((out_dir / "headline_table.md").read_text(encoding="utf-8"))
    print((out_dir / "per_category_recovery.md").read_text(encoding="utf-8"))
    if report["overlap"]:
        o = report["overlap"]
        print(f"Verdict: {o['verdict']}")
        print(f"  Module Jaccard: {o['module_jaccard']}")
        print(f"  Common to all:  {[short_module_name(p) for p in o['common_modules']] or 'none'}")
    for name, reason in report["skipped"].items():
        print(f"Skipped {name}: {reason}")
    print(f"\nWritten to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
