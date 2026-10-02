"""Every figure and table the committed reference runs support, in one CPU-only command.

Reads results/reference/{baseline_fp16, hqq_3bit, hqq_repaired} and results/flipped/hqq_3bit.json
-- all tracked by git, so this needs no GPU and no dataset download -- and writes under
results/figures/:

    usr_per_category_<run>.png    the Week 2 chart, once per reference run
    flipped_per_category.png      Stage 3: how many safe-set items flipped, per category
    conditions_usr.png            baseline vs 3-bit vs repaired, USR per category
    conditions_srs.png            same, SRS
    recovery_curve.png            only with --search-summary (search runs aren't committed)
    results_table.{csv,md}        the per-category comparison with bootstrap CIs

    python -m src.plotting.make_figures
    python -m src.plotting.make_figures --search-summary results/search_<id>/summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.data.safe_set import read_results_file  # noqa: E402
from src.experiments.logger import read_category_summary  # noqa: E402
from src.plotting.condition_chart import plot_conditions_per_category  # noqa: E402
from src.plotting.flipped_chart import plot_flipped_per_category, safe_set_sizes  # noqa: E402
from src.plotting.recovery_curve import plot_recovery_curves  # noqa: E402
from src.plotting.style import FIGURES_DIR, save_figure, set_house_style  # noqa: E402
from src.plotting.tables import comparison_rows, write_table  # noqa: E402
from src.plotting.usr_chart import plot_usr_per_category  # noqa: E402

REFERENCE_DIR = Path("results/reference")
# Reference directory -> the condition label used on charts (and in CONDITION_PALETTE).
DEFAULT_RUNS = {"baseline_fp16": "baseline", "hqq_3bit": "hqq_3bit", "hqq_repaired": "hqq_repaired"}


def _save(fig: plt.Figure, name: str) -> Path:
    path = save_figure(fig, name)
    plt.close(fig)
    print(f"  {path}")
    return path


def _prefix_points() -> list[dict]:
    """Full-benchmark recovery after restoring the search's top 1..k modules, from promoted
    prefix repairs plus the committed full repair. Empty until the prefix runs exist."""
    from src.stats.metrics import is_unknown
    from src.stats.recovery import recovery_fraction

    runs = {}
    for summary_path in REFERENCE_DIR.glob("*/summary.json"):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        source = summary.get("patch_source") or {}
        if summary.get("condition") != "hqq_repaired":
            continue
        if source.get("kind") == "prefix":
            runs[int(source["k"])] = (summary_path.parent, summary["modules"])
        elif summary_path.parent.name == "hqq_repaired":
            runs[len(summary["modules"])] = (summary_path.parent, summary["modules"])
    if len(runs) < 2:
        return []
    base = read_results_file(REFERENCE_DIR / "baseline_fp16" / "question_results.jsonl")
    q3 = read_results_file(REFERENCE_DIR / "hqq_3bit" / "question_results.jsonl")
    points = []
    for k in sorted(runs):
        directory, modules = runs[k]
        r = recovery_fraction(base, q3, read_results_file(directory / "question_results.jsonl"), is_unknown)
        points.append({"k": k, "point": r.point, "lower": r.lower, "upper": r.upper, "module": modules[k - 1]})
    return points


def _revised_plan_figures(capability_trials: Path | None, fairness_trials: Path | None) -> None:
    """Figures for the controls, groups, recovery-by-modules and concentration -- each drawn only
    once the analysis behind it has produced its results file."""
    from src.plotting.control_charts import (
        plot_baseline_comparison,
        plot_concentration,
        plot_group_changes,
        plot_recovery_by_modules,
    )

    stats_dir = Path("results/stats")
    controls = stats_dir / "controls.json"
    if controls.exists():
        data = json.loads(controls.read_text(encoding="utf-8"))
        if len(data.get("baselines", {})) > 1:
            _save(plot_baseline_comparison(data), "controls_baselines")
    groups = stats_dir / "groups.json"
    if groups.exists():
        _save(plot_group_changes(json.loads(groups.read_text(encoding="utf-8"))), "groups_srs")
    points = _prefix_points()
    if points:
        _save(plot_recovery_by_modules(points), "recovery_by_modules")
    if capability_trials and fairness_trials:
        from src.stats.dissociation import read_trials

        _save(plot_concentration(read_trials(fairness_trials), read_trials(capability_trials)), "concentration")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--flipped", type=Path, default=Path("results/flipped/hqq_3bit.json"))
    parser.add_argument("--search-summary", type=Path, action="append", default=[],
                        help="a run_search summary.json to draw a recovery curve for (repeatable)")
    parser.add_argument("--skip-table", action="store_true",
                        help="skip the results table (its bootstrap CIs take a minute or two)")
    parser.add_argument("--fairness-trials", type=Path, help="trials.csv of the fairness search, for the concentration chart")
    parser.add_argument("--capability-trials", type=Path, help="trials.csv of the general-ability search")
    args = parser.parse_args()

    set_house_style()
    print(f"Writing to {FIGURES_DIR}/")

    summaries = {}
    for directory, label in DEFAULT_RUNS.items():
        if not (REFERENCE_DIR / directory / "category_summary.csv").exists():
            print(f"  (no reference run {directory!r}, skipping it)")
            continue
        summaries[label] = read_category_summary(f"{REFERENCE_DIR.name}/{directory}")
        _save(plot_usr_per_category(summaries[label]), f"usr_per_category_{label}")

    if args.flipped.exists():
        flipped = json.loads(args.flipped.read_text(encoding="utf-8"))
        safe = safe_set_sizes(summaries["baseline"]) if "baseline" in summaries else None
        _save(plot_flipped_per_category(flipped, safe), "flipped_per_category")

    if len(summaries) >= 2:
        _save(plot_conditions_per_category(summaries, "usr", title="USR per category: baseline vs 3-bit vs repaired"),
              "conditions_usr")
        _save(plot_conditions_per_category(summaries, "srs", title="SRS per category: baseline vs 3-bit vs repaired"),
              "conditions_srs")

    if args.search_summary:
        searches = {path.parent.name: json.loads(path.read_text(encoding="utf-8")) for path in args.search_summary}
        _save(plot_recovery_curves(searches), "recovery_curve")

    _revised_plan_figures(args.capability_trials, args.fairness_trials)

    if not args.skip_table and len(summaries) == 3:
        from src.stats.repair_report import build_comparison

        print("Bootstrapping CIs for the results table...")
        rows = {label: read_results_file(REFERENCE_DIR / d / "question_results.jsonl") for d, label in DEFAULT_RUNS.items()}
        comparison = build_comparison(rows["baseline"], rows["hqq_3bit"], rows["hqq_repaired"])
        for path in write_table(comparison_rows(comparison), FIGURES_DIR / "results_table"):
            print(f"  {path}")


if __name__ == "__main__":
    main()
