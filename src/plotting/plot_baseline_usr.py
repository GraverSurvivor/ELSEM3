"""Loads one run's category_summary.csv and saves the USR-per-category bar chart.

Run against a real run_id (e.g. the one `run_baseline.py` printed):
    python -m src.plotting.plot_baseline_usr bf16_20260920T171540Z
"""

from __future__ import annotations

import sys

from src.experiments.logger import read_category_summary
from src.plotting.style import save_figure, set_house_style
from src.plotting.usr_chart import plot_usr_per_category


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"Usage: python -m {__name__} <run_id>")
    run_id = sys.argv[1]

    summaries = read_category_summary(run_id)
    set_house_style()
    fig = plot_usr_per_category(summaries)
    path = save_figure(fig, "usr_per_category", run_id=run_id)
    print(f"Saved to {path}")


if __name__ == "__main__":
    main()
