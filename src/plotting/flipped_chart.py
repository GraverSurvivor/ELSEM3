"""Flipped-set chart: how many safe-set questions compression flipped to the stereotype, per
category.

Reads the flipped-set payload `src.data.safe_set.write_flipped_set` produces
(`results/flipped/hqq_3bit.json`). Raw counts alone mislead here -- the categories differ ~6x in
size (Religion has 600 items, Race_ethnicity 3,440), so Religion's small bar would read as "barely
affected". Each bar is therefore labelled with its share of that category's safe set too, when the
baseline's per-category safe-set sizes are passed in.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from src.data.bbq_loader import CATEGORIES
from src.experiments.schema import CategorySummary
from src.plotting.style import CATEGORY_PALETTE


def safe_set_sizes(baseline_summaries: list[CategorySummary]) -> dict[str, int]:
    """Per-category safe-set size: the baseline's "unknown" answers, which is exactly how
    `build_safe_set` defines the safe set."""
    return {s.category: s.n_unknown for s in baseline_summaries if s.category != "overall"}


def plot_flipped_per_category(flipped: dict, safe_sizes: dict[str, int] | None = None) -> plt.Figure:
    """Bar chart of flipped items per category, in `CATEGORIES` order.

    `flipped` is the flipped-set payload. `safe_sizes` (from `safe_set_sizes`) adds each bar's
    share of its category's safe set to the label.
    """
    counts = flipped["flipped_per_category"]
    categories = [c for c in CATEGORIES if c in counts]
    if not categories:
        raise ValueError("flipped set has no per-category counts to plot")
    values = [counts[c] for c in categories]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(categories, values, color=[CATEGORY_PALETTE[c] for c in categories])
    if safe_sizes is not None:
        labels = [f"{v:,}\n({v / safe_sizes[c]:.0%} of safe)" for c, v in zip(categories, values)]
    else:
        labels = [f"{v:,}" for v in values]
    ax.bar_label(bars, labels=labels, padding=3, fontsize=9)
    ax.margins(y=0.2)

    ax.set_ylabel("Safe-set items flipped to the stereotype")
    ax.set_title(
        f"Flipped set: {flipped['total_flipped']:,} of {flipped['safe_set_size']:,} safe-set items "
        f"— {flipped['quantized_condition']} vs {flipped['baseline_condition']}"
    )
    fig.tight_layout()
    return fig
