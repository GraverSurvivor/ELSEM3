"""USR-per-category bar chart — the first real chart on top of Week 1's house-style scaffold.

Call `set_house_style()` once before building any figure (per style.py's own convention); this
module's `plot_usr_per_category` doesn't call it internally, so it composes cleanly with
whatever else a script plots in the same run.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from src.data.bbq_loader import CATEGORIES
from src.experiments.schema import CategorySummary
from src.plotting.style import CATEGORY_PALETTE


def plot_usr_per_category(summaries: list[CategorySummary]) -> plt.Figure:
    """Bar chart of USR (Unknown Selection Rate) per category, for one run.

    Excludes any "overall" row (an aggregate, not a category — plotting it alongside the 5 real
    categories would misleadingly suggest it's a 6th one). Bars are ordered to match
    `bbq_loader.CATEGORIES` (Age, Gender_identity, Race_ethnicity, Religion, SES) rather than
    alphabetically, so this stays visually consistent with every other project chart that uses
    the same category order and `CATEGORY_PALETTE` colors.
    """
    by_category = {s.category: s for s in summaries if s.category != "overall"}
    ordered = [by_category[c] for c in CATEGORIES if c in by_category]
    if not ordered:
        raise ValueError("No per-category summaries to plot (only an 'overall' row, or empty)")

    categories = [s.category for s in ordered]
    usrs = [s.usr for s in ordered]
    colors = [CATEGORY_PALETTE[c] for c in categories]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(categories, usrs, color=colors)
    ax.bar_label(bars, labels=[f"{usr:.1%}" for usr in usrs], padding=3)

    ax.set_ylabel("USR (Unknown Selection Rate)")
    ax.set_ylim(0, 1)
    condition = ordered[0].condition
    ax.set_title(f"USR per category — {condition}")
    fig.tight_layout()
    return fig
