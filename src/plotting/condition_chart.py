"""Grouped bar chart comparing model conditions per category -- baseline vs 3-bit vs repaired.

This is the one-picture version of the whole project: grey bars (what the model does
uncompressed), red bars (what 3-bit compression breaks), blue bars (what the repair gets back).
Works for either metric: USR (higher is better) or SRS (lower is better).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from src.data.bbq_loader import CATEGORIES
from src.experiments.schema import CategorySummary
from src.plotting.style import CONDITION_PALETTE

METRIC_LABELS = {
    "usr": "USR (Unknown Selection Rate) — higher is better",
    "srs": "SRS (Stereotype Rate Score) — lower is better",
}
_FALLBACK_COLORS = ("#999999", "#555555", "#BBBBBB")


def plot_conditions_per_category(
    runs: dict[str, list[CategorySummary]],
    metric: str = "usr",
    include_overall: bool = True,
    title: str | None = None,
    colors: dict[str, str] | None = None,
) -> plt.Figure:
    """One group of bars per category, one bar per run.

    `runs` maps a display label to that run's category summaries; insertion order is bar order,
    so pass baseline, then hqq_3bit, then hqq_repaired. Labels found in `colors`, then
    `CONDITION_PALETTE`, get that colour. "overall" is drawn last, after a gap, when
    `include_overall` is set -- it is an aggregate, so it's kept visually apart from the 5 real
    categories.
    """
    palette = {**CONDITION_PALETTE, **(colors or {})}
    if metric not in METRIC_LABELS:
        raise ValueError(f"metric must be one of {sorted(METRIC_LABELS)}, got {metric!r}")
    if not runs:
        raise ValueError("no runs to plot")

    tables = {label: {s.category: s for s in summaries} for label, summaries in runs.items()}
    groups = [c for c in CATEGORIES if all(c in table for table in tables.values())]
    if include_overall and all("overall" in table for table in tables.values()):
        groups.append("overall")
    if not groups:
        raise ValueError("the runs share no categories")

    values = {}
    for label, table in tables.items():
        row = [getattr(table[g], metric) for g in groups]
        if any(v is None for v in row):
            raise ValueError(f"run {label!r} has no {metric} values (unlabelled stereotype data?)")
        values[label] = row

    # A gap before "overall" so it doesn't read as a sixth category.
    x = np.arange(len(groups), dtype=float)
    if groups[-1] == "overall":
        x[-1] += 0.5
    n = len(runs)
    width = 0.8 / n

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for i, (label, row) in enumerate(values.items()):
        color = palette.get(label, _FALLBACK_COLORS[i % len(_FALLBACK_COLORS)])
        bars = ax.bar(x + (i - (n - 1) / 2) * width, row, width, label=label, color=color)
        ax.bar_label(bars, labels=[f"{v:.0%}" for v in row], padding=2, fontsize=8)

    ax.set_xticks(x, [g if g != "overall" else "Overall" for g in groups])
    ax.set_ylim(0, 1.08)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_ylabel(METRIC_LABELS[metric])
    ax.set_title(title or f"{metric.upper()} per category by condition")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=n, frameon=False)
    fig.tight_layout()
    return fig
