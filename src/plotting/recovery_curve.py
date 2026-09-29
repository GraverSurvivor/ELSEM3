"""Recovery curve: USR on the search sample as the greedy search restores modules one by one.

x = number of modules restored (0 = plain 3-bit), y = USR on the flipped-set sample. Read off a
search run's summary.json (`src/attribution/run_search.py`): `start_usr` is the x=0 point and
`usr_after_round[i]` is the model with `kept[:i+1]` restored. Each point is labelled with the
module kept at that step.

This is the SAMPLE's USR, which starts near 0 by construction (every sample item is one the 3-bit
model gets wrong). It shows how fast the search finds its fix, not the final fairness number --
that comes from the full-benchmark verification, and the axis label says so.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from src.plotting.restored_blocks import short_module_name

_DEFAULT_COLORS = ("#1F3A68", "#C0392B", "#029E73", "#DE8F05")


def curve_points(summary: dict) -> tuple[list[int], list[float], list[str]]:
    """(x, y, labels) for one search summary. Labels are "" for x=0 (nothing restored yet)."""
    for key in ("start_usr", "usr_after_round", "kept"):
        if key not in summary:
            raise ValueError(f"search summary is missing {key!r} -- is this a run_search summary.json?")
    kept, after = summary["kept"], summary["usr_after_round"]
    if len(kept) != len(after):
        raise ValueError(f"{len(kept)} modules kept but {len(after)} USR measurements -- summary is inconsistent")
    xs = list(range(len(kept) + 1))
    ys = [summary["start_usr"], *after]
    labels = ["", *(short_module_name(path) for path in kept)]
    return xs, ys, labels


def plot_recovery_curves(
    searches: dict[str, dict],
    colors: dict[str, str] | None = None,
    annotate: bool | None = None,
) -> plt.Figure:
    """One line per search run. `searches` maps a label (e.g. a failure window) to its summary.

    Points are labelled with the module kept at each step by default only for a single run: with
    several, the labels pile up wherever the curves coincide, and the restored-blocks figure
    already shows which modules each window kept.
    """
    if not searches:
        raise ValueError("no search runs to plot")
    if annotate is None:
        annotate = len(searches) == 1
    colors = colors or {}

    fig, ax = plt.subplots(figsize=(8, 5))
    max_x = 0
    for i, (label, summary) in enumerate(searches.items()):
        xs, ys, point_labels = curve_points(summary)
        max_x = max(max_x, xs[-1])
        color = colors.get(label, _DEFAULT_COLORS[i % len(_DEFAULT_COLORS)])
        ax.plot(xs, ys, marker="o", color=color, label=f"{label} — {summary.get('stop_reason', '')}".rstrip(" —"))
        if annotate:
            for x, y, text in zip(xs, ys, point_labels):
                if text:
                    ax.annotate(text, (x, y), textcoords="offset points", xytext=(4, -12), fontsize=7, color=color)

    ax.set_xticks(range(max_x + 1))
    ax.set_xlim(-0.2, max_x + 0.6)
    ax.set_ylim(0, 1.02)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel("Modules restored to higher precision")
    ax.set_ylabel("USR on the flipped-set search sample")
    ax.set_title("Recovery curve: greedy attribution search")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    return fig
