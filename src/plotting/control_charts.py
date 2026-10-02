"""Charts for the revised plan's results: controls, recovery by modules, groups, concentration.

Each takes the JSON a src/stats analysis already wrote (or the trials a search wrote), so the
figures always show exactly the numbers in the results files -- never hand-copied values.

    plot_baseline_comparison   USR recovered per patch at the same weight budget, with 95% CIs
    plot_recovery_by_modules   recovery on the full benchmark after restoring the top 1..5 modules
    plot_group_changes         stereotype rate per targeted group: original vs 3-bit vs repaired
    plot_concentration         cumulative share of round-1 gain by module rank, fairness vs general
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from src.plotting.style import CONDITION_PALETTE

SEARCH_COLOR = CONDITION_PALETTE["hqq_repaired"]
OTHER_COLOR = "#9AA5B1"
CAPABILITY_COLOR = "#B8860B"


def _percent_axis(axis) -> None:
    axis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))


def plot_baseline_comparison(controls: dict, search_name: str = "search_top5") -> plt.Figure:
    rows = controls["baselines"]
    names = sorted(rows, key=lambda n: (n != search_name, -rows[n]["usr_recovery"]["point"]))
    points = [rows[n]["usr_recovery"]["point"] for n in names]
    lower = [p - rows[n]["usr_recovery"]["lower"] for n, p in zip(names, points)]
    upper = [rows[n]["usr_recovery"]["upper"] - p for n, p in zip(names, points)]
    colors = [SEARCH_COLOR if n == search_name else OTHER_COLOR for n in names]

    fig, ax = plt.subplots(figsize=(8, 0.5 * len(names) + 1.5))
    ax.barh(names, points, xerr=[lower, upper], color=colors, capsize=3)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    _percent_axis(ax.xaxis)
    ax.set_xlabel("Share of the lost 'unknown' answers recovered (95% CI)")
    ax.set_title("Same memory budget, different choice of modules")
    for y, p in enumerate(points):
        ax.text(min(p + 0.02, 0.93), y, f"{p:.0%}", va="center", fontsize=9)
    fig.tight_layout()
    return fig


def plot_recovery_by_modules(points: list[dict]) -> plt.Figure:
    """points: [{"k": 1, "point": .., "lower": .., "upper": .., "module": ".."}, ...] sorted by k."""
    ks = [p["k"] for p in points]
    ys = [p["point"] for p in points]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(ks, [p["lower"] for p in points], [p["upper"] for p in points], color=SEARCH_COLOR, alpha=0.15)
    ax.plot(ks, ys, marker="o", color=SEARCH_COLOR)
    for p in points:
        label = p.get("module", "").replace("model.layers.", "L").replace(".self_attn.", " ").replace(".mlp.", " ")
        ax.annotate(f"+{label}" if label else "", (p["k"], p["point"]), textcoords="offset points",
                    xytext=(4, -14), fontsize=8, color=SEARCH_COLOR)
    ax.set_xticks(ks)
    ax.set_ylim(0, 1.02)
    _percent_axis(ax.yaxis)
    ax.set_xlabel("Modules restored to 8-bit (in the order the search picked them)")
    ax.set_ylabel("Lost 'unknown' answers recovered, full benchmark")
    ax.set_title("Recovery as modules are added")
    fig.tight_layout()
    return fig


def plot_group_changes(groups: list[dict], min_n: int = 100) -> plt.Figure:
    rows = [g for g in groups if g["n"] >= min_n]
    labels = [f"{g['category'].split('_')[0]}: {g['group'][:28]}" for g in rows]
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(rows) + 1.6))
    for y, g in enumerate(rows):
        base, q3, rep = g["srs"]["baseline"], g["srs"]["hqq_3bit"], g["srs"]["repaired"]
        ax.plot([base, rep], [y, y], color="#CBD2D9", zorder=1)
        ax.scatter(base, y, color=CONDITION_PALETTE["baseline"], zorder=2, label="Original" if y == 0 else None)
        ax.scatter(q3, y, color=CONDITION_PALETTE["hqq_3bit"], marker="x", zorder=2, label="3-bit" if y == 0 else None)
        ax.scatter(rep, y, color=CONDITION_PALETTE["hqq_repaired"], zorder=3, label="Repaired" if y == 0 else None)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels, fontsize=8)
    for tick, g in zip(ax.get_yticklabels(), rows):
        if g["worse_than_baseline"]:
            tick.set_color(CONDITION_PALETTE["hqq_3bit"])
    ax.invert_yaxis()
    _percent_axis(ax.xaxis)
    ax.set_xlabel("Stereotyped answers")
    ax.set_title("Who the repair helps, and who it doesn't")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.text(0.01, 0.0, "Red labels: after repair, significantly more stereotyped than the original model.",
             fontsize=8, color=CONDITION_PALETTE["hqq_3bit"], ha="left", va="bottom")
    fig.subplots_adjust(bottom=max(fig.subplotpars.bottom, 0.09))
    return fig


def cumulative_shares(trials: list[dict]) -> list[float]:
    gains = sorted((max(t["gain"], 0.0) for t in trials if t["round"] == 1), reverse=True)
    total = sum(gains) or 1.0
    shares, running = [], 0.0
    for gain in gains:
        running += gain
        shares.append(running / total)
    return shares


def plot_concentration(fairness_trials: list[dict], capability_trials: list[dict]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, trials, color in (("Fairness damage", fairness_trials, SEARCH_COLOR),
                                 ("General-ability damage", capability_trials, CAPABILITY_COLOR)):
        shares = cumulative_shares(trials)
        ax.plot(range(1, len(shares) + 1), shares, color=color, label=label)
    ax.axvline(5, color="#CBD2D9", linestyle="--")
    ax.set_xscale("log")
    ax.set_ylim(0, 1.02)
    _percent_axis(ax.yaxis)
    ax.set_xlabel("Top-k modules (by single-module gain, log scale)")
    ax.set_ylabel("Share of total single-module gain")
    ax.set_title("Is the damage concentrated or spread out?")
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    return fig
