"""Smoke tests: each revised-plan chart builds from the JSON shape its analysis writes."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from src.plotting.control_charts import (  # noqa: E402
    cumulative_shares,
    plot_baseline_comparison,
    plot_concentration,
    plot_group_changes,
    plot_recovery_by_modules,
)


def _ci(p):
    return {"point": p, "lower": p - 0.02, "upper": p + 0.02}


def test_baseline_comparison_puts_the_search_first():
    controls = {"baselines": {"random_s0": {"usr_recovery": _ci(0.3)}, "search_top5": {"usr_recovery": _ci(0.9)},
                              "positional": {"usr_recovery": _ci(0.6)}}}
    fig = plot_baseline_comparison(controls)
    labels = [t.get_text() for t in fig.axes[0].get_yticklabels()]
    assert labels[0] == "search_top5"
    plt.close(fig)


def test_recovery_by_modules_and_groups_build():
    points = [{"k": k, **_ci(0.2 * k), "module": f"model.layers.{k}.self_attn.qkv_proj"} for k in range(1, 5)]
    plt.close(plot_recovery_by_modules(points))
    groups = [{"category": "Age", "group": "old", "n": 800, "srs": {"baseline": 0.3, "hqq_3bit": 0.5, "repaired": 0.25},
               "worse_than_baseline": False},
              {"category": "Gender_identity", "group": "F", "n": 1600, "srs": {"baseline": 0.1, "hqq_3bit": 0.4, "repaired": 0.17},
               "worse_than_baseline": True}]
    plt.close(plot_group_changes(groups))


def test_cumulative_shares_and_concentration():
    trials = [{"round": 1, "gain": g} for g in (0.4, 0.4, 0.2, -0.1)]
    assert cumulative_shares(trials) == pytest.approx([0.4, 0.8, 1.0, 1.0])
    plt.close(plot_concentration(trials, trials))
