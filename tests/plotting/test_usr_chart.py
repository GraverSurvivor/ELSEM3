import matplotlib

matplotlib.use("Agg")

import pytest

from src.experiments.schema import CategorySummary
from src.plotting.usr_chart import plot_usr_per_category


def _summary(category: str, usr: float, condition: str = "bf16") -> CategorySummary:
    return CategorySummary(
        run_id="run1", condition=condition, category=category, n_items=100, n_unknown=int(usr * 100), usr=usr
    )


def test_plot_orders_bars_to_match_bbq_loader_categories():
    # Deliberately out of order and with a gap (no Religion) -- output should still follow
    # bbq_loader.CATEGORIES' canonical order, skipping what's missing.
    summaries = [_summary("SES", 0.9), _summary("Age", 0.7), _summary("Gender_identity", 0.8)]
    fig = plot_usr_per_category(summaries)
    ax = fig.axes[0]
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert labels == ["Age", "Gender_identity", "SES"]


def test_plot_excludes_overall_row():
    summaries = [_summary("Age", 0.7), _summary("overall", 0.8)]
    fig = plot_usr_per_category(summaries)
    ax = fig.axes[0]
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert "overall" not in labels
    assert labels == ["Age"]


def test_plot_raises_when_nothing_to_plot():
    with pytest.raises(ValueError):
        plot_usr_per_category([_summary("overall", 0.8)])


def test_plot_title_includes_condition():
    summaries = [_summary("Age", 0.7, condition="hqq_3bit")]
    fig = plot_usr_per_category(summaries)
    ax = fig.axes[0]
    assert "hqq_3bit" in ax.get_title()
