import matplotlib

matplotlib.use("Agg")

import pytest

from src.experiments.schema import CategorySummary
from src.plotting.condition_chart import plot_conditions_per_category
from src.plotting.flipped_chart import plot_flipped_per_category, safe_set_sizes
from src.plotting.recovery_curve import curve_points, plot_recovery_curves
from src.plotting.restored_blocks import parse_module_path, plot_restored_blocks, short_module_name
from src.plotting.style import CONDITION_PALETTE


def _summary(category, usr, condition="fp16", n_unknown=None, srs=0.1):
    return CategorySummary(run_id="r", condition=condition, category=category, n_items=100,
                           n_unknown=n_unknown if n_unknown is not None else int(usr * 100), usr=usr,
                           n_stereotyped=int(srs * 100), srs=srs)


FLIPPED = {
    "total_flipped": 30, "safe_set_size": 200, "baseline_condition": "fp16", "quantized_condition": "hqq_3bit",
    "flipped_per_category": {"SES": 20, "Age": 10},
}


# --- flipped chart -------------------------------------------------------------------------------

def test_flipped_chart_orders_categories_canonically():
    ax = plot_flipped_per_category(FLIPPED).axes[0]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["Age", "SES"]


def test_flipped_chart_labels_share_of_safe_set():
    safe = safe_set_sizes([_summary("Age", 0.5, n_unknown=50), _summary("SES", 0.8, n_unknown=80),
                           _summary("overall", 0.6, n_unknown=130)])
    assert safe == {"Age": 50, "SES": 80}
    ax = plot_flipped_per_category(FLIPPED, safe).axes[0]
    texts = [t.get_text() for t in ax.texts]
    assert "10\n(20% of safe)" in texts
    assert "20\n(25% of safe)" in texts


def test_flipped_chart_raises_when_empty():
    with pytest.raises(ValueError):
        plot_flipped_per_category({**FLIPPED, "flipped_per_category": {}})


# --- condition chart -----------------------------------------------------------------------------

def _runs():
    return {
        "baseline": [_summary("Age", 0.7), _summary("SES", 0.9), _summary("overall", 0.8)],
        "hqq_3bit": [_summary("Age", 0.3, "hqq_3bit"), _summary("SES", 0.4, "hqq_3bit"),
                     _summary("overall", 0.35, "hqq_3bit")],
    }


def test_condition_chart_puts_overall_last_and_uses_house_colours():
    ax = plot_conditions_per_category(_runs()).axes[0]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["Age", "SES", "Overall"]
    first_bar = ax.containers[0][0]
    assert matplotlib.colors.to_hex(first_bar.get_facecolor()) == CONDITION_PALETTE["baseline"].lower()


def test_condition_chart_can_drop_overall():
    ax = plot_conditions_per_category(_runs(), include_overall=False).axes[0]
    assert "Overall" not in [t.get_text() for t in ax.get_xticklabels()]


def test_condition_chart_only_plots_shared_categories():
    runs = _runs()
    runs["baseline"].append(_summary("Religion", 0.9))
    ax = plot_conditions_per_category(runs).axes[0]
    assert "Religion" not in [t.get_text() for t in ax.get_xticklabels()]


def test_condition_chart_rejects_missing_srs_and_bad_metric():
    runs = {"a": [CategorySummary("r", "fp16", "Age", 10, 5, 0.5)]}
    with pytest.raises(ValueError, match="no srs"):
        plot_conditions_per_category(runs, "srs")
    with pytest.raises(ValueError):
        plot_conditions_per_category(runs, "accuracy")


# --- restored blocks / module names --------------------------------------------------------------

def test_parse_and_shorten_module_paths():
    assert parse_module_path("model.layers.14.self_attn.qkv_proj") == (14, "self_attn.qkv_proj")
    assert short_module_name("model.layers.3.mlp.gate_up_proj") == "L3 gate_up"
    for bad in ("model.layers.x.self_attn.qkv_proj", "model.layers.1.self_attn.q_proj", "layers.1.mlp.down_proj"):
        with pytest.raises(ValueError):
            parse_module_path(bad)


def test_restored_blocks_one_panel_per_window_numbered_in_keep_order():
    kept = {
        "age_heavy": ["model.layers.1.self_attn.qkv_proj", "model.layers.14.self_attn.qkv_proj"],
        "ses_heavy": ["model.layers.14.self_attn.qkv_proj"],
    }
    fig = plot_restored_blocks(kept)
    assert len(fig.axes) == 2
    assert [t.get_text() for t in fig.axes[0].texts] == ["1", "2"]
    assert "every window" in fig._suptitle.get_text()


# --- recovery curve ------------------------------------------------------------------------------

SEARCH = {"start_usr": 0.02, "usr_after_round": [0.4, 0.6], "stop_reason": "reached max_rounds",
          "kept": ["model.layers.1.self_attn.qkv_proj", "model.layers.10.self_attn.o_proj"]}


def test_curve_points_starts_at_plain_3bit():
    xs, ys, labels = curve_points(SEARCH)
    assert xs == [0, 1, 2]
    assert ys == [0.02, 0.4, 0.6]
    assert labels == ["", "L1 qkv", "L10 o"]


def test_curve_points_rejects_inconsistent_summary():
    with pytest.raises(ValueError, match="inconsistent"):
        curve_points({**SEARCH, "usr_after_round": [0.4]})
    with pytest.raises(ValueError, match="missing"):
        curve_points({"kept": []})


def test_recovery_curves_one_line_per_search():
    fig = plot_recovery_curves({"a": SEARCH, "b": {**SEARCH, "kept": [], "usr_after_round": []}})
    assert len(fig.axes[0].lines) == 2
