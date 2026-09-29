"""Which modules each repair restored, drawn on the model's own layout -- the headline figure.

One panel per failure window, each a 4 x 32 grid: rows are the four candidate module types
(qkv_proj, o_proj, gate_up_proj, down_proj), columns are the 32 layers. A restored module is a
filled cell numbered by the round the search kept it in. Stacking the windows with a shared layer
axis makes the headline question answerable at a glance: the same cells lit in every panel means
the damage is category-invariant; different cells means it depends on the symptom, which static
pre-deployment protection cannot anticipate.

Cells restored in every window are outlined, so the overlap doesn't have to be found by eye.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from src.quant.candidates import ATTENTION_MODULES, LAYER_PREFIX, MLP_MODULES

MODULE_TYPES = ATTENTION_MODULES + MLP_MODULES
_DEFAULT_COLOR = "#1F3A68"


def parse_module_path(path: str) -> tuple[int, str]:
    """"model.layers.14.self_attn.qkv_proj" -> (14, "self_attn.qkv_proj")."""
    prefix = LAYER_PREFIX + "."
    if not path.startswith(prefix):
        raise ValueError(f"not a candidate module path: {path!r}")
    layer, _, suffix = path[len(prefix):].partition(".")
    if not layer.isdigit() or suffix not in MODULE_TYPES:
        raise ValueError(f"not a candidate module path: {path!r}")
    return int(layer), suffix


def short_module_name(path: str) -> str:
    """"model.layers.14.self_attn.qkv_proj" -> "L14 qkv" -- short enough for a plot label."""
    layer, suffix = parse_module_path(path)
    return f"L{layer} {suffix.rsplit('.', 1)[-1].removesuffix('_proj')}"


def plot_restored_blocks(
    kept_by_window: dict[str, list[str]],
    n_layers: int = 32,
    colors: dict[str, str] | None = None,
) -> plt.Figure:
    """One 4 x `n_layers` grid per window; filled cells are restored modules, numbered by round.

    `kept_by_window` maps a window label to its search's `kept` list, in keep order.
    """
    if not kept_by_window:
        raise ValueError("no windows to plot")
    colors = colors or {}
    parsed = {label: [parse_module_path(p) for p in kept] for label, kept in kept_by_window.items()}
    common = set.intersection(*(set(cells) for cells in parsed.values())) if len(parsed) > 1 else set()

    n = len(parsed)
    fig, axes = plt.subplots(n, 1, figsize=(13, 1.35 * n + 0.9), sharex=True, squeeze=False)
    for ax, (label, cells) in zip(axes[:, 0], parsed.items()):
        color = colors.get(label, _DEFAULT_COLOR)
        for row in range(len(MODULE_TYPES)):
            for layer in range(n_layers):
                ax.add_patch(Rectangle((layer, row), 1, 1, facecolor="#F2F2F2", edgecolor="white", lw=1))
        for order, (layer, suffix) in enumerate(cells, start=1):
            row = MODULE_TYPES.index(suffix)
            outline = {"edgecolor": "black", "lw": 2} if (layer, suffix) in common else {"edgecolor": "white", "lw": 1}
            ax.add_patch(Rectangle((layer, row), 1, 1, facecolor=color, **outline))
            ax.text(layer + 0.5, row + 0.5, str(order), ha="center", va="center", color="white",
                    fontsize=8, fontweight="bold")

        ax.set_xlim(0, n_layers)
        ax.set_ylim(len(MODULE_TYPES), 0)
        ax.set_yticks([r + 0.5 for r in range(len(MODULE_TYPES))],
                      [s.rsplit(".", 1)[-1] for s in MODULE_TYPES], fontsize=8)
        ax.set_ylabel(f"{label}\n({len(cells)} kept)", rotation=0, ha="right", va="center", fontsize=9)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)

    bottom = axes[-1, 0]
    bottom.set_xticks([layer + 0.5 for layer in range(n_layers)], [str(l) for l in range(n_layers)], fontsize=7)
    bottom.set_xlabel("Layer")
    title = "Modules restored to higher precision, by failure window (number = round kept)"
    if common:
        title += "\nOutlined: restored in every window"
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig
