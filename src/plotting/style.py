"""House style for every plot the project makes, so the deck's figures look like one system
instead of five different default-matplotlib charts glued together.

Scaffolding only (Week 1) -- the actual charts (USR-per-category bar, flipped-set breakdown,
recovery curve, headline comparison) get built in later weeks on top of this.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

FIGURES_DIR = Path("results") / "figures"

# Colorblind-safe, consistent across every chart -- used to color the 5 BBQ categories the same
# way everywhere (bar charts, recovery curves, headline comparison) so a reader doesn't have to
# re-learn the color mapping per slide.
CATEGORY_PALETTE = {
    "Age": "#0173B2",
    "Gender_identity": "#DE8F05",
    "Race_ethnicity": "#029E73",
    "Religion": "#D55E00",
    "SES": "#CC78BC",
}


def set_house_style() -> None:
    """Call once at the top of any plotting script, before creating figures."""
    sns.set_theme(style="whitegrid", palette="colorblind", font_scale=1.1)
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.bbox"] = "tight"


def save_figure(fig: plt.Figure, name: str, run_id: str | None = None) -> Path:
    """Saves `fig` as a PNG under results/figures/ (or results/figures/{run_id}/ if given).

    `name` should be a plain filename stem, e.g. "usr_per_category" -- ".png" is added here.
    """
    out_dir = FIGURES_DIR / run_id if run_id else FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path)
    return path
