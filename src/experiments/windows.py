"""The headline experiment's three failure windows.

A failure window is the symptom evidence the search localizes from: a fixed sample of the flipped
set, skewed toward one category. The question the headline experiment answers is whether the
modules the search restores depend on which failures it was shown:

    same modules in every window  -> localization is category-invariant
    different modules per window  -> the damage is symptom-specific, so static pre-deployment
                                     protection can't be sufficient

Both outcomes are reportable. The windows only change *which flipped items* the search sees --
the model, the candidates, the threshold and the sample size are identical -- so any difference
in the kept modules is attributable to the evidence.

Weights and sizes, against the committed flipped set (Age 472, Gender 802, Race 944, Religion 131,
SES 1043) at n=300:

    age_heavy  Age 60%, the other four 10% each  -> Age 180 of 472 available
    ses_heavy  SES 60%, the other four 10% each  -> SES 180 of 1043
    balanced   20% each                          -> 60 per category (Religion 60 of 131)

"Balanced" means balanced across categories, deliberately NOT proportional to the flipped set:
the proportional sample is the existing Stage 4 run (hqq_3bit_n300_seed0.json), which already
leans on SES and Race. Equal weights make the balanced window the neutral point between the two
skewed ones.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from src.attribution.search_sample import DEFAULT_SEED, DEFAULT_SIZE, draw_search_sample, write_search_sample
from src.data.bbq_loader import CATEGORIES

SAMPLES_DIR = Path("results/search_samples")
HEAVY_SHARE = 0.6


@dataclass(frozen=True)
class FailureWindow:
    name: str
    description: str
    weights: dict[str, float]


def _heavy(category: str) -> dict[str, float]:
    rest = (1 - HEAVY_SHARE) / (len(CATEGORIES) - 1)
    return {c: (HEAVY_SHARE if c == category else rest) for c in CATEGORIES}


WINDOWS: dict[str, FailureWindow] = {
    w.name: w
    for w in (
        FailureWindow("age_heavy", f"Age {HEAVY_SHARE:.0%}, other categories equal", _heavy("Age")),
        FailureWindow("ses_heavy", f"SES {HEAVY_SHARE:.0%}, other categories equal", _heavy("SES")),
        FailureWindow("balanced", "all five categories equal", {c: 1 / len(CATEGORIES) for c in CATEGORIES}),
    )
}


def sample_path(window: str, n: int = DEFAULT_SIZE, seed: int = DEFAULT_SEED) -> Path:
    """Where a window's sample lives -- next to the Stage 4 sample, named so it can't be confused
    with it (that one is hqq_3bit_n300_seed0.json)."""
    return SAMPLES_DIR / f"hqq_3bit_{window}_n{n}_seed{seed}.json"


def draw_window_sample(flipped: dict, window: str, n: int = DEFAULT_SIZE, seed: int = DEFAULT_SEED) -> dict:
    if window not in WINDOWS:
        raise ValueError(f"unknown window {window!r}; expected one of {sorted(WINDOWS)}")
    sample = draw_search_sample(flipped, n=n, seed=seed, category_weights=WINDOWS[window].weights)
    return {"window": window, **sample}


def ensure_window_sample(flipped: dict, window: str, n: int = DEFAULT_SIZE, seed: int = DEFAULT_SEED) -> Path:
    """Writes the window's sample if missing; if present, checks it is the draw we'd make now.

    A sample left over from an earlier weighting would still load and run, and the search would
    localize from different evidence than the report claims. The draw is deterministic, so a
    fresh draw is the ground truth to compare against.
    """
    path = sample_path(window, n, seed)
    expected = draw_window_sample(flipped, window, n, seed)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("question_ids") != expected["question_ids"]:
            raise ValueError(
                f"{path} exists but doesn't match the {window!r} draw from this flipped set "
                "(different weights or a different flipped set?). Move it aside to re-draw."
            )
        return path
    return write_search_sample(expected, path)
