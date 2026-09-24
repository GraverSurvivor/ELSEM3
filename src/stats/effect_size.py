"""Effect sizes -- how big is the difference, not just whether it is real.

With 12,140 questions almost any difference comes out statistically
significant, so a p-value on its own stops being interesting. Two answers:

  Cramer's V       strength of association in the chi-squared table, 0 to 1.
  proportion delta the plain "USR dropped 0.061 (95% CI 0.048-0.074)" line.
                   This is the one that goes on the slide, because it is the
                   one a reader can interpret without a stats background.

Cohen's h is included alongside the delta: it is the standardised difference
between two proportions, and unlike a raw delta it does not shrink as the rates
approach 0 or 1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from .bootstrap import (
    DEFAULT_ALPHA,
    DEFAULT_N_RESAMPLES,
    DEFAULT_SEED,
    bootstrap_paired_diff_ci,
)
from .chisq import ChiSquaredResult, chisq_from_table
from .metrics import (
    Predicate,
    Row,
    align_by_question,
    assert_comparable,
    indicators,
    is_stereotype,
    is_unknown,
)


@dataclass(frozen=True)
class ProportionDelta:
    """A before/after rate change with an interval and a standardised size."""

    name: str
    before: float
    after: float
    delta: float
    lower: float
    upper: float
    cohens_h: float
    n_pairs: int
    alpha: float
    seed: int
    n_resamples: int

    @property
    def confidence(self) -> float:
        return 1.0 - self.alpha

    @property
    def excludes_zero(self) -> bool:
        """True when the interval does not straddle "no change"."""
        return self.lower > 0.0 or self.upper < 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "before": self.before,
            "after": self.after,
            "delta": self.delta,
            "lower": self.lower,
            "upper": self.upper,
            "cohens_h": self.cohens_h,
            "n_pairs": self.n_pairs,
            "alpha": self.alpha,
            "seed": self.seed,
            "n_resamples": self.n_resamples,
        }

    def __str__(self) -> str:
        return (
            f"{self.name} {self.before:.3f} -> {self.after:.3f}, "
            f"delta {self.delta:+.3f} "
            f"({self.confidence:.0%} CI {self.lower:+.3f} to {self.upper:+.3f}), "
            f"h={self.cohens_h:+.3f}"
        )


def cramers_v(table: Sequence[Sequence[int]] | np.ndarray) -> float:
    """Strength of association in a contingency table, 0 (none) to 1 (perfect).

    For a 2x2 table this is the phi coefficient. Rough reading for 2x2: 0.1
    small, 0.3 medium, 0.5 large -- but quote the raw rate change too, since
    these bands are conventions rather than facts about this benchmark.
    """
    result = chisq_from_table(table, correction=False)
    return cramers_v_from_result(result)


def cramers_v_from_result(result: ChiSquaredResult) -> float:
    """Cramer's V from an already-computed chi-squared, no refitting."""
    if result.correction:
        raise ValueError(
            "Cramer's V from a continuity-corrected statistic understates the "
            "association; recompute the test with correction=False"
        )
    k = min(result.table.shape) - 1
    if k < 1 or result.n == 0:
        raise ValueError(
            f"table shape {result.table.shape} has no association to measure"
        )
    return float(math.sqrt(result.statistic / (result.n * k)))


def cohens_h(before: float, after: float) -> float:
    """Standardised difference between two proportions.

    Negative means the rate fell. Rough reading: 0.2 small, 0.5 medium, 0.8 large.
    """
    for value, label in ((before, "before"), (after, "after")):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{label} must be a proportion in [0, 1], got {value}")

    def phi(p: float) -> float:
        return 2.0 * math.asin(math.sqrt(p))

    return float(phi(after) - phi(before))


def proportion_delta(
    rows_before: Iterable[Row],
    rows_after: Iterable[Row],
    predicate: Predicate,
    *,
    name: str,
    check_conditions: bool = True,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    seed: int = DEFAULT_SEED,
    alpha: float = DEFAULT_ALPHA,
) -> ProportionDelta:
    """Change in the share of rows satisfying `predicate`, with a paired CI."""
    rows_before, rows_after = list(rows_before), list(rows_after)
    if check_conditions:
        assert_comparable(rows_before, rows_after)
    before_rows, after_rows = align_by_question(rows_before, rows_after)

    before = float(indicators(before_rows, predicate).mean())
    after = float(indicators(after_rows, predicate).mean())
    ci = bootstrap_paired_diff_ci(
        before_rows,
        after_rows,
        predicate,
        check_conditions=False,
        n_resamples=n_resamples,
        seed=seed,
        alpha=alpha,
    )
    return ProportionDelta(
        name=name,
        before=before,
        after=after,
        delta=ci.point,
        lower=ci.lower,
        upper=ci.upper,
        cohens_h=cohens_h(before, after),
        n_pairs=len(before_rows),
        alpha=alpha,
        seed=seed,
        n_resamples=n_resamples,
    )


def usr_delta(
    rows_before: Iterable[Row], rows_after: Iterable[Row], **kwargs: Any
) -> ProportionDelta:
    """Change in Unknown Selection Rate. Negative means fairness got worse."""
    kwargs.setdefault("name", "USR")
    return proportion_delta(rows_before, rows_after, is_unknown, **kwargs)


def srs_delta(
    rows_before: Iterable[Row], rows_after: Iterable[Row], **kwargs: Any
) -> ProportionDelta:
    """Change in Stereotype Rate Score. Positive means fairness got worse."""
    kwargs.setdefault("name", "SRS")
    return proportion_delta(rows_before, rows_after, is_stereotype, **kwargs)
