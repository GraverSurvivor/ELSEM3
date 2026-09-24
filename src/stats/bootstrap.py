"""Bootstrap confidence intervals.

You measured USR = 0.840 on 12,140 questions. That is one sample. A different
12,140 questions would have given a slightly different number, and "how
different" is what a confidence interval answers.

The bootstrap fakes having many samples: draw n rows from your own data *with
replacement* (so a row can be drawn twice, and some rows not at all), compute
the metric on that resample, write it down, repeat 1,000 times. The 2.5th and
97.5th percentiles of those 1,000 values are the 95% interval.

Everything here is seeded, and the seed travels in the result, because an
unseeded CI changes slightly on every run and then two people quote different
numbers for the same result with no way to find out why.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from .metrics import Predicate, Row, align_by_question, assert_comparable, indicators

# The plan specifies 1,000 resamples. Do not quietly lower this to make tests
# fast -- pass n_resamples explicitly in the test instead.
DEFAULT_N_RESAMPLES = 1000
DEFAULT_SEED = 0
DEFAULT_ALPHA = 0.05

# Resample indices are generated in blocks: the full (n_resamples x n) index
# matrix for a 12,140-item run would be ~97 MB.
_BLOCK = 256


@dataclass(frozen=True)
class BootstrapResult:
    """A point estimate with a percentile bootstrap interval around it."""

    point: float
    lower: float
    upper: float
    n_rows: int
    n_resamples: int
    seed: int
    alpha: float
    replicates: np.ndarray = dc_field(repr=False, compare=False)

    @property
    def width(self) -> float:
        return self.upper - self.lower

    @property
    def confidence(self) -> float:
        return 1.0 - self.alpha

    def as_dict(self) -> dict[str, Any]:
        """JSON-safe form for results/ -- replicates deliberately omitted."""
        return {
            "point": self.point,
            "lower": self.lower,
            "upper": self.upper,
            "n_rows": self.n_rows,
            "n_resamples": self.n_resamples,
            "seed": self.seed,
            "alpha": self.alpha,
        }

    def __str__(self) -> str:
        return (
            f"{self.point:.3f} "
            f"({self.confidence:.0%} CI {self.lower:.3f}-{self.upper:.3f})"
        )


def _validate(n_rows: int, n_resamples: int, alpha: float) -> None:
    if n_rows < 1:
        raise ValueError("no rows to resample")
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be >= 1, got {n_resamples}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")


def _finish(
    point: float,
    replicates: np.ndarray,
    n_rows: int,
    n_resamples: int,
    seed: int,
    alpha: float,
) -> BootstrapResult:
    lower, upper = np.percentile(
        replicates, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)]
    )
    return BootstrapResult(
        point=float(point),
        lower=float(lower),
        upper=float(upper),
        n_rows=n_rows,
        n_resamples=n_resamples,
        seed=seed,
        alpha=alpha,
        replicates=replicates,
    )


def bootstrap_values(
    values: np.ndarray,
    *,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    seed: int = DEFAULT_SEED,
    alpha: float = DEFAULT_ALPHA,
) -> BootstrapResult:
    """Bootstrap the mean of a per-row numeric array. The fast path."""
    values = np.asarray(values, dtype=np.float64)
    n = values.size
    _validate(n, n_resamples, alpha)

    rng = np.random.default_rng(seed)
    replicates = np.empty(n_resamples, dtype=np.float64)
    for start in range(0, n_resamples, _BLOCK):
        size = min(_BLOCK, n_resamples - start)
        idx = rng.integers(0, n, size=(size, n))
        replicates[start : start + size] = values[idx].mean(axis=1)

    return _finish(values.mean(), replicates, n, n_resamples, seed, alpha)


def bootstrap_proportion_ci(
    rows: Iterable[Row],
    predicate: Predicate,
    *,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    seed: int = DEFAULT_SEED,
    alpha: float = DEFAULT_ALPHA,
) -> BootstrapResult:
    """CI for the share of rows satisfying `predicate` -- USR, SRS, and friends."""
    values = indicators(rows, predicate).astype(np.float64)
    return bootstrap_values(
        values, n_resamples=n_resamples, seed=seed, alpha=alpha
    )


def bootstrap_ci(
    rows: Sequence[Row],
    metric_fn: Callable[[Sequence[Row]], float],
    *,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    seed: int = DEFAULT_SEED,
    alpha: float = DEFAULT_ALPHA,
) -> BootstrapResult:
    """CI for any metric over rows.

    General but slow: `metric_fn` is called once per resample on a rebuilt list.
    Use `bootstrap_proportion_ci` for anything that is a share of rows; reach for
    this one for metrics that are not a simple mean, such as a ratio of two
    rates.
    """
    rows = list(rows)
    n = len(rows)
    _validate(n, n_resamples, alpha)

    rng = np.random.default_rng(seed)
    replicates = np.empty(n_resamples, dtype=np.float64)
    for k in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        replicates[k] = metric_fn([rows[i] for i in idx])

    return _finish(metric_fn(rows), replicates, n, n_resamples, seed, alpha)


def bootstrap_paired_diff_ci(
    rows_before: Iterable[Row],
    rows_after: Iterable[Row],
    predicate: Predicate,
    *,
    check_conditions: bool = True,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    seed: int = DEFAULT_SEED,
    alpha: float = DEFAULT_ALPHA,
) -> BootstrapResult:
    """CI for (rate after) - (rate before), resampling questions, not rows.

    The two runs score the same questions, so the comparison is paired: each
    resample draws a set of *questions* and uses both runs' answers for them.
    Resampling the runs independently would ignore that pairing and give a
    needlessly wide interval.

    A negative point estimate means the rate fell from before to after.
    """
    rows_before, rows_after = list(rows_before), list(rows_after)
    if check_conditions:
        assert_comparable(rows_before, rows_after)
    before, after = align_by_question(rows_before, rows_after)

    per_question = indicators(after, predicate).astype(
        np.float64
    ) - indicators(before, predicate).astype(np.float64)
    return bootstrap_values(
        per_question, n_resamples=n_resamples, seed=seed, alpha=alpha
    )
