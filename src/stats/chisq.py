"""Chi-squared test on stereotype counts, before vs after.

The test operates on a 2x2 table of counts:

                    picked stereotype   didn't
    before (fp16)              1,420    10,720
    after  (3-bit)             2,050    10,090

and asks: if compression changed nothing, how surprising is this table? The
answer comes back as a p-value.

Caveat, stated once and deliberately: the chi-squared test of independence
assumes independent observations, and these two rows are the *same questions*
scored twice. The test is still informative about whether the marginal
stereotype rates differ, but for the paired structure prefer the paired
bootstrap interval in bootstrap.py, and report both.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any, Iterable, Sequence

import numpy as np
from scipy.stats import chi2_contingency

from .metrics import Row, assert_comparable, indicators, is_stereotype

# Minimum expected cell count below which the chi-squared approximation is
# shaky. At BBQ's sample sizes this should never fire; if it does, something is
# wrong with the inputs rather than with the test.
_MIN_EXPECTED = 5.0


@dataclass(frozen=True)
class ChiSquaredResult:
    """Test output, carrying the table that produced it."""

    statistic: float
    p_value: float
    dof: int
    n: int
    correction: bool
    table: np.ndarray = dc_field(compare=False)
    expected: np.ndarray = dc_field(compare=False)
    row_labels: tuple[str, ...] = ("before", "after")
    col_labels: tuple[str, ...] = ("stereotype", "other")

    def as_dict(self) -> dict[str, Any]:
        """JSON-safe form for results/, table included."""
        return {
            "statistic": self.statistic,
            "p_value": self.p_value,
            "dof": self.dof,
            "n": self.n,
            "correction": self.correction,
            "row_labels": list(self.row_labels),
            "col_labels": list(self.col_labels),
            "table": self.table.tolist(),
            "expected": self.expected.tolist(),
        }

    def __str__(self) -> str:
        rows = "; ".join(
            f"{label}={[int(c) for c in counts]}"
            for label, counts in zip(self.row_labels, self.table)
        )
        return (
            f"chi2={self.statistic:.2f} dof={self.dof} p={self.p_value:.3g} "
            f"[{', '.join(self.col_labels)}] {rows}"
        )


def stereotype_table(
    rows_before: Iterable[Row], rows_after: Iterable[Row]
) -> np.ndarray:
    """Counts of stereotyped vs non-stereotyped answers, one row per condition."""
    counts = []
    for rows in (rows_before, rows_after):
        picked = indicators(rows, is_stereotype)
        counts.append([int(picked.sum()), int((~picked).sum())])
    return np.array(counts, dtype=int)


def chisq_from_table(
    table: Sequence[Sequence[int]] | np.ndarray,
    *,
    correction: bool = False,
    row_labels: tuple[str, ...] = ("before", "after"),
    col_labels: tuple[str, ...] = ("stereotype", "other"),
) -> ChiSquaredResult:
    """Chi-squared test of independence on a contingency table.

    `correction=False` overrides scipy's default, which applies Yates' continuity
    correction to 2x2 tables. Yates is conservative and is meant for small
    samples; at BBQ's sample sizes it only biases the statistic downwards, and it
    would also distort the Cramer's V computed from it.
    """
    table = np.asarray(table, dtype=int)
    if table.ndim != 2:
        raise ValueError(f"expected a 2-D table, got shape {table.shape}")
    if table.size == 0 or table.sum() == 0:
        raise ValueError("contingency table is empty")
    if (table < 0).any():
        raise ValueError(f"negative counts in table:\n{table}")
    if (table.sum(axis=1) == 0).any() or (table.sum(axis=0) == 0).any():
        raise ValueError(
            "chi-squared is undefined for a table with an all-zero row or "
            f"column:\n{table}"
        )

    statistic, p_value, dof, expected = chi2_contingency(
        table, correction=correction
    )
    if (expected < _MIN_EXPECTED).any():
        warnings.warn(
            f"expected cell count below {_MIN_EXPECTED}; the chi-squared "
            f"approximation is unreliable here:\n{expected}",
            stacklevel=2,
        )

    return ChiSquaredResult(
        statistic=float(statistic),
        p_value=float(p_value),
        dof=int(dof),
        n=int(table.sum()),
        correction=correction,
        table=table,
        expected=np.asarray(expected, dtype=float),
        row_labels=tuple(row_labels),
        col_labels=tuple(col_labels),
    )


def stereotype_chisq(
    rows_before: Iterable[Row],
    rows_after: Iterable[Row],
    *,
    correction: bool = False,
    check_conditions: bool = True,
    row_labels: tuple[str, ...] = ("before", "after"),
) -> ChiSquaredResult:
    """Did the stereotype rate change between two runs?"""
    rows_before, rows_after = list(rows_before), list(rows_after)
    if check_conditions:
        assert_comparable(rows_before, rows_after)
    table = stereotype_table(rows_before, rows_after)
    return chisq_from_table(
        table, correction=correction, row_labels=row_labels
    )
