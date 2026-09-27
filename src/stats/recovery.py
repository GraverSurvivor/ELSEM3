"""Recovery fraction: how much of the damage compression did, a repair undoes.

    recovery = (repaired - degraded) / (reference - degraded)

1.0 means the repaired model is back to the reference (uncompressed) rate; 0.0 means the repair
changed nothing; above 1.0 means it went past the reference. The same formula works for both
metrics: for USR the gap is a drop, for SRS a rise, and the signs cancel.

The interval is a paired bootstrap over questions. All three runs score the same questions, so
each resample draws question ids once and uses all three runs' answers for them -- the same
reasoning as `bootstrap_paired_diff_ci`, extended to three runs.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .bootstrap import DEFAULT_ALPHA, DEFAULT_N_RESAMPLES, DEFAULT_SEED, BootstrapResult
from .metrics import Predicate, Row, align_by_question, assert_comparable, indicators

# Below this, the reference and degraded runs are too close for a ratio to mean anything: a
# small gap makes the fraction swing wildly on noise.
MIN_GAP = 1e-9


def recovery_fraction(
    reference: Iterable[Row],
    degraded: Iterable[Row],
    repaired: Iterable[Row],
    predicate: Predicate,
    *,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    seed: int = DEFAULT_SEED,
    alpha: float = DEFAULT_ALPHA,
    check_conditions: bool = True,
) -> BootstrapResult:
    """Share of the reference->degraded change in `predicate`'s rate that `repaired` undoes."""
    reference, degraded, repaired = list(reference), list(degraded), list(repaired)
    if check_conditions:
        assert_comparable(reference, degraded)
        assert_comparable(reference, repaired)
    ref_rows, deg_rows = align_by_question(reference, degraded)
    _, rep_rows = align_by_question(reference, repaired)

    ref = indicators(ref_rows, predicate).astype(np.float64)
    deg = indicators(deg_rows, predicate).astype(np.float64)
    rep = indicators(rep_rows, predicate).astype(np.float64)

    gap = ref.mean() - deg.mean()
    if abs(gap) < MIN_GAP:
        raise ValueError("reference and degraded runs have the same rate; there is nothing to recover")
    point = (rep.mean() - deg.mean()) / gap

    n = len(ref)
    rng = np.random.default_rng(seed)
    replicates = np.empty(n_resamples, dtype=np.float64)
    for k in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        resampled_gap = ref[idx].mean() - deg[idx].mean()
        replicates[k] = (rep[idx].mean() - deg[idx].mean()) / resampled_gap if abs(resampled_gap) >= MIN_GAP else np.nan
    if np.isnan(replicates).any():
        raise ValueError("some resamples had no gap to recover; the difference is too small for a ratio")

    lower, upper = np.percentile(replicates, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return BootstrapResult(
        point=float(point),
        lower=float(lower),
        upper=float(upper),
        n_rows=n,
        n_resamples=n_resamples,
        seed=seed,
        alpha=alpha,
        replicates=replicates,
    )
