"""Stage 3 in one call: the flipped set plus the statistics that say the damage is real.

Everything here already exists -- the flipped set in `src.data.safe_set`, the tests in
`src.stats` -- and this only runs them together on one baseline/compressed pair, so the headline
experiment can detect on each failure window without re-assembling the steps each time:

    from src.stats.detect import run_detection

    result = run_detection(baseline_rows, compressed_rows)
    result.total_flipped, result.usr.delta, result.chisq.p_value, result.cramers_v

Rows are QuestionResults, e.g. from `src.data.safe_set.read_results_file(...)`. Both runs must be
at the same compute dtype and cover the same questions; each underlying step raises otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.data.safe_set import flipped_set_payload
from src.experiments.schema import QuestionResult

from .bootstrap import DEFAULT_N_RESAMPLES, DEFAULT_SEED
from .chisq import ChiSquaredResult, stereotype_chisq
from .effect_size import ProportionDelta, cramers_v_from_result, srs_delta, usr_delta


@dataclass(frozen=True)
class DetectionResult:
    """What compression did to one run, relative to the baseline."""

    baseline_run_id: str
    compressed_run_id: str
    safe_set_size: int
    flipped: list[dict]  # safe-set items now answered with the stereotype
    flipped_per_category: dict[str, int]
    usr: ProportionDelta  # USR change, with a paired bootstrap CI
    srs: ProportionDelta  # SRS change, with a paired bootstrap CI
    chisq: ChiSquaredResult  # stereotype vs other, before vs after
    cramers_v: float  # effect size of that shift

    @property
    def total_flipped(self) -> int:
        return len(self.flipped)

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe form for results/. The flipped items are included in full."""
        return {
            "baseline_run_id": self.baseline_run_id,
            "compressed_run_id": self.compressed_run_id,
            "safe_set_size": self.safe_set_size,
            "total_flipped": self.total_flipped,
            "flipped_per_category": self.flipped_per_category,
            "usr": self.usr.as_dict(),
            "srs": self.srs.as_dict(),
            "chisq": self.chisq.as_dict(),
            "cramers_v": self.cramers_v,
            "flipped": self.flipped,
        }


def run_detection(
    baseline: list[QuestionResult],
    compressed: list[QuestionResult],
    *,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> DetectionResult:
    """Detects what compression broke: the flipped set, and whether the shift is significant."""
    payload = flipped_set_payload(baseline, compressed)
    chi = stereotype_chisq(baseline, compressed)
    return DetectionResult(
        baseline_run_id=payload["baseline_run_id"],
        compressed_run_id=payload["quantized_run_id"],
        safe_set_size=payload["safe_set_size"],
        flipped=payload["items"],
        flipped_per_category=payload["flipped_per_category"],
        usr=usr_delta(baseline, compressed, n_resamples=n_resamples, seed=seed),
        srs=srs_delta(baseline, compressed, n_resamples=n_resamples, seed=seed),
        chisq=chi,
        cramers_v=cramers_v_from_result(chi),
    )
