"""Statistics for the ELSEM3 fairness-repair experiments.

Everything here reads committed result files off disk. No GPU, no model.
"""

from .bootstrap import (
    BootstrapResult,
    bootstrap_ci,
    bootstrap_paired_diff_ci,
    bootstrap_proportion_ci,
    bootstrap_values,
)
from .chisq import (
    ChiSquaredResult,
    chisq_from_table,
    stereotype_chisq,
    stereotype_table,
)
from .effect_size import (
    ProportionDelta,
    cohens_h,
    cramers_v,
    cramers_v_from_result,
    proportion_delta,
    srs_delta,
    usr_delta,
)
from .metrics import (
    ConditionMismatchError,
    align_by_question,
    assert_comparable,
    by_category,
    indicators,
    is_stereotype,
    is_unknown,
    srs,
    usr,
)

__all__ = [
    "BootstrapResult",
    "ChiSquaredResult",
    "ConditionMismatchError",
    "ProportionDelta",
    "align_by_question",
    "assert_comparable",
    "bootstrap_ci",
    "bootstrap_paired_diff_ci",
    "bootstrap_proportion_ci",
    "bootstrap_values",
    "by_category",
    "chisq_from_table",
    "cohens_h",
    "cramers_v",
    "cramers_v_from_result",
    "indicators",
    "is_stereotype",
    "is_unknown",
    "proportion_delta",
    "srs",
    "srs_delta",
    "stereotype_chisq",
    "stereotype_table",
    "usr",
    "usr_delta",
]
