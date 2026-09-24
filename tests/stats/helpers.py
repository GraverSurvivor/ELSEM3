"""Synthetic QuestionResult rows for the statistics tests.

A real run comes off a GPU. Everything in src/stats/ only ever sees rows, so
the tests build rows by hand with known answers.

Answer pattern letters:
    u -- picked "unknown"          (correct; counts towards USR)
    s -- picked the stereotype     (counts towards SRS)
    o -- picked the other person   (wrong, but not a fairness symptom)
"""

from __future__ import annotations

from dataclasses import dataclass

UNKNOWN_INDEX = 0
STEREOTYPE_INDEX = 1
OTHER_INDEX = 2

_LETTER_TO_INDEX = {
    "u": UNKNOWN_INDEX,
    "s": STEREOTYPE_INDEX,
    "o": OTHER_INDEX,
}


@dataclass
class Row:
    """Stands in for experiments.schema.QuestionResult."""

    question_id: str
    category: str
    condition: str
    predicted_index: int
    unknown_index: int
    stereotype_index: int
    is_stereotyped_predicted: bool = False


def make_rows(pattern, *, condition="fp16", category="Age", prefix="q"):
    """Build one row per letter in `pattern`, ids `q0`, `q1`, ..."""
    rows = []
    for i, letter in enumerate(pattern):
        pred_idx = _LETTER_TO_INDEX[letter]
        rows.append(
            Row(
                question_id=f"{prefix}{i}",
                category=category,
                condition=condition,
                predicted_index=pred_idx,
                unknown_index=UNKNOWN_INDEX,
                stereotype_index=STEREOTYPE_INDEX,
                is_stereotyped_predicted=(pred_idx == STEREOTYPE_INDEX),
            )
        )
    return rows


def as_dicts(rows):
    """The same rows as plain dicts, to prove get_field handles both."""
    return [vars(row).copy() for row in rows]


def pattern(n_unknown=0, n_stereotype=0, n_other=0):
    """A pattern string with the given counts, e.g. pattern(7, 2, 1) -> 10 rows."""
    return "u" * n_unknown + "s" * n_stereotype + "o" * n_other