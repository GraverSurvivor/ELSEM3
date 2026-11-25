"""How version 2 of Fast Repair decides: which questions each decision may see, and how a trial is scored.

Splits. Every question belongs to one, fixed by a hash of its id and the seed:

    search      40%   finding the parts
    validation  20%   choosing between finished repairs (reserved: the re-measured repair, PR A2, uses it)
    final       40%   the numbers and the verdict; no decision ever reads them

`search` lies inside version 1's search half and `final` inside its held-out half, so the two versions
can be compared on the same `final` questions when both are run with the same `--questions` and seed (the
sample depends on both; their defaults differ).

Trial scores. Version 1 counts right answers on about 60 questions, so a part's gain moves in steps of
a point or two, and a part that moves "can't tell" from 10% to 45% without winning counts as nothing.
Version 2 uses the probabilities behind the answers. Three scores are computed for every trial:

    soft    average probability of "can't tell" on the sample, minus what the part takes away from the
            right answers of the stated-answer questions
    kl      minus how far the sample's probabilities are from the original model's (KL divergence), minus
            what the part adds to that distance on the stated-answer questions
    count   version 1's: share answered "can't tell", minus the hedging the part adds

One of them (`FastRepairConfig.objective`) decides; all three go to scan.jsonl. Which predicts the full
result best is measured, not assumed: `study` and src/product/compare_objectives.py.

On stated-answer questions a part can only lose points: it is charged for harm it adds to what the
compressed model did, and gets nothing for general repair. Rewarding that too is what made an early
version keep parts that raised Qwen2.5-3B's accuracy (31% -> 68%) while fair answers fell.
"""

from __future__ import annotations

import hashlib
import json
import math
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

SPLITS = ("search", "validation", "final")
OBJECTIVES = ("soft", "kl", "count")
SURE = 0.98          # a stand-in backend without probabilities: how sure its picks are taken to be
CLAMP = 0.01         # KL: smaller probabilities count as this, so a few extreme questions don't decide
BALANCED_CAP = 400   # final-check questions of each set also asked with their choices rotated


def split_of(question_id: str, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}:{question_id}".encode()).digest()
    if digest[1] % 5 == 4:
        return "validation"
    return "search" if digest[0] % 2 == 0 else "final"


# --- asking ------------------------------------------------------------------------------------------

def ask(backend, view: str, items: list, on_progress=None) -> tuple[list[int], list[tuple[float, ...]]]:
    """Each question's pick and the probability of every choice. A backend without probabilities (the
    tests' stand-ins) is taken to be `SURE` of its picks."""
    if hasattr(backend, "probs"):
        probs = backend.probs(view, items, on_progress)
        return [max(range(len(p)), key=p.__getitem__) for p in probs], probs
    picks = backend.score(view, items, on_progress)
    return picks, [tuple(SURE if i == pick else (1 - SURE) / (len(it.options) - 1) for i in range(len(it.options)))
                   for it, pick in zip(items, picks)]


def rotate(item, k: int):
    """The question with its choices moved `k` places up: choice `i` is shown at position `(i - k) % n`."""
    n = len(item.options)
    place = lambda i: (i - k) % n if i >= 0 else i  # noqa: E731 -- -1 means "none" and stays
    return replace(item, options=tuple(item.options[(pos + k) % n] for pos in range(n)), label=place(item.label),
                   unknown_index=place(item.unknown_index), stereotype_index=place(item.stereotype_index))


def ask_balanced(backend, view: str, items: list, printed: list[tuple[float, ...]]) -> list[int]:
    """Each question's pick when its choices are asked in every rotation and the probabilities averaged:
    what the model answers once the letters' positions can't help or hurt. `printed`: the probabilities
    already measured as printed (rotation 0). Picks are in the original choice order."""
    if not items:
        return []
    totals = [list(p) for p in printed]
    for k in range(1, len(items[0].options)):
        _, probs = ask(backend, view, [rotate(it, k) for it in items])
        for total, row in zip(totals, probs):
            for i in range(len(total)):
                total[i] += row[(i - k) % len(total)]
    return [max(range(len(t)), key=t.__getitem__) for t in totals]


# --- scoring a trial ---------------------------------------------------------------------------------

def kl(reference: tuple[float, ...], p: tuple[float, ...]) -> float:
    """KL(reference || p) over the choices, with small probabilities raised to CLAMP."""
    a, b = ([max(v, CLAMP) for v in row] for row in (reference, p))
    sum_a, sum_b = sum(a), sum(b)
    return sum((x / sum_a) * math.log((x / sum_a) / (y / sum_b)) for x, y in zip(a, b))


def mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def combine(sample: list, stated: list, weight: float = 1.0) -> dict[str, float]:
    """The three scores from per-question results: `sample` rows (right, probability of the right answer,
    KL from the original) and `stated` rows (hedging added, probability taken away, KL added)."""
    part = lambda rows, k: mean(row[k] for row in rows)  # noqa: E731
    return {"soft": part(sample, 1) - weight * part(stated, 1),
            "kl": -part(sample, 2) - weight * part(stated, 2),
            "count": part(sample, 0) - weight * part(stated, 0)}


class TrialScorer:
    """Scores the model as it stands on (sample, stated-answer) questions, all three ways.

    `reference`: id(question) -> the original's probabilities; `before`: the compressed model's, before
    the scan; `hedged_before`: question id -> the compressed model already said "can't tell"."""

    def __init__(self, reference: dict, before: dict, hedged_before: dict, weight: float = 1.0):
        self.reference, self.before, self.hedged_before, self.weight = reference, before, hedged_before, weight

    def rows(self, items, picks, probs, stated, stated_picks, stated_probs) -> tuple[list, list]:
        sample_rows = [(int(pick == it.label), p[it.label], kl(self.reference[id(it)], p))
                       for it, pick, p in zip(items, picks, probs)]
        stated_rows = []
        for it, pick, p in zip(stated, stated_picks, stated_probs):
            ref, before = self.reference[id(it)], self.before[id(it)]
            added = pick == it.unknown_index and not self.hedged_before.get(it.question_id, False)
            stated_rows.append((int(added), max(0.0, before[it.label] - p[it.label]),
                                max(0.0, kl(ref, p) - kl(ref, before))))
        return sample_rows, stated_rows

    def scores(self, items, picks, probs, stated, stated_picks, stated_probs) -> dict[str, float]:
        return combine(*self.rows(items, picks, probs, stated, stated_picks, stated_probs), self.weight)


class WatchedRegistry:
    """The swap registry, remembering which part is being tried, so the scan log can name it."""

    def __init__(self, registry):
        self._registry, self.trying = registry, None

    def __getattr__(self, name):
        return getattr(self._registry, name)

    @contextmanager
    def trial(self, path, nbits):
        with self._registry.trial(path, nbits) as model:
            self.trying = path
            try:
                yield model
            finally:
                self.trying = None


class ScanLog:
    """scan.jsonl: one line per scoring of the scan: the part being tried (null: the model as it stands),
    the parts already kept, how many questions, and the three scores."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.write_text("", encoding="utf-8")

    def write(self, registry, n_sample: int, n_stated: int, scores: dict) -> None:
        record = {"path": getattr(registry, "trying", None), "kept": sorted(registry.active), "sample": n_sample,
                  "stated": n_stated, **{name: round(value, 6) for name, value in scores.items()}}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")


def study(backend, candidates: list[str], n: int, sample: list, stated: list, scorer: TrialScorer, nbits: int,
          seed: int, path: Path, on_progress=None, screen: tuple[int, int] = (60, 40)) -> list[str]:
    """For choosing the trial score (src/product/compare_objectives.py): every question's result for `n`
    parts restored one at a time, and for the model with none (path null), one line each. The first line
    also records how the scan itself scores: how many questions a trial sees (`screen`) and the weight.

    The parts are chosen at random, not by any score, so the comparison favours none of them."""
    import random

    parts = random.Random(seed).sample(candidates, min(n, len(candidates)))

    def measure(name) -> str:
        rows = scorer.rows(sample, *ask(backend, "compressed", sample), stated, *ask(backend, "compressed", stated))
        head = {"screen": list(screen), "weight": scorer.weight} if name is None else {}
        return json.dumps({"path": name, **head, **{key: [[r[0], round(r[1], 4), round(r[2], 4)] for r in value]
                                            for key, value in zip(("sample", "stated"), rows)}}) + "\n"

    with Path(path).open("w", encoding="utf-8") as handle:
        handle.write(measure(None))
        for done, part in enumerate(parts, start=1):
            with backend.registry.trial(part, nbits):
                handle.write(measure(part))
            handle.flush()
            if on_progress:
                on_progress(done, len(parts))
    return parts
