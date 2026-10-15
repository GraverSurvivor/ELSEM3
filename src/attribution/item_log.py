"""Per-item search log (contract C1): `results/<run_id>/items.jsonl`.

`trials.csv` keeps one score per trial; this keeps *which questions* each trial got right, so Fast
Scan can be replayed offline on subsets of the sample (`replay.py`). One JSON line per trial:

    {"round": int, "path": str | null, "sample": "0110...", "probe": "0010..."}

`path` is null (round 0) for the model before any trial. `sample` has one character per sample
question in the sample's `question_ids` order, 1 = answered the question's label. `probe` has one
character per probe question, 1 = said "unknown" although the answer is stated; "" with no probe.

Appends are keyed by (round, path), so a resumed search that re-measures a trial whose line was
already written does not duplicate it, and a line cut off by a dead session is dropped on reopening.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

ITEMS_FILE = "items.jsonl"
BASELINE_ROUND = 0


def sample_bits(predictions: list[int], items: list) -> str:
    """One character per item: 1 if the prediction is the item's label."""
    return "".join("1" if p == item.label else "0" for p, item in zip(predictions, items))


def probe_bits(predictions: list[int], items: list) -> str:
    """One character per item: 1 if the model said "unknown" although the answer is stated."""
    return "".join("1" if item.label != item.unknown_index and p == item.unknown_index else "0"
                   for p, item in zip(predictions, items))


def read_items(path: Path) -> list[dict]:
    """All complete lines of an items.jsonl; a truncated final line is ignored."""
    records = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


class ItemLog:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        records = read_items(self.path) if self.path.exists() else []
        if self.path.exists():  # rewrite without any partial line, so appending never glues onto it
            self.path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
        self._seen = {(r["round"], r["path"]) for r in records}

    def __contains__(self, key: tuple[int, str | None]) -> bool:
        return key in self._seen

    def append(self, round: int, path: str | None, sample: str, probe: str) -> bool:
        """Writes one line and returns True, or returns False if this (round, path) is already logged."""
        if (round, path) in self._seen:
            return False
        record = {"round": round, "path": path, "sample": sample, "probe": probe}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._seen.add((round, path))
        return True


class SearchItemLogger:
    """Wires an ItemLog into a greedy search without changing its interfaces.

    `wrap(score_fn)` returns a scorer that remembers each call's per-question bits; the first call
    of a fresh search is logged as the baseline. `on_trial(result)` is called from the search's
    on_progress: when the trial count has grown it logs the newest trial with the bits of the call
    that just produced it. A keep leaves the count unchanged, so the re-measure after it is not logged.
    """

    def __init__(self, log: ItemLog, resumed_trials: int = 0, fresh: bool = True) -> None:
        self.log = log
        self.logged_trials = resumed_trials  # trials already in the checkpoint were logged by their session
        self._baseline_pending = fresh
        self.bits = {"sample": "", "probe": ""}

    def wrap(self, score_fn):
        def scored() -> float:
            value = score_fn(self.bits)
            if self._baseline_pending:
                self.log.append(BASELINE_ROUND, None, self.bits["sample"], self.bits["probe"])
                self._baseline_pending = False
            return value

        return scored

    def on_trial(self, result) -> None:
        if len(result.trials) <= self.logged_trials:
            return
        latest = result.trials[-1]
        self.log.append(latest.round, latest.path, self.bits["sample"], self.bits["probe"])
        self.logged_trials = len(result.trials)
