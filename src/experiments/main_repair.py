"""Which repair is the project's main result: one pointer file every tool reads.

`results/reference/MAIN_REPAIR.json` names the main repair's patch and its full-benchmark runs on
both question sets, plus the previous main repair for comparison. The final tables, the FairGate
report, the Studio app, the groups and sensitivity analyses, the figures and the deck all read it,
so they can't drift apart, and changing the main repair is a one-line edit.

Since Phase 2 (October 2026) the main repair is the hedging-aware one: the first search's patch
(search_top5) recovers 92% of the lost "unknown" answers but says "unknown" on 20.3% of
stated-answer questions (the original: 7.3%); the search re-run with a hedging penalty recovers 86%
with 10.9%. search_top5 stays as `previous`, and as the patch the same-size baselines were built for.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

POINTER = Path("results/reference/MAIN_REPAIR.json")
REFERENCE_DIR = Path("results/reference")


@dataclass(frozen=True)
class Repair:
    name: str
    label: str
    patch: Path
    ambig: str      # folder under results/reference/ with the ambiguous-question run
    disambig: str   # ... and the stated-answer run

    def ambig_dir(self, reference_dir: Path = REFERENCE_DIR) -> Path:
        return Path(reference_dir) / self.ambig

    def disambig_dir(self, reference_dir: Path = REFERENCE_DIR) -> Path:
        return Path(reference_dir) / self.disambig


def _repair(entry: dict) -> Repair:
    return Repair(name=entry["name"], label=entry["label"], patch=Path(entry["patch"]),
                  ambig=entry["ambig"], disambig=entry["disambig"])


# Used when the pointer file is missing (an old checkout): the first repair, as before.
_FALLBACK = {"name": "search_top5", "label": "3-bit + search patch", "patch": "results/patches/search_top5.json",
             "ambig": "hqq_repaired", "disambig": "hqq_repaired_disambig"}


def main_repair(pointer: Path = POINTER) -> Repair:
    data = json.loads(Path(pointer).read_text(encoding="utf-8")) if Path(pointer).exists() else _FALLBACK
    return _repair(data)


def previous_repair(pointer: Path = POINTER) -> Repair | None:
    if not Path(pointer).exists():
        return None
    previous = json.loads(Path(pointer).read_text(encoding="utf-8")).get("previous")
    return _repair(previous) if previous else None
