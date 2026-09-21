"""Writes QuestionResult/CategorySummary records to results/, so no script ever hand-copies
numbers into the deck (per CLAUDE.md convention).

Plain JSON/CSV rather than Weights & Biases: no account/API key setup needed for a 3-person
8-week project, and every downstream consumer (plotting, the headline experiment, the deck) just
needs to read files back off disk. `results/` is gitignored, so nothing here gets committed --
each teammate's runs stay local unless explicitly shared.
"""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from src.experiments.schema import CategorySummary, QuestionResult

RESULTS_DIR = Path("results")


def new_run_id(condition: str) -> str:
    """Timestamp-based run ID, e.g. "bf16_20260921T143000Z" -- sortable, unique per run."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{condition}_{timestamp}"


def _run_dir(run_id: str, create: bool = True) -> Path:
    run_dir = RESULTS_DIR / run_id
    if create:
        run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _write_atomically(path: Path, write) -> Path:
    """Writes via a temp file and renames, so an interrupt cannot leave a half-written file.

    These files are checkpoints for long runs, so they get rewritten repeatedly while a run
    is in progress -- which makes an interrupt landing mid-write a real possibility rather
    than a theoretical one. A truncated JSONL line would make the run unresumable and would
    look like corrupt data rather than an interrupted write. os.replace is atomic on both
    POSIX and Windows, so what is on disk is always either the previous complete version or
    the new one.
    """
    temp_path = path.with_name(path.name + ".tmp")
    with temp_path.open("w", encoding="utf-8", newline="") as handle:
        write(handle)
    os.replace(temp_path, path)
    return path


def write_question_results(results: list[QuestionResult], run_id: str) -> Path:
    """Writes per-question results as JSONL (one JSON object per line) -- easy to append to
    incrementally during a long eval run, unlike a single JSON array.
    """
    path = _run_dir(run_id) / "question_results.jsonl"

    def write(handle) -> None:
        for result in results:
            handle.write(json.dumps(result.to_dict()) + "\n")

    return _write_atomically(path, write)


def read_question_results(run_id: str, missing_ok: bool = False) -> list[QuestionResult]:
    """Reads back a run's per-question records.

    `missing_ok` returns [] instead of raising when the run has nothing written yet, which
    is what resuming needs: "no such run" and "interrupted before the first checkpoint"
    should both just mean "start from the beginning".
    """
    path = _run_dir(run_id, create=False) / "question_results.jsonl"
    if missing_ok and not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [QuestionResult(**json.loads(line)) for line in f if line.strip()]


def write_category_summary(summaries: list[CategorySummary], run_id: str) -> Path:
    """Writes per-category summaries as CSV -- this is what the plotting/reporting code reads."""
    path = _run_dir(run_id) / "category_summary.csv"
    fieldnames = list(summaries[0].to_dict().keys())

    def write(handle) -> None:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            writer.writerow(summary.to_dict())

    return _write_atomically(path, write)


def read_category_summary(run_id: str) -> list[CategorySummary]:
    path = _run_dir(run_id, create=False) / "category_summary.csv"
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    summaries = []
    for row in rows:
        row["n_items"] = int(row["n_items"])
        row["n_unknown"] = int(row["n_unknown"])
        row["usr"] = float(row["usr"])
        row["n_stereotyped"] = int(row["n_stereotyped"]) if row["n_stereotyped"] else None
        row["srs"] = float(row["srs"]) if row["srs"] else None
        summaries.append(CategorySummary(**row))
    return summaries
