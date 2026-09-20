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
from datetime import datetime, timezone
from pathlib import Path

from src.experiments.schema import CategorySummary, QuestionResult

RESULTS_DIR = Path("results")


def new_run_id(condition: str) -> str:
    """Timestamp-based run ID, e.g. "bf16_20260921T143000Z" -- sortable, unique per run."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{condition}_{timestamp}"


def _run_dir(run_id: str) -> Path:
    run_dir = RESULTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_question_results(results: list[QuestionResult], run_id: str) -> Path:
    """Writes per-question results as JSONL (one JSON object per line) -- easy to append to
    incrementally during a long eval run, unlike a single JSON array.
    """
    path = _run_dir(run_id) / "question_results.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result.to_dict()) + "\n")
    return path


def read_question_results(run_id: str) -> list[QuestionResult]:
    path = _run_dir(run_id) / "question_results.jsonl"
    with path.open("r", encoding="utf-8") as f:
        return [QuestionResult(**json.loads(line)) for line in f if line.strip()]


def write_category_summary(summaries: list[CategorySummary], run_id: str) -> Path:
    """Writes per-category summaries as CSV -- this is what the plotting/reporting code reads."""
    path = _run_dir(run_id) / "category_summary.csv"
    fieldnames = list(summaries[0].to_dict().keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            writer.writerow(summary.to_dict())
    return path


def read_category_summary(run_id: str) -> list[CategorySummary]:
    path = _run_dir(run_id) / "category_summary.csv"
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
