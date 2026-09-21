"""Promotes a completed run to `results/reference/`, the one place results are committed.

Why an exception to "results/ is gitignored"
--------------------------------------------
The convention exists so nobody's ad-hoc runs end up in the repo, and it's right for almost
everything. The baseline is the exception, for three reasons:

- Every later stage is defined *relative* to it. The safe set is "what the baseline got right",
  the flipped set is "what changed against the baseline". Losing it doesn't cost one run, it
  invalidates the comparison every other run is built on.
- All three of us need the identical file. Two teammates regenerating "the baseline"
  separately is how you end up diffing against two slightly different references.
- It unblocks CPU-only work. Flipped-set logic, statistics and plots all just read this file.
  With it in the repo, that work doesn't need a GPU session at all -- which matters when free
  GPU quota is the scarcest resource on the project.

It has already been lost three times (an interrupt, a dead Colab session, a dead Kaggle
session) at ~8 GPU-minutes and a working GPU each time. It's 3 MB of text.

Promotion is a deliberate step, not automatic, and it validates first: committing a truncated
or duplicated run as "the reference" would silently corrupt everything downstream, and it would
look like real data rather than a broken file.

    python -m src.experiments.promote_reference fp16_20260921T180125Z --name baseline_fp16
    python -m src.experiments.promote_reference fp16_20260921T180125Z --name baseline_fp16 --check-only
"""

from __future__ import annotations

import argparse
import shutil
from collections import Counter
from pathlib import Path

from src.data.bbq_loader import CATEGORIES
from src.experiments.logger import RESULTS_DIR, read_category_summary, read_question_results

REFERENCE_DIR_NAME = "reference"


def reference_dir() -> Path:
    return RESULTS_DIR / REFERENCE_DIR_NAME


def validate_run(run_id: str) -> list[str]:
    """Returns the reasons `run_id` isn't fit to be a reference; empty means it is.

    Checks self-consistency rather than comparing against a hardcoded item count, so this
    doesn't need the dataset (or a network) to run.
    """
    problems: list[str] = []

    try:
        results = read_question_results(run_id)
    except FileNotFoundError:
        return [f"no question_results.jsonl for run {run_id!r}"]
    if not results:
        return [f"run {run_id!r} has no question results"]

    present = {result.category for result in results}
    missing = [category for category in CATEGORIES if category not in present]
    if missing:
        problems.append(f"incomplete: missing categories {missing}")

    # A resumed run that appended instead of replacing would double its rows, inflating every
    # count derived from the file while still looking like a valid run.
    duplicates = [qid for qid, count in Counter(r.question_id for r in results).items() if count > 1]
    if duplicates:
        problems.append(f"{len(duplicates)} duplicate question_id(s), e.g. {duplicates[:3]}")

    conditions = {result.condition for result in results}
    if len(conditions) > 1:
        problems.append(f"mixed conditions in one run: {sorted(conditions)}")
    elif not run_id.startswith(f"{next(iter(conditions))}_"):
        problems.append(f"condition {next(iter(conditions))!r} doesn't match run id {run_id!r}")

    try:
        summaries = read_category_summary(run_id)
    except FileNotFoundError:
        problems.append("no category_summary.csv")
        return problems

    counts = Counter(result.category for result in results)
    for summary in summaries:
        if summary.category == "overall":
            if summary.n_items != len(results):
                problems.append(f"overall n_items {summary.n_items} != {len(results)} records")
        elif summary.n_items != counts[summary.category]:
            problems.append(
                f"{summary.category}: summary says {summary.n_items}, found {counts[summary.category]}"
            )

    return problems


def promote(run_id: str, name: str) -> Path:
    """Copies a validated run into `results/reference/<name>/`. Refuses an invalid run."""
    problems = validate_run(run_id)
    if problems:
        raise ValueError(
            f"Run {run_id!r} is not fit to be a reference:\n  - " + "\n  - ".join(problems)
        )

    source = RESULTS_DIR / run_id
    destination = reference_dir() / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id", help="the run to promote, e.g. fp16_20260921T180125Z")
    parser.add_argument("--name", required=True, help="name under results/reference/, e.g. baseline_fp16")
    parser.add_argument("--check-only", action="store_true", help="validate without copying")
    args = parser.parse_args()

    problems = validate_run(args.run_id)
    if problems:
        raise SystemExit(f"FAIL: {args.run_id} is not fit to be a reference:\n  - " + "\n  - ".join(problems))

    results = read_question_results(args.run_id)
    counts = Counter(result.category for result in results)
    print(f"{args.run_id}: {len(results)} records across {len(counts)} categories")
    for category in CATEGORIES:
        print(f"  {category:<20} {counts[category]:>6}")

    if args.check_only:
        print("\nPASS (--check-only, nothing copied).")
        return

    destination = promote(args.run_id, args.name)
    print(f"\nPromoted to {destination}")
    print("This path is tracked by git -- commit and push it so the whole team shares one reference.")


if __name__ == "__main__":
    main()
