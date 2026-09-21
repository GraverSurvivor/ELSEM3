"""Stage 1: full-precision baseline evaluation on BBQ ambiguous condition, all 5 categories.

Runs the deterministic option-scoring evaluator over every item, writes per-question and
per-category results via Person C's logger (results/<run_id>/), and prints USR/SRS per category
and overall. This is the uncompressed reference run everything else (the safe set, the
flipped-set diff after compression) gets compared against.

`--dtype` is a methodology choice, not just a speed knob — it's recorded as the run's condition
and is what Stage 3 diffs against. See `src/eval/model_loader.py` for why it's parameterized
(bf16 runs ~6.5x slower than fp16 on a Turing-era GPU like Colab's T4).

Results are checkpointed after each category, so an interrupted run can be continued with
`--resume <RUN_ID>` instead of rescoring the categories that already finished.

Run after `pip install -r requirements.txt`:
    python -m src.eval.run_baseline
    python -m src.eval.run_baseline --batch-size 16
    python -m src.eval.run_baseline --resume fp16_20260921T175733Z
"""

from __future__ import annotations

import argparse

import torch

from src.data.bbq_loader import CATEGORIES, load_all_categories
from src.eval.model_loader import DEFAULT_DTYPE, DTYPES, load_model
from src.eval.option_scorer import DEFAULT_BATCH_SIZE, score_items
from src.experiments.logger import (
    RESULTS_DIR,
    new_run_id,
    read_question_results,
    write_category_summary,
    write_question_results,
)
from src.experiments.schema import CategorySummary, QuestionResult, summarize_category


def _aggregate_overall(results: list[QuestionResult], run_id: str, condition: str) -> CategorySummary:
    """Same aggregation as `summarize_category`, but across every category at once — that
    function's category filter would require relabeling every record just to match "overall".
    """
    n_items = len(results)
    n_unknown = sum(r.is_unknown_predicted for r in results)

    stereotype_flags = [r.is_stereotyped_predicted for r in results]
    has_stereotype_data = all(flag is not None for flag in stereotype_flags)
    n_stereotyped = sum(stereotype_flags) if has_stereotype_data else None
    srs = (n_stereotyped / n_items) if has_stereotype_data else None

    return CategorySummary(
        run_id=run_id,
        condition=condition,
        category="overall",
        n_items=n_items,
        n_unknown=n_unknown,
        usr=n_unknown / n_items,
        n_stereotyped=n_stereotyped,
        srs=srs,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--dtype", default=DEFAULT_DTYPE, choices=list(DTYPES))
    parser.add_argument(
        "--resume",
        metavar="RUN_ID",
        help="continue an interrupted run, skipping categories it already finished",
    )
    args = parser.parse_args()

    # The condition label follows the dtype: it's written into every result record and is what
    # Stage 3 diffs against, so a run must never be filed under a precision it wasn't run at.
    condition = args.dtype

    if args.resume:
        run_id = args.resume
        # The dtype is baked into the run id, and mixing precisions inside one run would
        # produce a results file that looks uniform but is not -- the exact confound the
        # condition label exists to make visible. Refuse rather than warn.
        resumed_dtype = run_id.split("_")[0]
        if resumed_dtype != args.dtype:
            raise SystemExit(
                f"Run {run_id!r} was started at {resumed_dtype}, but --dtype is {args.dtype}. "
                f"Resume it with --dtype {resumed_dtype}, or start a fresh run."
            )
        completed = read_question_results(run_id, missing_ok=True)
    else:
        run_id = new_run_id(condition)
        completed = []

    print(f"CUDA available: {torch.cuda.is_available()}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\nLoading Phi-3.5-mini-Instruct in {args.dtype}...")
    model, tokenizer = load_model(args.dtype)

    print("\nLoading BBQ ambiguous condition, all 5 categories...")
    items_by_category = load_all_categories()

    print(f"\nRun ID: {run_id} (dtype {args.dtype}, batch size {args.batch_size})")

    # A category is only ever written once it is fully scored, so anything already in the
    # file is complete -- which is what makes "skip it" safe rather than a source of
    # half-scored categories.
    finished_categories = {result.category for result in completed}
    if finished_categories:
        print(f"Resuming: {sorted(finished_categories)} already scored, skipping.")

    all_results: list[QuestionResult] = list(completed)
    for category, items in items_by_category.items():
        if category in finished_categories:
            continue
        predictions = score_items(model, tokenizer, items, device, batch_size=args.batch_size, desc=category)
        all_results.extend(
            QuestionResult(
                run_id=run_id,
                condition=condition,
                question_id=item.question_id,
                category=item.category,
                predicted_index=predicted_index,
                label=item.label,
                unknown_index=item.unknown_index,
                is_unknown_predicted=(predicted_index == item.unknown_index),
                is_correct=(predicted_index == item.label),
                is_stereotyped_predicted=(predicted_index == item.stereotype_index),
            )
            for item, predicted_index in zip(items, predictions)
        )
        # Checkpoint after every category. A full run is ~8 minutes and sessions do get
        # interrupted; without this, a stop partway through the fourth category throws
        # away the three that finished.
        write_question_results(all_results, run_id)

    summaries = [summarize_category(all_results, run_id, condition, category) for category in CATEGORIES]
    overall = _aggregate_overall(all_results, run_id, condition)
    write_category_summary(summaries + [overall], run_id)

    print(f"\n{'Category':<20} {'N':>6} {'USR':>8} {'SRS':>8}")
    for summary in summaries:
        srs_display = f"{summary.srs:.3f}" if summary.srs is not None else "n/a"
        print(f"{summary.category:<20} {summary.n_items:>6} {summary.usr:>8.3f} {srs_display:>8}")
    overall_srs_display = f"{overall.srs:.3f}" if overall.srs is not None else "n/a"
    print(f"{'OVERALL':<20} {overall.n_items:>6} {overall.usr:>8.3f} {overall_srs_display:>8}")
    print(f"\nResults written to {(RESULTS_DIR / run_id).resolve()}")
    print("On Kaggle, Save Version to keep them past this session.")


if __name__ == "__main__":
    main()
