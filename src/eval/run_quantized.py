"""Stage 2: evaluate the HQQ 3-bit model on BBQ ambiguous condition, all 5 categories.

The compressed counterpart to `run_baseline.py`: same items, same deterministic option-scoring,
same checkpoint-per-category and `--resume`, written through the same logger. The only
differences are the model (`load_hqq_3bit_model()` instead of `load_model()`) and the condition
label, `hqq_3bit`. Stage 3's flipped set is the diff between this run and the fp16 baseline.

The compute dtype is deliberately NOT a flag. `hqq_3bit` means "3-bit weights, fp16 compute",
matching the fp16 baseline; running it at another dtype would file a different experiment under
the same label, and ~1.7% of ambiguous answers flip from precision alone -- flips nothing
downstream could tell apart from quantization damage. See CLAUDE.md > Precision.

Run after `pip install -r requirements.txt`, on a GPU:
    python -m src.eval.run_quantized
    python -m src.eval.run_quantized --batch-size 16
    python -m src.eval.run_quantized --resume hqq_3bit_20260924T120000Z
"""

from __future__ import annotations

import argparse

import torch

from src.data.bbq_loader import CATEGORIES, load_all_categories
from src.eval.option_scorer import DEFAULT_BATCH_SIZE, score_items
from src.eval.run_baseline import _aggregate_overall
from src.experiments.logger import (
    RESULTS_DIR,
    new_run_id,
    read_question_results,
    write_category_summary,
    write_question_results,
)
from src.experiments.schema import QuestionResult, summarize_category

CONDITION = "hqq_3bit"
# Must match the baseline's dtype. src/stats/metrics.py maps "hqq_3bit" -> fp16 on the same
# assumption, so changing this without changing that mapping would slip past the guard.
COMPUTE_DTYPE = "fp16"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--resume",
        metavar="RUN_ID",
        help="continue an interrupted run, skipping categories it already finished",
    )
    args = parser.parse_args()

    if args.resume:
        run_id = args.resume
        # run_baseline reads the dtype off the run id's first "_" field; that doesn't work here
        # ("hqq_3bit_..." splits to "hqq"), so check the whole condition prefix instead.
        if not run_id.startswith(f"{CONDITION}_"):
            raise SystemExit(
                f"Run {run_id!r} is not a {CONDITION} run. Resume only runs this script started."
            )
        completed = read_question_results(run_id, missing_ok=True)
    else:
        run_id = new_run_id(CONDITION)
        completed = []

    if not torch.cuda.is_available():
        raise SystemExit("No GPU found. HQQ 3-bit evaluation needs CUDA -- on Kaggle, set Accelerator to GPU.")
    device = "cuda"

    # Imported here so `--help` and the tests work on a machine without hqq installed.
    from src.quant.hqq_quantizer import load_hqq_3bit_model

    print(f"\nLoading Phi-3.5-mini-Instruct, HQQ 3-bit, {COMPUTE_DTYPE} compute (quantizes on load)...")
    model, tokenizer = load_hqq_3bit_model(device=device, dtype=COMPUTE_DTYPE)

    print("\nLoading BBQ ambiguous condition, all 5 categories...")
    items_by_category = load_all_categories()

    print(f"\nRun ID: {run_id} (condition {CONDITION}, batch size {args.batch_size})")

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
                condition=CONDITION,
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
        # Checkpoint after every category, as in run_baseline.
        write_question_results(all_results, run_id)

    summaries = [summarize_category(all_results, run_id, CONDITION, category) for category in CATEGORIES]
    overall = _aggregate_overall(all_results, run_id, CONDITION)
    write_category_summary(summaries + [overall], run_id)

    print(f"\n{'Category':<20} {'N':>6} {'USR':>8} {'SRS':>8}")
    for summary in summaries + [overall]:
        name = "OVERALL" if summary.category == "overall" else summary.category
        srs_display = f"{summary.srs:.3f}" if summary.srs is not None else "n/a"
        print(f"{name:<20} {summary.n_items:>6} {summary.usr:>8.3f} {srs_display:>8}")
    print(f"\nResults written to {(RESULTS_DIR / run_id).resolve()}")
    print(f"Next: make promote-reference RUN_ID={run_id} NAME={CONDITION}")
    print("On Kaggle, Save Version to keep them past this session.")


if __name__ == "__main__":
    main()
