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
    python -m src.eval.run_quantized --context disambig    # control set
    python -m src.eval.run_quantized --nbits 4             # 4-bit comparison
"""

from __future__ import annotations

import argparse

import torch

from src.data.bbq_loader import CATEGORIES, CONTEXTS, load_all_categories
from src.eval.option_scorer import DEFAULT_BATCH_SIZE
from src.eval.scoring_run import check_resume_id, condition_for, score_categories
from src.eval.run_baseline import _aggregate_overall
from src.experiments.logger import (
    RESULTS_DIR,
    new_run_id,
    read_question_results,
    write_category_summary,
)
from src.experiments.schema import summarize_category

COMPUTE_DTYPE = "fp16"
# The compressed run's condition name carries its bit-width: hqq_3bit is the main experiment,
# hqq_4bit the "beyond one setting" check. Compute stays fp16 for both, matching the baseline;
# src/stats/metrics.py maps each of these names to fp16 on that assumption.
SUPPORTED_NBITS = (3, 4)


def base_condition(nbits: int) -> str:
    return f"hqq_{nbits}bit"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--nbits", type=int, default=3, choices=SUPPORTED_NBITS, help="HQQ weight bit-width")
    parser.add_argument(
        "--context",
        default="ambig",
        choices=CONTEXTS,
        help="ambig = the fairness benchmark; disambig = the control set where the text states the answer",
    )
    parser.add_argument(
        "--resume",
        metavar="RUN_ID",
        help="continue an interrupted run, skipping categories it already finished",
    )
    args = parser.parse_args()

    condition = condition_for(base_condition(args.nbits), args.context)

    if args.resume:
        run_id = args.resume
        check_resume_id(run_id, condition)
        completed = read_question_results(run_id, missing_ok=True)
    else:
        run_id = new_run_id(condition)
        completed = []

    if not torch.cuda.is_available():
        raise SystemExit("No GPU found. HQQ evaluation needs CUDA -- on Kaggle, set Accelerator to GPU.")
    device = "cuda"

    # Imported here so `--help` and the tests work on a machine without hqq installed.
    from src.quant.hqq_quantizer import load_hqq_3bit_model

    print(f"\nLoading Phi-3.5-mini-Instruct, HQQ {args.nbits}-bit, {COMPUTE_DTYPE} compute (quantizes on load)...")
    model, tokenizer = load_hqq_3bit_model(device=device, nbits=args.nbits, dtype=COMPUTE_DTYPE)

    print(f"\nLoading BBQ {args.context} condition, all 5 categories...")
    items_by_category = load_all_categories(args.context)

    print(f"\nRun ID: {run_id} (condition {condition}, batch size {args.batch_size})")
    all_results = score_categories(
        model, tokenizer, items_by_category, device, run_id, condition, args.batch_size, completed
    )

    summaries = [summarize_category(all_results, run_id, condition, category) for category in CATEGORIES]
    overall = _aggregate_overall(all_results, run_id, condition)
    write_category_summary(summaries + [overall], run_id)

    print(f"\n{'Category':<20} {'N':>6} {'USR':>8} {'SRS':>8} {'Acc':>8}")
    for summary in summaries + [overall]:
        name = "OVERALL" if summary.category == "overall" else summary.category
        srs_display = f"{summary.srs:.3f}" if summary.srs is not None else "n/a"
        rows = all_results if summary.category == "overall" else [r for r in all_results if r.category == summary.category]
        accuracy = sum(r.is_correct for r in rows) / len(rows)
        print(f"{name:<20} {summary.n_items:>6} {summary.usr:>8.3f} {srs_display:>8} {accuracy:>8.3f}")
    print(f"\nResults written to {(RESULTS_DIR / run_id).resolve()}")
    print(f"Next: make promote-reference RUN_ID={run_id} NAME={condition}")
    print("On Kaggle, Save Version to keep them past this session.")


if __name__ == "__main__":
    main()
