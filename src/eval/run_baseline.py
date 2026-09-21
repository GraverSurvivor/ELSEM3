"""Stage 1: BF16 baseline evaluation on BBQ ambiguous condition, all 5 categories.

Runs the deterministic option-scoring evaluator over every item, writes per-question and
per-category results via Person C's logger (results/<run_id>/), and prints USR/SRS per category
and overall. This is the BF16 reference run everything else (the safe set, the flipped-set
diff after compression) gets compared against.

Run in Colab after `pip install -r requirements.txt`:
    python -m src.eval.run_baseline
"""

from __future__ import annotations

import torch
from tqdm import tqdm

from src.data.bbq_loader import CATEGORIES, load_all_categories
from src.eval.model_loader import load_bf16_model
from src.eval.option_scorer import score_item
from src.experiments.logger import new_run_id, write_category_summary, write_question_results
from src.experiments.schema import CategorySummary, QuestionResult, summarize_category

CONDITION = "bf16"


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
    print(f"CUDA available: {torch.cuda.is_available()}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\nLoading BF16 Phi-3.5-mini-Instruct...")
    model, tokenizer = load_bf16_model()

    print("\nLoading BBQ ambiguous condition, all 5 categories...")
    items_by_category = load_all_categories()

    run_id = new_run_id(CONDITION)
    print(f"\nRun ID: {run_id}")

    all_results: list[QuestionResult] = []
    for category, items in items_by_category.items():
        for item in tqdm(items, desc=category):
            predicted_index = score_item(model, tokenizer, item, device)
            all_results.append(
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
            )

    write_question_results(all_results, run_id)

    summaries = [summarize_category(all_results, run_id, CONDITION, category) for category in CATEGORIES]
    overall = _aggregate_overall(all_results, run_id, CONDITION)
    write_category_summary(summaries + [overall], run_id)

    print(f"\n{'Category':<20} {'N':>6} {'USR':>8} {'SRS':>8}")
    for summary in summaries:
        srs_display = f"{summary.srs:.3f}" if summary.srs is not None else "n/a"
        print(f"{summary.category:<20} {summary.n_items:>6} {summary.usr:>8.3f} {srs_display:>8}")
    overall_srs_display = f"{overall.srs:.3f}" if overall.srs is not None else "n/a"
    print(f"{'OVERALL':<20} {overall.n_items:>6} {overall.usr:>8.3f} {overall_srs_display:>8}")
    print(f"\nResults written to results/{run_id}/")


if __name__ == "__main__":
    main()
