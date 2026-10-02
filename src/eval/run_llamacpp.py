"""llama.cpp transfer test: does the repair found with HQQ still help in llama.cpp's own 3-bit format?

FairGate exports a patch as llama.cpp `--tensor-type` overrides, but the patch was found and verified
with HQQ, and llama.cpp quantizes differently (Q3_K_M mixes 3- to 5-bit k-quants per tensor). This
scores BBQ with three llama.cpp models, all built from the same 16-bit GGUF:

    f16             the original weights                         condition llamacpp_f16
    q3km            plain Q3_K_M                                 condition llamacpp_q3km
    q3km_fairgate   Q3_K_M with the patch's modules kept at q8_0 condition llamacpp_q3km_fairgate

The three are compared with each other only -- recovery is measured from llamacpp_q3km towards
llamacpp_f16 -- because llama.cpp computes differently from the transformers runs, and comparing
across engines would book engine differences as quantization damage. src/stats/metrics.py files
these conditions under their own compute type, "llamacpp", so the comparison guard refuses mixing.

The prompt and the A/B/C token ids come from the transformers tokenizer (the GGUF carries the same
vocabulary), and the token ids are handed to llama.cpp directly, so every question is scored on
exactly the tokens the HQQ runs scored. Building the GGUF files is in RUNBOOK.md.

    python -m src.eval.run_llamacpp --gguf /content/phi35-q3km.gguf --variant q3km
    python -m src.eval.run_llamacpp --gguf ... --variant q3km --resume llamacpp_q3km_20261027T120000Z
"""

from __future__ import annotations

import argparse
import math

from src.data.bbq_loader import CATEGORIES, CONTEXTS, BBQItem, load_all_categories
from src.eval.option_scorer import _resolve_label_token_ids, format_prompt
from src.eval.scoring_run import build_results, check_resume_id, condition_for
from src.experiments.logger import (
    RESULTS_DIR,
    new_run_id,
    read_question_results,
    write_category_summary,
    write_question_results,
)
from src.experiments.schema import summarize_category

VARIANTS = ("f16", "q3km", "q3km_fairgate")


def condition_name(variant: str) -> str:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {VARIANTS}")
    return f"llamacpp_{variant}"


def prompt_token_ids(items: list[BBQItem], tokenizer) -> tuple[list[list[int]], list[int]]:
    """Each item's prompt as token ids, plus the A/B/C label token ids (resolved as option_scorer does)."""
    prompts = [format_prompt(item, tokenizer) for item in items]
    label_ids = _resolve_label_token_ids(prompts, tokenizer)
    return [tokenizer.encode(p, add_special_tokens=True) for p in prompts], label_ids


def score_with_llama(llm, token_lists: list[list[int]], label_ids: list[int]) -> tuple[list[int], list[float]]:
    """Next-token logits after each prompt, restricted to the three label tokens: the same argmax and
    the same softmax-over-three confidence as option_scorer, computed by llama.cpp."""
    predictions, confidences = [], []
    for ids in token_lists:
        llm.reset()
        llm.eval(ids)
        logits = [float(llm.scores[llm.n_tokens - 1][i]) for i in label_ids]
        top = max(logits)
        exps = [math.exp(x - top) for x in logits]
        best = max(range(len(logits)), key=lambda i: logits[i])
        predictions.append(best)
        confidences.append(exps[best] / sum(exps))
    return predictions, confidences


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gguf", required=True, help="path to the .gguf model file")
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--context", default="ambig", choices=CONTEXTS)
    parser.add_argument("--n-gpu-layers", type=int, default=-1, help="-1 = all layers on the GPU")
    parser.add_argument("--resume", metavar="RUN_ID")
    args = parser.parse_args()

    from llama_cpp import Llama
    from transformers import AutoTokenizer

    from src.eval.model_loader import MODEL_ID

    condition = condition_for(condition_name(args.variant), args.context)
    if args.resume:
        run_id = args.resume
        check_resume_id(run_id, condition)
        completed = read_question_results(run_id, missing_ok=True)
    else:
        run_id, completed = new_run_id(condition), []

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    llm = Llama(model_path=args.gguf, n_ctx=2048, n_batch=512, n_gpu_layers=args.n_gpu_layers,
                logits_all=False, verbose=False)
    items_by_category = load_all_categories(args.context)

    print(f"Run ID: {run_id} (condition {condition}, model {args.gguf})")
    results = list(completed)
    finished = {r.category for r in results}
    for category, items in items_by_category.items():
        if category in finished:
            continue
        print(f"Scoring {category} ({len(items)} questions)...")
        token_lists, label_ids = prompt_token_ids(items, tokenizer)
        predictions, confidences = score_with_llama(llm, token_lists, label_ids)
        results.extend(build_results(items, predictions, confidences, run_id, condition))
        write_question_results(results, run_id)

    summaries = [summarize_category(results, run_id, condition, c) for c in CATEGORIES]
    from src.eval.run_baseline import _aggregate_overall

    overall = _aggregate_overall(results, run_id, condition)
    write_category_summary(summaries + [overall], run_id)
    print(f"USR {overall.usr:.3f}  SRS {overall.srs:.3f}  ({overall.n_items} questions)")
    print(f"Results: {(RESULTS_DIR / run_id).resolve()}")
    print(f"Next: make promote-reference RUN_ID={run_id} NAME={condition}")


if __name__ == "__main__":
    main()
