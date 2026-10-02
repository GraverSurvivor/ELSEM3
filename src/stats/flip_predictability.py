"""Can the original model's own confidence predict which answers compression will flip?

If yes, a deployment check doesn't need to probe every question: it can probe the ones the
original model was least sure about first, and need no human labels at all -- a flip is simply the
compressed model disagreeing with the original. Hua et al. (COLM 2026) report flips are 3-11x
likelier on uncertain items across 50 quantized models; this tests whether that holds here.

Inputs: an fp16 run that recorded confidence (any run made after the scorer started recording it)
and the committed hqq_3bit run. The fp16 run's predictions are checked against the committed
baseline first -- same model, same scorer, so they must match; if they don't, the confidence
belongs to a different model and the analysis refuses.

Outputs (results/stats/flip_predictability.json):
    auroc                  how well low confidence ranks flipped items above unflipped (0.5 = chance)
    flip_rate_by_quartile  flip rate in each confidence quartile, and the low-vs-high ratio
    coverage               probing the least-confident X% of safe items catches Y% of all flips

    python -m src.stats.flip_predictability --fp16 results/reference/baseline_fp16_confidence
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.safe_set import read_results_file

from .metrics import align_by_question, is_stereotype

REFERENCE_DIR = Path("results/reference")
COVERAGE_POINTS = (0.1, 0.2, 0.3, 0.5, 0.75, 1.0)


def auroc(scores: list[float], positives: list[bool]) -> float:
    """Area under the ROC curve via the Mann-Whitney U statistic, with ties counted as half."""
    pos = [s for s, p in zip(scores, positives) if p]
    neg = [s for s, p in zip(scores, positives) if not p]
    if not pos or not neg:
        raise ValueError("need both flipped and unflipped items")
    ranked = sorted((s, i) for i, s in enumerate(scores))
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(ranked):
        j = i
        while j + 1 < len(ranked) and ranked[j + 1][0] == ranked[i][0]:
            j += 1
        for k in range(i, j + 1):
            ranks[ranked[k][1]] = (i + j) / 2 + 1  # average rank for ties
        i = j + 1
    rank_sum = sum(r for r, p in zip(ranks, positives) if p)
    return (rank_sum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def coverage_curve(confidence: list[float], flipped: list[bool], points=COVERAGE_POINTS) -> list[dict]:
    """Probe the least-confident items first: what share of all flips does each budget catch?"""
    order = sorted(range(len(confidence)), key=lambda i: confidence[i])
    total = sum(flipped)
    curve = []
    for fraction in points:
        probed = order[: max(1, round(fraction * len(order)))]
        caught = sum(flipped[i] for i in probed)
        curve.append({"probed": fraction, "flips_caught": caught / total if total else 0.0})
    return curve


def quartile_rates(confidence: list[float], flipped: list[bool]) -> dict:
    order = sorted(range(len(confidence)), key=lambda i: confidence[i])
    size = len(order) // 4
    quartiles = [order[q * size: (q + 1) * size if q < 3 else len(order)] for q in range(4)]
    rates = [sum(flipped[i] for i in idx) / len(idx) for idx in quartiles]
    return {
        "least_to_most_confident": rates,
        "low_vs_high_ratio": rates[0] / rates[3] if rates[3] > 0 else float("inf"),
    }


def analyze(fp16_with_confidence: list, committed_fp16: list, quantized: list) -> dict:
    a, b = align_by_question(committed_fp16, fp16_with_confidence)
    mismatched = sum(x.predicted_index != y.predicted_index for x, y in zip(a, b))
    if mismatched:
        raise ValueError(
            f"{mismatched} predictions differ between the confidence run and the committed baseline -- "
            "they are not the same model/scorer, so this confidence can't be used"
        )
    if any(r.confidence is None for r in fp16_with_confidence):
        raise ValueError("the fp16 run has no confidence values -- re-run it with the current scorer")

    base, quant = align_by_question(fp16_with_confidence, quantized)
    safe = [(b, q) for b, q in zip(base, quant) if b.is_unknown_predicted]
    confidence = [b.confidence for b, _ in safe]
    flipped = [is_stereotype(q) for _, q in safe]
    return {
        "safe_items": len(safe),
        "flipped": sum(flipped),
        "auroc": auroc([1 - c for c in confidence], flipped),
        "flip_rate_by_quartile": quartile_rates(confidence, flipped),
        "coverage": coverage_curve(confidence, flipped),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fp16", type=Path, required=True, help="results dir of an fp16 run that recorded confidence")
    parser.add_argument("--out", type=Path, default=Path("results/stats/flip_predictability.json"))
    args = parser.parse_args()

    result = analyze(
        read_results_file(args.fp16 / "question_results.jsonl"),
        read_results_file(REFERENCE_DIR / "baseline_fp16" / "question_results.jsonl"),
        read_results_file(REFERENCE_DIR / "hqq_3bit" / "question_results.jsonl"),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    q = result["flip_rate_by_quartile"]
    print(f"{result['flipped']} of {result['safe_items']} safe items flipped")
    print(f"AUROC of low confidence predicting a flip: {result['auroc']:.3f}  (0.5 = chance)")
    print(f"Flip rate, least to most confident quartile: {', '.join(f'{r:.1%}' for r in q['least_to_most_confident'])}")
    print(f"Least-confident quartile flips {q['low_vs_high_ratio']:.1f}x as often as the most confident")
    for point in result["coverage"]:
        print(f"  probing the least-confident {point['probed']:.0%} catches {point['flips_caught']:.0%} of flips")
    print(f"Written to {args.out}")


if __name__ == "__main__":
    main()
