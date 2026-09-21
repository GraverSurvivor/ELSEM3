"""Measures scoring throughput across dtypes and batch sizes, to pick both from data.

Batching the scorer turned out to give only ~1.1x on a Colab T4, which means the GPU was
already saturated and the bottleneck is elsewhere. The suspicion this script exists to test:
the T4 is Turing (compute capability 7.5), and native BF16 tensor cores only arrived with
Ampere (8.0), so `torch.bfloat16` may be running on an unaccelerated path there while
`torch.float16` would use the tensor cores.

Why it matters: Stage 4's greedy search re-runs the flipped set once per candidate module, so
throughput is the difference between the headline experiment finishing overnight and not
finishing. Pick the dtype and batch size from this table, not from assumption.

FP16 is not a free swap — it has a much smaller exponent range than BF16, and this model was
trained in BF16, so overflow to inf/nan is a real (not theoretical) risk. This script reports
how many predictions change per dtype; anything beyond a handful of near-tie flips means FP16
is numerically unsafe here and the speed is irrelevant.

    python -m src.eval.benchmark_throughput
    python -m src.eval.benchmark_throughput --n-items 200 --batch-sizes 1 8 16 32
"""

from __future__ import annotations

import argparse
import time

import torch

from src.data.bbq_loader import load_bbq_ambiguous
from src.eval.model_loader import DTYPES, load_model
from src.eval.option_scorer import score_items


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-items", type=int, default=120)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 16])
    parser.add_argument("--dtypes", nargs="+", default=["bf16", "fp16"], choices=list(DTYPES))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("Needs a GPU.")

    capability = torch.cuda.get_device_capability()
    print(f"GPU: {torch.cuda.get_device_name()}  compute capability {capability[0]}.{capability[1]}")
    print(f"Native BF16 tensor cores require 8.0+ — this card: {'yes' if capability[0] >= 8 else 'NO'}")
    print(f"torch.cuda.is_bf16_supported(): {torch.cuda.is_bf16_supported()}\n")

    items = load_bbq_ambiguous("Age")[: args.n_items]

    results: dict[tuple[str, int], float] = {}
    predictions_by_dtype: dict[str, list[int]] = {}

    for dtype_name in args.dtypes:
        print(f"Loading model in {dtype_name}...")
        model, tokenizer = load_model(dtype_name)

        for batch_size in args.batch_sizes:
            start = time.perf_counter()
            predictions = score_items(model, tokenizer, items, "cuda", batch_size=batch_size)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            results[(dtype_name, batch_size)] = len(items) / elapsed
            predictions_by_dtype.setdefault(dtype_name, predictions)
            print(f"  {dtype_name} batch={batch_size:<3} {elapsed:6.1f}s  {len(items)/elapsed:5.2f} items/s")

        del model
        torch.cuda.empty_cache()

    print(f"\n{'dtype':<8}" + "".join(f"{'batch ' + str(b):>14}" for b in args.batch_sizes))
    for dtype_name in args.dtypes:
        row = "".join(f"{results[(dtype_name, b)]:>11.2f} it/s" for b in args.batch_sizes)
        print(f"{dtype_name:<8}{row}")

    if len(predictions_by_dtype) > 1:
        names = list(predictions_by_dtype)
        baseline = predictions_by_dtype[names[0]]
        for other in names[1:]:
            differing = sum(a != b for a, b in zip(baseline, predictions_by_dtype[other]))
            print(f"\n{names[0]} vs {other}: {differing}/{len(baseline)} predictions differ")
            if differing > len(baseline) * 0.02:
                print("  WARNING: more than a couple of near-tie flips — check for fp16 overflow.")


if __name__ == "__main__":
    main()
