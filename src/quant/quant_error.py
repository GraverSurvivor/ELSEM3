"""How much each candidate module's weights changed when quantized -- for the "largest error" baseline.

One of the fairness controls asks whether the search is just finding "whatever compression damaged
most". This measures that directly and without any data: for every candidate, quantize its
original weights exactly as the 3-bit model does (HQQ, same bit-width and group size), dequantize,
and record how far the result is from the original. The baseline patch then restores the modules
with the largest error, at the same parameter budget as the search's patch.

Relative error (||W - Q(W)|| / ||W||) is the ranking key, so large and small modules compete on how
badly they were damaged rather than on how many weights they have; the absolute error is kept too.

Weights are read one module at a time from the checkpoint (CheckpointWeightSource), so this never
holds the full model. A GPU makes it a few minutes; on CPU it works but is slow.

    python -m src.quant.quant_error                         # -> results/quant_error/hqq_3bit_g64.json
    python -m src.quant.quant_error --device cpu --limit 4  # quick smoke test
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn

from src.quant.candidates import candidate_paths
from src.quant.hqq_quantizer import DEFAULT_GROUP_SIZE, DEFAULT_NBITS, build_quant_config

DEFAULT_OUT = Path("results/quant_error/hqq_3bit_g64.json")


def quantization_error(weight: torch.Tensor, nbits: int, group_size: int, device: str) -> dict:
    """Quantizes `weight` with HQQ, dequantizes it, and returns the relative and absolute error."""
    from hqq.core.quantize import HQQLinear

    original = weight.detach().float().cpu()
    linear = nn.Linear(original.shape[1], original.shape[0], bias=False)
    linear.weight.data = original.clone()
    quantized = HQQLinear(
        linear,
        quant_config=build_quant_config(nbits=nbits, group_size=group_size),
        compute_dtype=torch.float32 if device == "cpu" else torch.float16,
        device=device,
    )
    restored = quantized.dequantize().float().cpu()
    if restored.shape != original.shape:
        raise RuntimeError(f"dequantized shape {tuple(restored.shape)} != original {tuple(original.shape)}")
    abs_error = (original - restored).norm().item()
    return {"rel_error": abs_error / original.norm().item(), "abs_error": abs_error}


def rank_by_error(entries: list[dict]) -> list[dict]:
    """Largest relative error first; ties broken by path so the order is reproducible."""
    return sorted(entries, key=lambda e: (-e["rel_error"], e["path"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nbits", type=int, default=DEFAULT_NBITS)
    parser.add_argument("--group-size", type=int, default=DEFAULT_GROUP_SIZE)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, help="only the first N candidates (smoke test)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    from src.quant.weight_source import CheckpointWeightSource

    source = CheckpointWeightSource()
    paths = candidate_paths()[: args.limit] if args.limit else candidate_paths()
    entries = []
    started = time.time()
    for i, path in enumerate(paths, start=1):
        weight = source.get_linear(path, dtype=torch.float16).weight.data
        entries.append({"path": path, **quantization_error(weight, args.nbits, args.group_size, args.device)})
        print(f"\r{i}/{len(paths)} {path:40} rel {entries[-1]['rel_error']:.4f}  ({time.time() - started:.0f}s)", end="")
    print()

    payload = {
        "nbits": args.nbits,
        "group_size": args.group_size,
        "n": len(entries),
        "ranked": rank_by_error(entries),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("Top 5 by relative error:")
    for entry in payload["ranked"][:5]:
        print(f"  {entry['path']:40} {entry['rel_error']:.4f}")
    print(f"Written to {args.out}")


if __name__ == "__main__":
    main()
