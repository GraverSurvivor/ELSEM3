"""Verifies the two mechanisms Stage 4's search is built on, against the real model.

The unit tests cover these on CPU with small tensors. This checks the parts that only show up
at full scale and with the real checkpoint:

1. Lazy loading is EXACT. `CheckpointWeightSource` must return bit-identical weights to what
   `snapshot_linear` captures off the loaded model. The whole memory strategy rests on this --
   if the checkpoint read differs even slightly, every promotion injects weights the baseline
   never had, and the search attributes fairness recovery to modules on the strength of a
   difference it introduced itself.
2. Revert is exact at scale. Trial a promotion, revert, and the model's logits must come back
   bit-for-bit. Across 128 candidates x several rounds, any drift here compounds silently.
3. The real memory numbers, for the report: what plain 3-bit costs, what one promotion adds,
   and what holding fp16 snapshots of all 128 candidates would have cost instead.
4. Whether the search is affordable -- one trial's wall-clock, extrapolated to a full round.

Checks 2-4 need a GPU, because HQQ quantizes onto a CUDA device. Check 1 does not, and it
is the load-bearing one -- so on a CPU-only runtime this runs check 1 alone and says so,
rather than refusing outright. Useful when Colab's GPU quota is exhausted.

Run in Colab after `pip install -r requirements.txt`:
    python -m src.quant.verify_repair_state
    python -m src.quant.verify_repair_state --dtype bf16 --promoted-nbits 8
"""

from __future__ import annotations

import argparse
import time

import torch
from hqq.models.hf.base import AutoHQQHFModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.eval.model_loader import DEFAULT_DTYPE, DTYPES, MODEL_ID
from src.quant.candidates import candidate_paths_for_model
from src.quant.hqq_quantizer import DEFAULT_GROUP_SIZE, DEFAULT_NBITS, build_quant_config
from src.quant.module_swap import SwapRegistry, get_module_by_path, module_nbytes, snapshot_linear
from src.quant.weight_source import CheckpointWeightSource

TARGET_PATH = "model.layers.0.self_attn.qkv_proj"
PROMPT = "The capital of France is"
GB = 1024**3
MB = 1024**2


def _logits(model, tokenizer, device: str) -> torch.Tensor:
    inputs = tokenizer(PROMPT, return_tensors="pt").to(device)
    with torch.no_grad():
        return model(**inputs).logits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", default=DEFAULT_DTYPE, choices=list(DTYPES))
    parser.add_argument("--promoted-nbits", type=int, default=8)
    parser.add_argument("--n-commit", type=int, default=4, help="modules to commit for the cost report")
    args = parser.parse_args()

    has_gpu = torch.cuda.is_available()
    device = "cuda" if has_gpu else "cpu"
    torch_dtype = DTYPES[args.dtype]
    if not has_gpu:
        print("No GPU visible -- running check [1/4] only. HQQ quantizes onto a CUDA")
        print("device, so the revert and cost checks need one. Check 1 is the")
        print("load-bearing one: the whole memory strategy rests on it.\n")

    print(f"Loading Phi-3.5-mini-Instruct in {args.dtype} (pre-quantization)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch_dtype)

    # --- 1. lazy load must equal the eager snapshot ------------------------------------------
    print(f"\n[1/4] Checking the checkpoint read matches the loaded model at {TARGET_PATH!r}...")
    eager = snapshot_linear(model, TARGET_PATH)
    source = CheckpointWeightSource(MODEL_ID)
    lazy = source.get_linear(TARGET_PATH, dtype=torch_dtype)

    if not torch.equal(eager.weight.data, lazy.weight.data):
        difference = (eager.weight.data.float() - lazy.weight.data.float()).abs().max().item()
        raise SystemExit(
            f"FAIL: checkpoint read differs from the loaded weights (max abs {difference:.3e}). "
            "Promotions would inject weights the baseline never had, and the search would "
            "attribute recovery to a difference it created itself."
        )
    print(f"      exact match on {eager.weight.numel():,} weights ({eager.weight.dtype})")

    candidates = candidate_paths_for_model(model)
    print(f"      candidate set: {len(candidates)} modules")

    # Exact, not sampled: candidates are grouped by module type, so the first N are all
    # qkv_proj and would badly under-represent gate_up_proj, which is 1.8x larger. Read straight
    # off the unquantized weights -- no cloning, so measuring the cost doesn't incur it.
    snapshot_all_bytes = sum(
        get_module_by_path(model, path).weight.numel() * get_module_by_path(model, path).weight.element_size()
        for path in candidates
    )

    if not has_gpu:
        print(f"\n      snapshotting all {len(candidates)} candidates instead would cost "
              f"{snapshot_all_bytes / GB:.2f} GB at {args.dtype}")
        print(f"\nPARTIAL PASS: the checkpoint read is exact, which is what lets the search")
        print("avoid holding that. Re-run with a GPU for the revert and cost checks.")
        return

    print(f"\nQuantizing to {DEFAULT_NBITS}-bit (group_size={DEFAULT_GROUP_SIZE})...")
    AutoHQQHFModel.quantize_model(
        model,
        quant_config=build_quant_config(nbits=DEFAULT_NBITS, group_size=DEFAULT_GROUP_SIZE),
        compute_dtype=torch_dtype,
        device=device,
    )
    model.eval()

    registry = SwapRegistry(model, source, device=device)

    # --- 2. revert must be exact --------------------------------------------------------------
    print(f"\n[2/4] Trialling {TARGET_PATH!r} at {args.promoted_nbits}-bit, then reverting...")
    before = _logits(model, tokenizer, device)

    started = time.perf_counter()
    with registry.trial(TARGET_PATH, nbits=args.promoted_nbits):
        promoted = _logits(model, tokenizer, device)
    trial_seconds = time.perf_counter() - started

    after = _logits(model, tokenizer, device)

    if not torch.equal(before, after):
        drift = (before - after).abs().max().item()
        raise SystemExit(
            f"FAIL: logits changed after revert (max abs {drift:.3e}) -- the model did not "
            "return to its pre-trial state. Every candidate measured after a trial would be "
            "scored against a drifting model."
        )
    if torch.equal(before, promoted):
        raise SystemExit("FAIL: promotion changed nothing -- the swap is a no-op.")

    print(f"      promotion changed logits (max abs {(promoted - before).abs().max().item():.6f})")
    print("      revert restored them bit-for-bit")
    print(f"      registry state after trial: {registry.active} (a trial must not commit)")

    # --- 3. what it costs ---------------------------------------------------------------------
    print(f"\n[3/4] Committing {args.n_commit} modules and measuring cost...")
    quantized_bytes = module_nbytes(model)
    for path in candidates[: args.n_commit]:
        registry.commit(path, nbits=args.promoted_nbits)

    deployed = registry.deployed_overhead_bytes()
    search = registry.search_overhead_bytes()

    print(f"      plain {DEFAULT_NBITS}-bit model:        {quantized_bytes / GB:6.2f} GB")
    print(f"      + {args.n_commit} modules at {args.promoted_nbits}-bit:      {deployed / MB:6.1f} MB"
          f"  ({100 * deployed / quantized_bytes:+.2f}%)")
    print(f"      retained originals (search):  {search / MB:6.1f} MB  (freed by release_baselines)")
    print(f"      snapshotting all {len(candidates)} instead: {snapshot_all_bytes / GB:6.2f} GB"
          f"  ({snapshot_all_bytes / quantized_bytes:.1f}x the compressed model)")

    # --- 4. is the search affordable ----------------------------------------------------------
    print("\n[4/4] Search cost estimate (swap only -- excludes scoring the flipped set):")
    print(f"      one trial: {trial_seconds:.2f}s")
    print(f"      one full round over {len(candidates)} candidates: {trial_seconds * len(candidates) / 60:.1f} min")

    registry.revert_all()
    print(f"\n      reverted everything; registry state: {registry.active}")
    print("\nPASS: lazy checkpoint loading is exact, revert is exact, costs measured above.")


if __name__ == "__main__":
    main()
