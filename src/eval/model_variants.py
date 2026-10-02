"""Load the three models every comparison uses: original fp16, plain 3-bit, and 3-bit with a patch.

The CrowS-Pairs and open-ended evaluations score the same three conditions as the main benchmark.
Loading them in one place keeps the rules identical everywhere: fp16 compute for all three (never
compare across dtypes), and a patch applied with the same SwapRegistry commit run_repair uses.

Each loader returns (model, tokenizer, condition), where condition is the label written into the
results -- fp16, hqq_3bit, or hqq_repaired.
"""

from __future__ import annotations

from pathlib import Path

VARIANTS = ("fp16", "hqq_3bit", "patch")
COMPUTE_DTYPE = "fp16"


def load_variant(variant: str, device: str, patch_path: Path | None = None):
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {VARIANTS}")
    if variant == "patch" and patch_path is None:
        raise ValueError("variant 'patch' needs a patch file")

    if variant == "fp16":
        from src.eval.model_loader import load_model

        model, tokenizer = load_model(COMPUTE_DTYPE)
        return model, tokenizer, "fp16"

    from src.quant.hqq_quantizer import load_hqq_3bit_model

    model, tokenizer = load_hqq_3bit_model(device=device, dtype=COMPUTE_DTYPE)
    if variant == "hqq_3bit":
        return model, tokenizer, "hqq_3bit"

    from src.attribution.patch import load_patch
    from src.eval.model_loader import MODEL_ID
    from src.quant.module_swap import SwapRegistry
    from src.quant.weight_source import CheckpointWeightSource

    patch = load_patch(patch_path)
    registry = SwapRegistry(model, CheckpointWeightSource(MODEL_ID), device=device)
    for path in patch.modules:
        registry.commit(path, patch.nbits)
    registry.release_baselines()
    print(f"Applied patch {patch.name}: {len(patch.modules)} modules at {patch.nbits}-bit")
    return model, tokenizer, "hqq_repaired"
