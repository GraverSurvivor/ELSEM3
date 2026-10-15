"""Loads Phi-3.5-mini-Instruct at full precision — the reference all quantized variants compare to.

Dtype is a parameter rather than a constant because it turned out to be a 6.5x throughput
decision on the hardware this project actually runs on. Measured on a Colab T4
(`src/eval/benchmark_throughput.py`): bf16 3.88 items/s vs fp16 23.27 items/s at batch 8. The
T4 is Turing (compute capability 7.5) and native BF16 tensor cores only arrived with Ampere
(8.0), so bf16 runs there without tensor-core acceleration.

Watch out for `torch.cuda.is_bf16_supported()` — it returns True on a T4. It means bf16 will
run, not that it will run fast, which is why this was invisible until throughput was measured
directly.

Changing dtype is a methodology decision, not just a performance knob: the baseline's precision
is part of what the project reports. Validate any change with `src/eval/validate_scorer.py
--dtype <x>` before trusting numbers produced under it.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

MODEL_ID = "microsoft/Phi-3.5-mini-instruct"

# Named dtypes for CLI flags, so scripts can expose --dtype without importing torch types.
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}
DEFAULT_DTYPE = "fp16"


def load_model(
    dtype: str = DEFAULT_DTYPE, device_map: str = "auto", model_id: str = MODEL_ID
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Loads the full-precision baseline model + tokenizer at the named dtype.

    Requires a CUDA device for `device_map="auto"` to place the model on GPU; falls back to
    whatever `accelerate` resolves (CPU) if no GPU is visible, which will be slow but correct.

    Deliberately does NOT pass `trust_remote_code=True`: `transformers` has shipped a native,
    maintained `Phi3ForCausalLM` implementation since ~4.41 (this model's config.json declares
    `model_type: "phi3"`), so omitting the flag makes `from_pretrained` use that instead of the
    Hub's own frozen `modeling_phi3.py`. The Hub copy calls APIs (e.g. `DynamicCache.seen_tokens`)
    that newer `transformers` versions have since removed, causing an AttributeError in
    `generate()` that the native implementation doesn't have, since it's kept in sync with the
    library's own Cache API.
    """
    if dtype not in DTYPES:
        raise ValueError(f"Unknown dtype {dtype!r}; expected one of {list(DTYPES)}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=DTYPES[dtype],
        device_map=device_map,
    )
    model.eval()
    return model, tokenizer
