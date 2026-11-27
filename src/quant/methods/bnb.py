"""bitsandbytes NF4 (4-bit) and int8 behind the `Method` interface.

The most widely used compression, and the one whose repaired model loads with the ordinary
`from_pretrained`. A restored part is a plain fp16 `nn.Linear` (restore bits 16): bitsandbytes has no
mixed bit-widths, so a part is either quantized or not.

To rebuild a repaired model the kept parts are listed in `llm_int8_skip_modules`, which transformers
matches as regular expressions and as suffixes (`should_convert_module`). So `skip_modules` passes
escaped, full paths and always adds `lm_head`: an unescaped "model.layers.1.mlp" would also match
layer 11. The set of methods in `src/quant/methods/__init__.py` imports this file lazily, so
bitsandbytes is only needed when one of these methods is used.

bitsandbytes needs a GPU for real use. Everything runs on CPU in recent versions, which the tests use
when it works; the Colab cell in RUNBOOK.md runs them on the T4."""

from __future__ import annotations

import re
from collections.abc import Iterable

import torch
import torch.nn as nn

from src.quant.methods.hqq import dtype_name, torch_dtype


def skip_modules(paths: Iterable[str] = ()) -> list[str]:
    """`llm_int8_skip_modules` that keeps exactly `paths` (and `lm_head`) unquantized: escaped, full
    paths, so layer 1 never also matches layer 11."""
    return [re.escape(path) for path in paths] + ["lm_head"]


def load_tokenizer(model_id: str, revision: str | None = None):
    from transformers import AutoTokenizer

    from src.eval.option_scorer import prepare_tokenizer

    return prepare_tokenizer(AutoTokenizer.from_pretrained(model_id, revision=revision))


class _BnbMethod:
    name: str
    label: str
    bits: tuple[int, ...]
    restore_bits = (16,)
    restores_on_load = True  # the kept parts are listed at load time (llm_int8_skip_modules), not swapped in

    def quantization_config(self, dtype, restored: Iterable[str] = ()):
        raise NotImplementedError

    def load(self, model_id: str, nbits: int, device: str, dtype, streaming: bool = False,
             restored: Iterable[str] = (), revision: str | None = None):
        """(model, tokenizer). `restored` lists the parts kept at 16-bit (a repair's patch paths)."""
        if nbits not in self.bits:
            raise ValueError(f"{self.label} compresses to {list(self.bits)} bits, not {nbits}.")
        if streaming:
            raise ValueError(f"{self.label} can't be built layer by layer: it needs the whole 16-bit model "
                             "loaded first. Use HQQ or RTN for models too big for that.")
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=self.quantization_config(dtype, restored),
            dtype=torch_dtype(dtype), device_map={"": device}, revision=revision)
        model.eval()
        return model, load_tokenizer(model_id, revision)

    def restore(self, linear: nn.Linear, nbits: int, device: str, dtype: torch.dtype) -> nn.Module:
        if nbits not in self.restore_bits:
            raise ValueError(f"{self.label} restores parts only as plain 16-bit, not {nbits}-bit.")
        part = nn.Linear(linear.in_features, linear.out_features, bias=linear.bias is not None,
                         device=device, dtype=dtype)
        part.weight.data = linear.weight.data.to(device=device, dtype=dtype, copy=True)
        if linear.bias is not None:
            part.bias.data = linear.bias.data.to(device=device, dtype=dtype, copy=True)
        part.requires_grad_(False)
        return part

    def nbytes(self, module: nn.Module) -> int:
        from src.quant.module_swap import module_nbytes

        return module_nbytes(module)

    def can_publish(self, model) -> str:
        # Saved with save_pretrained and loaded with the normal from_pretrained. A GPT-2 style model keeps its
        # layers as Conv1D (weights stored transposed), so a restored part (a plain Linear) would not load
        # back into its slot: refused rather than published broken.
        if getattr(model.config, "model_type", "") in ("gpt2", "gpt_bigcode", "imagegpt"):
            return ("This model's layers are GPT-2 style, which bitsandbytes can't save with restored parts, so it "
                    "can't be published as a ready model. Use it now (the API) or download the repair kit instead.")
        return ""


class BnbNF4Method(_BnbMethod):
    name = "bnb_nf4"
    label = "bitsandbytes NF4"
    bits = (4,)

    def quantization_config(self, dtype, restored: Iterable[str] = ()):
        from transformers import BitsAndBytesConfig

        return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                  bnb_4bit_compute_dtype=torch_dtype(dtype),
                                  llm_int8_skip_modules=skip_modules(restored))


class BnbInt8Method(_BnbMethod):
    name = "bnb_int8"
    label = "bitsandbytes int8"
    bits = (8,)

    def quantization_config(self, dtype, restored: Iterable[str] = ()):
        dtype_name(dtype)  # int8 matmuls compute in the model's dtype; refuse unknown names early
        from transformers import BitsAndBytesConfig

        return BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=skip_modules(restored))
