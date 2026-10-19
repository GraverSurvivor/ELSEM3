"""Build a quantized model one layer at a time, straight from its checkpoint, for models too big to
load whole first.

`load_hqq_3bit_model` loads the full 16-bit model on the CPU and then quantizes it: simple, and what
the research runs use, but it needs 2 bytes a weight of system memory (16 GB for an 8B model, more than
free Colab's 12.7 GB). Here the model is created on the "meta" device (shapes, no memory), and each
linear layer is read from the checkpoint (`CheckpointWeightSource`, one tensor at a time) and
HQQ-quantized straight onto the GPU; every other weight (embeddings, norms, the output head) is read
and placed in 16-bit. System memory holds one layer at a time.

The result is the same model `load_hqq_3bit_model` builds -- the same tensors, cast the same way, and
the same HQQ settings -- just assembled in a different order (tested). At 8 bits it is the reference
the product uses when the 16-bit original doesn't fit the GPU.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.quant.hqq_quantizer import DEFAULT_GROUP_SIZE, build_quant_config

LINEAR_TYPES = ("Linear", "Conv1D")


def load_streaming(model_id: str, nbits: int, device: str = "cuda", dtype: torch.dtype = torch.float16,
                   group_size: int = DEFAULT_GROUP_SIZE, keep: dict | None = None):
    """(model, tokenizer): every linear layer but the output head HQQ-quantized to `nbits` (or to
    `keep[path]` for a repair's parts), built layer by layer from the checkpoint."""
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from hqq.core.quantize import HQQLinear
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    from src.eval.option_scorer import prepare_tokenizer
    from src.models.adapter import check_generative
    from src.quant.weight_source import CheckpointWeightSource

    config = AutoConfig.from_pretrained(model_id)
    check_generative(config)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, dtype=dtype)
    model.eval()

    linears = [name for name, module in model.named_modules()
               if type(module).__name__ in LINEAR_TYPES and name and not name.endswith("lm_head")]
    conv1d = {name for name in linears if type(model.get_submodule(name)).__name__ == "Conv1D"}
    source = CheckpointWeightSource(model_id, transposed=lambda path: path in conv1d)
    configs = {bits: build_quant_config(nbits=bits, group_size=group_size) for bits in {nbits, *(keep or {}).values()}}
    for name in linears:
        linear = source.get_linear(name, dtype=dtype)
        parent, _, child = name.rpartition(".")
        bits = (keep or {}).get(name, nbits)
        setattr(model.get_submodule(parent) if parent else model, child,
                HQQLinear(linear, configs[bits], compute_dtype=dtype, device=device))

    tied = bool(getattr(config, "tie_word_embeddings", False))
    for name, param in list(model.named_parameters()):
        if param.device.type != "meta":
            continue  # already real: the quantized layers
        if tied and name.endswith("lm_head.weight") and not source.has(name):
            continue  # shares the input embeddings: tied below
        set_module_tensor_to_device(model, name, device, value=source.get_tensor(name).to(dtype))
    if tied:
        model.tie_weights()
    for name, buffer in list(model.named_buffers()):
        if buffer.device.type == "meta":
            raise RuntimeError(f"{name} has no value to load")  # not expected: buffers are made real at init
        module_name, _, buffer_name = name.rpartition(".")
        owner = model.get_submodule(module_name) if module_name else model
        # Floating buffers (rotary inv_freq) in the compute dtype, exactly as HQQ's own quantize_model
        # casts every non-linear module: the model is then bit-identical to load_hqq_3bit_model's.
        owner._buffers[buffer_name] = buffer.to(device, dtype=dtype) if buffer.is_floating_point() else buffer.to(device)
    leftover = [name for name, param in model.named_parameters() if param.device.type == "meta"]
    if leftover:
        raise RuntimeError(f"weights missing from the checkpoint: {leftover[:5]}")
    return model, prepare_tokenizer(AutoTokenizer.from_pretrained(model_id))
