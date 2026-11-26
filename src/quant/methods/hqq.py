"""HQQ behind the `Method` interface: wraps what `hqq_quantizer`, `streaming` and `module_swap`
already do, and changes none of it. HQQ 3-bit must stay byte-identical (tested)."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.eval.model_loader import DTYPES
from src.quant.hqq_quantizer import DEFAULT_GROUP_SIZE


def dtype_name(dtype) -> str:
    """The `DTYPES` key ("fp16") for a name or a torch dtype."""
    if isinstance(dtype, str):
        if dtype not in DTYPES:
            raise ValueError(f"Unknown dtype {dtype!r}; expected one of {list(DTYPES)}")
        return dtype
    for name, value in DTYPES.items():
        if value == dtype:
            return name
    raise ValueError(f"Unsupported dtype {dtype!r}; expected one of {list(DTYPES)}")


def torch_dtype(dtype) -> torch.dtype:
    return DTYPES[dtype_name(dtype)]


class HQQMethod:
    name = "hqq"
    label = "HQQ"
    bits = (2, 3, 4)
    restore_bits = (4, 8)  # HQQ stores 5- and 6-bit as 8-bit, so those are not offered

    def __init__(self, group_size: int = DEFAULT_GROUP_SIZE) -> None:
        self.group_size = group_size

    def load(self, model_id: str, nbits: int, device: str, dtype, streaming: bool = False):
        if nbits not in self.bits:
            raise ValueError(f"HQQ compresses to {list(self.bits)} bits, not {nbits}.")
        if streaming:
            from src.quant.streaming import load_streaming

            return load_streaming(model_id, nbits, device=device, dtype=torch_dtype(dtype),
                                  group_size=self.group_size)
        from src.quant.hqq_quantizer import load_hqq_3bit_model

        return load_hqq_3bit_model(device=device, nbits=nbits, group_size=self.group_size,
                                   dtype=dtype_name(dtype), model_id=model_id)

    def restore(self, linear: nn.Linear, nbits: int, device: str, dtype: torch.dtype) -> nn.Module:
        if nbits not in self.restore_bits:
            raise ValueError(f"HQQ restores parts at {list(self.restore_bits)} bits, not {nbits}.")
        from hqq.core.quantize import HQQLinear

        from src.quant.hqq_quantizer import build_quant_config

        quant_config = build_quant_config(nbits=nbits, group_size=self.group_size)
        return HQQLinear(linear, quant_config=quant_config, compute_dtype=dtype, device=device)

    def nbytes(self, module: nn.Module) -> int:
        from src.quant.module_swap import module_nbytes

        return module_nbytes(module)

    def can_publish(self, model) -> str:
        from src.product.hub_publish import PublishError, check_publishable

        try:
            check_publishable(model)
        except PublishError as error:
            return str(error)
        return ""
