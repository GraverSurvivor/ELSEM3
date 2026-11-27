"""The one interface every compression method sits behind (contract C8).

Until now the project compressed with HQQ only, and HQQ was assumed in many places. A `Method` is
everything the rest of the code needs to know about a compression method: how to build the compressed
model, how to make one restored (higher-precision) part, how big a part really is, and whether a
repaired model can be published. Code that repairs, saves or reports a model asks the method; it does
not look for `HQQLinear`.

A method is a small object with no state about any one model, so one instance serves every model.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch
import torch.nn as nn


@runtime_checkable
class Method(Protocol):
    name: str                      # short id used in patches, job folders and the command line ("hqq")
    label: str                     # what a person reads on screen ("HQQ")
    bits: tuple[int, ...]          # bit-widths the whole model can be compressed to
    restore_bits: tuple[int, ...]  # bit-widths a restored part can have (16 = plain fp16)

    def load(self, model_id: str, nbits: int, device: str, dtype, streaming: bool = False):
        """(model, tokenizer): `model_id` compressed to `nbits`. `dtype` is the compute dtype (a name
        such as "fp16" or a torch dtype). `streaming` builds it layer by layer from the checkpoint,
        for models too big to load whole first."""
        ...

    def restore(self, linear: nn.Linear, nbits: int, device: str, dtype: torch.dtype) -> nn.Module:
        """A restored part: the original `linear` at `nbits` (16 means the plain fp16 layer)."""
        ...

    def nbytes(self, module: nn.Module) -> int:
        """Real bytes of storage the module holds, measured, including the method's extra state."""
        ...

    def can_publish(self, model) -> str:
        """"" if the compressed or repaired model can be published, otherwise why not, in plain words."""
        ...
