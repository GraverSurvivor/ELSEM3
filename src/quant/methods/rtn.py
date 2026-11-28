"""RTN: plain round-to-nearest, the simplest compression and the reference for the cleverer ones.

It is HQQ's container (`HQQLinear`, packing, group scale and zero, restoring parts at 4 or 8 bits)
with HQQ's optimiser switched off: each group of 64 weights is rounded onto the grid between its own
min and max. Tested to equal hand-written min-max rounding. Restored parts are RTN too, so the only
difference from HQQ is how the numbers were rounded, never the format."""

from __future__ import annotations

from src.quant.methods.hqq import HQQMethod


class RTNMethod(HQQMethod):
    name = "rtn"
    label = "RTN"
    bits = (3, 4)
    restore_bits = (4, 8)
    optimize = False
