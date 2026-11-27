"""Compression methods behind one interface (`base.Method`). `get_method("hqq")` is the way in."""

from __future__ import annotations

from src.quant.methods.base import Method

DEFAULT_METHOD = "hqq"


def _hqq(**kwargs) -> Method:
    from src.quant.methods.hqq import HQQMethod

    return HQQMethod(**kwargs)


# name -> factory. Each method registers here; imports stay lazy so a method's library (bitsandbytes)
# is only needed when that method is used.
FACTORIES = {"hqq": _hqq}


def method_names() -> list[str]:
    return list(FACTORIES)


def get_method(name: str | Method = DEFAULT_METHOD, **kwargs) -> Method:
    """The method called `name` (an instance passes through). `kwargs` go to its constructor."""
    if not isinstance(name, str):
        return name
    if name not in FACTORIES:
        raise ValueError(f"Unknown compression method {name!r}; known methods: {method_names()}")
    return FACTORIES[name](**kwargs)
