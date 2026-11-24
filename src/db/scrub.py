"""Nothing secret reaches a row or the screen: tokens, API keys and database addresses are cut out."""

from __future__ import annotations

import re

REMOVED = "[removed]"
_SECRETS = re.compile(
    r"postgres(?:ql)?(?:\+\w+)?://\S*"  # a database address holds its password
    r"|(?:hf_|ghp_|gho_|ghs_|github_pat_)[A-Za-z0-9_]*"
    r"|(?<![A-Za-z0-9])fg-[A-Za-z0-9_-]*"  # the Studio's own API keys
)


def scrub(value):
    """`value` with every secret cut out, inside strings, lists and dicts (keys too)."""
    if isinstance(value, str):
        # NUL can't be stored by Postgres, and a pipeline's output can hold one.
        return _SECRETS.sub(REMOVED, value).replace("\x00", "")
    if isinstance(value, dict):
        return {scrub(k): scrub(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [scrub(v) for v in value]
    return value


def safe_error(error: BaseException) -> str:
    """An error in words fit for the console. Driver errors can quote the whole address, so only the
    kind of error and its scrubbed first line are kept."""
    root = getattr(error, "orig", None) or error
    lines = str(root).strip().splitlines()
    return f"{type(root).__name__}: {scrub(lines[0])[:200]}" if lines else type(root).__name__
