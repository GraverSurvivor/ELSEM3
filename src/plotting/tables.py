"""Results tables for the deck, written as CSV (for re-use) and Markdown (for pasting).

Every number comes from a structure the stats code already produced -- `build_comparison` from
`src.stats.repair_report` -- so nothing is ever hand-copied into a slide (CLAUDE.md > Conventions).
Formatting lives only here, so the CSV and the Markdown can't disagree.
"""

from __future__ import annotations

import csv
from pathlib import Path

from src.data.bbq_loader import CATEGORIES


def _pct(x: float) -> str:
    return f"{x:.1%}"


def _ci_pct(d: dict) -> str:
    """A recovery fraction with its bootstrap CI, e.g. "91% [88%, 94%]"."""
    return f"{d['point']:.0%} [{d['lower']:.0%}, {d['upper']:.0%}]"


def _ci_delta(d: dict) -> str:
    """A paired rate difference with its CI, in percentage points, e.g. "-3.6 pp [-4.1, -3.1]"."""
    return f"{100 * d['delta']:+.1f} pp [{100 * d['lower']:+.1f}, {100 * d['upper']:+.1f}]"


def _p(p: float) -> str:
    return "<0.001" if p < 1e-3 else f"{p:.3f}"


def comparison_rows(comparison: dict) -> list[dict]:
    """One row per category (in `CATEGORIES` order) then Overall, from a `build_comparison` dict."""
    order = [c for c in CATEGORIES if c in comparison] + (["overall"] if "overall" in comparison else [])
    rows = []
    for name in order:
        block = comparison[name]
        rows.append({
            "Category": "Overall" if name == "overall" else name,
            "N": block["n_items"],
            "USR baseline": _pct(block["baseline"]["usr"]),
            "USR 3-bit": _pct(block["hqq_3bit"]["usr"]),
            "USR repaired": _pct(block["repaired"]["usr"]),
            "USR recovered": _ci_pct(block["recovery"]["usr"]),
            "SRS baseline": _pct(block["baseline"]["srs"]),
            "SRS 3-bit": _pct(block["hqq_3bit"]["srs"]),
            "SRS repaired": _pct(block["repaired"]["srs"]),
            "SRS recovered": _ci_pct(block["recovery"]["srs"]),
            "USR gap left": _ci_delta(block["gap_left"]["usr"]),
            "Repair p (chi2)": _p(block["repair_effect"]["chisq_p"]),
        })
    return rows


def to_markdown(rows: list[dict]) -> str:
    if not rows:
        raise ValueError("no rows")
    headers = list(rows[0])
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(row[h]) for h in headers) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def write_table(rows: list[dict], out_stem: Path) -> tuple[Path, Path]:
    """Writes `<out_stem>.csv` and `<out_stem>.md`; returns both paths."""
    if not rows:
        raise ValueError("no rows")
    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    csv_path, md_path = out_stem.with_suffix(".csv"), out_stem.with_suffix(".md")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    md_path.write_text(to_markdown(rows), encoding="utf-8")
    return csv_path, md_path
