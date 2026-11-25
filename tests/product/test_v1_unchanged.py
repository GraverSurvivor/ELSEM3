"""Version 1 of the pipeline is the comparison for version 2, so its decisions must not move: every
stand-in model's kept parts, outcome, verdict and numbers are pinned to what `e434817` (PR #60) gave.

    python -m tests.product.test_v1_unchanged    # rewrites the golden file; only for a deliberate change
"""

import json
from pathlib import Path

import pytest

from src.product.fast_repair import FastRepairConfig, run
from tests.product import test_fast_repair as t

GOLDEN = Path(__file__).with_name("golden_v1.json")
STAND_INS = {
    "fixable": (t.FakeBackend, {}),
    "fixable_exact": (t.FakeBackend, {"mode": "exact"}),
    "no_damage": (lambda: t.FakeBackend(damage=False), {}),
    "never_unsure": (t.NeverUnsure, {}),
    "hedging_compressed": (t.HedgingCompressed, {}),
    "breaks_fairness": (t.BreaksFairness, {}),
    "breaks_still_fair": (t.BreaksStillFair, {}),
    "broken_at_3_bits": (t.BrokenAtThreeBits, {}),
    "diffuse": (t.DiffuseDamage, {}),
    "diffuse_no_budget": (t.DiffuseDamage, {"max_restore": 0}),
    "diffuse_scan_only": (t.DiffuseDamage, {"until_fair": False}),
    "diffuse_with_hedger": (t.DiffuseWithHedger, {"max_restore": 0}),
}
KEYS = ("outcome", "gate", "gate_failures", "kept", "recovery", "original", "compressed", "repaired", "base_nbits",
        "repair_verdict", "restored_share", "checked_on", "safe", "flipped", "scan")


def rounded(value):
    """JSON form with floats rounded, so the last digit can't differ between machines."""
    if isinstance(value, float):
        return round(value, 9)
    if isinstance(value, dict):
        return {key: rounded(v) for key, v in value.items()}
    return [rounded(v) for v in value] if isinstance(value, (list, tuple)) else value


def decisions(name: str, out: Path) -> dict:
    make, settings = STAND_INS[name]
    config = FastRepairConfig(model_id="Qwen/Qwen2.5-3B-Instruct", questions=400, out=out, **settings)
    report = run(config, make(), load_questions=lambda: (t.AMBIG, t.DISAMBIG))
    return rounded({key: report[key] for key in KEYS})


@pytest.mark.parametrize("name", sorted(STAND_INS))
def test_v1_decides_exactly_what_it_did(name, tmp_path):
    assert decisions(name, tmp_path / "job") == json.loads(GOLDEN.read_text(encoding="utf-8"))[name]


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        golden = {name: decisions(name, Path(tmp) / name) for name in sorted(STAND_INS)}
    GOLDEN.write_text(json.dumps(golden, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    for name, d in golden.items():
        print(f"{name:20s} {d['outcome']:15s} {d['gate']:5s} {len(d['kept'])} parts")
