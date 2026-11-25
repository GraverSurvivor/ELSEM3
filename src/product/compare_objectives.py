"""Which trial score should version 2's scan use? Measured on a real model, from a run with `--study`.

    python -m src.product.fast_repair --model Qwen/Qwen2.5-3B-Instruct --pipeline v2 --study 60 --out DIR
    python -m src.product.compare_objectives DIR        # CPU, seconds -> DIR/objectives.json

The run measured random parts, one at a time, on every search question, and saved every question's
result. Here each score sees only a small random set of questions, as it does in the scan, and ranks the
parts; the truth is the part's real effect on answers (version 1's count) on the questions that set does
NOT contain. Repeated over many random sets, this gives for each score:

    rank correlation   does it order the parts as the truth does?
    regret             how much true gain is lost by restoring its favourite instead of the best part?
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from src.product.decisions import OBJECTIVES, combine, mean

SIZES = ((36, 24), (60, 40), (120, 80))  # (sample, stated-answer) questions a trial sees; the scan's is the middle


def load(path: Path) -> list[dict]:
    records = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(records) < 6 or records[0]["path"] is not None:
        raise ValueError(f"{path} isn't a finished study: it needs the model as it stands and at least 5 parts")
    return records


def gains(records: list[dict], sample: list[int], stated: list[int]) -> dict[str, list[float]]:
    """Each score's gain for every part, on these questions."""
    def scores(record):
        return combine([record["sample"][i] for i in sample], [record["stated"][i] for i in stated])

    start = scores(records[0])
    table = [scores(r) for r in records[1:]]
    return {name: [row[name] - start[name] for row in table] for name in OBJECTIVES}


def rank_correlation(a: list[float], b: list[float]) -> float:
    from scipy.stats import spearmanr

    value = spearmanr(a, b)[0]
    return 0.0 if value != value else float(value)  # a constant ranking says nothing


def compare(records: list[dict], sizes=SIZES, repeats: int = 200, seed: int = 0) -> dict:
    rng = random.Random(seed)
    n_sample, n_stated = len(records[0]["sample"]), len(records[0]["stated"])
    out = {"parts": len(records) - 1, "sample": n_sample, "stated": n_stated, "repeats": repeats, "sizes": {}}
    for want_sample, want_stated in sizes:
        k, m = min(want_sample, n_sample // 2), min(want_stated, n_stated // 2)
        found = {name: {"rank_correlation": [], "regret": []} for name in OBJECTIVES}
        for _ in range(repeats):
            seen, seen_stated = set(rng.sample(range(n_sample), k)), set(rng.sample(range(n_stated), m))
            truth = gains(records, [i for i in range(n_sample) if i not in seen],
                          [i for i in range(n_stated) if i not in seen_stated])["count"]
            for name, value in gains(records, sorted(seen), sorted(seen_stated)).items():
                found[name]["rank_correlation"].append(rank_correlation(value, truth))
                favourite = max(range(len(value)), key=value.__getitem__)
                found[name]["regret"].append(max(truth) - truth[favourite])
        out["sizes"][f"{k}+{m}"] = {name: {key: round(mean(values), 4) for key, values in result.items()}
                                    for name, result in found.items()}
    middle = out["sizes"][list(out["sizes"])[len(out["sizes"]) // 2]]
    out["best"] = max(OBJECTIVES, key=lambda name: middle[name]["rank_correlation"])
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path, help="a run's folder with study.jsonl")
    parser.add_argument("--repeats", type=int, default=200)
    args = parser.parse_args(argv)
    result = compare(load(args.folder / "study.jsonl"), repeats=args.repeats)
    (args.folder / "objectives.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"{result['parts']} random parts, {result['repeats']} random question sets of each size")
    print(f"{'questions':>10}  {'score':6}  {'rank correlation':>16}  {'regret (points)':>15}")
    for size, table in result["sizes"].items():
        for name in OBJECTIVES:
            print(f"{size:>10}  {name:6}  {table[name]['rank_correlation']:16.3f}  {100 * table[name]['regret']:15.2f}")
    print(f"Best at the scan's own size: {result['best']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
