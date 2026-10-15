"""Replay Fast Scan on the logs of a finished full search (CPU only, no model).

(a) `trials`: from trials.csv, "rounds 2..N restricted to round 1's top K" -- does the kept set still
    match the full search, for K = kmin..kmax? Every kept module's rank in round 1 is reported too:
    the smallest K that works is the largest of those ranks.

(b) `items`: from items.jsonl (run_search --log-items), Monte-Carlo the whole Fast Scan over many
    seeds, drawing the screening subset exactly as `fast_scan` does, and report how often it picks the
    same modules in the same order.

Replay can only use what the full search measured. Round r's trials were measured on the model with
the full search's first r-1 kept modules, so a replayed run is followed only while it agrees with the
full search; at the first round where it would pick something else it is counted as a disagreement
and stops there (what it would have done next was never measured). Two things follow: "same
modules, different order" can't be told apart from "different modules" and is reported as a
disagreement at that round, and the stop-on-threshold rule is not replayed -- exactly as many rounds
run as the full search kept.

    python -m src.attribution.replay trials results/reference/search_hedging_penalty/trials.csv
    python -m src.attribution.replay items results/<search run>/ --mode fast --seeds 1000
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

from src.attribution.fast_scan import FastScanConfig, seeded_subset
from src.attribution.item_log import read_items


# ---------------------------------------------------------------- (a) trials.csv

def read_trial_rows(path: Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return [{"round": int(r["round"]), "path": r["path"], "usr": float(r["usr"]), "gain": float(r["gain"]),
                 "kept": r["kept"].strip().lower() == "true"} for r in csv.DictReader(handle)]


def kept_in_order(rows: list[dict]) -> list[str]:
    return [r["path"] for r in sorted((r for r in rows if r["kept"]), key=lambda r: r["round"])]


def round_one_ranking(rows: list[dict]) -> list[str]:
    """Round 1's candidates, best first by the objective; ties to the earlier candidate (file order)."""
    first = [r for r in rows if r["round"] == 1]
    return [r["path"] for r in sorted(first, key=lambda r: -r["usr"])]  # sorted() is stable


def replay_pool(rows: list[dict], k: int) -> dict:
    """Rounds 2.. restricted to round 1's top k. Matches iff each round's winner is still in the pool."""
    actual = kept_in_order(rows)
    ranking = round_one_ranking(rows)
    pool = set(ranking[:k])
    order = {r["path"]: i for i, r in enumerate(x for x in rows if x["round"] == 1)}
    kept: list[str] = []
    diverged_at = None
    for round_number in range(1, len(actual) + 1):
        in_round = [r for r in rows if r["round"] == round_number and r["path"] not in kept
                    and (round_number == 1 or r["path"] in pool)]
        best = max(in_round, key=lambda r: (r["usr"], -order[r["path"]])) if in_round else None  # K too small
        if best is None or best["path"] != actual[round_number - 1]:
            diverged_at = round_number
            break
        kept.append(best["path"])
    return {"k": k, "matches": diverged_at is None, "diverged_at_round": diverged_at, "kept": kept}


def replay_trials(rows: list[dict], kmin: int = 5, kmax: int = 32) -> dict:
    actual = kept_in_order(rows)
    ranking = round_one_ranking(rows)
    ranks = {path: ranking.index(path) + 1 for path in actual}
    runs = [replay_pool(rows, k) for k in range(kmin, kmax + 1)]
    needed = max(ranks.values()) if ranks else 0
    return {"kept": actual, "round1_rank_of_kept": ranks, "smallest_k_that_matches": needed,
            "n_round1_candidates": len(ranking), "runs": runs,
            "all_match_from_k": next((r["k"] for r in runs if all(x["matches"] for x in runs if x["k"] >= r["k"])), None)}


# ---------------------------------------------------------------- (b) items.jsonl

class ItemData:
    """items.jsonl indexed by (round, path), with the objective computable on any subset of questions."""

    def __init__(self, records: list[dict], penalty_weight: float = 1.0) -> None:
        self.by_key = {(r["round"], r["path"]): r for r in records}
        self.weight = penalty_weight
        first = next(iter(self.by_key.values()))
        self.n_sample, self.n_probe = len(first["sample"]), len(first["probe"])
        self.candidates = [r["path"] for r in records if r["round"] == 1]
        if not self.candidates or (0, None) not in self.by_key:
            raise ValueError("items.jsonl needs the baseline line and a full round 1")

    def objective(self, key: tuple, idx_s: list[int], idx_p: list[int]) -> float:
        record = self.by_key[key]
        value = sum(record["sample"][i] == "1" for i in idx_s) / len(idx_s)
        if self.n_probe and idx_p:
            value -= self.weight * sum(record["probe"][i] == "1" for i in idx_p) / len(idx_p)
        return value


def simulate_fast_scan(data: ItemData, kept: list[str], config: FastScanConfig, seed: int) -> dict:
    """One Fast Scan run replayed on the logs. Returns the modules picked and where it left the full search."""
    rng = random.Random(seed)
    all_s, all_p = list(range(data.n_sample)), list(range(data.n_probe))
    screen_s = seeded_subset(all_s, config.screen_sample, rng)
    screen_p = seeded_subset(all_p, config.screen_probe, rng)
    exact = config.mode == "exact"
    work = (all_s, all_p) if exact else (screen_s, screen_p)
    order = {p: i for i, p in enumerate(data.candidates)}

    def best_of(paths, key_round, subset):
        return max(paths, key=lambda p: (data.objective((key_round, p), *subset), -order[p]))

    def top(paths, key_round, subset, n):
        ranked = sorted(paths, key=lambda p: (-data.objective((key_round, p), *subset), order[p]))[:n]
        return sorted(ranked, key=order.__getitem__)

    survivors = top(data.candidates, 1, (screen_s, screen_p), config.survivors) if exact else list(data.candidates)
    pool = top(survivors, 1, work, config.pool)
    picked = []
    for round_number in range(1, len(kept) + 1):
        scope = survivors if round_number == 1 else [p for p in pool if p not in picked]
        left = [p for p in scope if p not in picked]
        winner = best_of(left, round_number, work) if left else None  # an exhausted pool has nothing to pick
        if winner != kept[round_number - 1]:
            return {"picked": picked, "same": False, "diverged_at_round": round_number, "chose": winner,
                    "wanted_in_pool": kept[round_number - 1] in (survivors if round_number == 1 else pool)}
        picked.append(winner)
    return {"picked": picked, "same": True, "diverged_at_round": None, "chose": None, "wanted_in_pool": True}


def monte_carlo(data: ItemData, kept: list[str], config: FastScanConfig, seeds: int) -> dict:
    runs = [simulate_fast_scan(data, kept, config, seed) for seed in range(seeds)]
    same = sum(r["same"] for r in runs)
    where = Counter(r["diverged_at_round"] for r in runs if not r["same"])
    dropped = sum(not r["wanted_in_pool"] for r in runs)
    return {"mode": config.mode, "seeds": seeds, "agree": same, "agreement": same / seeds,
            "diverged_at_round": dict(sorted(where.items())), "true_module_missing_from_pool": dropped,
            "config": {"screen_sample": config.screen_sample, "screen_probe": config.screen_probe,
                       "survivors": config.survivors, "pool": config.pool}}


def load_run(path: Path) -> tuple[ItemData, list[str]]:
    """Reads a search directory (items.jsonl + trials.csv, optional summary.json) or an items.jsonl path."""
    path = Path(path)
    run_dir = path if path.is_dir() else path.parent
    items_path = run_dir / "items.jsonl" if path.is_dir() else path
    weight = 1.0
    summary = run_dir / "summary.json"
    if summary.exists():
        penalty = json.loads(summary.read_text(encoding="utf-8")).get("penalty")
        weight = penalty["weight"] if penalty else 0.0
    return ItemData(read_items(items_path), weight), kept_in_order(read_trial_rows(run_dir / "trials.csv"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("trials", help="(a) restrict rounds 2.. to round 1's top K, from trials.csv")
    a.add_argument("trials_csv", type=Path)
    a.add_argument("--kmin", type=int, default=5)
    a.add_argument("--kmax", type=int, default=32)
    b = sub.add_parser("items", help="(b) Monte-Carlo Fast Scan on a search's items.jsonl")
    b.add_argument("run", type=Path, help="search directory (items.jsonl + trials.csv) ")
    b.add_argument("--mode", choices=("fast", "exact"), default="fast")
    b.add_argument("--seeds", type=int, default=1000)
    for name in ("screen_sample", "screen_probe", "survivors", "pool"):
        b.add_argument(f"--{name.replace('_', '-')}", type=int, default=getattr(FastScanConfig, name))
    for sp in (a, b):
        sp.add_argument("--out", type=Path, help="also write the result as JSON")
    args = parser.parse_args()

    if args.command == "trials":
        result = replay_trials(read_trial_rows(args.trials_csv), args.kmin, args.kmax)
        print(f"kept (full search): {result['kept']}")
        for path, rank in result["round1_rank_of_kept"].items():
            print(f"  round-1 rank {rank:>3} of {result['n_round1_candidates']}  {path}")
        print(f"smallest K that keeps every winner in the pool: {result['smallest_k_that_matches']}")
        print("K:    " + " ".join(f"{r['k']:>2}" for r in result["runs"]))
        print("same: " + " ".join(f"{'✓' if r['matches'] else '✗':>2}" for r in result["runs"]))
    else:
        data, kept = load_run(args.run)
        config = FastScanConfig(mode=args.mode, screen_sample=args.screen_sample, screen_probe=args.screen_probe,
                                survivors=args.survivors, pool=args.pool)
        result = monte_carlo(data, kept, config, args.seeds)
        print(f"{args.mode} mode, {args.seeds} seeds: picked the same {len(kept)} modules in the same order "
              f"{result['agreement']:.1%} of the time")
        print(f"first disagreement by round: {result['diverged_at_round'] or 'none'}; "
              f"a true module missing from the pool in {result['true_module_missing_from_pool']} runs")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
