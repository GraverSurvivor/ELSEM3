"""Headline experiment: detect -> localize -> repair on each failure window. Needs a GPU.

For every window in `src.experiments.windows`:

    1. sample   draw the window's skewed flipped-set sample (CPU, deterministic, written once)
    2. search   python -m src.attribution.run_search --sample <window sample>        (Stage 4)
    3. repair   python -m src.attribution.run_repair --search-summary <that search>  (Stage 5)

Detection (Stage 3) is shared: all windows are drawn from the one committed flipped set, which is
what `src.stats.detect.run_detection` computes from the two reference runs, and it's written once
per experiment as detection.json.

Stages 4 and 5 run as subprocesses of Person B's own drivers rather than re-implemented here, so
the headline experiment runs exactly the code the single-window results came from. Each gets a
fresh process and so a freshly loaded model -- one window's promotions can never leak into the
next window's search.

Everything is recorded in results/headline_<ts>/manifest.json after every step. A full run is
3 x (search ~6-7 h with the hedging penalty + repair ~20 min) on a T4 -- longer than one session is
safe for -- so `--resume` skips finished steps, continues an interrupted search where it stopped
(run_search --resume) and an interrupted repair in place.

The searches use the hedging penalty by default, like the project's main repair (Phase 2 found the
unpenalised search's repair hedges); `--no-penalty` runs the original search. On Colab, pass
`--mirror <folder on Google Drive>`: the manifest, every search's progress and every finished run are
copied there, and `--resume` restores whatever the dead session's disk lost.

    python -m src.experiments.headline --windows age_heavy --mirror /content/drive/MyDrive/ELSEM3_headline
    python -m src.experiments.headline --resume headline_<ts> --mirror /content/drive/MyDrive/ELSEM3_headline
    python -m src.experiments.headline --dry-run                    # print the plan, run nothing

Don't run two of these (or a separate run_search/run_repair) at once in the same results/: each
step identifies its run by the one new results/<prefix>_* directory it created.

Then, CPU only: python -m src.experiments.headline_report headline_<ts>
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from src.attribution.greedy_search import SearchConfig
from src.attribution.search_sample import DEFAULT_SEED, DEFAULT_SIZE
from src.experiments.logger import _write_atomically, new_run_id
from src.experiments.windows import WINDOWS, ensure_window_sample

REFERENCE_DIR = Path("results/reference")
DEFAULT_FLIPPED = Path("results/flipped/hqq_3bit.json")
DEFAULT_PENALTY_SAMPLE = Path("results/search_samples/hedging_probe_n200_seed0.json")

Runner = Callable[[list[str]], None]


def _run_subprocess(argv: list[str]) -> None:
    subprocess.run([sys.executable, *argv], check=True)


def _run_dirs(results_dir: Path, prefix: str) -> set[str]:
    return {p.name for p in results_dir.glob(f"{prefix}_*") if p.is_dir()}


def _run_step(argv: list[str], results_dir: Path, prefix: str, runner: Runner) -> tuple[str | None, BaseException | None]:
    """Runs one step; returns (the run dir it created, the exception it died with, if any).

    The run id is recovered as the one new `<prefix>_*` directory even when the step fails --
    including Ctrl-C -- and the error is handed back rather than raised, so the caller can record
    the id in the manifest before re-raising. That record is what lets `--resume` continue it.
    """
    before = _run_dirs(results_dir, prefix)
    error = None
    try:
        runner(argv)
    except BaseException as exc:  # noqa: BLE001 -- re-raised by the caller after saving
        error = exc
    new = sorted(_run_dirs(results_dir, prefix) - before)
    if len(new) > 1:
        raise RuntimeError(f"expected one new {prefix}_* run, found {new} -- was something else running?")
    return (new[0] if new else None), error


class Manifest:
    """The experiment's state on disk. Saved after every step."""

    def __init__(self, path: Path, data: dict):
        self.path = path
        self.data = data

    @classmethod
    def load(cls, path: Path) -> "Manifest":
        return cls(path, json.loads(path.read_text(encoding="utf-8")))

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        _write_atomically(self.path, lambda handle: json.dump(self.data, handle, indent=2))
        mirror = self.data.get("config", {}).get("mirror")
        if mirror:
            target = Path(mirror) / self.path.parent.name
            target.mkdir(parents=True, exist_ok=True)
            for file in self.path.parent.iterdir():
                if file.is_file():
                    shutil.copy2(file, target / file.name)

    def window(self, name: str) -> dict:
        return self.data["windows"].setdefault(
            name, {"sample_path": None, "search_run_id": None, "repair_run_id": None, "repair_skipped": None}
        )


def _finished(results_dir: Path, run_id: str | None) -> bool:
    return run_id is not None and (results_dir / run_id / "summary.json").exists()


def _restore(results_dir: Path, mirror: str | None, run_id: str | None) -> None:
    """Bring a run back from the mirror when this machine's disk no longer has it."""
    if mirror and run_id and not (results_dir / run_id).exists() and (Path(mirror) / run_id).exists():
        shutil.copytree(Path(mirror) / run_id, results_dir / run_id)
        print(f"Restored {run_id} from {mirror}")


def _save_to_mirror(results_dir: Path, mirror: str | None, run_id: str | None) -> None:
    if mirror and run_id and (results_dir / run_id).exists():
        shutil.copytree(results_dir / run_id, Path(mirror) / run_id, dirs_exist_ok=True)


def search_argv(config: dict, sample_path: str, resume_id: str | None = None) -> list[str]:
    argv = ["-m", "src.attribution.run_search", "--sample", sample_path,
            "--threshold", str(config["threshold"]), "--nbits", str(config["nbits"]),
            "--max-rounds", str(config["max_rounds"]), "--batch-size", str(config["batch_size"])]
    if config.get("penalty_sample"):
        argv += ["--penalty-sample", config["penalty_sample"], "--penalty-weight", str(config["penalty_weight"])]
    if config.get("mirror"):
        argv += ["--mirror", config["mirror"]]
    if resume_id:
        argv += ["--resume", resume_id]
    return argv


def run_window(
    manifest: Manifest,
    name: str,
    flipped: dict,
    results_dir: Path,
    runner: Runner = _run_subprocess,
    dry_run: bool = False,
) -> None:
    config = manifest.data["config"]
    state = manifest.window(name)

    # 1. Sample
    sample = ensure_window_sample(flipped, name, n=config["n"], seed=config["seed"])
    state["sample_path"] = sample.as_posix()
    state["weights"] = WINDOWS[name].weights
    if not dry_run:
        manifest.save()

    # 2. Search
    mirror = config.get("mirror")
    _restore(results_dir, mirror, state["search_run_id"])
    if _finished(results_dir, state["search_run_id"]):
        print(f"[{name}] search {state['search_run_id']} already finished, skipping")
    else:
        previous = state["search_run_id"]
        places = [results_dir] + ([Path(mirror)] if mirror else [])
        checkpoint = bool(previous) and any((d / previous / "progress.json").exists() for d in places)
        argv = search_argv(config, state["sample_path"], resume_id=previous if checkpoint else None)
        print(f"[{name}] search: python {' '.join(argv)}")
        if dry_run:
            return
        if previous and not checkpoint:
            print(f"[{name}] previous search {previous} left no checkpoint; starting it over")
        run_id, error = _run_step(argv, results_dir, "search", runner)
        state["search_run_id"] = run_id or (previous if checkpoint else None)  # a resume makes no new directory
        manifest.save()
        if error is not None:
            raise error
        if not _finished(results_dir, state["search_run_id"]):
            raise RuntimeError(f"[{name}] search did not produce a summary.json (run {state['search_run_id']})")

    summary_path = results_dir / state["search_run_id"] / "summary.json"
    search_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not search_summary["kept"]:
        # Nothing cleared the threshold -- itself a result, and run_repair refuses an empty plan.
        state["repair_skipped"] = "search kept no modules"
        manifest.save()
        print(f"[{name}] search kept no modules; no repair to run")
        return

    # 3. Repair
    _restore(results_dir, mirror, state["repair_run_id"])
    if _finished(results_dir, state["repair_run_id"]):
        print(f"[{name}] repair {state['repair_run_id']} already finished, skipping")
        return
    argv = ["-m", "src.attribution.run_repair", "--search-summary", summary_path.as_posix(),
            "--batch-size", str(config["batch_size"])]
    if state["repair_run_id"] and (results_dir / state["repair_run_id"]).exists():
        argv += ["--resume", state["repair_run_id"]]
    print(f"[{name}] repair: python {' '.join(argv)}")
    if dry_run:
        return
    run_id, error = _run_step(argv, results_dir, "hqq_repaired", runner)
    state["repair_run_id"] = run_id or state["repair_run_id"]  # a resume creates no new directory
    manifest.save()
    if error is not None:
        raise error
    if not _finished(results_dir, state["repair_run_id"]):
        raise RuntimeError(f"[{name}] repair did not produce a summary.json (run {state['repair_run_id']})")
    _save_to_mirror(results_dir, mirror, state["repair_run_id"])
    _save_to_mirror(results_dir, mirror, state["search_run_id"])


def write_detection(out_path: Path) -> None:
    """Stage 3 on the reference runs, once per experiment. Flipped items are left out -- they're
    already in results/flipped/, and every window's sample records which ones it drew."""
    from src.data.safe_set import read_results_file
    from src.stats.detect import run_detection

    baseline = read_results_file(REFERENCE_DIR / "baseline_fp16" / "question_results.jsonl")
    compressed = read_results_file(REFERENCE_DIR / "hqq_3bit" / "question_results.jsonl")
    detection = run_detection(baseline, compressed).to_dict()
    detection.pop("flipped")
    _write_atomically(out_path, lambda handle: json.dump(detection, handle, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--windows", nargs="+", choices=sorted(WINDOWS), default=list(WINDOWS))
    parser.add_argument("--resume", metavar="HEADLINE_ID", help="continue an experiment, e.g. headline_20261020T101500Z")
    parser.add_argument("--flipped", type=Path, default=DEFAULT_FLIPPED)
    parser.add_argument("--n", type=int, default=DEFAULT_SIZE, help="sample size per window")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--threshold", type=float, default=SearchConfig.threshold)
    parser.add_argument("--nbits", type=int, default=SearchConfig.nbits)
    parser.add_argument("--max-rounds", type=int, default=SearchConfig.max_rounds)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--penalty-sample", type=Path, default=DEFAULT_PENALTY_SAMPLE,
                        help="hedging probe for the search's penalty (default: the committed probe)")
    parser.add_argument("--penalty-weight", type=float, default=1.0)
    parser.add_argument("--no-penalty", action="store_true", help="run the search without the hedging penalty")
    parser.add_argument("--mirror", type=Path, help="folder (e.g. on Google Drive) to keep a copy of everything in")
    parser.add_argument("--dry-run", action="store_true", help="draw samples and print the plan; run no GPU step")
    args = parser.parse_args()

    results_dir = Path("results")
    flipped = json.loads(args.flipped.read_text(encoding="utf-8"))

    if args.resume:
        local = results_dir / args.resume / "manifest.json"
        if not local.exists() and args.mirror and (args.mirror / args.resume / "manifest.json").exists():
            shutil.copytree(args.mirror / args.resume, results_dir / args.resume, dirs_exist_ok=True)
            print(f"Restored {args.resume} from {args.mirror}")
        manifest = Manifest.load(local)
        if args.mirror:  # a new session's Drive path can differ; the recorded settings otherwise stand
            manifest.data["config"]["mirror"] = args.mirror.as_posix()
        print(f"Resuming {args.resume} with its recorded config {manifest.data['config']}")
    else:
        SearchConfig(threshold=args.threshold, nbits=args.nbits, max_rounds=args.max_rounds)  # validates
        headline_id = new_run_id("headline")
        manifest = Manifest(results_dir / headline_id / "manifest.json", {
            "headline_id": headline_id,
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "flipped_path": args.flipped.as_posix(),
            "flipped_quantized_run_id": flipped["quantized_run_id"],
            "config": {"n": args.n, "seed": args.seed, "threshold": args.threshold, "nbits": args.nbits,
                       "max_rounds": args.max_rounds, "batch_size": args.batch_size,
                       "penalty_sample": None if args.no_penalty else args.penalty_sample.as_posix(),
                       "penalty_weight": None if args.no_penalty else args.penalty_weight,
                       "mirror": args.mirror.as_posix() if args.mirror else None},
            "windows": {},
        })
        if not args.dry_run:
            manifest.save()
            write_detection(manifest.path.parent / "detection.json")
        print(f"Headline experiment {headline_id}: windows {args.windows}")

    for name in args.windows:
        run_window(manifest, name, flipped, results_dir, dry_run=args.dry_run)

    if args.dry_run:
        print("\nDry run: samples written, no GPU steps run, no manifest saved.")
        return
    headline_id = manifest.data["headline_id"]
    print(f"\nDone. Manifest: {manifest.path}")
    print(f"Next (CPU): python -m src.experiments.headline_report {headline_id}")
    print("On Kaggle, download results/ before closing the session.")


if __name__ == "__main__":
    main()
