"""Files into the store: a diagnosis's folder, or the committed reference runs' summaries.

    python -m src.db.importer --jobs results/byom     # every diagnosis folder
    python -m src.db.importer --references             # results/reference/*/summary.json

Running it again adds nothing: a row's id is worked out from what it is (src/db/records.py). Per-question
research files stay in git, where the freeze manifest hashes them. `record_job` is also how a running
diagnosis gets recorded (byom.JobRunner): the pipeline itself never talks to a database.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from src.db import records
from src.db.schema import artifacts, metrics, models, parts, runs
from src.db.store import Store, open_store

LIBRARIES = ("hqq", "transformers", "torch", "accelerate", "tokenizers", "safetensors")
MAX_JSON_BYTES = 20_000_000  # nothing the pipeline writes comes close


def library_versions() -> dict[str, str | None]:
    """What HQQ's output can depend on: a repair rebuilt with others may not be the model measured."""
    from importlib import metadata

    out = {}
    for name in LIBRARIES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def read_json(path: Path):
    """The file's contents, or None if it isn't there, is still being written, or isn't JSON."""
    try:
        if Path(path).stat().st_size > MAX_JSON_BYTES:
            return None
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def record_job(store: Store, job_dir: Path, status: str, *, message: str = "", source: str = "import",
               seen: dict | None = None) -> str:
    """Brings the store up to date with one diagnosis folder; returns its run id. `seen` remembers how
    far an earlier call got, so a tick of the screen writes only what is new."""
    from src.product.byom import read_events

    job_dir, seen = Path(job_dir), {} if seen is None else seen
    info = read_json(job_dir / "job.json") or {}
    run_id = records.job_run_id(job_dir.name, info)
    if not seen.get("started"):
        store.start_run(
            run_id, str(info.get("model_id") or "unknown"), job=job_dir.name, source=source,
            pipeline=info.get("pipeline"), base_bits=info.get("base_nbits"), budget=info.get("budget"),
            mode=info.get("mode"), questions=info.get("questions"), versions=info.get("versions"),
            started_at=records.parse_time(info.get("started")) or _folder_time(job_dir),
            ran_by=info.get("ran_by") or ("imported" if source == "import" else None))
        seen["started"] = True
    events = read_events(job_dir / "events.jsonl")
    offset = seen.get("events", 0)
    if len(events) > offset:
        store.save_events(run_id, events[offset:], offset)
        seen["events"] = len(events)
    if status == "running":
        return run_id

    report, patch = read_json(job_dir / "report.json"), read_json(job_dir / "patch.json")
    kit = job_dir / "repair_kit.zip"
    ended = records.parse_time(info.get("ended")) or (records.parse_time(events[-1].get("time")) if events else None)
    store.finish_run(run_id, status, report=report, patch=patch, message=message, ended_at=ended,
                     hub_id=info.get("model_id"), versions=info.get("versions"),
                     kit=(kit.name, records.sha256(kit.read_bytes())) if kit.exists() else None)
    answers = read_json(job_dir / "answers.json")
    if isinstance(answers, dict):
        store.save_answers(run_id, answers)
    return run_id


def _folder_time(job_dir: Path) -> datetime:
    try:  # 20261125T100000123456Z_Qwen_...
        return datetime.strptime(job_dir.name.split("_")[0], "%Y%m%dT%H%M%S%fZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return datetime.fromtimestamp(job_dir.stat().st_mtime, tz=timezone.utc)


def import_jobs(store: Store, jobs_dir: Path, source: str = "import") -> list[str]:
    """Every diagnosis folder; a folder that can't be read is skipped, and said."""
    from src.product.byom import JobRunner, failure_message

    runner, done = JobRunner(jobs_dir), []
    folders = sorted(p for p in Path(jobs_dir).glob("*") if (p / "job.json").exists()) if Path(jobs_dir).exists() else []
    for job in folders:
        try:
            state = runner.state(job)
            status = "running" if state["status"] == "idle" else state["status"]
            done.append(record_job(store, job, status, message=failure_message(state), source=source))
        except Exception as error:  # noqa: BLE001 -- one odd folder must not stop the rest
            from src.db.scrub import safe_error

            print(f"[database] {job.name} was not imported ({safe_error(error)})")
    return done


def import_references(store: Store, reference_dir: Path = Path("results/reference")) -> list[str]:
    """The committed research runs' summaries: which parts, what they cost, what they measured."""
    from src.attribution.patch import MODEL_ID, Patch, total_candidate_params

    done = []
    for path in sorted(Path(reference_dir).glob("*/summary.json")):
        summary = read_json(path)
        kept = (summary or {}).get("modules") or (summary or {}).get("kept")
        if not kept:
            continue
        name, nbits = path.parent.name, int(summary.get("nbits") or (summary.get("config") or {}).get("nbits") or 8)
        run_id = records.stable_id("reference", name)
        base = int(summary.get("base_nbits") or 3)
        patch = Patch(name=name, modules=tuple(kept), nbits=nbits, source=summary.get("patch_source") or {},
                      base={"method": "hqq", "nbits": base, "group_size": 64, "compute_dtype": "fp16"}).to_dict()
        clock, memory = summary.get("wall_clock_seconds"), summary.get("memory") or {}
        seconds = clock.get("cycle") if isinstance(clock, dict) else clock
        overhead = memory.get("deployed_overhead_bytes", summary.get("deployed_overhead_bytes"))
        model = records.model_row(MODEL_ID)
        row = {"id": run_id, "model_id": model["id"], "job": name, "source": "reference", "pipeline": "real",
               "method": "hqq", "base_bits": base, "repair_bits": nbits, "status": "done",
               "mode": summary.get("context") or "ambig", "ran_by": "research",
               "questions": ((summary.get("fairness") or {}).get("overall") or {}).get("n_items"),
               "extra_mb": overhead / 2**20 if overhead is not None else None,
               "restored_share": patch["params"] / total_candidate_params(),
               "minutes": seconds / 60 if seconds is not None else None,
               "started_at": _run_time(summary.get("run_id"))}
        store.save_rows(models, [{**model, "family": "phi3"}])
        store.save_rows(runs, [row])
        store.save_rows(metrics, records.reference_metric_rows(run_id, summary))
        store.save_rows(parts, records.part_rows(run_id, kept, patch))
        store.save_rows(artifacts, [records.artifact_row(run_id, patch, None, None, None)])
        done.append(run_id)
    return done


def _run_time(run_id) -> datetime | None:
    try:  # hqq_repaired_20261005T180658Z
        return datetime.strptime(str(run_id).rsplit("_", 1)[-1], "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jobs", type=Path, help="a folder of diagnosis folders, e.g. results/byom")
    parser.add_argument("--references", action="store_true", help="the committed results/reference summaries")
    parser.add_argument("--reference-dir", type=Path, default=Path("results/reference"))
    parser.add_argument("--db", type=Path, help="the local database file")
    args = parser.parse_args(argv)
    if args.jobs is None and not args.references:
        parser.error("nothing to import: give --jobs and/or --references")
    store = open_store(path=args.db)
    print(store.wait_settled())
    before = store.counts()["runs"]
    seen = import_jobs(store, args.jobs) if args.jobs else []
    seen += import_references(store, args.reference_dir) if args.references else []
    print(f"{len(seen)} runs read, {store.counts()['runs'] - before} of them new")
    store.sync_now()
    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
