"""Job folders and the committed reference runs, into the store: complete, and only once."""

import json
from pathlib import Path

import pytest

pytest.importorskip("sqlalchemy")

from src.db import records
from src.db.importer import import_jobs, import_references, main, record_job
from tests.db.test_store import ANSWERS, EVENTS, KEPT, MODEL, PATCH, REPORT, every_value

STAGES = ("download", "quantize", "score_original", "score_3bit", "find_damage", "scan", "verify", "report")


def make_job(jobs, name="20261125T100000000000Z_Qwen", info=None, report=REPORT, patch=PATCH, answers=ANSWERS,
             events=None):
    job = Path(jobs) / name
    job.mkdir(parents=True)
    (job / "job.json").write_text(json.dumps({
        "model_id": MODEL, "mode": "fast", "base_nbits": 3, "questions": 1000, "pipeline": "real", "pid": None,
        "started": "2026-11-25T10:00:00+00:00", "ended": None, "cancelled": False, **(info or {})}))
    if events is None:
        events = [{**EVENTS[0], "stage": s, "status": st, "time": f"2026-11-25T10:{i:02d}:00+00:00"}
                  for i, s in enumerate(STAGES) for st in ("started", "done")]
    (job / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    for file, value in (("report.json", report), ("patch.json", patch), ("answers.json", answers)):
        if value is not None:
            (job / file).write_text(json.dumps(value))
    (job / "repair_kit.zip").write_bytes(b"kit")
    return job


def test_a_job_folder_is_imported_whole_and_only_once(store, tmp_path):
    job = make_job(tmp_path / "jobs")
    [run_id] = import_jobs(store, tmp_path / "jobs")
    run = store.run(run_id)
    assert (run["status"], run["outcome"], run["source"], run["ran_by"]) == ("done", "repaired", "import", "imported")
    assert run["hub_id"] == MODEL and run["revision"] == "abc123" and len(run["events"]) == 16
    assert run["started_at"].isoformat() == "2026-11-25T10:00:00+00:00"
    assert run["ended_at"].isoformat() == "2026-11-25T10:07:00+00:00"  # the last thing it said
    assert run["versions"] is None  # not known: this machine's versions aren't claimed for it
    assert run["artifact"]["kit_sha256"] == records.sha256(b"kit") and len(store.answers(run_id)) == 2
    counts, waiting = store.counts(), store.pending()
    assert import_jobs(store, tmp_path / "jobs") == [run_id]
    assert store.counts() == counts and store.pending() == waiting

    moved = tmp_path / "elsewhere"  # the same folder on another machine is the same run
    moved.mkdir()
    job.rename(moved / job.name)
    assert import_jobs(store, moved) == [run_id] and store.counts() == counts


def test_jobs_from_before_the_database_and_odd_folders(store, tmp_path, capsys):
    jobs = tmp_path / "jobs"
    old = make_job(jobs, "20261016T090000Z_old", report={k: v for k, v in REPORT.items() if k not in
                                                        ("outcome", "fingerprint")}, answers=None)
    (old / "patch.json").write_text(json.dumps({k: v for k, v in PATCH.items() if k not in ("module_bits", "revision")}))
    make_job(jobs, "20261125T110000000000Z_failed", report=None, patch=None, answers=None, events=[
        EVENTS[0], {**EVENTS[0], "status": "failed", "message": "model not found"}])
    make_job(jobs, "20261125T120000000000Z_cancelled", info={"cancelled": True}, report=None, patch=None, answers=None,
             events=EVENTS[:1])
    odd = make_job(jobs, "20261125T130000000000Z_odd")
    (odd / "report.json").write_text('{"model_id": "x", ')  # cut off
    (odd / "answers.json").write_text("[]")
    (jobs / "not a job").mkdir()
    import_jobs(store, jobs)
    by_job = {r["job"]: r for r in store.runs()}
    assert len(by_job) == 4 and by_job[old.name]["outcome"] == "repaired" and by_job[old.name]["revision"] is None
    failed = by_job["20261125T110000000000Z_failed"]
    assert failed["status"] == "failed" and "model not found" in failed["message"]
    assert by_job["20261125T120000000000Z_cancelled"]["status"] == "cancelled"
    assert by_job[odd.name]["status"] == "done" and by_job[odd.name]["outcome"] is None
    assert store.run(by_job[old.name]["id"])["artifact"]["fingerprint"] is None


def test_the_id_the_runner_gave_is_used(store, tmp_path):
    job = make_job(tmp_path / "jobs", info={"run_id": "33333333-3333-4333-8333-333333333333", "ran_by": "Asha"})
    assert record_job(store, job, "done") == "33333333-3333-4333-8333-333333333333"
    assert store.runs()[0]["ran_by"] == "Asha"


def test_the_reference_runs_are_imported_once(store):
    if not Path("results/reference/MAIN_REPAIR.json").exists():
        pytest.skip("run from the repository root")
    ids = import_references(store)
    by_job = {r["job"]: r for r in store.runs()}
    main_run = by_job["repair_hedging_penalty"]
    assert main_run["source"] == "reference" and main_run["hub_id"] == "microsoft/Phi-3.5-mini-instruct"
    assert main_run["base_bits"] == 3 and main_run["started_at"].isoformat() == "2026-10-05T18:06:58+00:00"
    run = store.run(main_run["id"])
    assert len(run["parts"]) == 5 and {p["bits"] for p in run["parts"]} == {8}
    values = {(m["metric"], m["scope"], m["name"]): m["value"] for m in run["metrics"]}
    assert values["usr", "overall", None] == pytest.approx(0.7734, abs=1e-4)
    assert values["usr", "category", "Age"] == pytest.approx(0.7054, abs=1e-4)
    assert 0.03 < main_run["restored_share"] < 0.05 and main_run["extra_mb"] == pytest.approx(81, abs=1)
    stated = {(m["metric"], m["scope"]): m["value"] for m in store.run(by_job["repair_hedging_penalty_disambig"]["id"])
              ["metrics"]}
    assert stated["false_unknown", "overall"] == pytest.approx(0.109, abs=0.002)  # not a "fair answer" rate
    assert stated["accuracy", "overall"] == pytest.approx(0.837, abs=0.002) and ("usr", "overall") not in stated
    assert store.counts()["run_answers"] == 0  # per-question files stay in git
    counts, waiting = store.counts(), store.pending()
    assert import_references(store) == ids and store.counts() == counts and store.pending() == waiting


def test_the_command_says_what_was_new(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    make_job(tmp_path / "jobs")
    args = ["--jobs", str(tmp_path / "jobs"), "--db", str(tmp_path / "x.sqlite")]
    assert main(args) == 0 and "1 runs read, 1 of them new" in capsys.readouterr().out
    assert main(args) == 0 and "1 runs read, 0 of them new" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main([])


def test_nothing_secret_is_imported(store, tmp_path):
    make_job(tmp_path / "jobs", info={"command": ["--token", "hf_abc123"]},
             events=[{**EVENTS[0], "message": "401 for hf_abc123 and postgres://u:p@h/d"}],
             report={**REPORT, "outcome_reason": "ghp_abc"})
    import_jobs(store, tmp_path / "jobs")
    for name, value in every_value(store):
        assert not any(bad in value for bad in ("hf_", "fg-", "ghp_", "postgres://")), name
    assert KEPT[0] in json.dumps([p["path"] for p in store.run(store.runs()[0]["id"])["parts"]])
