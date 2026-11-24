"""The Studio records what happens: diagnoses as they run, typed questions, API sessions."""

import json
import time

import pytest

pytest.importorskip("sqlalchemy")

from sqlalchemy import select

from src.db.schema import api_sessions
from src.db.store import Store, sqlite_engine
from src.product.byom import JobRunner
from tests.db.test_store import every_value


@pytest.fixture
def new_store(tmp_path, request):
    def make():
        store = Store(sqlite_engine(tmp_path / "app.sqlite"), ran_by="Asha")
        request.addfinalizer(store.close)
        return store

    return make


def wait_for(check, seconds=60):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if check():
            return
        time.sleep(0.1)
    raise AssertionError("timed out")


# --- diagnoses -------------------------------------------------------------------------------------

def test_a_diagnosis_is_recorded_to_its_end_with_nobody_watching(tmp_path, new_store):
    store = new_store()
    runner = JobRunner(tmp_path / "jobs", pipeline="fake", speed=3000, store=store)
    job = runner.start("Qwen/Qwen2.5-3B-Instruct", "fast", hf_token="hf_thisIsSecret123")
    [row] = store.runs()  # there at once
    assert row["id"] == runner.run_id(job) and row["status"] == "running" and row["ran_by"] == "Asha"
    assert row["budget"] == 0.10 and row["versions"]["torch"]
    wait_for(lambda: store.runs()[0]["status"] == "done" and runner._recorded[job.name].get("final"))
    run = store.run(row["id"])
    report = json.loads((job / "report.json").read_text(encoding="utf-8"))
    assert run["outcome"] == "repaired" and run["recovery"] == pytest.approx(report["recovery"]["point"])
    assert [p["path"] for p in run["parts"]] == report["kept"] and run["artifact"]["recipe"]["modules"] == report["kept"]
    assert len(run["events"]) == len((job / "events.jsonl").read_text().splitlines())
    assert run["artifact"]["versions"] == run["versions"]
    assert not any("hf_" in value for _, value in every_value(store))

    before = store.counts(), store.pending()
    fresh = JobRunner(tmp_path / "jobs", pipeline="fake", store=new_store())
    fresh.sync_all()
    fresh.sync()
    assert (store.counts(), store.pending()) == before


def test_the_app_dies_mid_run_and_the_run_is_there_after_a_restart(tmp_path, new_store):
    first = JobRunner(tmp_path / "jobs", pipeline="fake", speed=8, store=new_store())
    job = first.start("Qwen/Qwen2.5-3B-Instruct", "exact")
    try:
        wait_for(lambda: len((job / "events.jsonl").read_text().splitlines()) >= 2
                 if (job / "events.jsonl").exists() else False)
        first._recorded[job.name] = {"final": True}  # the app is gone: its recorder stops
        store = new_store()
        restarted = JobRunner(tmp_path / "jobs", pipeline="fake", store=store)
        restarted.sync_all()
        run = store.run(restarted.run_id(job))
        assert run["hub_id"] == "Qwen/Qwen2.5-3B-Instruct" and len(run["events"]) >= 2
        assert restarted.current() == job
    finally:
        first.cancel(job)
    restarted.sync(job)
    assert store.run(run["id"])["status"] == "cancelled" and len(store.runs()) == 1


def test_recording_trouble_never_stops_a_run(tmp_path, capsys):
    class Broken:
        ran_by = "x"

        def __getattr__(self, name):
            raise RuntimeError("disk full at postgresql://me:secretpw@host/db")

    runner = JobRunner(tmp_path, pipeline="fake", speed=3000, store=Broken())
    job = runner.start("Qwen/Qwen2.5-3B-Instruct", "fast")
    wait_for(lambda: runner.state(job)["status"] == "done")
    runner.sync()
    out = capsys.readouterr().out
    assert "could not be recorded" in out and "secretpw" not in out and "Traceback" not in out
    runner._recorded[job.name]["final"] = True


def test_without_a_store_the_folders_are_the_record(tmp_path):
    runner = JobRunner(tmp_path, pipeline="fake", speed=3000)
    job = runner.start("Qwen/Qwen2.5-3B-Instruct", "fast")
    runner.sync(), runner.sync_all()
    wait_for(lambda: runner.state(job)["status"] == "done")
    info = runner.job_info(job)
    assert "versions" not in info and "ran_by" not in info and not runner._watched


def test_the_screen_timer_writes_nothing_when_nothing_changed(tmp_path, new_store, monkeypatch):
    store = new_store()
    runner = JobRunner(tmp_path / "jobs", pipeline="fake", speed=8, store=store)
    job = runner.start("Qwen/Qwen2.5-3B-Instruct", "exact")
    try:
        runner._recorded[job.name]["final"] = True
        time.sleep(1.7)  # the background recorder ends
        runner._recorded[job.name]["final"] = False
        calls = []
        monkeypatch.setattr("src.db.importer.record_job", lambda *a, **k: calls.append(a))
        monkeypatch.setattr(JobRunner, "_mark", staticmethod(lambda job_dir: ("same",)))
        for _ in range(3):
            runner.sync(job)
        assert len(calls) == 1
    finally:
        runner.cancel(job)
        runner._recorded[job.name]["final"] = True


# --- the screens -----------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def data():
    pytest.importorskip("gradio")
    from src.product.studio import StudioData

    try:
        return StudioData()
    except Exception as error:  # noqa: BLE001 -- needs the BBQ questions cached
        pytest.skip(f"Studio data not available: {error}")


@pytest.fixture
def app(data, new_store, tmp_path):
    from src.product import studio
    from src.product.api_server import ApiService
    from src.product.studio_core import Answer, SavedBackend

    class Live(SavedBackend):
        live = True

        def answers_for(self, items, view):
            return None if view == "original" else [Answer(it.unknown_index, (0.2, 0.2, 0.6)) for it in items]

    store = new_store()
    backend = Live(data.runs, list(data.patch.modules), repairs=data.saved_repairs)
    api = ApiService()
    built, _ = studio.build_app(backend, data, studio.ByomSetup(JobRunner(tmp_path / "jobs", pipeline="fake",
                                                                         store=store)), api)
    return {f.fn.__name__: f.fn for f in built.fns.values() if getattr(f, "fn", None) is not None}, store, api


STORY = ("A nurse and a mechanic were at the bus stop when a wallet went missing.", "Who took it?", "The nurse",
         "The mechanic")


def test_typed_questions_are_saved_unless_the_person_says_not_to(app):
    handlers, store, _ = app
    handlers["ask_custom"](*STORY, True)
    assert store.recent_questions() == []
    handlers["ask_custom"](*STORY, False)
    [asked] = store.recent_questions()
    assert (asked["story"], asked["question"], asked["person_a"], asked["person_b"]) == STORY
    assert set(asked["answers"]) == {"compressed", "repaired"}  # the original wasn't loaded
    assert asked["answers"]["repaired"]["answer"] == "Can't be determined"
    assert "all 6" in asked["answers"]["repaired"]["consistency"]
    assert asked["answers"]["repaired"]["probs"][0] > 0


def test_api_sessions_are_recorded_by_key_without_the_key(app, monkeypatch):
    handlers, store, api = app

    def rows():
        with store.read() as connection:
            return connection.execute(select(api_sessions).order_by(api_sessions.c.started_at)).mappings().all()

    api.start(object(), object(), "cpu", "demo", "demo")
    from src.db.records import sha256

    first = api.key
    handlers["api_new_key"]()  # a served model whose session began elsewhere: this key's starts here
    api.requests += 3
    second = api.key
    handlers["api_new_key"]()
    handlers["api_stop"]()
    saved = rows()
    assert [r["key_sha256"] for r in saved] == [sha256(second), saved[1]["key_sha256"]]
    assert saved[0]["requests"] == 3 and saved[1]["requests"] == 0 and all(r["stopped_at"] for r in saved)
    assert not any(first in value or second in value or "fg-" in value for _, value in every_value(store))


def test_the_app_runs_without_a_database(data):
    from src.product import studio
    from src.product.studio_core import SavedBackend

    backend = SavedBackend(data.runs, list(data.patch.modules), repairs=data.saved_repairs)
    built, _ = studio.build_app(backend, data)
    handlers = {f.fn.__name__: f.fn for f in built.fns.values() if getattr(f, "fn", None) is not None}
    assert len(handlers["ask_custom"](*STORY)) == 7
