"""The store (contract C9): the same tests on SQLite and, when TEST_DATABASE_URL is set, on Postgres."""

import json
import sqlite3
import threading
from datetime import datetime, timezone

import pytest
from sqlalchemy import select, text

from src.db import records, schema
from src.db.schema import SYNCED, SchemaTooNew, upgrade
from src.db.scrub import safe_error, scrub
from src.db.store import Store, open_store, postgres_engine, sqlite_engine

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
KEPT = ["model.layers.3.self_attn.q_proj", "model.layers.0.mlp.down_proj"]
PATCH = {"name": "fast_repair_fast", "model_id": MODEL, "modules": KEPT, "nbits": 8,
         "base": {"method": "hqq", "nbits": 3, "group_size": 64, "compute_dtype": "fp16"},
         "part_shapes": {"self_attn.q_proj": [64, 64], "mlp.down_proj": [64, 128]}, "n_layers": 4,
         "module_bits": {KEPT[1]: 4}, "revision": "abc123"}
REPORT = {"model_id": MODEL, "arch": "qwen2", "mode": "fast", "questions": 1000, "gpu": "Tesla T4", "minutes": 31.5,
          "original": {"usr": 0.84, "srs": 0.11, "false_unknown": 0.07, "accuracy": 0.9},
          "compressed": {"usr": 0.36, "srs": 0.37, "false_unknown": 0.2, "accuracy": 0.8},
          "repaired": {"usr": 0.8, "srs": 0.13, "false_unknown": 0.1, "accuracy": 0.85},
          "recovery": {"point": 0.9, "lower": 0.85, "upper": 0.95}, "kept": KEPT, "gate": "PASS", "gate_failures": [],
          "outcome": "repaired", "repair_verdict": "IMPROVED", "memory_mb": 12.5, "restored_share": 0.04,
          "base_nbits": 3, "fingerprint": [{"question_id": "Age:1", "set": "ambig", "pick": 2}]}
EVENTS = [{"time": "2026-11-25T10:00:00+00:00", "stage": "download", "status": "started", "done": 0, "total": 0,
           "message": ""},
          {"time": "2026-11-25T10:01:00+00:00", "stage": "download", "status": "done", "done": 1, "total": 1,
           "message": "3.1 GB of weights"}]
ANSWERS = {"ambig": {"Age:1": {"split": "final", "original": 2, "compressed": 0, "repaired": 2}},
           "stated": {"Age:1": {"split": "search", "original": 1, "compressed": 1, "repaired": None}}}


def save_everything(store, run_id="11111111-1111-5111-8111-111111111111"):
    store.start_run(run_id, MODEL, job="20261125T100000000000Z_Qwen", mode="fast", base_bits=3, budget=0.1,
                    questions=1000, versions={"hqq": "0.2.8.post1", "torch": "2.10.0"})
    store.save_events(run_id, EVENTS)
    store.finish_run(run_id, "done", report=REPORT, patch=PATCH, kit=("repair_kit.zip", "0" * 64))
    store.save_answers(run_id, ANSWERS)
    return run_id


# --- saving and reading ----------------------------------------------------------------------------

def test_a_run_is_saved_and_read_back_whole(store):
    run_id = save_everything(store)
    run = store.run(run_id)
    assert (run["hub_id"], run["revision"], run["family"]) == (MODEL, "abc123", "qwen2")
    assert run["weights"] == 4 * (64 * 64 + 64 * 128)
    assert (run["status"], run["outcome"], run["verdict"], run["method"]) == ("done", "repaired", "PASS", "hqq")
    assert (run["base_bits"], run["repair_bits"], run["budget"]) == (3, 8, 0.1)
    assert run["recovery"] == 0.9 and run["extra_mb"] == 12.5 and run["ran_by"] == "tester"
    assert run["versions"] == {"hqq": "0.2.8.post1", "torch": "2.10.0"}
    assert run["started_at"].tzinfo is not None
    assert run["report"]["gate"] == "PASS" and "fingerprint" not in run["report"]
    assert [e["message"] for e in run["events"]] == ["", "3.1 GB of weights"]
    assert [(p["path"], p["bits"]) for p in run["parts"]] == [(KEPT[0], 8), (KEPT[1], 4)]  # per-part bits (C7)
    assert run["artifact"]["recipe"] == PATCH and run["artifact"]["fingerprint"] == REPORT["fingerprint"]
    assert run["artifact"]["versions"]["hqq"] == "0.2.8.post1" and run["artifact"]["kit_name"] == "repair_kit.zip"
    values = {(m["view"], m["metric"]): m for m in run["metrics"]}
    assert values["compressed", "usr"]["value"] == 0.36 and values["repaired", "recovery"]["lower"] == 0.85
    answers = {(a["question_set"], a["question_id"]): a for a in store.answers(run_id)}
    assert answers["ambig", "Age:1"]["repaired"] == 2 and answers["stated", "Age:1"]["repaired"] is None
    assert store.run("missing") is None


def test_a_failed_run_keeps_its_progress_and_its_reason(store):
    store.start_run("r", MODEL, mode="exact")
    store.save_events("r", EVENTS)
    store.finish_run("r", "failed", message="It stopped at Compress the model: out of memory")
    run = store.run("r")
    assert run["status"] == "failed" and "out of memory" in run["message"]
    assert run["outcome"] is None and run["report"] is None and len(run["events"]) == 2 and run["artifact"] is None


def test_saving_the_same_thing_again_changes_nothing(store):
    save_everything(store)
    before, waiting = store.counts(), store.pending()
    assert store.start_run("11111111-1111-5111-8111-111111111111", MODEL) == 0  # not restarted
    assert store.save_events("11111111-1111-5111-8111-111111111111", EVENTS) == 0
    assert store.finish_run("11111111-1111-5111-8111-111111111111", "done", report=REPORT, patch=PATCH,
                            kit=("repair_kit.zip", "0" * 64)) == 0
    assert store.save_answers("11111111-1111-5111-8111-111111111111", ANSWERS) == 0
    assert store.counts() == before and store.pending() == waiting


def test_search_and_filters(store):
    save_everything(store, "a")
    store.start_run("b", "meta-llama/Llama-3.2-3B-Instruct", method="gptq")
    store.finish_run("b", "failed")
    assert [r["id"] for r in store.runs()] == ["b", "a"]  # newest first
    assert [r["id"] for r in store.runs(search="QWEN")] == ["a"]
    assert [r["id"] for r in store.runs(search="qwen pass")] == ["a"]
    assert [r["id"] for r in store.runs(search="tester llama")] == ["b"]
    assert store.runs(search="100%") == [] and store.runs(search="Qwen_Qwen2.5") == []  # taken literally
    assert [r["id"] for r in store.runs(outcome="repaired")] == ["a"]
    assert [r["id"] for r in store.runs(outcome="failed")] == ["b"]  # no outcome: its status
    assert [r["id"] for r in store.runs(method="gptq")] == ["b"] and store.methods() == ["gptq", "hqq"]


def test_weights_link_and_delete(store):
    run_id = save_everything(store)
    assert store.runs()[0]["weights_url"] is None
    store.set_weights(run_id, "https://huggingface.co/me/q-fairgate", private=True)
    assert store.runs()[0]["weights_url"].endswith("q-fairgate")
    assert store.run(run_id)["artifact"]["recipe"] == PATCH  # the link is added to the recipe, not instead of it
    assert store.delete_run(run_id) is True and store.delete_run(run_id) is False
    assert store.run(run_id) is None and store.answers(run_id) == []
    counts = store.counts()
    assert all(counts[t.name] == 0 for t in (schema.runs, *schema.RUN_TABLES)) and counts["deletions"] == 1
    save_everything(store, run_id)  # an import of its folder, say
    assert store.run(run_id) is None and store.counts()["run_events"] == 0


def test_ask_and_api_history(store):
    store.save_question("A nurse and a mechanic waited.", "Who left?", "The nurse", "The mechanic", {
        "original": {"pick": 2, "answer": "Can't be determined", "consistency": "all 6", "probs": [0.1, 0.1, 0.8]},
        "compressed": {"pick": 1, "answer": "The mechanic", "consistency": "4 of 6"}}, model_name="Phi-3.5")
    asked = store.recent_questions()[0]
    assert asked["person_b"] == "The mechanic" and asked["asked_by"] == "tester"
    assert asked["answers"]["original"]["probs"] == [0.1, 0.1, 0.8]
    assert asked["answers"]["compressed"]["answer"] == "The mechanic" and "repaired" not in asked["answers"]

    key = "fg-" + "k" * 43
    session = store.start_api_session(key, "qwen-fairgate", run_id="r")
    store.end_api_session(session, 7)
    with store.read() as connection:
        row = connection.execute(select(schema.api_sessions)).mappings().one()
    assert row["requests"] == 7 and row["stopped_at"] >= row["started_at"] and row["model_name"] == "qwen-fairgate"
    assert row["key_sha256"] == records.sha256(key)


def test_odd_values_are_stored_safely(store):
    report = {**REPORT, "gpu": "x" * 500, "minutes": float("nan"), "outcome_reason": "a\x00b",
              "other_damage": [float("inf")], "original": {"usr": None}}
    store.start_run("r", MODEL)
    store.finish_run("r", "done", report=report, patch=PATCH)
    run = store.run("r")
    assert len(run["gpu"]) == 100 and run["minutes"] is None and run["outcome_reason"] == "ab"
    assert run["report"]["other_damage"] == [None]


# --- secrets ---------------------------------------------------------------------------------------

SECRETS = ["hf_AbCdEf123456", "fg-abc_DEF-123", "ghp_0123456789", "postgres://me:pw@host/db",
           "postgresql+psycopg://me:pw@host/db"]


def every_value(store):
    with store.read() as connection:
        for table in SYNCED:
            for row in connection.execute(select(table)).mappings():
                yield table.name, json.dumps(dict(row), default=str)


def test_no_secret_reaches_any_row(store):
    said = " and ".join(SECRETS)
    run_id = "r"
    store.start_run(run_id, MODEL, versions={"note": said}, job=said)
    store.save_events(run_id, [{**EVENTS[0], "message": f"401 for url with token {said}"}])
    store.finish_run(run_id, "done", report={**REPORT, "outcome_reason": said, "extra": {said: [said]}},
                     patch={**PATCH, "source": {"kind": said}}, message=said)
    store.set_weights(run_id, f"https://huggingface.co/me/q?token={SECRETS[0]}")
    store.save_question(said, said, said, said, {"original": {"answer": said, "consistency": said}}, model_name=said)
    store.start_api_session(SECRETS[1], said)
    rows = list(every_value(store))
    assert len(rows) > 8
    for name, value in rows:
        for bad in ("hf_", "fg-", "ghp_", "postgres://", "postgresql+psycopg://", ":pw@"):
            assert bad not in value, (name, bad)


def test_scrubbing_leaves_ordinary_text_alone():
    for text_ in ("Qwen/Qwen2.5-3B-Instruct", "https://huggingface.co/me/model-fairgate", "model.layers.3.self_attn.q_proj",
                  "It stopped at \"Compress the model\".", "hf-internal-testing/tiny"):
        assert scrub(text_) == text_
    assert scrub({"a": ["hf_x1", 3, None]}) == {"a": ["[removed]", 3, None]}


def test_an_error_never_shows_the_address():
    error = RuntimeError("could not connect to postgresql://me:secretpw@host/db\nmore")
    assert "secretpw" not in safe_error(error)
    wrapped = RuntimeError("outer")
    wrapped.orig = ValueError("password=hf_abc")
    assert safe_error(wrapped) == "ValueError: password=[removed]"


# --- opening ---------------------------------------------------------------------------------------

def test_without_an_address_the_store_is_local_only(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    store = open_store(path=tmp_path / "db" / "x.sqlite")
    assert store.status_line() == "database: local only" and store.wait_settled(0.1) == "database: local only"
    save_everything(store)
    assert store.sync_now() is False and len(store.runs()) == 1
    store.close()
    again = open_store(path=tmp_path / "db" / "x.sqlite")  # the app restarted
    assert again.runs()[0]["hub_id"] == MODEL
    again.close()


def test_a_bad_address_is_local_only_and_never_printed(tmp_path, monkeypatch, capsys):
    for bad in ("mysql://me:secretpw@host/db", "not an address secretpw", "postgresql://me:secretpw@host:notaport/db"):
        monkeypatch.setenv("DATABASE_URL", bad)
        store = open_store(path=tmp_path / "x.sqlite")
        assert store.remote is None and "local only" in store.status_line()
        assert "secretpw" not in store.status_line()
        store.close()
    assert "secretpw" not in capsys.readouterr().out


def test_a_missing_driver_is_named_not_blamed_on_the_address(tmp_path, monkeypatch):
    from src.db import store as store_module

    def no_driver(url):
        raise ModuleNotFoundError("No module named 'psycopg'")

    monkeypatch.setattr(store_module, "postgres_engine", no_driver)
    store = open_store(url="postgresql://me:secretpw@host/db", path=tmp_path / "x.sqlite")
    assert "driver" in store.status_line() and "secretpw" not in store.status_line()
    store.close()


def test_the_shared_database_engine_is_set_up_for_free_pooled_servers():
    pytest.importorskip("psycopg", exc_type=ImportError)
    engine = postgres_engine("postgres://me:pw@ep-x-pooler.eu.aws.neon.tech/db")
    assert engine.url.drivername == "postgresql+psycopg" and engine.url.query["sslmode"] == "require"
    assert engine.pool._pre_ping is True and engine.hide_parameters is True
    assert "sslmode" not in postgres_engine("postgresql://me:pw@localhost/db").url.query
    assert postgres_engine("postgres://u:p@h/d?sslmode=verify-full").url.query["sslmode"] == "verify-full"
    assert "pw" not in repr(engine)


def test_a_server_that_cannot_be_reached_stops_nothing(tmp_path, capsys):
    class Down:
        url = "x"

        def begin(self):
            raise OSError("connection to postgresql://me:secretpw@host failed")

        connect = begin

        def _run_ddl_visitor(self, *a, **k):
            raise OSError("connection to postgresql://me:secretpw@host failed")

        def dispose(self):
            pass

    store = Store(sqlite_engine(tmp_path / "x.sqlite"), Down(), ran_by="me")
    store.start()
    save_everything(store)
    assert "can't be reached" in store.wait_settled(10)
    assert len(store.runs()) == 1 and store.pending() > 0
    store.close(1)
    out = capsys.readouterr().out
    assert "secretpw" not in out and "Traceback" not in out and out.count("working locally") == 1


# --- schema versions -------------------------------------------------------------------------------

def test_upgrade_runs_each_missing_step_once(engine, monkeypatch):
    assert upgrade(engine) == 1 and upgrade(engine) == 1
    ran = []
    monkeypatch.setattr(schema, "SCHEMA_VERSION", 3)
    monkeypatch.setattr(schema, "MIGRATIONS", {2: lambda c: ran.append(2), 3: lambda c: ran.append(3)})
    assert upgrade(engine) == 3 and ran == [2, 3]
    assert upgrade(engine) == 3 and ran == [2, 3]
    monkeypatch.setattr(schema, "SCHEMA_VERSION", 2)
    with pytest.raises(SchemaTooNew, match="update the code"):
        upgrade(engine)


def test_a_newer_shared_database_leaves_the_app_local(engine, tmp_path, capsys):
    upgrade(engine)
    with engine.begin() as connection:
        connection.execute(schema.schema_version.update().values(version=99))
    store = Store(sqlite_engine(tmp_path / "l.sqlite"), engine, ran_by="me")
    save_everything(store)
    assert store.sync_now() is False and "update the code" in store.status_line()
    assert len(store.runs()) == 1
    store.close()


# --- sharing the local file ------------------------------------------------------------------------

def test_threads_and_a_second_store_share_the_file(tmp_path):
    first, second = (Store(sqlite_engine(tmp_path / "x.sqlite"), ran_by=n) for n in ("one", "two"))
    errors = []

    def work(store, name):
        try:
            for i in range(15):
                store.start_run(f"{name}-{i}", MODEL)
                store.save_events(f"{name}-{i}", EVENTS)
                store.finish_run(f"{name}-{i}", "done", report=REPORT, patch=PATCH)
                store.runs(search="qwen")
        except Exception as error:  # noqa: BLE001
            errors.append(error)

    threads = [threading.Thread(target=work, args=(s, f"t{k}")) for k, s in enumerate((first, first, second, second))]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert errors == [] and len(first.runs()) == 60 and first.counts()["run_events"] == 120
    first.close(), second.close()


def test_a_backup_is_a_whole_database(tmp_path):
    store = Store(sqlite_engine(tmp_path / "x.sqlite"), ran_by="me")
    save_everything(store)
    copy = store.backup(tmp_path / "drive" / "copy.sqlite")
    with sqlite3.connect(copy) as connection:
        assert connection.execute("select count(*) from runs").fetchone() == (1,)
    connection.close()
    store.close()


def test_times_need_a_time_zone(engine):
    upgrade(engine)
    moment = datetime(2026, 11, 25, 12, 0, 0, 123456, tzinfo=timezone.utc)
    with engine.begin() as connection:
        row = {"id": "d", "table_name": "runs", "deleted_at": moment, "updated_at": moment}
        connection.execute(schema.deletions.insert().values(**row))
        assert connection.execute(select(schema.deletions.c.deleted_at)).scalar() == moment
        assert connection.execute(text("select 1")).scalar() == 1
    with pytest.raises(Exception, match="time zone"):
        with engine.begin() as connection:
            connection.execute(schema.deletions.insert().values(
                **{**row, "id": "e", "deleted_at": datetime(2026, 1, 1)}))


def test_the_statements_and_tables_are_valid_for_postgres_without_a_server():
    """Compiled only: catches a construct one database lacks even where no Postgres is available."""
    from sqlalchemy.dialects import postgresql
    from sqlalchemy.schema import CreateTable

    from src.db.store import _bound, _server_time

    dialect = postgresql.dialect()
    for table in schema.metadata.sorted_tables:
        assert "CREATE TABLE" in str(CreateTable(table).compile(dialect=dialect))
    connection = type("C", (), {"dialect": dialect})()
    statement = schema.dialect_insert(connection, schema.runs)
    moment = _server_time(connection)
    upsert = statement.values(id="a", received_at=moment).on_conflict_do_update(
        index_elements=["id"], set_={"status": statement.excluded.status, "received_at": moment},
        where=statement.excluded.updated_at >= schema.runs.c.updated_at)
    sql = str(upsert.compile(dialect=dialect))
    assert "ON CONFLICT (id) DO UPDATE" in sql and "timezone(" in sql and "clock_timestamp()" in sql
    from sqlalchemy import tuple_

    page = select(schema.runs.c.id).where(tuple_(schema.runs.c.received_at, schema.runs.c.id)
                                          > tuple_(_bound(schema.runs, datetime.now(timezone.utc)), "x"))
    assert "(runs.received_at, runs.id) >" in str(page.compile(dialect=dialect))
