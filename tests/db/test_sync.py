"""Two machines and a shared server: rows arrive once, late ones too, and a deleted run stays deleted."""

import time
from datetime import timedelta

from sqlalchemy import func, select

from src.db import schema, store as store_module
from src.db.schema import SYNCED, utcnow
from tests.db.test_store import ANSWERS, EVENTS, MODEL, PATCH, REPORT, every_value, save_everything


def server_counts(engine):
    with engine.begin() as connection:
        return {t.name: connection.execute(select(func.count()).select_from(t)).scalar() for t in SYNCED}


def test_rows_arrive_once_however_often_they_are_sent(machine, engine):
    one = machine("one")
    save_everything(one)
    expected = one.counts()
    assert one.sync_now() and one.pending() == 0 and one.status_line() == "database: connected"
    assert server_counts(engine) == expected
    assert one.sync_now() and server_counts(engine) == expected
    with one.write() as connection:  # as if the connection dropped after the server took them
        one._queue(connection, "runs", [r["id"] for r in one.runs()], utcnow())
    assert one.sync_now() and server_counts(engine) == expected


def test_started_offline_then_the_network_comes_back(machine, engine, tmp_path):
    local_only = machine("laptop", remote=None)
    run_id = save_everything(local_only)
    assert local_only.sync_now() is False and local_only.pending() > 0
    local_only.close()
    later = machine("laptop")  # the same local file, now with an address
    assert later.sync_now() and server_counts(engine) == later.counts() and later.pending() == 0
    assert later.run(run_id)["verdict"] == "PASS"


def test_a_connection_that_drops_half_way_loses_and_doubles_nothing(machine, engine, monkeypatch):
    one = machine("one")
    save_everything(one)
    real, sent = one._send, []

    def flaky(table, rows):
        real(table, rows)
        sent.append(table.name)
        if len(sent) == 4:
            raise OSError("server closed the connection unexpectedly")

    monkeypatch.setattr(one, "_send", flaky)
    assert one.sync_now() is False and "can't be reached" in one.status_line() and one.pending() > 0
    monkeypatch.setattr(one, "_send", real)
    assert one.sync_now() and one.pending() == 0 and server_counts(engine) == one.counts()


def test_another_machine_sees_the_run(machine):
    one, two = machine("one"), machine("two")
    run_id = save_everything(one)
    one.set_weights(run_id, "https://huggingface.co/me/q-fairgate")
    assert one.sync_now() and two.sync_now()
    run = two.run(run_id)
    assert run["hub_id"] == MODEL and run["ran_by"] == "one" and run["artifact"]["recipe"] == PATCH
    assert run["artifact"]["weights_url"].endswith("q-fairgate") and len(run["events"]) == 2
    assert two.pending() == 0  # fetched rows are not sent back
    assert two.answers(run_id) == []  # 2,000 rows a run: only when asked for
    assert len(two.fetch_answers(run_id)) == 2 and len(two.answers(run_id)) == 2


def test_progress_from_another_machine_keeps_coming(machine):
    one, two = machine("one"), machine("two")
    one.start_run("r", MODEL)
    one.save_events("r", EVENTS[:1])
    assert one.sync_now() and two.sync_now() and len(two.run("r")["events"]) == 1
    one.save_events("r", EVENTS[1:], offset=1)
    one.finish_run("r", "done", report=REPORT, patch=PATCH)
    assert one.sync_now() and two.sync_now()
    assert len(two.run("r")["events"]) == 2 and two.run("r")["status"] == "done"


def test_the_newer_write_wins_whoever_sends_last(machine):
    one, two = machine("one"), machine("two")
    run_id = save_everything(one)
    assert one.sync_now() and two.sync_now()
    old, new = utcnow() + timedelta(minutes=1), utcnow() + timedelta(minutes=2)
    with one.write() as connection:
        one._put(connection, schema.runs, [{"id": run_id, "message": "newer"}], new)
    with two.write() as connection:
        two._put(connection, schema.runs, [{"id": run_id, "message": "older"}], old)
    assert one.sync_now() and two.sync_now() and one.sync_now() and two.sync_now()
    assert one.run(run_id)["message"] == two.run(run_id)["message"] == "newer"


def test_a_deleted_run_stays_deleted_everywhere(machine, engine):
    one, two = machine("one"), machine("two")
    run_id = save_everything(one)
    keep = save_everything(one, "22222222-2222-5222-8222-222222222222")
    assert one.sync_now() and two.sync_now()
    two.fetch_answers(run_id)
    two.finish_run(run_id, "done", message="edited while the other machine deletes it")
    assert one.delete_run(run_id) and one.sync_now()
    assert two.sync_now() and two.run(run_id) is None and two.answers(run_id) == []
    save_everything(two, run_id)  # its folder imported on the other machine
    assert two.sync_now() and one.sync_now()
    assert one.run(run_id) is None and two.run(run_id) is None and one.run(keep) and two.run(keep)
    counts = server_counts(engine)
    assert counts["runs"] == 1 and counts["run_answers"] == 2 and counts["deletions"] == 1


def test_a_row_changed_while_it_was_being_sent_is_sent_again(machine, engine, monkeypatch):
    one = machine("one")
    run_id = save_everything(one)
    real, changed = one._send, []

    def send(table, rows):
        real(table, rows)
        if table is schema.runs and not changed:
            changed.append(one.finish_run(run_id, "done", message="changed meanwhile"))

    monkeypatch.setattr(one, "_send", send)
    assert one.sync_now() and one.sync_now() and one.pending() == 0
    with engine.begin() as connection:
        assert connection.execute(select(schema.runs.c.message)).scalar() == "changed meanwhile"


def test_a_row_the_server_refuses_does_not_hold_up_the_rest(machine, engine, monkeypatch, capsys):
    from sqlalchemy.exc import DataError

    one = machine("one")
    save_everything(one, "good")
    save_everything(one, "bad")
    real = one._send

    def picky(table, rows):
        if table is schema.runs and any(r["id"] == "bad" for r in rows):
            raise DataError("insert", {}, ValueError("value not accepted: hf_secret123"))
        real(table, rows)

    monkeypatch.setattr(one, "_send", picky)
    assert one.sync_now() and server_counts(engine)["runs"] == 1 and one.pending() == 1
    assert "hf_secret123" not in capsys.readouterr().out and one.run("bad")


def test_many_rows_go_through_in_batches(machine, engine, monkeypatch):
    monkeypatch.setattr(store_module, "BATCH", 50)
    one, two = machine("one"), machine("two")
    one.start_run("r", MODEL)
    one.save_events("r", [{**EVENTS[0], "done": i} for i in range(333)])
    one.save_answers("r", {"ambig": {f"Age:{i}": ANSWERS["ambig"]["Age:1"] for i in range(170)}})
    assert one.sync_now() and one.pending() == 0 and server_counts(engine)["run_events"] == 333
    assert two.sync_now() and [e["done"] for e in two.run("r")["events"]] == list(range(333))
    assert len(two.fetch_answers("r")) == 170


def test_the_background_thread_sends_without_being_asked(machine, engine):
    one = machine("one")
    one.start()
    assert one.wait_settled(10) == "database: connected"
    save_everything(one)
    deadline = time.monotonic() + 15
    while one.pending() and time.monotonic() < deadline:
        time.sleep(0.1)
    assert one.pending() == 0 and server_counts(engine)["runs"] == 1
    one.close()


def test_closing_sends_what_is_left(machine, engine):
    one = machine("one")
    one.start()
    one.wait_settled(10)
    save_everything(one)
    one.close(10)
    assert server_counts(engine)["runs"] == 1


def test_a_write_never_waits_for_the_network(machine, monkeypatch):
    one = machine("one")
    monkeypatch.setattr(one, "_pull", lambda: time.sleep(3))
    one.start()
    started = time.monotonic()
    save_everything(one)
    assert one.runs() and time.monotonic() - started < 2
    one._stop.set()
    one.close(5)


def test_no_secret_reaches_the_server(machine, engine):
    one, two = machine("one"), machine("two")
    one.start_run("r", MODEL)
    one.save_events("r", [{**EVENTS[0], "message": "token hf_AbC123 refused"}])
    assert one.sync_now() and two.sync_now()
    for _, value in every_value(two):
        assert "hf_" not in value
