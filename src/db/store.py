"""The store (contract C9): what the Studio saves and reads.

    store = open_store()                 # DATABASE_URL if set, else local only
    store.start_run(run_id, "Qwen/Qwen2.5-3B-Instruct", mode="fast")
    store.save_events(run_id, events)
    store.finish_run(run_id, "done", report=report, patch=patch)
    store.runs(search="qwen")            # for a screen: local, never waits on the network

Two databases, one set of tables (src/db/schema.py). Every write goes to a local SQLite file, together
with a line in its `outbox`, in one transaction. A background thread sends the outbox to Postgres
(upserts by id, so sending twice is harmless) and fetches what other machines wrote. Every read comes
from the local file. A dropped connection therefore delays rows and stops nothing; without a
DATABASE_URL, or without a network, the app works the same and the outbox waits.

The database's address holds its password: it is never printed, and errors are scrubbed (src/db/scrub.py).
"""

from __future__ import annotations

import json
import math
import os
import threading
import uuid
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path

from sqlalchemy import Boolean, Float, Integer, String, Text, and_, bindparam, create_engine, delete, event, func, literal_column, or_, select, tuple_
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.exc import DataError, IntegrityError, StatementError

from src.db import records
from src.db.schema import (
    BY_NAME,
    RUN_TABLES,
    SYNCED,
    SchemaTooNew,
    api_sessions,
    artifacts,
    deletions,
    dialect_insert,
    metrics,
    models,
    outbox,
    parts,
    question_answers,
    questions,
    run_answers,
    run_events,
    runs,
    strategies,
    sync_state,
    upgrade,
    utcnow,
)
from src.db.scrub import safe_error, scrub

DEFAULT_PATH = Path("results/db/fairgate.sqlite")
BATCH = 500                      # rows sent or fetched at a time
SYNC_EVERY = 20.0                # seconds between looks at the server
OVERLAP = timedelta(minutes=2)   # re-read this much: a row can be committed after a later-stamped one
# Not fetched from other machines (2,000 rows a run, and no screen lists them): `fetch_answers` asks for one run's.
NOT_PULLED = (run_answers,)


# --- engines ---------------------------------------------------------------------------------------

def sqlite_engine(path: Path) -> Engine:
    """A local file shared by the app's threads and by a second process: write-ahead log, and a writer
    waits its turn (BEGIN IMMEDIATE takes the write lock first; a read that turns into a write can't
    wait, and would fail with "database is locked")."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{path}", connect_args={"timeout": 10}, hide_parameters=True)

    @event.listens_for(engine, "connect")
    def _connect(connection, _record):
        connection.isolation_level = None  # transactions are begun below, not by the driver
        for pragma in ("journal_mode=WAL", "busy_timeout=10000", "synchronous=NORMAL"):
            connection.execute(f"PRAGMA {pragma}")

    @event.listens_for(engine, "begin")
    def _begin(connection):
        write = connection.get_execution_options().get("fairgate_write")
        connection.exec_driver_sql("BEGIN IMMEDIATE" if write else "BEGIN")

    return engine


def postgres_engine(url: str) -> Engine:
    """The shared database. `pool_pre_ping`: free servers go to sleep and drop connections. No prepared
    statements: pooled addresses (pgbouncer) hand each transaction to a different server connection."""
    parsed = make_url(url)
    if parsed.drivername in ("postgres", "postgresql"):
        parsed = parsed.set(drivername="postgresql+psycopg")
    if parsed.drivername != "postgresql+psycopg":
        raise ValueError("not a Postgres address")
    if parsed.host not in (None, "localhost", "127.0.0.1", "::1") and "sslmode" not in parsed.query:
        parsed = parsed.update_query_dict({"sslmode": "require"})
    return create_engine(parsed, pool_pre_ping=True, pool_size=2, max_overflow=1, pool_recycle=300,
                         hide_parameters=True, connect_args={"connect_timeout": 10, "prepare_threshold": None})


def open_store(url: str | None = None, path: Path | str | None = None, ran_by: str | None = None,
               start: bool = True) -> "Store":
    """The app's store. `url` defaults to the DATABASE_URL environment variable; without one (or with one
    that isn't a Postgres address) the store is local only. Never waits on the network."""
    url = os.environ.get("DATABASE_URL", "") if url is None else url
    remote, problem = None, ""
    if url.strip():
        try:
            remote = postgres_engine(url.strip())
        except ImportError:
            problem = "the Postgres driver isn't available: pip install \"psycopg[binary]\""
        except Exception:  # noqa: BLE001 -- the message could quote the address
            problem = "DATABASE_URL isn't a Postgres address this app can use"
    store = Store(sqlite_engine(Path(path or DEFAULT_PATH)), remote, ran_by=ran_by, problem=problem)
    if start:
        store.start()
    return store


def who_runs() -> str:
    """Who ran it: FAIRGATE_USER if set, else the machine's user name. Never an e-mail address."""
    name = os.environ.get("FAIRGATE_USER", "").strip()
    if not name:
        import getpass

        try:
            name = getpass.getuser()
        except Exception:  # noqa: BLE001 -- no user name on this machine
            name = ""
        name = {"root": "colab", "": "unknown"}.get(name, name)
    return name.split("@")[0][:100]


# --- the store -------------------------------------------------------------------------------------

class Store:
    def __init__(self, local: Engine, remote: Engine | None = None, ran_by: str | None = None, problem: str = ""):
        self.local, self.remote = local, remote
        self.ran_by = ran_by or who_runs()
        self.problem = problem
        self.state = "connecting" if remote is not None else "local only"
        self._write_lock = threading.RLock()
        self._sync_lock = threading.Lock()
        self._wake, self._stop, self._settled = threading.Event(), threading.Event(), threading.Event()
        self._thread: threading.Thread | None = None
        self._refused: set = set()    # (table, id) the server will never take: skipped for this session
        self._remote_ready = False
        self._last_message = ""
        upgrade(local, local=True)

    # connection ------------------------------------------------------------------------------------

    def status_line(self) -> str:
        if self.state == "connected":
            return "database: connected"
        if self.remote is None:
            return "database: local only" + (f" ({self.problem})" if self.problem else "")
        if self.state == "connecting":
            return "database: local only (connecting to the shared database)"
        why = f": {self.problem}" if self.problem else ", will keep trying"
        return f"database: local only (the shared database can't be reached{why})"

    def start(self) -> None:
        if self.remote is None or self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True, name="fairgate-db-sync")
        self._thread.start()

    def wait_settled(self, timeout: float = 15.0) -> str:
        """For start-up only: until the first try to connect has ended. Returns the status line."""
        if self.remote is not None and self._thread is not None:
            self._settled.wait(timeout)
        return self.status_line()

    def close(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout)
        self.local.dispose()
        if self.remote is not None:
            self.remote.dispose()

    def _loop(self) -> None:
        wait, retry = 0.0, 5.0
        while not self._stop.is_set():
            self._wake.wait(wait)
            self._wake.clear()
            ok = self.sync_now()
            self._settled.set()
            if ok:
                wait, retry = SYNC_EVERY, 5.0
            else:  # a write must not hammer a server that is down
                self._stop.wait(retry)
                wait, retry = 0.0, min(retry * 2, 120.0)
        # One last try, so what was written just before closing arrives.
        if self.state == "connected":
            self.sync_now()

    def sync_now(self) -> bool:
        """Fetches, then sends. True if the server was reached. Never raises."""
        if self.remote is None:
            return False
        with self._sync_lock:
            try:
                if not self._remote_ready:
                    upgrade(self.remote)
                    self._remote_ready = True
                self._pull()
                self._push()
                self.state, self.problem = "connected", ""
                self._last_message = ""
                return True
            except SchemaTooNew as error:
                self.state, self.problem = "offline", str(error)
                self._say(str(error))
            except Exception as error:  # noqa: BLE001 -- the app goes on; the outbox keeps the rows
                self.state, self.problem = "offline", ""
                self._say(safe_error(error))
            return False

    def _say(self, message: str) -> None:
        if message != self._last_message:  # once, not every few seconds
            print(f"[database] shared database not reached ({message}); working locally, will retry", flush=True)
            self._last_message = message

    def pending(self) -> int:
        """Rows still to be sent."""
        with self.read() as connection:
            return connection.execute(select(func.count()).select_from(outbox)).scalar()

    # transactions ----------------------------------------------------------------------------------

    @contextmanager
    def read(self):
        with self.local.begin() as connection:
            yield connection

    @contextmanager
    def write(self):
        with self._write_lock, self.local.execution_options(fairgate_write=True).begin() as connection:
            yield connection

    # writing ---------------------------------------------------------------------------------------

    def _clean(self, table, row: dict) -> dict:
        out = {}
        for key, value in row.items():
            column = table.c[key]
            if key not in ("id", "run_id", "model_id", "question_row_id"):
                value = scrub(value)
            if column.type.__class__.__name__ == "JSON":
                value = _plain_json(value)
            elif isinstance(column.type, String) and not isinstance(column.type, Text) and column.type.length:
                value = None if value is None else str(value)[:column.type.length]
            elif isinstance(column.type, Text):
                value = None if value is None else str(value)
            elif isinstance(column.type, (Integer, Float)):
                value = _number(value, whole=isinstance(column.type, Integer))
            elif isinstance(column.type, Boolean):
                value = None if value is None else bool(value)
            out[key] = value
        return out

    def _put(self, connection, table, rows: list[dict], moment=None) -> int:
        """Inserts or updates rows (only the given columns) and queues the ones that changed. A row that
        is already exactly that, or was deleted, is left alone. Returns how many changed."""
        rows = [self._clean(table, row) for row in rows]
        moment = moment or utcnow()
        changed = []
        for chunk in _chunks(rows, BATCH):
            ids = [row["id"] for row in chunk]
            gone = self._deleted(connection, ids + [row["run_id"] for row in chunk if row.get("run_id")])
            have = {r["id"]: r for r in connection.execute(select(table).where(table.c.id.in_(ids))).mappings()}
            for row in chunk:
                if row["id"] in gone or row.get("run_id") in gone:
                    continue
                old = have.get(row["id"])
                if old is not None and all(_same(old[key], value) for key, value in row.items()):
                    continue
                changed.append({**row, "updated_at": moment})
        for group in _by_columns(changed):
            _upsert(connection, table, group)
        self._queue(connection, table.name, [row["id"] for row in changed], moment)
        return len(changed)

    @staticmethod
    def _deleted(connection, ids: list[str]) -> set:
        ids = list(dict.fromkeys(ids))
        found = set()
        for chunk in _chunks(ids, BATCH):
            found.update(connection.execute(select(deletions.c.id).where(deletions.c.id.in_(chunk))).scalars())
        return found

    @staticmethod
    def _queue(connection, name: str, ids: list[str], moment) -> None:
        if not ids:
            return
        rows = [{"table_name": name, "row_id": i, "token": str(uuid.uuid4()), "queued_at": moment} for i in ids]
        statement = dialect_insert(connection, outbox)
        connection.execute(statement.on_conflict_do_update(
            index_elements=["table_name", "row_id"],
            set_={"token": statement.excluded.token, "queued_at": statement.excluded.queued_at}), rows)

    def _save(self, *writes) -> int:
        """Several tables' rows in one transaction; then the sync thread is told."""
        with self.write() as connection:
            changed = sum(self._put(connection, table, rows) for table, rows in writes if rows)
        if changed:
            self._wake.set()
        return changed

    # runs ------------------------------------------------------------------------------------------

    def start_run(self, run_id: str, hub_id: str, *, job: str | None = None, source: str = "studio",
                  pipeline: str = "real", method: str = "hqq", base_bits: int | None = None,
                  budget: float | None = None, mode: str | None = None, questions: int | None = None,
                  versions: dict | None = None, started_at=None, ran_by: str | None = None) -> int:
        if self.run_row(run_id):
            return 0  # known already: its status and times are not started over
        model = records.model_row(hub_id)
        run = {"id": run_id, "model_id": model["id"], "job": job, "source": source, "pipeline": pipeline,
               "method": method, "base_bits": base_bits or None, "budget": budget, "mode": mode,
               "questions": questions, "versions": versions, "status": "running",
               "started_at": started_at or utcnow(), "ran_by": ran_by or self.ran_by}
        return self._save((models, [model]), (runs, [run]))

    def save_events(self, run_id: str, events: list[dict], offset: int = 0) -> int:
        """Progress lines, `offset` being the first one's place in the whole list."""
        return self._save((run_events, records.event_rows(run_id, events, offset)))

    def finish_run(self, run_id: str, status: str, *, report: dict | None = None, patch: dict | None = None,
                   hub_id: str | None = None, message: str = "", ended_at=None, kit: tuple | None = None,
                   versions: dict | None = None) -> int:
        """The run's end: its status and, if it got that far, its report and repair. `kit` is the
        repair kit's (file name, SHA-256)."""
        run = {"id": run_id, "status": status, "message": message or None}
        if ended_at is not None:
            run["ended_at"] = ended_at
        writes = []
        if report is not None:
            run.update(records.run_columns(report))
            writes += [(metrics, records.metric_rows(run_id, report)),
                       (strategies, records.strategy_rows(run_id, report)),
                       (parts, records.part_rows(run_id, report.get("kept") or [], patch))]
        if patch is not None:
            run["repair_bits"] = int(patch["nbits"])
            hub_id = hub_id or patch.get("model_id")
        if versions is None:
            versions = self.run_row(run_id).get("versions")
        if patch is not None:
            writes.append((artifacts, [records.artifact_row(run_id, patch, report, versions, kit)]))
        if hub_id:
            model = records.model_row(hub_id, (patch or {}).get("revision"), report, patch)
            run["model_id"] = model["id"]
            writes.append((models, [model]))
        return self._save((runs, [run]), *writes)

    def save_answers(self, run_id: str, answers: dict) -> int:
        return self._save((run_answers, records.answer_rows(run_id, answers)))

    def save_rows(self, table, rows: list[dict]) -> int:
        """For the importer: rows already in a table's shape."""
        return self._save((table, rows))

    def set_weights(self, run_id: str, url: str, private: bool = True) -> int:
        return self._save((artifacts, [{"id": records.stable_id(run_id, "artifact"), "run_id": run_id,
                                        "weights_url": url, "weights_private": bool(private)}]))

    def delete_run(self, run_id: str) -> bool:
        """Removes a run and everything recorded about it, here and (through the outbox) on the server.
        Nothing on Hugging Face is touched."""
        with self.write() as connection:
            if connection.execute(select(runs.c.id).where(runs.c.id == run_id)).first() is None:
                return False
            _drop_run(connection, run_id)
            connection.execute(delete(outbox).where(outbox.c.row_id == run_id, outbox.c.table_name == "runs"))
            moment = utcnow()
            _upsert(connection, deletions, [{"id": run_id, "table_name": "runs", "deleted_at": moment,
                                             "updated_at": moment}])
            self._queue(connection, "deletions", [run_id], moment)
        self._wake.set()
        return True

    # Ask and API history ---------------------------------------------------------------------------

    def save_question(self, story: str, question: str, person_a: str, person_b: str, answers: dict[str, dict],
                      *, run_id: str | None = None, model_name: str = "") -> str:
        """A typed question and what each model answered: view -> {"pick", "answer", "consistency", "probs"}."""
        row_id = str(uuid.uuid4())
        row = {"id": row_id, "run_id": run_id, "model_name": model_name, "story": story, "question": question,
               "person_a": person_a, "person_b": person_b, "asked_at": utcnow(), "asked_by": self.ran_by}
        rows = [{"id": records.stable_id(row_id, view), "question_row_id": row_id, "view": view,
                 "pick": answer.get("pick"), "answer": answer.get("answer"),
                 "consistency": answer.get("consistency"), "probs": answer.get("probs")}
                for view, answer in answers.items()]
        self._save((questions, [row]), (question_answers, rows))
        return row_id

    def start_api_session(self, key: str, model_name: str, run_id: str | None = None) -> str:
        session_id = str(uuid.uuid4())
        self._save((api_sessions, [{"id": session_id, "run_id": run_id, "model_name": model_name,
                                    "started_at": utcnow(), "requests": 0, "key_sha256": records.sha256(key)}]))
        return session_id

    def end_api_session(self, session_id: str, requests: int) -> None:
        self._save((api_sessions, [{"id": session_id, "requests": int(requests), "stopped_at": utcnow()}]))

    # reading (local only) --------------------------------------------------------------------------

    def run_row(self, run_id: str) -> dict:
        with self.read() as connection:
            row = connection.execute(select(runs).where(runs.c.id == run_id)).mappings().first()
        return dict(row) if row else {}

    def runs(self, search: str = "", outcome: str | None = None, method: str | None = None,
             limit: int = 500) -> list[dict]:
        """Newest first. `search` looks at the model, who ran it, the verdict and the job."""
        query = (select(runs, models.c.hub_id, models.c.revision, models.c.family, artifacts.c.weights_url)
                 .join(models, models.c.id == runs.c.model_id, isouter=True)
                 .join(artifacts, artifacts.c.run_id == runs.c.id, isouter=True))
        for word in search.split():
            pattern = "%" + word.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
            query = query.where(or_(*(column.ilike(pattern, escape="\\") for column in (
                models.c.hub_id, models.c.family, runs.c.ran_by, runs.c.verdict, runs.c.outcome, runs.c.job,
                runs.c.method))))
        if outcome:
            query = query.where(or_(runs.c.outcome == outcome, and_(runs.c.outcome.is_(None), runs.c.status == outcome)))
        if method:
            query = query.where(runs.c.method == method)
        query = query.order_by(runs.c.started_at.desc(), runs.c.id).limit(limit)
        with self.read() as connection:
            return [dict(row) for row in connection.execute(query).mappings()]

    def run(self, run_id: str) -> dict | None:
        """One run with everything its page shows."""
        with self.read() as connection:
            row = connection.execute(
                select(runs, models.c.hub_id, models.c.revision, models.c.family, models.c.weights)
                .join(models, models.c.id == runs.c.model_id, isouter=True).where(runs.c.id == run_id)).mappings().first()
            if row is None:
                return None

            def children(table, order):
                return [dict(r) for r in connection.execute(
                    select(table).where(table.c.run_id == run_id).order_by(*order)).mappings()]

            out = dict(row)
            out["events"] = children(run_events, [run_events.c.position])
            out["metrics"] = children(metrics, [metrics.c.view, metrics.c.metric, metrics.c.scope, metrics.c.name])
            out["strategies"] = children(strategies, [strategies.c.position])
            out["parts"] = children(parts, [parts.c.position])
            artifact = connection.execute(select(artifacts).where(artifacts.c.run_id == run_id)).mappings().first()
            out["artifact"] = dict(artifact) if artifact else None
        return out

    def last_news(self, run_ids: list[str]) -> dict:
        """run id -> when its newest progress line was written."""
        found = {}
        with self.read() as connection:
            for chunk in _chunks(list(run_ids), BATCH):
                found.update(connection.execute(
                    select(run_events.c.run_id, func.max(run_events.c.updated_at))
                    .where(run_events.c.run_id.in_(chunk)).group_by(run_events.c.run_id)).all())
        return found

    def answers(self, run_id: str) -> list[dict]:
        with self.read() as connection:
            return [dict(r) for r in connection.execute(
                select(run_answers).where(run_answers.c.run_id == run_id).order_by(run_answers.c.id)).mappings()]

    def fetch_answers(self, run_id: str) -> list[dict]:
        """One run's answers, from the server if this machine hasn't got them. Waits on the network: for a
        button press, never for the timer."""
        have = self.answers(run_id)
        if have or self.remote is None or self.state != "connected":
            return have
        try:
            with self.remote.begin() as connection:
                rows = [dict(r) for r in connection.execute(
                    select(run_answers).where(run_answers.c.run_id == run_id)).mappings()]
        except Exception as error:  # noqa: BLE001
            self._say(safe_error(error))
            return have
        with self.write() as connection:
            self._take(connection, run_answers, rows)
        return self.answers(run_id)

    def recent_questions(self, limit: int = 50) -> list[dict]:
        with self.read() as connection:
            rows = [dict(r) for r in connection.execute(
                select(questions).order_by(questions.c.asked_at.desc()).limit(limit)).mappings()]
            for row in rows:
                row["answers"] = {r["view"]: dict(r) for r in connection.execute(
                    select(question_answers).where(question_answers.c.question_row_id == row["id"])).mappings()}
        return rows

    def methods(self) -> list[str]:
        with self.read() as connection:
            return sorted(m for m in connection.execute(select(runs.c.method).distinct()).scalars() if m)

    def counts(self) -> dict[str, int]:
        with self.read() as connection:
            return {t.name: connection.execute(select(func.count()).select_from(t)).scalar() for t in SYNCED}

    def backup(self, target: Path) -> Path:
        """A copy of the local file that is safe to put on Google Drive (SQLite's own backup, which never
        copies a half-written page; the live file itself must stay on the machine's disk)."""
        import sqlite3

        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        with self.local.connect() as connection:
            with sqlite3.connect(target) as out:
                connection.connection.driver_connection.backup(out)
            out.close()
        return target

    # syncing ---------------------------------------------------------------------------------------

    def _push(self) -> None:
        for table in SYNCED:
            while True:
                with self.read() as connection:
                    entries = connection.execute(
                        select(outbox.c.row_id, outbox.c.token).where(outbox.c.table_name == table.name)
                        .order_by(outbox.c.queued_at)).all()
                    entries = [e for e in entries if (table.name, e.row_id) not in self._refused][:BATCH]
                    ids = [e.row_id for e in entries]
                    rows = [dict(r) for r in connection.execute(
                        select(table).where(table.c.id.in_(ids))).mappings()] if ids else []
                if not entries:
                    break
                try:
                    self._send(table, rows)
                except (DataError, IntegrityError, StatementError) as error:
                    if getattr(error, "connection_invalidated", False):
                        raise
                    self._send_one_by_one(table, rows)
                refused = {i for name, i in self._refused if name == table.name}
                with self.write() as connection:
                    for entry in entries:
                        if entry.row_id not in refused:  # unless it changed again meanwhile: then it stays queued
                            connection.execute(delete(outbox).where(
                                outbox.c.table_name == table.name, outbox.c.row_id == entry.row_id,
                                outbox.c.token == entry.token))

    def _send(self, table, rows: list[dict]) -> None:
        if not rows:
            return
        with self.remote.begin() as connection:
            gone = self._deleted(connection, [r["id"] for r in rows] + [r["run_id"] for r in rows if r.get("run_id")])
            if table is deletions:
                for row in rows:
                    _drop_run(connection, row["id"])
            else:
                rows = [r for r in rows if r["id"] not in gone and r.get("run_id") not in gone]
            moment = _server_time(connection)
            rows = [{k: v for k, v in r.items() if k != "received_at"} for r in rows]
            if rows:
                statement = dialect_insert(connection, table)
                newer = {k: statement.excluded[k] for k in rows[0] if k != "id"}
                connection.execute(statement.values(received_at=moment).on_conflict_do_update(
                    index_elements=["id"], set_={**newer, "received_at": moment},
                    where=statement.excluded.updated_at >= table.c.updated_at), rows)

    def _send_one_by_one(self, table, rows: list[dict]) -> None:
        for row in rows:
            try:
                self._send(table, [row])
            except (DataError, IntegrityError, StatementError) as error:
                if getattr(error, "connection_invalidated", False):
                    raise
                self._refused.add((table.name, row["id"]))  # kept locally; the rest goes on
                print(f"[database] a row of {table.name} was not accepted ({safe_error(error)})", flush=True)

    def _pull(self) -> None:
        for table in SYNCED:
            if table in NOT_PULLED:
                continue
            with self.read() as connection:
                since = connection.execute(
                    select(sync_state.c.pulled_to).where(sync_state.c.table_name == table.name)).scalar()
            cursor, latest = None, since
            while True:
                query = select(table).where(table.c.received_at.is_not(None))
                if cursor is not None:
                    query = query.where(tuple_(table.c.received_at, table.c.id) > tuple_(*cursor))
                elif since is not None:
                    query = query.where(table.c.received_at >= since - OVERLAP)
                with self.remote.begin() as connection:
                    rows = [dict(r) for r in connection.execute(
                        query.order_by(table.c.received_at, table.c.id).limit(BATCH)).mappings()]
                if not rows:
                    break
                cursor = (_bound(table, rows[-1]["received_at"]), rows[-1]["id"])
                latest = max(filter(None, (latest, rows[-1]["received_at"])))
                with self.write() as connection:
                    self._take(connection, table, rows)
                if len(rows) < BATCH:
                    break
            if latest is not None and latest != since:
                with self.write() as connection:
                    statement = dialect_insert(connection, sync_state)
                    connection.execute(statement.values(table_name=table.name, pulled_to=latest)
                                       .on_conflict_do_update(index_elements=["table_name"],
                                                              set_={"pulled_to": latest}))

    def _take(self, connection, table, rows: list[dict]) -> None:
        """Rows from the server into the local file. A row this machine still has to send is kept."""
        if not rows:
            return
        ids = [r["id"] for r in rows]
        if table is deletions:
            for row in rows:
                _drop_run(connection, row["id"])
                connection.execute(delete(outbox).where(outbox.c.table_name == "runs", outbox.c.row_id == row["id"]))
        waiting = set(connection.execute(select(outbox.c.row_id).where(
            outbox.c.table_name == table.name, outbox.c.row_id.in_(ids))).scalars())
        have = dict(connection.execute(select(table.c.id, table.c.updated_at).where(table.c.id.in_(ids))).all())
        gone = set() if table is deletions else self._deleted(
            connection, ids + [r["run_id"] for r in rows if r.get("run_id")])
        fresh = [r for r in rows if r["id"] not in waiting and r["id"] not in gone and r.get("run_id") not in gone
                 and (r["id"] not in have or r["updated_at"] > have[r["id"]])]
        _upsert(connection, table, fresh)


# --- helpers ---------------------------------------------------------------------------------------

def _chunks(items: list, size: int):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def _by_columns(rows: list[dict]) -> list[list[dict]]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault(tuple(sorted(row)), []).append(row)
    return list(groups.values())


def _upsert(connection, table, rows: list[dict]) -> None:
    """Rows with the same columns: inserted, or those columns updated. A row with only some of a table's
    columns (a run's new status) updates an existing row and is not inserted on its own."""
    if not rows:
        return
    required = {c.name for c in table.columns if not c.nullable}
    if not required <= set(rows[0]):
        for row in rows:
            values = {k: v for k, v in row.items() if k != "id"}
            connection.execute(table.update().where(table.c.id == row["id"]).values(**values))
        return
    statement = dialect_insert(connection, table)
    connection.execute(statement.on_conflict_do_update(
        index_elements=["id"], set_={k: statement.excluded[k] for k in rows[0] if k != "id"}), rows)


def _drop_run(connection, run_id: str) -> None:
    for table in RUN_TABLES:
        connection.execute(delete(table).where(table.c.run_id == run_id))
    connection.execute(delete(runs).where(runs.c.id == run_id))


def _server_time(connection):
    """The server's own clock, in UTC: machines' clocks differ, and fetching goes by this one."""
    if connection.dialect.name == "postgresql":
        return func.timezone("UTC", func.clock_timestamp())
    return literal_column("strftime('%Y-%m-%d %H:%M:%f', 'now') || '000'")  # microseconds, as DateTime is stored


def _bound(table, moment):
    return bindparam(None, moment, type_=table.c.received_at.type)


def _plain_json(value):
    """What the JSON column will hold: plain types, and no NaN (Postgres refuses it)."""
    if value is None:
        return None

    def finite(v):
        if isinstance(v, float) and not math.isfinite(v):
            return None
        if isinstance(v, dict):
            return {str(k): finite(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [finite(x) for x in v]
        return v

    return json.loads(json.dumps(finite(value), default=str))


def _number(value, whole: bool):
    """A number the column can hold, or None: SQLite would keep anything, Postgres refuses the row."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return (int(value) if abs(value) < 2**31 else None) if whole else float(value)


def _same(old, new) -> bool:
    if isinstance(old, float) and isinstance(new, (int, float)) and not isinstance(new, bool):
        return math.isclose(old, new, rel_tol=1e-12, abs_tol=0.0) or old == new
    return old == new


__all__ = ["Store", "open_store", "sqlite_engine", "postgres_engine", "who_runs", "BY_NAME"]
