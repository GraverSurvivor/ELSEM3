"""The database's tables (contract C9): every diagnosis, its numbers, its repair, and the Ask history.

Records and links only, never model weights. One schema for SQLite (the local copy every screen reads)
and Postgres (the shared one). Every row has a UUID string key, chosen by the writer, so the same row
written twice -- by an import, or by a retry after a dropped connection -- is the same row.

There are no foreign-key constraints: rows reach Postgres from several machines, each with its own
outbox, so a run's progress line may arrive before the run. `Store.delete_run` removes a run's rows
from every table in `RUN_TABLES` instead.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    select,
    update,
)
from sqlalchemy.types import TypeDecorator

SCHEMA_VERSION = 1

metadata = MetaData()        # on both sides
local_metadata = MetaData()  # only in the local SQLite file: what still has to reach Postgres


class UTCDateTime(TypeDecorator):
    """Aware UTC datetimes in Python, plain UTC in the database (SQLite keeps no time zone)."""

    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError("times must carry a time zone")
        return value.astimezone(timezone.utc).replace(tzinfo=None)

    def process_result_value(self, value, dialect):
        return None if value is None else value.replace(tzinfo=timezone.utc)


def _json(name: str) -> Column:
    return Column(name, JSON(none_as_null=True))


def _table(name: str, *columns) -> Table:
    return Table(name, metadata, Column("id", String(36), primary_key=True), *columns,
                 # When the row's content last changed, by its writer's clock: the newer write wins.
                 Column("updated_at", UTCDateTime, nullable=False),
                 # When Postgres received it, by the server's clock: what another machine fetches by.
                 Column("received_at", UTCDateTime, index=True))


def _run_id() -> Column:
    return Column("run_id", String(36), nullable=False, index=True)


models = _table(
    "models",
    Column("hub_id", String(300), nullable=False, index=True),  # the Hugging Face id
    Column("revision", String(64)),                             # its commit, once known
    Column("family", String(64)),
    Column("weights", Float),                                   # weights in the parts a repair can restore
)

runs = _table(
    "runs",
    Column("model_id", String(36), nullable=False, index=True),
    Column("job", String(200), index=True),         # the job folder's name
    Column("source", String(20), nullable=False),   # studio | import | reference
    Column("pipeline", String(20)),                 # real | fake | replay
    Column("method", String(40)),                   # hqq
    Column("base_bits", Integer),                   # what the model was compressed to
    Column("repair_bits", Integer),                 # what its restored parts are kept at
    Column("budget", Float),
    Column("mode", String(20)),
    Column("questions", Integer),
    _json("versions"),
    Column("gpu", String(100)),
    Column("status", String(20), nullable=False),   # running | done | failed | cancelled | stopped
    Column("message", Text),                        # why it failed, in plain words
    Column("outcome", String(30), index=True),      # repaired | rejected | no_gain | no_damage | not_applicable
    Column("outcome_reason", Text),
    Column("verdict", String(10)),                  # PASS | FAIL | N/A
    Column("repair_verdict", String(20)),
    Column("recovery", Float),
    Column("recovery_lower", Float),
    Column("recovery_upper", Float),
    Column("extra_mb", Float),
    Column("restored_share", Float),
    Column("minutes", Float),
    Column("started_at", UTCDateTime, index=True),
    Column("ended_at", UTCDateTime),
    Column("ran_by", String(100)),
    _json("report"),                                # report.json, so the full page can be shown again
)

run_events = _table(
    "run_events", _run_id(),
    Column("position", Integer, nullable=False),
    Column("time", String(40)),                     # as the pipeline wrote it
    Column("stage", String(40)),
    Column("status", String(20)),
    Column("done", Integer),
    Column("total", Integer),
    Column("message", Text),
)

metrics = _table(
    "metrics", _run_id(),
    Column("view", String(20), nullable=False),     # original | compressed | repaired
    Column("metric", String(40), nullable=False),   # usr | srs | false_unknown | accuracy | recovery
    Column("scope", String(20), nullable=False),    # overall | category | group | letter_balanced
    Column("name", String(100)),                    # which category or group
    Column("value", Float),
    Column("lower", Float),
    Column("upper", Float),
)

strategies = _table(
    "strategies", _run_id(),
    Column("position", Integer, nullable=False),
    Column("name", String(100), nullable=False),
    Column("extra_mb", Float),
    Column("n_parts", Integer),
    _json("validation"),
    Column("chosen", Boolean),
)

parts = _table(
    "parts", _run_id(),
    Column("position", Integer, nullable=False),    # the order it was kept in
    Column("path", String(300), nullable=False),
    Column("bits", Integer),
)

run_answers = _table(
    "run_answers", _run_id(),
    Column("question_id", String(100), nullable=False),
    Column("question_set", String(20), nullable=False),  # ambig | stated
    Column("split", String(20)),                         # search | final (and what v2 adds)
    Column("original", Integer),
    Column("compressed", Integer),
    Column("repaired", Integer),
)

artifacts = _table(
    "artifacts", _run_id(),
    _json("recipe"),                                # patch.json: enough to rebuild the model
    Column("recipe_sha256", String(64)),
    Column("kit_name", String(200)),
    Column("kit_sha256", String(64)),
    Column("weights_url", String(400)),
    Column("weights_private", Boolean),
    _json("versions"),                              # libraries the repair was measured with
    _json("fingerprint"),                           # 50 questions and the repaired model's picks
)

questions = _table(
    "questions",
    Column("run_id", String(36), index=True),       # the model asked; null = the Phi-3.5 demo
    Column("model_name", String(300)),
    Column("story", Text),
    Column("question", Text),
    Column("person_a", String(200)),
    Column("person_b", String(200)),
    Column("asked_at", UTCDateTime, index=True),
    Column("asked_by", String(100)),
)

question_answers = _table(
    "question_answers",
    Column("question_row_id", String(36), nullable=False, index=True),
    Column("view", String(20), nullable=False),
    Column("pick", Integer),
    Column("answer", Text),
    Column("consistency", Text),                    # how many of the 6 answer orders agreed, in words
    _json("probs"),
)

api_sessions = _table(
    "api_sessions",
    Column("run_id", String(36), index=True),
    Column("model_name", String(300)),
    Column("started_at", UTCDateTime),
    Column("stopped_at", UTCDateTime),
    Column("requests", Integer),
    Column("key_sha256", String(64)),               # the hash only, never the key
)

# A deleted row's id, so another machine's copy (or a second import) can't bring it back.
deletions = _table("deletions", Column("table_name", String(40), nullable=False),
                   Column("deleted_at", UTCDateTime, nullable=False))

schema_version = Table("schema_version", metadata, Column("id", Integer, primary_key=True),
                       Column("version", Integer, nullable=False))

# Local only. One row per (table, row) still to be pushed: a second write before the push replaces the
# first, and `token` lets the pusher notice a row that changed while it was being sent.
outbox = Table("outbox", local_metadata, Column("table_name", String(40), primary_key=True),
               Column("row_id", String(36), primary_key=True), Column("token", String(36), nullable=False),
               Column("queued_at", UTCDateTime, nullable=False))
sync_state = Table("sync_state", local_metadata, Column("table_name", String(40), primary_key=True),
                   Column("pulled_to", UTCDateTime))

# Synced tables. `deletions` leads, so a pull learns what is gone before it adds anything.
SYNCED = (deletions, models, runs, run_events, metrics, strategies, parts, run_answers, artifacts, questions,
          question_answers, api_sessions)
BY_NAME = {table.name: table for table in SYNCED}
RUN_TABLES = (run_events, metrics, strategies, parts, run_answers, artifacts)  # removed with their run


class SchemaTooNew(RuntimeError):
    """The database was made by a newer version of this code."""


# version n -> what to run on a database at version n - 1 (a function of the connection).
MIGRATIONS: dict = {}


def dialect_insert(connection, table):
    """INSERT that knows ON CONFLICT, for the two databases this project uses."""
    if connection.dialect.name == "postgresql":
        from sqlalchemy.dialects.postgresql import insert
    else:
        from sqlalchemy.dialects.sqlite import insert
    return insert(table)


def upgrade(engine, local: bool = False) -> int:
    """Creates what is missing and brings an older database up to date. Returns its version."""
    metadata.create_all(engine)
    if local:
        local_metadata.create_all(engine)
    with engine.begin() as connection:
        found = connection.execute(select(schema_version.c.version).where(schema_version.c.id == 1)).scalar()
        if found is None:
            # Two apps can start on an empty database at once: the second one's insert is ignored.
            connection.execute(dialect_insert(connection, schema_version)
                               .values(id=1, version=SCHEMA_VERSION).on_conflict_do_nothing())
            return SCHEMA_VERSION
        if found > SCHEMA_VERSION:
            raise SchemaTooNew(f"The database is at version {found} and this code knows version {SCHEMA_VERSION}: "
                               "update the code.")
        for version in range(found + 1, SCHEMA_VERSION + 1):
            MIGRATIONS[version](connection)
            connection.execute(update(schema_version).where(schema_version.c.id == 1).values(version=version))
    return SCHEMA_VERSION


def utcnow() -> datetime:
    return datetime.now(timezone.utc)
