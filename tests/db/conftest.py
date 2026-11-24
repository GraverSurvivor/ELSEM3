"""Databases for the store tests: SQLite always, and Postgres in a throw-away schema when
TEST_DATABASE_URL is set (e.g. `TEST_DATABASE_URL=postgresql://... pytest tests/db`)."""

import os
import uuid

import pytest

pytest.importorskip("sqlalchemy")

from sqlalchemy import text  # noqa: E402

from src.db.store import Store, postgres_engine, sqlite_engine  # noqa: E402


@pytest.fixture
def pg_schema():
    """A Postgres engine that sees only a new, empty schema; dropped afterwards."""
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL is not set")
    engine = postgres_engine(url)
    name = "fairgate_test_" + uuid.uuid4().hex[:12]
    with engine.begin() as connection:
        connection.execute(text(f"CREATE SCHEMA {name}"))
    try:
        yield engine.execution_options(schema_translate_map={None: name})
    finally:
        with engine.begin() as connection:
            connection.execute(text(f"DROP SCHEMA {name} CASCADE"))
        engine.dispose()


@pytest.fixture(params=["sqlite", "postgres"])
def engine(request, tmp_path):
    """The database under test, in both flavours."""
    if request.param == "postgres":
        return request.getfixturevalue("pg_schema")
    made = sqlite_engine(tmp_path / "main.sqlite")
    request.addfinalizer(made.dispose)
    return made


@pytest.fixture
def store(engine):
    return Store(engine, ran_by="tester")


@pytest.fixture
def machine(engine, tmp_path, request):
    """Makes a machine: its own local SQLite file, sharing `engine` as the server."""

    def make(name, remote=engine):
        local = sqlite_engine(tmp_path / f"{name}.sqlite")
        request.addfinalizer(local.dispose)
        return Store(local, remote, ran_by=name)

    return make
