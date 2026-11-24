"""The launch notebook and the app's start-up: the database's address is used and never shown."""

import ast
import json
import sys
from pathlib import Path

import pytest

NOTEBOOK = Path("notebooks/fairgate_studio.ipynb")


@pytest.fixture
def launch_cell():
    if not NOTEBOOK.exists():
        pytest.skip("run from the repository root")
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    [source] = ["".join(c["source"]) for c in cells if c["cell_type"] == "code" and "Launch FairGate Studio" in "".join(c["source"])]
    return source


def test_the_notebook_reads_the_secret_into_the_environment_only(launch_cell):
    python = "\n".join("pass" if line.startswith("!") else line for line in launch_cell.splitlines())
    ast.parse(python)
    assert 'os.environ["DATABASE_URL"] = userdata.get("DATABASE_URL")' in launch_cell
    for line in launch_cell.splitlines():  # never printed, never on a command line
        if line.strip().startswith(("print", "!", "sh(")):
            assert "DATABASE_URL" not in line and "environ" not in line
    assert 'os.environ["FAIRGATE_USER"] = YOUR_NAME' in launch_cell


def test_the_database_lines_run_with_and_without_the_secret(launch_cell, monkeypatch):
    lines = launch_cell.splitlines()
    start = next(i for i, line in enumerate(lines) if "the shared database" in line)
    block = "\n".join(lines[start:start + 5])

    def run(secrets):
        class Userdata:
            @staticmethod
            def get(name):
                return secrets[name]

        import os

        monkeypatch.setenv("DATABASE_URL", "left over from before")
        exec(block, {"os": os, "userdata": Userdata, "YOUR_NAME": "Asha"})
        return os.environ.get("DATABASE_URL"), os.environ["FAIRGATE_USER"]

    monkeypatch.setenv("FAIRGATE_USER", "")
    assert run({"DATABASE_URL": "postgresql://u:p@h/d"}) == ("postgresql://u:p@h/d", "Asha")
    assert run({}) == (None, "Asha")  # no secret: local only, and nothing stale


def test_who_ran_it(monkeypatch):
    pytest.importorskip("sqlalchemy")
    from src.db.store import who_runs

    monkeypatch.setenv("FAIRGATE_USER", "asha@example.com")
    assert who_runs() == "asha"  # never an e-mail address
    monkeypatch.setenv("FAIRGATE_USER", "")
    monkeypatch.setattr("getpass.getuser", lambda: "root")
    assert who_runs() == "colab"


def test_the_app_says_connected_or_local_only_and_never_the_address(tmp_path, monkeypatch, capsys):
    pytest.importorskip("sqlalchemy")
    pytest.importorskip("gradio")
    from src.product import studio

    monkeypatch.setenv("DATABASE_URL", "mysql://me:secretpw@host/db")
    store = studio.open_database(tmp_path / "x.sqlite")
    assert store.wait_settled(1).startswith("database: local only")
    store.close()
    (tmp_path / "file").write_text("not a folder")
    assert studio.open_database(tmp_path / "file" / "x.sqlite") is None  # nowhere to write: the app still starts
    out = capsys.readouterr().out
    assert "database: off" in out and "secretpw" not in out and "Traceback" not in out


def test_the_requirements_name_the_database_libraries():
    text = Path("requirements.txt").read_text(encoding="utf-8") if Path("requirements.txt").exists() else pytest.skip("root")
    assert "sqlalchemy>=2.0" in text and "psycopg[binary]" in text


def test_this_python_can_read_every_new_file():
    """Colab's Python can be older than the one a file was written on: no 3.12-only syntax."""
    for path in [*Path("src/db").glob("*.py"), Path("src/product/my_models.py"), Path("src/product/reopen.py")]:
        ast.parse(path.read_text(encoding="utf-8"), feature_version=(3, 10))
    assert sys.version_info >= (3, 10)
