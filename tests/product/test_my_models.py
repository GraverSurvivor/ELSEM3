"""Tab 6's views: rows in, HTML out. No Gradio, no GPU."""

from datetime import datetime, timedelta, timezone

import pytest

pytest.importorskip("sqlalchemy")

from src.db.store import Store, sqlite_engine
from src.product import my_models
from src.product.reopen import (
    CannotReopen,
    Fingerprint,
    check_fingerprint,
    recipe_of,
    saved_weights_repo,
    version_differences,
)
from tests.db.test_store import EVENTS, KEPT, MODEL, PATCH, REPORT, save_everything

NOW = datetime(2026, 11, 25, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def store(tmp_path, request):
    made = Store(sqlite_engine(tmp_path / "x.sqlite"), ran_by="Asha")
    request.addfinalizer(made.close)
    return made


# --- the table -------------------------------------------------------------------------------------

def test_the_table_has_what_was_asked_for(store):
    save_everything(store)
    [row] = my_models.table(store.runs())
    shown = dict(zip(my_models.HEADERS, row))
    assert shown["Model"] == MODEL and shown["Method"] == "HQQ" and shown["Bits"] == "3-bit + 8-bit parts"
    assert shown["Verdict"] == "PASS" and shown["Fairness recovered"] == "90.0%" and shown["Extra MB"] == "+12"
    assert shown["Who ran it"] == "Asha" and shown["Outcome"] == "Repaired" and shown["Date"].endswith("UTC")
    [(label, run_id)] = my_models.choices(store.runs())
    assert MODEL in label and run_id == store.runs()[0]["id"]


def test_runs_without_results_still_show(store):
    store.start_run("a", MODEL)
    store.start_run("b", MODEL, pipeline="fake")
    store.finish_run("b", "cancelled")
    store.start_run("c", MODEL, source="reference", job="repair_random_s0")
    store.finish_run("c", "done")
    outcomes = {r["id"]: my_models.outcome_text(r) for r in store.runs()}
    assert outcomes == {"a": "Running", "b": "Cancelled (stand-in)", "c": "Research run"}
    assert "repair_random_s0" in my_models.title(store.run("c"))
    for run_id in "abc":
        assert "None" not in my_models.run_page(store.run(run_id))


def test_a_run_whose_machine_went_quiet_is_interrupted_not_running():
    run = {"status": "running", "updated_at": NOW - timedelta(hours=3), "started_at": NOW - timedelta(hours=3)}
    assert my_models.status_of(run, now=NOW) == "interrupted"
    assert my_models.status_of(run, NOW - timedelta(minutes=5), now=NOW) == "running"  # it spoke 5 minutes ago
    assert my_models.status_of({**run, "status": "done"}, now=NOW) == "done"


# --- pages -----------------------------------------------------------------------------------------

def test_a_runs_page_has_the_report_the_parts_the_versions_and_the_links(store):
    run_id = save_everything(store)
    store.set_weights(run_id, "https://huggingface.co/me/q-fairgate")
    page = my_models.run_page(store.run(run_id))
    assert "Fairness recovered: <b>90.0%</b>" in page and "by-grid" in page and "abc123" in page
    assert "hqq 0.2.8.post1" in page and 'href="https://huggingface.co/me/q-fairgate"' in page
    assert my_models.run_page(None) == "<p>Pick a run above.</p>"


def test_strategies_are_drawn_when_the_report_has_them(store):
    report = {**REPORT, "strategies": [
        {"name": "scan only", "extra_mb": 5, "parts": 3, "recovery": {"point": 0.45}},
        {"name": "until fair", "extra_mb": 12, "parts": KEPT, "recovery": 0.9, "chosen": True}, "junk"]}
    store.start_run("r", MODEL)
    store.finish_run("r", "done", report=report, patch=PATCH)
    run = store.run("r")
    assert [(s["name"], s["n_parts"], s["chosen"]) for s in run["strategies"]] == [
        ("scan only", 3, False), ("until fair", 2, True)]
    drawn = my_models.strategies_html(run)
    assert "width:50%" in drawn and "width:100%" in drawn and "until fair ✓ chosen" in drawn and "+12 MB" in drawn
    assert "Repairs compared" in my_models.run_page(run)
    assert "Only one repair" in my_models.strategies_html(store.run(save_everything(store)))


def test_a_report_this_version_cannot_draw_falls_back_to_its_numbers(store):
    store.start_run("r", MODEL)
    store.finish_run("r", "done", report={**REPORT, "questions": "many"}, patch=PATCH)
    assert "mm-metrics" in my_models.run_page(store.run("r"))


def test_what_comes_from_the_database_cannot_inject_markup(store):
    bad = "<script>alert(1)</script>"
    store.start_run("r", bad, job=bad, ran_by=bad, mode=bad, versions={bad: bad})
    store.save_events("r", [{**EVENTS[0], "message": bad, "stage": bad}])
    store.finish_run("r", "done", message=bad, report={
        **REPORT, "gate_failures": [bad], "outcome_reason": bad, "gpu": bad, "other_damage": [bad],
        "strategies": [{"name": bad, "recovery": 1}]}, patch={**PATCH, "model_id": bad, "revision": bad})
    store.set_weights("r", f"https://huggingface.co/{bad}")
    run = store.run("r")
    save_everything(store, "other")
    for shown in (my_models.run_page(run), my_models.compare_html(run, store.run("other")), my_models.audit_report(run)):
        assert "<script" not in shown


def test_two_runs_side_by_side(store):
    save_everything(store, "a")
    store.start_run("b", MODEL)
    store.finish_run("b", "done", report={**REPORT, "kept": KEPT[:1], "gate": "FAIL", "recovery": {"point": 0.5}},
                     patch={**PATCH, "modules": KEPT[:1], "module_bits": None})
    shown = my_models.compare_html(store.run("a"), store.run("b"))
    assert "<td>PASS</td><td>FAIL</td>" in shown and "<td>90.0%</td><td>50.0%</td>" in shown
    assert "Parts in both: <b>1</b> · only in the first: 1 · only in the second: 0" in shown
    assert "two different" in my_models.compare_html(store.run("a"), store.run("a"))
    assert "Pick two" in my_models.compare_html(store.run("a"), None)


def test_the_audit_report_is_one_self_contained_printable_page(store, tmp_path):
    run = store.run(save_everything(store))
    path = my_models.write_audit_report(run, tmp_path / "out", fingerprint="Matches what was measured, 50/50")
    page = path.read_text(encoding="utf-8")
    assert page.startswith("<!doctype html>") and "@media print" in page and "50/50" in page
    assert run["artifact"]["recipe_sha256"] in page and run["artifact"]["kit_sha256"] in page
    assert "http" not in page.replace("https://huggingface.co", "") and "<script" not in page  # nothing to fetch
    store.start_run("failed", MODEL)
    store.finish_run("failed", "failed", message="out of memory")
    assert "out of memory" in my_models.audit_report(store.run("failed"))


def test_the_kit_is_rebuilt_from_the_database(store, tmp_path):
    import json
    import zipfile

    run = store.run(save_everything(store))
    with zipfile.ZipFile(my_models.build_run_kit(run, tmp_path)) as kit:
        assert sorted(kit.namelist()) == ["README.md", "fairgate.py", "patch.json"]
        assert json.loads(kit.read("patch.json")) == PATCH
    store.start_run("none", MODEL)
    with pytest.raises(CannotReopen, match="no kit"):
        my_models.build_run_kit(store.run("none"), tmp_path)


# --- reopening: what can be opened, and is it the model that was measured ---------------------------

def test_only_a_real_repair_can_be_reopened(store):
    assert recipe_of(store.run(save_everything(store))) == PATCH
    store.start_run("n", MODEL)
    store.finish_run("n", "done", report={**REPORT, "kept": [], "outcome": "no_damage"})
    with pytest.raises(CannotReopen, match="without a repair"):
        recipe_of(store.run("n"))
    store.start_run("f", MODEL, pipeline="fake")
    store.finish_run("f", "done", report={**REPORT, "synthetic": True}, patch=PATCH)
    with pytest.raises(CannotReopen, match="stand-in"):
        recipe_of(store.run("f"))


def test_only_a_hugging_face_address_is_taken_as_saved_weights():
    def repo(url):
        return saved_weights_repo({"artifact": {"weights_url": url}})

    assert repo("https://huggingface.co/me/q-fairgate") == "me/q-fairgate"
    for bad in (None, "", "https://evil.example/me/q", "https://huggingface.co.evil/me/q", "../../etc/passwd",
                "https://huggingface.co/me/q/../../x", "file:///tmp/model", "http://huggingface.co/me/q"):
        assert repo(bad) is None
    assert saved_weights_repo({}) is None


def test_the_fingerprint_verdict():
    assert Fingerprint(50, 50, 0, []).text() == "✅ Matches what was measured, 50/50"
    assert Fingerprint(50, 48, 0, ["torch 3 (measured with 2)"]).ok
    worse = Fingerprint(50, 47, 0, ["hqq 0.3 (measured with 0.2.8)"])
    assert not worse.ok and "47/50" in worse.text() and "hqq 0.3 (measured with 0.2.8)" in worse.text()
    assert "the same, or weren't saved" in Fingerprint(50, 10, 0, []).text()
    assert "aren't in this copy" in Fingerprint(50, 49, 1, []).text()
    assert "No fingerprint was saved" in Fingerprint(0, 0, 0, []).text()


def test_version_differences_name_only_what_changed():
    was = {"hqq": "0.2.8", "torch": "2.10.0", "transformers": None}
    assert version_differences(was, {"hqq": "0.2.8", "torch": "2.11.0"}) == ["torch 2.11.0 (measured with 2.10.0)"]
    assert version_differences(was, {"torch": "2.10.0"}) == ["hqq not installed (measured with 0.2.8)"]
    assert version_differences(None) == []


def test_the_fingerprint_asks_the_right_questions_of_each_set(monkeypatch):
    asked = []

    def score(model, tokenizer, items, device, batch_size):
        asked.extend(items)
        return [2, 0]

    monkeypatch.setattr("src.eval.option_scorer.score_items", score)
    run = {"artifact": {"versions": {}, "fingerprint": [
        {"question_id": "Age:1", "set": "ambig", "pick": 2}, {"question_id": "Age:1", "set": "stated", "pick": 1},
        {"question_id": "Age:9", "set": "ambig", "pick": 0}]}}
    items = {"ambig": {"Age:1": "ambiguous one"}, "stated": {"Age:1": "stated one"}}
    mark = check_fingerprint(run, None, None, "cpu", items)
    assert asked == ["ambiguous one", "stated one"]
    assert (mark.asked, mark.matched, mark.missing, mark.ok) == (3, 1, 1, False)
    assert check_fingerprint({"artifact": None}, None, None, "cpu", items).asked == 0
