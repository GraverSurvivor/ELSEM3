"""The reproduction check's helpers: version comparison, sampling and agreement."""

from types import SimpleNamespace

from src.eval.check_reproduction import agreement, installed_vs_locked, sample_ids


def test_versions_are_read_from_the_lock(tmp_path):
    lock = tmp_path / "lock.txt"
    lock.write_text("pytest==0.0.1\nsome_pkg==2.0\n", encoding="utf-8")
    rows = dict((name, (have, locked)) for name, have, locked in installed_vs_locked(lock, ("pytest", "nope-pkg")))
    assert rows["pytest"][1] == "0.0.1" and rows["pytest"][0] != "not installed"
    assert rows["nope-pkg"] == ("not installed", "not in lock")


def test_sample_is_stratified_and_seeded():
    rows = [SimpleNamespace(category=c, question_id=f"{c}:{i}") for c, n in (("Age", 300), ("SES", 700)) for i in range(n)]
    ids = sample_ids(rows, 100)
    assert len(ids) == 100 and sum(q.startswith("Age") for q in ids) == 30
    assert ids == sample_ids(rows, 100) and ids != sample_ids(rows, 100, seed=1)


def test_agreement_counts_only_shared_questions():
    a = agreement({"q1": 0, "q2": 1, "q3": 2}, {"q1": 0, "q2": 2, "q9": 1})
    assert (a["n"], a["agree"], a["disagree"]) == (2, 1, ["q2"])
