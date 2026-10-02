"""Freeze: completeness checks and the hash manifest."""

from src.experiments.freeze import EXPECTED, check_run, hash_results, verify


def _jsonl(path, n):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join("{}\n" for _ in range(n)), encoding="utf-8")


def test_runs_must_match_their_reference_question_count(tmp_path):
    _jsonl(tmp_path / "baseline_fp16" / "question_results.jsonl", 10)
    _jsonl(tmp_path / "baseline_fp16_disambig" / "question_results.jsonl", 7)
    _jsonl(tmp_path / "repair_positional" / "question_results.jsonl", 10)
    _jsonl(tmp_path / "repair_random_s0" / "question_results.jsonl", 9)
    _jsonl(tmp_path / "hqq_repaired_disambig" / "question_results.jsonl", 7)
    (tmp_path / "headline").mkdir()
    (tmp_path / "headline" / "headline.json").write_text("{}", encoding="utf-8")

    assert check_run(tmp_path, "repair_positional") == (True, "10 questions")
    assert check_run(tmp_path, "repair_random_s0") == (False, "incomplete: 9 of 10 questions")
    assert check_run(tmp_path, "hqq_repaired_disambig")[0]  # compared with the disambiguated baseline
    assert check_run(tmp_path, "headline") == (True, "present")
    assert check_run(tmp_path, "hqq_4bit") == (False, "missing")


def test_required_runs_cover_the_main_claim():
    required = {name for name, _, req in EXPECTED if req}
    assert {"baseline_fp16", "hqq_3bit", "hqq_repaired", "hqq_repaired_disambig", "repair_positional"} <= required
    assert "llamacpp_q3km_fairgate" not in required  # stretch goal


def test_verify_spots_changed_missing_and_added_files(tmp_path):
    (tmp_path / "reference" / "a").mkdir(parents=True)
    (tmp_path / "reference" / "a" / "x.json").write_text("1", encoding="utf-8")
    (tmp_path / "patches").mkdir()
    (tmp_path / "patches" / "p.json").write_text("2", encoding="utf-8")
    manifest = {"files": hash_results(tmp_path, exclude=None)}
    assert set(manifest["files"]) == {"reference/a/x.json", "patches/p.json"}
    assert verify(manifest, tmp_path, exclude=None) == {"changed": [], "missing": [], "added": []}

    (tmp_path / "reference" / "a" / "x.json").write_text("changed", encoding="utf-8")
    (tmp_path / "patches" / "p.json").unlink()
    (tmp_path / "patches" / "new.json").write_text("3", encoding="utf-8")
    assert verify(manifest, tmp_path, exclude=None) == {
        "changed": ["reference/a/x.json"], "missing": ["patches/p.json"], "added": ["patches/new.json"]}
