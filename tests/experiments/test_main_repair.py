"""The main-repair pointer: what it names, the fallback, and the committed file."""

import json

from src.attribution.patch import load_patch
from src.experiments.main_repair import main_repair, previous_repair


def test_reads_the_pointer_and_falls_back_without_one(tmp_path):
    pointer = tmp_path / "MAIN_REPAIR.json"
    pointer.write_text(json.dumps({
        "name": "new", "label": "New", "patch": "p.json", "ambig": "repair_new", "disambig": "repair_new_disambig",
        "previous": {"name": "old", "label": "Old", "patch": "o.json", "ambig": "hqq_repaired",
                     "disambig": "hqq_repaired_disambig"}}), encoding="utf-8")
    main, previous = main_repair(pointer), previous_repair(pointer)
    assert (main.name, main.ambig, main.ambig_dir(tmp_path)) == ("new", "repair_new", tmp_path / "repair_new")
    assert previous.ambig == "hqq_repaired"
    missing = tmp_path / "none.json"
    assert main_repair(missing).ambig == "hqq_repaired" and previous_repair(missing) is None


def test_the_committed_pointer_names_runs_and_a_patch_that_exist():
    main = main_repair()
    assert load_patch(main.patch).modules  # a valid patch
    assert (main.ambig_dir() / "question_results.jsonl").exists()
    assert (main.disambig_dir() / "question_results.jsonl").exists()
    summary = json.loads((main.ambig_dir() / "summary.json").read_text(encoding="utf-8"))
    assert summary["modules"] == list(load_patch(main.patch).modules)  # the run verified this patch
