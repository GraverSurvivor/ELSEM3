"""The shared scoring loop: run naming per question set, resume safety, and checkpointing."""

from types import SimpleNamespace

import pytest

import src.eval.scoring_run as scoring_run
from src.eval.scoring_run import build_results, check_resume_id, condition_for, score_categories
from src.experiments import logger

UNKNOWN, STEREOTYPE, OTHER = 2, 0, 1


def _item(qid, category="Age", label=UNKNOWN):
    return SimpleNamespace(question_id=qid, category=category, label=label,
                           unknown_index=UNKNOWN, stereotype_index=STEREOTYPE)


def test_condition_names_carry_the_question_set():
    assert condition_for("fp16", "ambig") == "fp16"
    assert condition_for("fp16", "disambig") == "fp16_disambig"
    assert condition_for("hqq_repaired", "disambig") == "hqq_repaired_disambig"
    with pytest.raises(ValueError):
        condition_for("fp16", "both")


def test_resume_accepts_a_run_of_the_same_condition():
    check_resume_id("fp16_20260921T180125Z", "fp16")
    check_resume_id("fp16_disambig_20261005T101500Z", "fp16_disambig")


def test_resume_refuses_a_disambiguated_run_as_an_ambiguous_one():
    # "fp16_disambig_..." starts with "fp16_" too; a plain prefix test would let an ambiguous resume
    # append ambiguous answers to a disambiguated results file.
    with pytest.raises(SystemExit):
        check_resume_id("fp16_disambig_20261005T101500Z", "fp16")


def test_resume_refuses_a_different_model():
    with pytest.raises(SystemExit):
        check_resume_id("hqq_3bit_20260924T175317Z", "fp16")


def test_build_results_scores_correctness_against_the_label():
    items = [_item("Age:1"), _item("Age:2", label=OTHER)]  # a disambiguated item's answer is a person
    rows = build_results(items, [UNKNOWN, UNKNOWN], [0.9, 0.5], "r", "fp16_disambig")
    assert [r.is_correct for r in rows] == [True, False]
    assert [r.is_unknown_predicted for r in rows] == [True, True]
    assert [r.confidence for r in rows] == [0.9, 0.5]


def test_build_results_without_confidence_leaves_it_empty():
    rows = build_results([_item("Age:1")], [STEREOTYPE], None, "r", "fp16")
    assert rows[0].confidence is None and rows[0].is_stereotyped_predicted


def test_score_categories_checkpoints_and_skips_finished(tmp_path, monkeypatch):
    monkeypatch.setattr(logger, "RESULTS_DIR", tmp_path)
    calls = []

    def fake_scorer(model, tok, items, device, batch_size, desc):
        calls.append(desc)
        return [UNKNOWN] * len(items), [0.8] * len(items)

    monkeypatch.setattr(scoring_run, "score_items_with_confidence", fake_scorer)
    items = {"Age": [_item("Age:1")], "SES": [_item("SES:1", category="SES")]}
    already = build_results(items["Age"], [UNKNOWN], [0.7], "run1", "fp16")

    results = score_categories(None, None, items, "cpu", "run1", "fp16", 8, completed=already)

    assert calls == ["SES"]  # Age was already complete
    assert {r.question_id for r in results} == {"Age:1", "SES:1"}
    assert len(logger.read_question_results("run1")) == 2  # checkpointed to disk
