"""What the search ranks candidates by: correctness on the sample, optionally minus hedging."""

from types import SimpleNamespace

import pytest

from src.attribution.run_search import correct_rate, false_unknown_rate, search_objective
from src.quant.candidates import candidate_paths, filter_candidates

UNKNOWN, STEREOTYPE, OTHER = 2, 0, 1


def _item(label):
    return SimpleNamespace(label=label, unknown_index=UNKNOWN)


def test_correct_rate_is_usr_on_ambiguous_items():
    # Every ambiguous item's label is "unknown", so "correct" and "answered unknown" coincide.
    items = [_item(UNKNOWN)] * 4
    assert correct_rate([UNKNOWN, UNKNOWN, STEREOTYPE, OTHER], items) == 0.5


def test_correct_rate_is_accuracy_on_disambiguated_items():
    items = [_item(STEREOTYPE), _item(OTHER)]
    assert correct_rate([STEREOTYPE, UNKNOWN], items) == 0.5


def test_false_unknown_rate_counts_hedging_only_where_an_answer_was_stated():
    items = [_item(STEREOTYPE), _item(OTHER), _item(UNKNOWN)]
    # One hedge out of the two items that had a stated answer; the "unknown"-labelled one is ignored.
    assert false_unknown_rate([UNKNOWN, OTHER, UNKNOWN], items) == 0.5


def test_false_unknown_rate_needs_stated_answers():
    with pytest.raises(ValueError):
        false_unknown_rate([UNKNOWN], [_item(UNKNOWN)])


def test_objective_without_penalty_is_the_sample_score():
    assert search_objective(0.4, None, 1.0) == 0.4


def test_objective_penalises_hedging():
    # A module that recovers "unknown" by making the model hedge everywhere must not win.
    honest = search_objective(0.40, hedging=0.02, weight=1.0)
    hedger = search_objective(0.45, hedging=0.30, weight=1.0)
    assert honest > hedger


def test_filter_candidates_keeps_order_and_family():
    paths = candidate_paths(n_layers=2)
    mlp = filter_candidates(paths, "mlp")
    assert mlp == [p for p in paths if ".mlp." in p] and len(mlp) == 4
    assert all(".self_attn." in p for p in filter_candidates(paths, "attention"))
    assert filter_candidates(paths, "all") == paths
    with pytest.raises(ValueError):
        filter_candidates(paths, "heads")


def test_resume_state_refuses_different_settings_and_reads_the_checkpoint(tmp_path):
    import json

    import pytest

    from src.attribution.greedy_search import Trial
    from src.attribution.run_search import PROGRESS_FILE, _write_trials, resume_state

    meta = {"sample": "s.json", "penalty_sample": "p.json", "penalty_weight": 1.0, "candidates": "all",
            "config": {"threshold": 0.01, "nbits": 8, "max_rounds": 5}}
    trials = [Trial(1, "model.layers.1.self_attn.qkv_proj", 0.5, 0.5, True),
              Trial(2, "model.layers.2.self_attn.qkv_proj", 0.6, 0.1, False)]
    _write_trials(trials, tmp_path / "trials.csv")
    (tmp_path / PROGRESS_FILE).write_text(json.dumps(
        {"meta": meta, "start_usr": 0.0, "kept": [trials[0].path], "usr_after_round": [0.5]}), encoding="utf-8")

    state = resume_state(tmp_path, meta)
    assert state.kept == [trials[0].path] and state.trials == trials and state.usr_after_round == [0.5]
    with pytest.raises(SystemExit, match="Cannot resume"):
        resume_state(tmp_path, {**meta, "penalty_weight": 2.0})


def _checkpoint(tmp_path, trials, kept, usr_after_round):
    import json

    from src.attribution.run_search import PROGRESS_FILE, _write_trials

    meta = {"sample": "s.json", "penalty_sample": None, "penalty_weight": None, "candidates": "all",
            "config": {"threshold": 0.01, "nbits": 8, "max_rounds": 5}}
    _write_trials(trials, tmp_path / "trials.csv")
    (tmp_path / PROGRESS_FILE).write_text(json.dumps(
        {"meta": meta, "start_usr": 0.0, "kept": kept, "usr_after_round": usr_after_round}), encoding="utf-8")
    return meta


A, B, C = (f"model.layers.{i}.self_attn.qkv_proj" for i in (1, 14, 15))


def test_resume_follows_the_trial_log_when_progress_is_behind(tmp_path):
    from src.attribution.greedy_search import Trial
    from src.attribution.run_search import resume_state

    trials = [Trial(1, A, 0.5, 0.5, True), Trial(2, B, 0.6, 0.1, True), Trial(3, C, 0.62, 0.02, False)]
    meta = _checkpoint(tmp_path, trials, kept=[A], usr_after_round=[0.5])
    state = resume_state(tmp_path, meta)
    assert state.kept == [A, B] and state.usr_after_round == [0.5]  # round 2's score gets re-measured


def test_resume_follows_the_trial_log_when_progress_is_ahead(tmp_path):
    from src.attribution.greedy_search import Trial
    from src.attribution.run_search import resume_state

    trials = [Trial(1, A, 0.5, 0.5, True), Trial(2, C, 0.55, 0.05, False)]
    meta = _checkpoint(tmp_path, trials, kept=[A, B], usr_after_round=[0.5, 0.6])
    state = resume_state(tmp_path, meta)
    assert state.kept == [A] and state.usr_after_round == [0.5]  # round 2 continues from its log


def test_resume_refuses_files_that_contradict(tmp_path):
    import pytest

    from src.attribution.greedy_search import Trial
    from src.attribution.run_search import resume_state

    meta = _checkpoint(tmp_path, [Trial(1, A, 0.5, 0.5, True)], kept=[B], usr_after_round=[0.5])
    with pytest.raises(SystemExit, match="Cannot resume"):
        resume_state(tmp_path, meta)


def test_resume_refuses_a_missing_empty_or_stale_trial_log(tmp_path):
    import pytest

    from src.attribution.greedy_search import Trial
    from src.attribution.run_search import resume_state

    meta = _checkpoint(tmp_path, [], kept=[A, B, C], usr_after_round=[0.5, 0.6, 0.7])
    with pytest.raises(SystemExit, match="empty"):
        resume_state(tmp_path, meta)
    (tmp_path / "trials.csv").unlink()
    with pytest.raises(SystemExit, match="missing"):
        resume_state(tmp_path, meta)
    meta = _checkpoint(tmp_path, [Trial(1, A, 0.5, 0.5, True)], kept=[A, B, C], usr_after_round=[0.5, 0.6, 0.7])
    with pytest.raises(SystemExit, match="older copy"):
        resume_state(tmp_path, meta)


def test_a_fast_scan_checkpoint_resumes_with_its_pool(tmp_path):
    import json

    from src.attribution.fast_scan import FastScanResult
    from src.attribution.greedy_search import Trial
    from src.attribution.run_search import PROGRESS_FILE, _write_trials, resume_state

    meta = {"sample": "s.json", "penalty_sample": None, "penalty_weight": None, "candidates": "all",
            "config": {"threshold": 0.01, "nbits": 8, "max_rounds": 5}, "fast": {"mode": "exact"}}
    _write_trials([Trial(1, A, 0.5, 0.5, True)], tmp_path / "trials.csv")
    (tmp_path / PROGRESS_FILE).write_text(json.dumps(
        {"meta": meta, "start_usr": 0.0, "kept": [A], "usr_after_round": [0.5], "survivors": [A, B], "pool": [A, B]}),
        encoding="utf-8")
    state = resume_state(tmp_path, meta)
    assert isinstance(state, FastScanResult) and state.pool == [A, B] and state.kept == [A]
    with pytest.raises(SystemExit, match="Cannot resume"):  # a full-search resume must not pick up a Fast Scan run
        resume_state(tmp_path, {k: v for k, v in meta.items() if k != "fast"})
