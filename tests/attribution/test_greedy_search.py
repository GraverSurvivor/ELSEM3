"""Greedy search decision logic, against a fake registry whose "model" is a table of effects.

The real SwapRegistry's promote/revert is covered in tests/quant/. What's tested here is the
search's own logic: which candidate it keeps, when it stops, and that its log is complete --
none of which needs a GPU once scoring is replaced by a lookup.
"""

from contextlib import contextmanager

import pytest

from src.attribution.greedy_search import (
    STOP_EXHAUSTED,
    STOP_MAX_ROUNDS,
    STOP_RECOVERED,
    STOP_THRESHOLD,
    SearchConfig,
    greedy_search,
)

ATTN = [f"model.layers.{i}.self_attn.qkv_proj" for i in range(3)]
MLP = [f"model.layers.{i}.mlp.down_proj" for i in range(3)]
CANDIDATES = ATTN + MLP  # attention first, as candidate_paths_for_model orders them


class FakeRegistry:
    """Promoted modules each add a fixed USR effect; USR is the capped sum.

    Tracks trials so tests can check that every trial was put back.
    """

    def __init__(self, effects, start=0.0, combine=None):
        self.effects = effects
        self.start = start
        self.combine = combine
        self._committed = {}
        self._trying = None
        self.commits = []
        self.trial_count = 0

    @property
    def active(self):
        return dict(self._committed)

    @contextmanager
    def trial(self, path, nbits):
        assert self._trying is None, "trial started while another was still promoted"
        assert path not in self._committed, "tried a module that was already kept"
        self._trying = path
        self.trial_count += 1
        try:
            yield
        finally:
            self._trying = None

    def commit(self, path, nbits):
        self._committed[path] = nbits
        self.commits.append(path)

    def usr(self):
        promoted = list(self._committed) + ([self._trying] if self._trying else [])
        if self.combine is not None:
            return self.combine(promoted)
        return min(1.0, self.start + sum(self.effects.get(p, 0.0) for p in promoted))


def _run(registry, config=SearchConfig(), candidates=CANDIDATES, **kwargs):
    return greedy_search(candidates, registry, registry.usr, config, **kwargs)


# --- what gets kept -------------------------------------------------------------------------


def test_keeps_the_one_module_that_helps_and_stops_at_the_threshold():
    registry = FakeRegistry({MLP[1]: 0.30})
    result = _run(registry)
    assert result.kept == [MLP[1]]
    assert result.usr_after_round == [pytest.approx(0.30)]
    assert result.stop_reason == STOP_THRESHOLD


def test_keeps_the_largest_gain_each_round_in_order():
    registry = FakeRegistry({ATTN[2]: 0.10, MLP[0]: 0.40, ATTN[0]: 0.20})
    result = _run(registry)
    assert result.kept == [MLP[0], ATTN[0], ATTN[2]]
    assert result.usr_after_round == pytest.approx([0.40, 0.60, 0.70])


def test_an_exact_tie_goes_to_the_earlier_cheaper_attention_candidate():
    registry = FakeRegistry({MLP[0]: 0.25, ATTN[1]: 0.25})
    assert _run(registry).kept[0] == ATTN[1]


def test_nothing_clears_the_threshold_so_nothing_is_kept():
    registry = FakeRegistry({ATTN[0]: 0.005})
    result = _run(registry, SearchConfig(threshold=0.01))
    assert result.kept == []
    assert registry.commits == []
    assert result.final_usr == result.start_usr
    assert result.stop_reason == STOP_THRESHOLD


def test_a_gain_exactly_at_the_threshold_is_kept():
    registry = FakeRegistry({ATTN[0]: 0.25})
    assert _run(registry, SearchConfig(threshold=0.25)).kept == [ATTN[0]]


def test_a_module_that_hurts_is_never_kept():
    registry = FakeRegistry({ATTN[0]: -0.2, MLP[2]: 0.1})
    assert _run(registry).kept == [MLP[2]]


# --- stopping -------------------------------------------------------------------------------


def test_stops_at_max_rounds_even_when_more_modules_would_help():
    registry = FakeRegistry({p: 0.1 for p in CANDIDATES})
    result = _run(registry, SearchConfig(max_rounds=2))
    assert len(result.kept) == 2
    assert result.stop_reason == STOP_MAX_ROUNDS


def test_stops_once_the_sample_is_fully_recovered():
    registry = FakeRegistry({ATTN[0]: 0.6, ATTN[1]: 0.6, MLP[0]: 0.3})
    result = _run(registry)
    assert result.final_usr == 1.0
    assert result.stop_reason == STOP_RECOVERED
    assert len(result.kept) == 2


def test_stops_when_every_candidate_is_kept():
    registry = FakeRegistry({p: 0.1 for p in ATTN})
    result = _run(registry, SearchConfig(max_rounds=10), candidates=ATTN)
    assert result.kept == ATTN
    assert result.stop_reason == STOP_EXHAUSTED


# --- measurement --------------------------------------------------------------------------


def test_gains_are_measured_against_the_rescored_model_not_the_winning_trial():
    # Committing ATTN[0] is worth less than its trial suggested (e.g. requantization drift), so
    # round 2's gains must be relative to 0.25, not to the trial's 0.30.
    def combine(promoted):
        usr = 0.0
        if ATTN[0] in promoted:
            usr += 0.25 if ATTN[0] in registry.active else 0.30
        if MLP[0] in promoted:
            usr += 0.05
        return usr

    registry = FakeRegistry({}, combine=combine)
    result = _run(registry, SearchConfig(threshold=0.04))
    assert result.usr_after_round[0] == pytest.approx(0.25)
    round_two = [t for t in result.trials if t.round == 2 and t.path == MLP[0]][0]
    assert round_two.gain == pytest.approx(0.05)


def test_every_candidate_is_tried_once_per_round_and_already_kept_ones_are_skipped():
    registry = FakeRegistry({ATTN[0]: 0.3, MLP[0]: 0.2})
    result = _run(registry)
    per_round = [sum(t.round == r for t in result.trials) for r in (1, 2, 3)]
    assert per_round == [6, 5, 4]
    assert registry.trial_count == 15


def test_the_log_marks_exactly_the_kept_trials():
    registry = FakeRegistry({ATTN[1]: 0.3, MLP[2]: 0.2})
    result = _run(registry)
    kept_in_log = [t.path for t in result.trials if t.kept]
    assert kept_in_log == result.kept == [ATTN[1], MLP[2]]


def test_progress_is_reported_after_every_trial():
    registry = FakeRegistry({ATTN[0]: 0.3})
    seen = []
    result = _run(registry, on_progress=lambda r: seen.append(len(r.trials)))
    # 6 trials in round 1, a keep, 5 in round 2, then the threshold stop.
    assert seen == [1, 2, 3, 4, 5, 6, 6, 7, 8, 9, 10, 11, 11]
    assert len(result.trials) == 11


def test_duplicate_candidates_raise():
    with pytest.raises(ValueError, match="duplicates"):
        _run(FakeRegistry({}), candidates=[ATTN[0], ATTN[0]])


# --- config ----------------------------------------------------------------------------------


@pytest.mark.parametrize("threshold", [0.0, -0.1])
def test_threshold_must_be_positive(threshold):
    # At 0, a module that changes nothing would be "kept", inflating the repair's memory cost.
    with pytest.raises(ValueError, match="threshold"):
        SearchConfig(threshold=threshold)


def test_nbits_must_be_a_promotion_target():
    with pytest.raises(ValueError, match="nbits"):
        SearchConfig(nbits=3)


def test_max_rounds_must_be_at_least_one():
    with pytest.raises(ValueError, match="max_rounds"):
        SearchConfig(max_rounds=0)


def test_result_serialises_with_the_fields_the_driver_writes():
    result = _run(FakeRegistry({ATTN[0]: 0.3}))
    assert result.to_dict() == {
        "start_usr": 0.0,
        "final_usr": pytest.approx(0.3),
        "kept": [ATTN[0]],
        "usr_after_round": [pytest.approx(0.3)],
        "stop_reason": STOP_THRESHOLD,
        "n_trials": 11,
    }


# --- resuming an interrupted search --------------------------------------------------------------

class _Interrupted(Exception):
    pass


def _snapshot(result):
    from src.attribution.greedy_search import SearchResult

    return SearchResult(start_usr=result.start_usr, kept=list(result.kept),
                        usr_after_round=list(result.usr_after_round), trials=list(result.trials))


EFFECTS = {ATTN[2]: 0.10, MLP[0]: 0.40, ATTN[0]: 0.20, MLP[2]: 0.05}


@pytest.mark.parametrize("cut", range(1, 16))
def test_a_resumed_search_equals_an_uninterrupted_one(cut):
    full = _run(FakeRegistry(EFFECTS))
    calls, saved = [0], []

    def stop_after(result):
        calls[0] += 1
        if calls[0] == cut:
            saved.append(_snapshot(result))
            raise _Interrupted

    try:
        _run(FakeRegistry(EFFECTS), on_progress=stop_after)
    except _Interrupted:
        pass
    if not saved:  # the search finished before this cut
        return
    fresh = FakeRegistry(EFFECTS)
    resumed = _run(fresh, resume=saved[0])
    assert resumed.kept == full.kept
    assert resumed.trials == full.trials
    assert resumed.usr_after_round == pytest.approx(full.usr_after_round)
    assert resumed.stop_reason == full.stop_reason
    assert fresh.trial_count == len(full.trials) - len(saved[0].trials)  # nothing scored twice


def test_resume_right_after_a_keep_rescored_the_model():
    from src.attribution.greedy_search import SearchResult

    full = _run(FakeRegistry(EFFECTS))
    first_round = [t for t in full.trials if t.round == 1]
    state = SearchResult(start_usr=full.start_usr, kept=[full.kept[0]], usr_after_round=[], trials=first_round)
    resumed = _run(FakeRegistry(EFFECTS), resume=state)
    assert resumed.kept == full.kept and resumed.usr_after_round == pytest.approx(full.usr_after_round)
