"""Fast Scan: equals greedy_search when nothing is pruned, plus determinism, tie-break, pruning and resume."""

import copy
import random
from contextlib import contextmanager
from dataclasses import asdict

import pytest

from src.attribution.fast_scan import FastScanConfig, ScanScore, fast_scan, seeded_subset
from src.attribution.greedy_search import SearchConfig, greedy_search


class FakeRegistry:
    def __init__(self):
        self.active = {}
        self.current = None

    @contextmanager
    def trial(self, path, nbits):
        self.current = path
        try:
            yield
        finally:
            self.current = None

    def commit(self, path, nbits):
        self.active[path] = nbits

    def promoted(self):
        return set(self.active) | ({self.current} if self.current else set())


class World:
    """Random effect table: per candidate, the sample questions it fixes and the probe ones it breaks."""

    def __init__(self, n_candidates, n_sample, n_probe, seed):
        rng = random.Random(seed)
        self.candidates = [f"m{i:03d}" for i in range(n_candidates)]
        self.sample_ids = [f"s{i}" for i in range(n_sample)]
        self.probe_ids = [f"p{i}" for i in range(n_probe)]
        self.fixes = {c: {q for q in self.sample_ids if rng.random() < rng.choice([0.0, 0.02, 0.1])}
                      for c in self.candidates}
        self.breaks = {c: {q for q in self.probe_ids if rng.random() < 0.05} for c in self.candidates}
        self.registry = FakeRegistry()
        self.calls = 0
        self.asked = 0

    def scorer(self, sample_ids, probe_ids):
        self.calls += 1
        self.asked += len(sample_ids) + len(probe_ids)
        on = self.registry.promoted()
        fixed = set().union(*(self.fixes[c] for c in on)) if on else set()
        broken = set().union(*(self.breaks[c] for c in on)) if on else set()
        sample_bits = "".join("1" if q in fixed else "0" for q in sample_ids)
        probe_bits = "".join("1" if q in broken else "0" for q in probe_ids)
        objective = sample_bits.count("1") / len(sample_ids)
        if probe_ids:
            objective -= probe_bits.count("1") / len(probe_ids)
        return ScanScore(objective, sample_bits, probe_bits)


def _greedy(world, config):
    world.registry = FakeRegistry()
    return greedy_search(world.candidates, world.registry,
                         lambda: world.scorer(world.sample_ids, world.probe_ids).objective,
                         SearchConfig(threshold=config.threshold, nbits=config.nbits, max_rounds=config.max_rounds))


def _fast(world, config, **kwargs):
    world.registry = FakeRegistry()
    return fast_scan(world.candidates, world.registry, world.scorer, world.sample_ids, world.probe_ids, config,
                     **kwargs)


def _same_search(a, b):
    assert a.start_usr == b.start_usr
    assert a.kept == b.kept
    assert a.usr_after_round == b.usr_after_round
    assert a.stop_reason == b.stop_reason
    assert [t.to_dict() for t in a.trials] == [t.to_dict() for t in b.trials]


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("mode", ["exact", "fast"])
def test_equals_greedy_search_when_nothing_is_pruned(seed, mode):
    world = World(n_candidates=24, n_sample=50, n_probe=30, seed=seed)
    config = FastScanConfig(mode=mode, screen_sample=10**6, screen_probe=10**6, survivors=len(world.candidates),
                            pool=len(world.candidates), max_rounds=4, threshold=0.01)
    greedy = _greedy(world, config)
    greedy_calls = world.calls
    world.calls = 0
    fast = _fast(world, config)
    _same_search(greedy, fast)
    assert world.calls == greedy_calls  # and it is not doing extra scoring either


def test_equals_greedy_without_a_probe():
    world = World(12, 40, 0, seed=3)
    config = FastScanConfig(mode="exact", screen_sample=10**6, survivors=12, pool=12, max_rounds=3)
    _same_search(_greedy(world, config), _fast(world, config))


def test_deterministic_for_a_seed():
    world = World(30, 120, 60, seed=1)
    config = FastScanConfig(mode="exact", screen_sample=30, screen_probe=15, survivors=10, pool=6, seed=7)
    a, b = _fast(world, config), _fast(world, config)
    _same_search(a, b)
    assert a.survivors == b.survivors and a.pool == b.pool


def test_seeded_subset_is_ordered_and_seed_dependent():
    ids = [f"q{i}" for i in range(100)]
    a = seeded_subset(ids, 10, random.Random(0))
    assert a == sorted(a, key=ids.index) and len(set(a)) == 10
    assert a == seeded_subset(ids, 10, random.Random(0))
    assert a != seeded_subset(ids, 10, random.Random(1))
    assert seeded_subset(ids, 100, random.Random(0)) == ids and seeded_subset(ids, 500, random.Random(0)) == ids


def test_earliest_candidate_wins_a_tie():
    world = World(6, 20, 0, seed=0)
    world.fixes = {c: set(world.sample_ids[:5]) for c in world.candidates}  # every candidate equally good
    config = FastScanConfig(mode="exact", screen_sample=10**6, survivors=3, pool=3, max_rounds=1)
    assert _fast(world, config).kept == [world.candidates[0]]


def test_survivor_and_pool_ties_go_to_the_earlier_candidate():
    world = World(8, 20, 0, seed=0)
    world.fixes = {c: set(world.sample_ids[:4]) for c in world.candidates}
    config = FastScanConfig(mode="exact", screen_sample=10**6, survivors=3, pool=2, max_rounds=1)
    result = _fast(world, config)
    assert result.survivors == world.candidates[:3] and result.pool == world.candidates[:2]


def test_rounds_after_the_first_only_try_the_pool():
    world = World(30, 100, 0, seed=5)
    config = FastScanConfig(mode="exact", screen_sample=40, survivors=10, pool=5, max_rounds=4, threshold=0.001)
    result = _fast(world, config)
    assert {t.path for t in result.trials if t.round > 1} <= set(result.pool)
    assert sum(t.round == 1 for t in result.trials) == 30  # round 1 has one trial per candidate
    assert all(sum(t.round == r for t in result.trials) <= 4 for r in (2, 3, 4))


def test_exact_mode_asks_far_fewer_questions_than_greedy():
    world = World(40, 200, 100, seed=2)
    config = FastScanConfig(mode="exact", screen_sample=40, screen_probe=20, survivors=8, pool=4, max_rounds=3,
                            threshold=0.001)
    greedy = _greedy(world, config)
    greedy_asked = world.asked
    world.asked = 0
    fast = _fast(world, config)
    assert len(fast.kept) >= 2 and len(greedy.kept) >= 2
    assert world.asked < greedy_asked / 3


def test_fast_mode_never_scores_beyond_the_screening_subset():
    world = World(20, 100, 50, seed=4)
    sizes = set()
    original = world.scorer

    def watching(s, p):
        sizes.add((len(s), len(p)))
        return original(s, p)

    world.scorer = watching
    _fast(world, FastScanConfig(mode="fast", screen_sample=25, screen_probe=10, max_rounds=3))
    assert sizes == {(25, 10)}


def test_resume_matches_an_uninterrupted_run():
    world = World(20, 100, 40, seed=6)
    config = FastScanConfig(mode="exact", screen_sample=30, screen_probe=10, survivors=8, pool=5, max_rounds=4,
                            threshold=0.001)
    whole = _fast(world, config)
    assert len(whole.kept) >= 3

    class Died(Exception):
        pass

    checkpoints = []

    def snapshot_then_die(result):
        checkpoints.append(copy.deepcopy(result))
        if len(result.kept) == 2 and sum(t.round == 3 for t in result.trials) == 2:
            raise Died

    with pytest.raises(Died):
        _fast(world, config, on_progress=snapshot_then_die)
    resumed = _fast(world, config, resume=checkpoints[-1])
    _same_search(whole, resumed)
    assert resumed.pool == whole.pool


def test_a_checkpoint_from_before_round_one_finished_restarts_the_screen():
    world = World(15, 60, 0, seed=8)
    config = FastScanConfig(mode="exact", screen_sample=20, survivors=6, pool=4, max_rounds=3, threshold=0.001)
    whole = _fast(world, config)
    partial = []

    class Died(Exception):
        pass

    def die_in_round_one(result):
        partial.append(copy.deepcopy(result))
        if len(result.trials) == 5:
            raise Died

    with pytest.raises(Died):
        _fast(world, config, on_progress=die_in_round_one)
    _same_search(whole, _fast(world, config, resume=partial[-1]))


def test_resume_without_a_pool_is_refused():
    world = World(10, 40, 0, seed=9)
    config = FastScanConfig(mode="exact", screen_sample=10**6, survivors=10, pool=10, max_rounds=3)
    done = greedy_search(world.candidates, FakeRegistry(), lambda: 0.0, SearchConfig(max_rounds=1))
    done.kept.append(world.candidates[0])
    with pytest.raises(ValueError, match="no pool"):
        _fast(world, config, resume=done)


def test_stops_like_greedy_when_nothing_helps():
    world = World(5, 20, 0, seed=0)
    world.fixes = {c: set() for c in world.candidates}
    result = _fast(world, FastScanConfig(mode="exact", screen_sample=10**6, survivors=5, pool=5))
    assert result.kept == [] and "threshold" in result.stop_reason


def test_config_validation_and_duplicates():
    for bad in ({"mode": "slow"}, {"survivors": 0}, {"pool": 0}, {"threshold": 0}, {"nbits": 3},
                {"screen_probe": -1}):
        with pytest.raises(ValueError):
            FastScanConfig(**bad)
    world = World(3, 10, 0, seed=0)
    with pytest.raises(ValueError, match="duplicates"):
        fast_scan(["a", "a"], world.registry, world.scorer, world.sample_ids, [], FastScanConfig())
    assert asdict(FastScanConfig())["screen_sample"] == 60
