"""Per-item search log (contract C1): format, baseline line, resume without duplicates, headline pass-through."""

import json
from contextlib import contextmanager
from types import SimpleNamespace

from src.attribution.greedy_search import SearchConfig, greedy_search
from src.attribution.item_log import ItemLog, SearchItemLogger, probe_bits, read_items, sample_bits

UNKNOWN = 2


def _item(label):
    return SimpleNamespace(label=label, unknown_index=UNKNOWN)


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


SAMPLE = [_item(UNKNOWN)] * 4
PROBE = [_item(0), _item(1), _item(UNKNOWN)]
GOOD = {"m1": [UNKNOWN, UNKNOWN, 0, 0], "m2": [UNKNOWN, 0, 0, 0], "m3": [0, 0, 0, 0]}  # predictions per promoted module


def _score_fn_for(registry):
    def score(bits):
        predictions = GOOD.get(registry.current or next(iter(registry.active), None), [0, 0, 0, 0])
        if registry.active and registry.current is None:  # after a keep: model has the kept module
            predictions = GOOD[next(iter(registry.active))]
        bits["sample"] = sample_bits(predictions, SAMPLE)
        bits["probe"] = probe_bits([UNKNOWN, 1, UNKNOWN], PROBE)
        return sum(p == UNKNOWN for p in predictions) / len(predictions)

    return score


def test_bits():
    assert sample_bits([UNKNOWN, 0, UNKNOWN], [_item(UNKNOWN)] * 3) == "101"
    # probe: only stated-answer items that were answered "unknown" count
    assert probe_bits([UNKNOWN, 1, UNKNOWN], PROBE) == "100"


def test_search_writes_baseline_then_one_line_per_trial(tmp_path):
    registry = FakeRegistry()
    logger = SearchItemLogger(ItemLog(tmp_path / "items.jsonl"))
    result = greedy_search(["m1", "m2", "m3"], registry, logger.wrap(_score_fn_for(registry)),
                           SearchConfig(threshold=0.01, max_rounds=1), on_progress=logger.on_trial)
    records = read_items(tmp_path / "items.jsonl")
    assert len(records) == 1 + len(result.trials)  # baseline + trials; the post-keep re-measure is not a trial
    assert records[0] == {"round": 0, "path": None, "sample": "0000", "probe": "100"}
    assert [(r["round"], r["path"]) for r in records[1:]] == [(1, "m1"), (1, "m2"), (1, "m3")]
    assert records[1]["sample"] == "1100"
    assert all(len(r["sample"]) == len(SAMPLE) and len(r["probe"]) == len(PROBE) for r in records)


def test_resume_does_not_duplicate_or_relog_baseline(tmp_path):
    path = tmp_path / "items.jsonl"
    log = ItemLog(path)
    log.append(0, None, "0000", "")
    log.append(1, "m1", "1100", "")
    # a session that died after logging m1 but before checkpointing it re-measures m1
    reopened = ItemLog(path)
    assert not reopened.append(1, "m1", "1100", "")
    assert reopened.append(1, "m2", "1000", "")
    assert [r["path"] for r in read_items(path)] == [None, "m1", "m2"]


def test_truncated_last_line_is_dropped_so_appends_stay_valid(tmp_path):
    path = tmp_path / "items.jsonl"
    path.write_text(json.dumps({"round": 0, "path": None, "sample": "0", "probe": ""}) + '\n{"round": 1, "pa', encoding="utf-8")
    log = ItemLog(path)
    log.append(1, "m1", "1", "")
    assert [r["path"] for r in read_items(path)] == [None, "m1"]
    assert all(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())


def test_resumed_search_logs_only_new_trials(tmp_path):
    registry = FakeRegistry()
    path = tmp_path / "items.jsonl"
    logger = SearchItemLogger(ItemLog(path), resumed_trials=2, fresh=False)
    logger.bits.update(sample="1111", probe="")

    class R:  # a result whose trial count has not grown past the checkpoint
        trials = [object(), object()]

    logger.on_trial(R)
    assert not path.exists() or read_items(path) == []
