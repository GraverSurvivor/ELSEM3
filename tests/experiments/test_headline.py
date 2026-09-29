"""The headline driver's bookkeeping, with the GPU steps replaced by a fake runner that writes the
same directories run_search / run_repair would."""

import json
import subprocess
from pathlib import Path

import pytest

from src.data.bbq_loader import CATEGORIES
from src.experiments import windows as windows_mod
from src.experiments.headline import Manifest, run_window
from src.experiments.headline_report import (
    VERDICT_DISJOINT,
    VERDICT_IDENTICAL,
    VERDICT_PARTIAL,
    localization_overlap,
)
from src.experiments.windows import WINDOWS, draw_window_sample, ensure_window_sample

Q = "model.layers.{}.self_attn.qkv_proj"


def _flipped(per_category=40):
    items = [{"question_id": f"{c}:{i}", "category": c} for c in CATEGORIES for i in range(per_category)]
    return {"baseline_run_id": "b", "quantized_run_id": "q", "total_flipped": len(items), "items": items}


@pytest.fixture
def in_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(windows_mod, "SAMPLES_DIR", Path("results/search_samples"))
    return tmp_path


# --- windows -------------------------------------------------------------------------------------

def test_window_weights_sum_to_one_and_skew_the_named_category():
    for window in WINDOWS.values():
        assert sum(window.weights.values()) == pytest.approx(1)
        assert set(window.weights) == set(CATEGORIES)
    assert WINDOWS["age_heavy"].weights["Age"] == max(WINDOWS["age_heavy"].weights.values())
    assert WINDOWS["ses_heavy"].weights["SES"] == max(WINDOWS["ses_heavy"].weights.values())


def test_window_samples_have_the_intended_mix():
    sample = draw_window_sample(_flipped(), "age_heavy", n=50)
    assert sample["window"] == "age_heavy"
    assert sample["per_category"]["Age"] == 30
    assert sum(sample["per_category"].values()) == 50
    assert set(draw_window_sample(_flipped(), "balanced", n=50)["per_category"].values()) == {10}


def test_windows_fit_the_real_flipped_set():
    real = json.loads(Path("results/flipped/hqq_3bit.json").read_text(encoding="utf-8")) \
        if Path("results/flipped/hqq_3bit.json").exists() else None
    if real is None:
        pytest.skip("committed flipped set not found from this cwd")
    for name in WINDOWS:
        assert sum(draw_window_sample(real, name)["per_category"].values()) == 300


def test_ensure_window_sample_refuses_a_stale_file(in_tmp):
    path = ensure_window_sample(_flipped(), "balanced", n=20)
    assert ensure_window_sample(_flipped(), "balanced", n=20) == path  # same draw: reused
    data = json.loads(path.read_text(encoding="utf-8"))
    data["question_ids"] = data["question_ids"][::-1][:-1]
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="doesn't match"):
        ensure_window_sample(_flipped(), "balanced", n=20)


# --- driver --------------------------------------------------------------------------------------

class FakeRunner:
    """Writes what the real drivers write. `fail_repair_once` dies after creating the run dir."""

    def __init__(self, kept=(Q.format(1),), fail_repair_once=False):
        self.calls, self.kept, self.fail_repair_once, self.n = [], list(kept), fail_repair_once, 0

    def __call__(self, argv):
        self.calls.append(argv)
        self.n += 1
        if argv[1] == "src.attribution.run_search":
            out = Path("results") / f"search_{self.n}"
            out.mkdir(parents=True)
            (out / "summary.json").write_text(json.dumps({"kept": self.kept}), encoding="utf-8")
        else:
            if "--resume" in argv:
                out = Path("results") / argv[argv.index("--resume") + 1]
            else:
                out = Path("results") / f"hqq_repaired_{self.n}"
                out.mkdir(parents=True)
            if self.fail_repair_once:
                self.fail_repair_once = False
                raise subprocess.CalledProcessError(1, argv)
            (out / "summary.json").write_text("{}", encoding="utf-8")


def _manifest():
    config = {"n": 20, "seed": 0, "threshold": 0.01, "nbits": 8, "max_rounds": 5, "batch_size": 16}
    return Manifest(Path("results/headline_x/manifest.json"), {"headline_id": "headline_x", "config": config, "windows": {}})


def test_window_runs_search_then_repair_and_records_both(in_tmp):
    manifest, runner = _manifest(), FakeRunner()
    run_window(manifest, "age_heavy", _flipped(), Path("results"), runner=runner)
    state = Manifest.load(manifest.path).data["windows"]["age_heavy"]
    assert state["search_run_id"] == "search_1"
    assert state["repair_run_id"] == "hqq_repaired_2"
    assert runner.calls[1][runner.calls[1].index("--search-summary") + 1] == "results/search_1/summary.json"


def test_finished_window_is_skipped_on_resume(in_tmp):
    manifest, runner = _manifest(), FakeRunner()
    run_window(manifest, "age_heavy", _flipped(), Path("results"), runner=runner)
    run_window(manifest, "age_heavy", _flipped(), Path("results"), runner=runner)
    assert len(runner.calls) == 2


def test_interrupted_repair_is_resumed_in_place(in_tmp):
    manifest, runner = _manifest(), FakeRunner(fail_repair_once=True)
    with pytest.raises(subprocess.CalledProcessError):
        run_window(manifest, "ses_heavy", _flipped(), Path("results"), runner=runner)
    assert Manifest.load(manifest.path).data["windows"]["ses_heavy"]["repair_run_id"] == "hqq_repaired_2"

    run_window(manifest, "ses_heavy", _flipped(), Path("results"), runner=runner)
    assert runner.calls[-1][-2:] == ["--resume", "hqq_repaired_2"]
    assert manifest.data["windows"]["ses_heavy"]["repair_run_id"] == "hqq_repaired_2"
    assert len(runner.calls) == 3  # the search was not re-run


def test_empty_search_skips_repair(in_tmp):
    manifest, runner = _manifest(), FakeRunner(kept=())
    run_window(manifest, "balanced", _flipped(), Path("results"), runner=runner)
    assert manifest.data["windows"]["balanced"]["repair_skipped"] == "search kept no modules"
    assert len(runner.calls) == 1


def test_dry_run_runs_nothing_and_saves_no_manifest(in_tmp):
    manifest, runner = _manifest(), FakeRunner()
    run_window(manifest, "balanced", _flipped(), Path("results"), runner=runner, dry_run=True)
    assert runner.calls == []
    assert not manifest.path.exists()


# --- overlap verdict -----------------------------------------------------------------------------

def test_overlap_identical_ignores_keep_order():
    out = localization_overlap({"a": [Q.format(1), Q.format(2)], "b": [Q.format(2), Q.format(1)]})
    assert out["verdict"] == VERDICT_IDENTICAL
    assert out["module_jaccard"] == {"a vs b": 1.0}


def test_overlap_disjoint_but_layer_overlap_reported():
    out = localization_overlap({"a": [Q.format(1)], "b": ["model.layers.1.self_attn.o_proj"]})
    assert out["verdict"] == VERDICT_DISJOINT
    assert out["layer_jaccard"] == {"a vs b": 1.0}
    assert out["common_layers"] == [1]


def test_overlap_partial():
    out = localization_overlap({"a": [Q.format(1), Q.format(2)], "b": [Q.format(1)], "c": [Q.format(1), Q.format(3)]})
    assert out["verdict"] == VERDICT_PARTIAL
    assert out["common_modules"] == [Q.format(1)]
    assert out["module_jaccard"]["a vs b"] == pytest.approx(0.5)


def test_overlap_needs_two_windows():
    with pytest.raises(ValueError):
        localization_overlap({"a": [Q.format(1)]})
