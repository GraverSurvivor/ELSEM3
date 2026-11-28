"""Tab 5's new controls, run through the real Gradio app: the Method choice, "Compare methods", and "Repair with
this method". The app is built on stand-in data (the real one needs the BBQ download), and the jobs run the
stand-in pipelines, so this needs neither a GPU nor a network."""

import time
from unittest.mock import MagicMock

import pytest

pytest.importorskip("torch")
pytest.importorskip("gradio")


def wait(runner, job, seconds=60):
    end = time.time() + seconds
    while time.time() < end:
        if runner.state(job)["status"] in ("done", "failed", "stopped") and not runner.is_running(job):
            return
        time.sleep(0.2)
    raise AssertionError(runner.log_tail(job))


@pytest.fixture
def screen(tmp_path):
    from src.attribution.patch import load_patch
    from src.product import studio
    from src.product.api_server import ApiService
    from src.product.byom import JobRunner

    data, backend = MagicMock(), MagicMock()
    data.patch = load_patch(studio.SEARCH_PATCH)
    data.random_patches, data.examples, data.example_pool = [], [], []
    data.counter_items, data.trials, data.groups, data.saved_repairs, data.twins, data.items_by_id = [], [], [], {}, {}, {}
    gate = {"n": 1000, "verdict": "PASS", "usr": {"delta": 0.0}, "srs": {"delta": 0.0},
            "bias_score": {"candidate": 0.1, "reference": 0.1}, "failures": []}
    data.gate_compressed = data.gate_repaired = gate
    backend.mode_label, backend.live, backend.reloading = "test", False, False
    backend.overhead_mb.return_value = 81.0
    backend.restored.return_value = list(data.patch.modules)
    runner = JobRunner(tmp_path / "jobs", pipeline="fake", speed=3000)
    app, _ = studio.build_app(backend, data, studio.ByomSetup(runner), ApiService())
    handlers = {f.fn.__name__: f.fn for f in app.fns.values() if getattr(f, "fn", None) is not None}
    return app, handlers, runner


def test_the_new_controls_are_on_the_screen(screen):
    app, handlers, _ = screen
    labels = {getattr(b, "label", None): type(b).__name__ for b in app.blocks.values()}
    assert labels["Methods to compare"] == "CheckboxGroup" and labels["Repair with"] == "Dropdown"
    assert any(label and label.startswith("Compression method") for label in labels)
    buttons = [b.value for b in app.blocks.values() if type(b).__name__ == "Button"]
    assert "▶ Compare methods" in buttons and "Repair with this method" in buttons
    assert {"compare_start", "compare_refresh", "compare_repair_start", "byom_start"} <= set(handlers)


def test_compare_then_repair_with_the_fairest_method(screen):
    from src.product import byom

    _, handlers, runner = screen
    picked = ["HQQ 3-bit", "HQQ 4-bit", "RTN 4-bit", "bitsandbytes NF4 4-bit"]
    handlers["compare_start"]("Qwen/Qwen2.5-3B-Instruct", picked, "")
    job = runner.current()
    assert runner.job_info(job)["kind"] == "compare" and runner.job_info(job)["methods"] == "hqq:3,hqq:4,rtn:4,bnb_nf4:4"
    wait(runner, job)

    table, chart, choice, button = handlers["compare_refresh"]()
    assert "RTN 4-bit" in table and "<svg" in chart and button["visible"] is True
    assert [value for _, value in choice["choices"]] == ["hqq-4", "rtn-4", "hqq-3"]  # no bitsandbytes: can't repair yet

    handlers["compare_repair_start"]("rtn-4", byom.MODES["fast"], 0.10, "")
    repair = runner.current()
    try:
        assert repair != job and runner.job_info(repair)["method"] == "rtn" and runner.job_info(repair)["base_nbits"] == 4
    finally:
        runner.cancel(repair)


def test_a_diagnosis_takes_the_method_choice_and_refuses_what_the_repair_cannot_use(screen):
    from src.product import byom

    _, handlers, runner = screen
    handlers["byom_start"]("Qwen/Qwen2.5-3B-Instruct", byom.MODES["fast"], byom.BITS[3], byom.BUDGETS[0.10], "",
                           byom.METHODS["rtn"])
    job = runner.current()
    try:
        assert runner.job_info(job)["method"] == "rtn"
    finally:
        runner.cancel(job)
    before = runner.current()
    handlers["byom_start"]("Qwen/Qwen2.5-3B-Instruct", byom.MODES["fast"], byom.BITS[0], byom.BUDGETS[0.10], "",
                           byom.METHODS["bnb_nf4"])  # refused with a warning; nothing starts
    assert runner.current() == before


def test_compare_with_nothing_ticked_starts_nothing(screen):
    _, handlers, runner = screen
    handlers["compare_start"]("Qwen/Qwen2.5-3B-Instruct", [], "")
    assert runner.current() is None
