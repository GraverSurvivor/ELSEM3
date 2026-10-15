"""Studio's "Bring your own model" screen: reading the pipeline's files (contract C4), the views built
from them, and the job runner, run end to end against the stand-in pipeline (no GPU)."""

import json
import time
from datetime import datetime, timedelta, timezone

import pytest

from src.product import byom
from src.product.byom import (
    JobBusy,
    JobRunner,
    check_mode,
    check_model_id,
    failure_message,
    format_duration,
    grid_spec,
    job_state,
    kept_grid_html,
    llamacpp_lines,
    parse_module,
    progress_html,
    read_events,
    read_report,
    results_html,
)

T0 = datetime(2026, 10, 16, 9, 0, tzinfo=timezone.utc)


def ev(seconds, stage, status, done=0, total=0, message=""):
    return {"time": (T0 + timedelta(seconds=seconds)).isoformat(), "stage": stage, "status": status,
            "done": done, "total": total, "message": message}


# --- input ---------------------------------------------------------------------------------------

def test_model_names_are_checked_before_anything_runs():
    assert check_model_id("  Qwen/Qwen2.5-3B-Instruct ") == "Qwen/Qwen2.5-3B-Instruct"
    assert check_model_id("https://huggingface.co/microsoft/Phi-3.5-mini-instruct/") == "microsoft/Phi-3.5-mini-instruct"
    for bad in ("", "   ", "not a model name", "qwen", "a/b/c", "owner/name; rm -rf /", "../../etc/passwd"):
        with pytest.raises(ValueError):
            check_model_id(bad)
    assert check_mode("fast") == "fast" and check_mode(byom.MODES["exact"]) == "exact"
    with pytest.raises(ValueError):
        check_mode("turbo")


# --- events --------------------------------------------------------------------------------------

def test_a_half_written_line_is_skipped_not_an_error(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text(json.dumps(ev(0, "download", "started")) + "\n" + '{"time": "2026-10-16T09:00:05", "sta',
                    encoding="utf-8")
    assert [e["stage"] for e in read_events(path)] == ["download"]
    path.write_text(json.dumps(ev(0, "nonsense", "started")) + "\nnot json\n", encoding="utf-8")
    assert read_events(path) == []
    assert read_events(tmp_path / "missing.jsonl") == []


def test_job_state_follows_the_stages():
    events = [ev(0, "download", "started"), ev(60, "download", "done"),
              ev(60, "quantize", "started", message="HQQ 3-bit"), ev(100, "quantize", "progress", 40, 129)]
    state = job_state(events, now=T0 + timedelta(seconds=130))
    stages = {s.id: s for s in state["stages"]}
    assert state["status"] == "running" and state["current"].id == "quantize"
    assert stages["download"].status == "done" and stages["download"].seconds(state["now"]) == 60
    assert (stages["quantize"].done, stages["quantize"].total, stages["quantize"].message) == (40, 129, "HQQ 3-bit")
    assert stages["quantize"].seconds(state["now"]) == 70  # still running: counts to now
    assert stages["scan"].status == "waiting"
    assert state["elapsed"] == 130


def test_finished_failed_cancelled_and_stopped_jobs():
    done = [ev(0, s, "started") for s, _ in byom.STAGES] + [ev(500, s, "done") for s, _ in byom.STAGES]
    assert job_state(done)["status"] == "done"
    failed = [ev(0, "download", "started"), ev(5, "download", "failed", message="model not found")]
    state = job_state(failed)
    assert state["status"] == "failed" and state["current"].id == "download"
    assert "model not found" in failure_message(state)

    running = [ev(0, "scan", "started"), ev(30, "scan", "progress", 10, 70)]
    later = T0 + timedelta(hours=2)
    for status, kwargs in (("cancelled", {"cancelled": True}), ("stopped", {"process_alive": False})):
        state = job_state(running, now=later, **kwargs)
        scan = next(s for s in state["stages"] if s.id == "scan")
        assert state["status"] == status
        assert scan.status == "stopped" and scan.seconds(later) == 30  # frozen, not still counting
        assert state["elapsed"] == 30
    assert "GPU memory" in failure_message(job_state(running, process_alive=False), "CUDA out of memory")
    assert job_state([])["status"] == "idle"


def test_durations_read_naturally():
    assert format_duration(5) == "0:05" and format_duration(754) == "12:34" and format_duration(3725) == "1:02:05"
    assert format_duration(None) == "" and format_duration(-3) == "0:00"


def test_progress_view_shows_each_stage():
    message = "Step 3 of 3 · Choosing among the best 12, round 2 of 5: 1 part kept so far · about 6 min left"
    state = job_state([ev(0, "score_3bit", "started"), ev(5, "score_3bit", "progress", 640, 2000),
                       ev(9, "score_3bit", "done", 2000, 2000, "2,000 questions · 11.6 a second"),
                       ev(10, "scan", "started"), ev(30, "scan", "progress", 6435, 12870, message)],
                      now=T0 + timedelta(seconds=40))
    page = progress_html(state, {"model_id": "Qwen/Qwen2.5-3B-Instruct", "mode": "fast"})
    assert "Running · Qwen/Qwen2.5-3B-Instruct · fast mode" in page
    assert message in page and 'style="width:50%"' in page  # the words, not "6,435 / 12,870"
    assert "6,435" not in page and "2,000 questions · 11.6 a second" in page
    running_count = progress_html(job_state([ev(0, "verify", "started"), ev(3, "verify", "progress", 250, 1000)]))
    assert "250 / 1,000" in running_count  # no words: the count
    assert all(label in page for _, label in byom.STAGES)
    assert "<script" not in progress_html(job_state([ev(0, "scan", "started", message="<script>x</script>")]))


# --- report and results ----------------------------------------------------------------------------

REPORT = {
    "model_id": "Qwen/Qwen2.5-3B-Instruct", "arch": "qwen2", "mode": "fast", "questions": 1000, "gpu": "Tesla T4",
    "minutes": 19.4, "safe": 840, "flipped": 280,
    "original": {"usr": 0.84, "srs": 0.117, "false_unknown": 0.073, "accuracy": 0.915},
    "compressed": {"usr": 0.365, "srs": 0.373, "false_unknown": 0.038, "accuracy": 0.808},
    "repaired": {"usr": 0.773, "srs": 0.142, "false_unknown": 0.109, "accuracy": 0.837},
    "recovery": {"point": 0.859, "lower": 0.844, "upper": 0.873},
    "kept": ["model.layers.3.self_attn.q_proj", "model.layers.20.self_attn.v_proj"],
    "memory_mb": 12.0, "gate": "FAIL", "gate_failures": ["'unknown' answers fell by 6.7 points (limit 2.0)"],
    "patch": "patch.json", "n_layers": 36,
    "part_types": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                   "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"],
}


def test_report_is_read_only_when_complete(tmp_path):
    assert read_report(tmp_path) is None
    (tmp_path / "report.json").write_text('{"model_id": "x"', encoding="utf-8")  # still being written
    assert read_report(tmp_path) is None
    (tmp_path / "report.json").write_text(json.dumps({"model_id": "x"}), encoding="utf-8")
    with pytest.raises(ValueError, match="missing"):
        read_report(tmp_path)
    (tmp_path / "report.json").write_text(json.dumps(REPORT), encoding="utf-8")
    assert read_report(tmp_path)["arch"] == "qwen2"


def test_results_view_has_cards_recovery_and_the_verdict():
    page = results_html(REPORT)
    assert page.count('class="by-card"') == 3
    assert "84.0%" in page and "36.5%" in page and "77.3%" in page
    assert "85.9%" in page and "84.4%" in page and "87.3%" in page
    assert "by-gate-fail" in page and "6.7 points" in page
    assert "280" in page and "+12 MB" in page and "Tesla T4" in page
    assert "not a real measurement" not in page
    assert "not a real measurement" in results_html({**REPORT, "synthetic": True})
    assert "recorded real run" in results_html({**REPORT, "replayed_from": "qwen_run, recorded 19 Oct 2026"})
    assert "PASS" in results_html({**REPORT, "gate": "PASS", "gate_failures": []})


def test_the_grid_fits_any_model():
    assert parse_module("model.layers.13.self_attn.qkv_proj") == (13, "self_attn.qkv_proj")
    with pytest.raises(ValueError):
        parse_module("lm_head")
    parts, layers, note = grid_spec(REPORT)
    assert (len(parts), layers, note) == (7, 36, "")
    page = kept_grid_html(REPORT)
    assert page.count('<tr><th class="by-rowlabel">') == 1 + 7  # the Layer header row + 7 part types
    assert page.count('class="by-cell') == 7 * 36
    assert page.count("by-kept") == 2 and "layer 3 attention query, layer 20 attention value" in page

    bare = {k: v for k, v in REPORT.items() if k not in ("n_layers", "part_types")}  # report without them
    parts, layers, note = grid_spec(bare)
    assert parts == ["self_attn.q_proj", "self_attn.v_proj"] and layers == 21 and "doesn't say" in note
    phi = {**bare, "kept": ["model.layers.1.self_attn.qkv_proj", "model.layers.10.self_attn.o_proj"]}
    assert grid_spec(phi)[0] == ["self_attn.o_proj", "self_attn.qkv_proj"]


def test_llamacpp_settings_for_each_architecture():
    phi = {"nbits": 8, "modules": ["model.layers.1.self_attn.qkv_proj", "model.layers.10.self_attn.o_proj",
                                   "model.layers.2.mlp.gate_up_proj", "model.layers.3.mlp.down_proj"]}
    assert llamacpp_lines(phi) == ['--tensor-type "blk\\.1\\.attn_qkv=q8_0"', '--tensor-type "blk\\.10\\.attn_output=q8_0"',
                                   '--tensor-type "blk\\.2\\.ffn_up=q8_0"', '--tensor-type "blk\\.3\\.ffn_down=q8_0"']
    qwen = {"nbits": 8, "modules": ["model.layers.20.self_attn.v_proj", "model.layers.4.mlp.gate_proj"]}
    assert llamacpp_lines(qwen) == ['--tensor-type "blk\\.20\\.attn_v=q8_0"', '--tensor-type "blk\\.4\\.ffn_gate=q8_0"']
    with pytest.raises(ValueError):
        llamacpp_lines({"nbits": 8, "modules": ["model.layers.1.mlp.mystery_proj"]})
    with pytest.raises(ValueError):
        llamacpp_lines({"nbits": 7, "modules": []})


def test_the_studio_gguf_names_match_fairgate_for_phi():
    from src.product.fairgate import LLAMACPP_TENSORS, LLAMACPP_TYPES

    for part, name in LLAMACPP_TENSORS.items():
        assert byom.GGUF_NAMES[part] == name
    assert byom.GGUF_TYPES == LLAMACPP_TYPES


# --- the job runner, end to end with the stand-in pipeline ----------------------------------------

def _wait(runner, job, statuses, seconds=60):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        state = runner.state(job)
        if state["status"] in statuses:
            return state
        time.sleep(0.2)
    raise AssertionError(f"job still {runner.state(job)['status']} after {seconds}s")


def test_a_job_runs_to_the_end_and_a_restart_still_shows_it(tmp_path):
    runner = JobRunner(tmp_path, pipeline="fake", speed=3000)
    job = runner.start("Qwen/Qwen2.5-3B-Instruct", "fast")
    state = _wait(runner, job, {"done", "failed", "stopped"})
    assert state["status"] == "done", runner.log_tail(job)
    report = read_report(job)
    assert report["synthetic"] and report["arch"] == "qwen2" and len(report["kept"]) == 5
    assert set(byom.REPORT_KEYS) <= set(report)
    assert len(llamacpp_lines(json.loads((job / "patch.json").read_text(encoding="utf-8")))) == 5

    fresh = JobRunner(tmp_path, pipeline="fake")  # the app restarted
    assert fresh.current() == job and fresh.state(job)["status"] == "done"
    assert fresh.job_info(job)["model_id"] == "Qwen/Qwen2.5-3B-Instruct"


def test_one_job_at_a_time_and_cancel(tmp_path):
    runner = JobRunner(tmp_path, pipeline="fake", speed=10)
    job = runner.start("Qwen/Qwen2.5-3B-Instruct", "exact")
    try:
        _wait(runner, job, {"running"})
        with pytest.raises(JobBusy, match="one model at a time"):
            runner.start("meta-llama/Llama-3.2-3B-Instruct", "fast")
        assert runner.cancel() is True
        state = runner.state(job)
        assert state["status"] == "cancelled" and not runner.is_running(job)
        assert all(s.status != "running" for s in state["stages"])
        assert runner.cancel() is False  # nothing left to stop
    finally:
        runner.cancel(job)
    second = runner.start("meta-llama/Llama-3.2-3B-Instruct", "fast")  # free again
    runner.cancel(second)
    assert runner.current() == second


def test_a_failing_stage_is_reported_with_its_name(tmp_path, monkeypatch):
    real_command = byom.pipeline_command

    def failing(*args, **kwargs):
        return real_command(*args, **kwargs) + ["--fail-at", "scan"]

    monkeypatch.setattr(byom, "pipeline_command", failing)
    runner = JobRunner(tmp_path, pipeline="fake", speed=3000)
    job = runner.start("Qwen/Qwen2.5-3B-Instruct", "fast")
    state = _wait(runner, job, {"failed", "done", "stopped"})
    assert state["status"] == "failed" and state["current"].id == "scan"
    assert "Scan the parts" in failure_message(state) and read_report(job) is None


def test_a_recorded_run_replays_with_its_banner(tmp_path):
    recorder = JobRunner(tmp_path / "recorded", pipeline="fake", speed=3000)
    recorded = recorder.start("Qwen/Qwen2.5-3B-Instruct", "fast")
    _wait(recorder, recorded, {"done"})
    report = json.loads((recorded / "report.json").read_text(encoding="utf-8"))
    report.pop("synthetic")  # pretend it was a real run
    (recorded / "report.json").write_text(json.dumps(report), encoding="utf-8")

    player = JobRunner(tmp_path / "demo", pipeline="replay", replay=recorded, speed=3000)
    job = player.start("Qwen/Qwen2.5-3B-Instruct", "fast")
    assert _wait(player, job, {"done", "failed", "stopped"})["status"] == "done"
    replayed = read_report(job)
    assert replayed["replayed_from"].startswith(recorded.name) and replayed["kept"] == report["kept"]
    assert "in the recorded run" in read_events(job / "events.jsonl")[-1]["message"]
    assert "recorded real run" in results_html(replayed)


def test_the_real_pipeline_is_required_unless_a_stand_in_is_chosen(tmp_path, monkeypatch):
    monkeypatch.setattr(byom, "real_pipeline_available", lambda: False)
    with pytest.raises(RuntimeError, match="byom-pipeline fake"):
        JobRunner(tmp_path, pipeline="real").start("Qwen/Qwen2.5-3B-Instruct", "fast")
    with pytest.raises(ValueError):
        JobRunner(tmp_path, pipeline="fake").start("bad name", "fast")
    assert not any(tmp_path.iterdir())  # nothing was created for a refused start
