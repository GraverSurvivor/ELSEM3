"""Compare methods as a job: the runner starts it like a diagnosis, the Studio reads its files, "Repair with this
method" starts a repair, and the Method choice refuses what the repair can't do, with a reason."""

import json
import time

import pytest

from src.product import byom
from src.product.byom import JobBusy, JobRunner, check_method, pipeline_command, read_events
from src.product.compare_methods import read_comparison


def wait(runner, job, statuses, seconds=60):
    """Until the job reaches a status (and, for a finished one, its process has exited: a new job can then start)."""
    end = time.time() + seconds
    while time.time() < end:
        state = runner.state(job)
        if state["status"] in statuses and (state["status"] == "running" or not runner.is_running(job)):
            return state
        time.sleep(0.2)
    raise AssertionError(f"job still {runner.state(job)['status']} after {seconds}s: {runner.log_tail(job)}")


# --- the Method choice ------------------------------------------------------------------------------------

def test_the_method_choice_accepts_hqq_and_rtn_and_refuses_the_rest_with_a_reason():
    assert check_method("hqq", 0) == "hqq" and check_method(byom.METHODS["hqq"], 3) == "hqq"
    assert check_method("rtn", 4) == "rtn" and check_method(byom.METHODS["rtn"], 0) == "rtn"
    with pytest.raises(ValueError, match="restores parts only to plain 16-bit"):
        check_method("bnb_nf4", 0)
    with pytest.raises(ValueError, match="compresses to 4-bit, not 3-bit"):
        check_method(byom.METHODS["bnb_nf4"], 3)
    with pytest.raises(ValueError, match="Pick a compression method"):
        check_method("awq", 3)


def test_only_a_non_hqq_method_changes_the_pipeline_command(tmp_path):
    base = pipeline_command("real", "a/b", "fast", 1000, tmp_path)
    assert "--method" not in base  # exactly the command it always was
    assert pipeline_command("real", "a/b", "fast", 1000, tmp_path, method="rtn")[-2:] == ["--method", "rtn"]
    assert "--method" not in pipeline_command("fake", "a/b", "fast", 1000, tmp_path, method="rtn")


def test_a_repair_starts_with_the_method_and_refuses_one_it_cannot_use(tmp_path):
    runner = JobRunner(tmp_path, pipeline="fake", speed=3000)
    job = runner.start("Qwen/Qwen2.5-3B-Instruct", "fast", 3, 0.10, method="rtn")
    assert runner.job_info(job)["method"] == "rtn"
    runner.cancel(job)
    with pytest.raises(ValueError, match="restores parts only to plain 16-bit"):
        runner.start("Qwen/Qwen2.5-3B-Instruct", "fast", 0, 0.10, method="bnb_int8")
    assert len([p for p in tmp_path.iterdir() if p.is_dir()]) == 1  # nothing was started for the refused one


# --- a comparison job -------------------------------------------------------------------------------------

def test_a_comparison_runs_as_a_job_and_the_screen_reads_it(tmp_path):
    runner = JobRunner(tmp_path, pipeline="fake")
    job = runner.start_compare("Qwen/Qwen2.5-3B-Instruct", "hqq:3,hqq:4,rtn:3", questions=100)
    state = wait(runner, job, {"done", "failed", "stopped"})
    assert state["status"] == "done", runner.log_tail(job)
    info = runner.job_info(job)
    assert info["kind"] == "compare" and info["methods"] == "hqq:3,hqq:4,rtn:3" and "--fake" in info["command"]
    report = read_comparison(job)
    assert report["ranking"] == ["hqq-4", "hqq-3", "rtn-3"] and report["synthetic"] is True
    assert all(s.status == "done" for s in state["stages"])
    assert byom.read_report(job) is None  # a comparison has no report.json, and the repair view copes with that

    fresh = JobRunner(tmp_path, pipeline="fake")  # the app restarted
    assert fresh.current() == job and fresh.state(job)["status"] == "done"


def test_a_comparison_refuses_what_cannot_run_before_starting_and_takes_the_gpu_alone(tmp_path):
    runner = JobRunner(tmp_path, pipeline="fake", speed=10)
    with pytest.raises(ValueError, match="compresses to 4-bit, not 3-bit"):
        runner.start_compare("Qwen/Qwen2.5-3B-Instruct", "bnb_nf4:3")
    with pytest.raises(ValueError, match="isn't a Hugging Face model name"):
        runner.start_compare("not a model", "all")
    assert not list(tmp_path.iterdir())
    job = runner.start_compare("Qwen/Qwen2.5-3B-Instruct", "all", questions=500)
    try:
        with pytest.raises(JobBusy, match="one model at a time"):
            runner.start("meta-llama/Llama-3.2-3B-Instruct", "fast")
        with pytest.raises(JobBusy):
            runner.start_compare("meta-llama/Llama-3.2-3B-Instruct", "all")
    finally:
        runner.cancel(job)
    assert not runner.is_running(job) and runner.state(job)["status"] in ("idle", "cancelled", "done")


# --- what the Studio shows --------------------------------------------------------------------------------

@pytest.fixture
def studio():
    pytest.importorskip("gradio")
    from src.product import studio

    return studio


def test_the_screen_shows_the_table_chart_and_repair_choices_of_a_finished_comparison(tmp_path, studio):
    runner = JobRunner(tmp_path, pipeline="fake")
    shown: dict = {}
    assert studio.compare_view(runner, shown)[0] == ""  # no job yet
    job = runner.start_compare("Qwen/Qwen2.5-3B-Instruct", "hqq:3,hqq:4,bnb_nf4:4", questions=100)
    wait(runner, job, {"done"})
    table, chart, choice, button = studio.compare_view(runner, shown)
    assert "HQQ 4-bit" in table and "bitsandbytes NF4 4-bit" in table and "<svg" in chart
    assert choice["choices"] == [("HQQ 4-bit", "hqq-4"), ("HQQ 3-bit", "hqq-3")]  # bitsandbytes can't be repaired yet
    assert choice["value"] == "hqq-4" and choice["visible"] is True and button["visible"] is True
    # The next tick keeps the person's choice: it leaves the dropdown and the button alone.
    _, _, again_choice, again_button = studio.compare_view(runner, shown)
    assert "choices" not in again_choice and "value" not in again_choice and "visible" not in again_button
    # A page load fills the choices in again.
    assert studio.compare_view(runner, shown, force=True)[2]["value"] == "hqq-4"


def test_a_diagnosis_job_shows_no_comparison(tmp_path, studio):
    runner = JobRunner(tmp_path, pipeline="fake", speed=3000)
    job = runner.start("Qwen/Qwen2.5-3B-Instruct", "fast")
    wait(runner, job, {"done"})
    assert studio.compare_view(runner, {})[:2] == ("", "")


def test_the_ticked_boxes_become_the_methods_text(studio):
    assert studio.methods_text(["HQQ 3-bit", "bitsandbytes NF4 4-bit", "RTN 4-bit"]) == "hqq:3,rtn:4,bnb_nf4:4"
    with pytest.raises(ValueError, match="Tick at least one method"):
        studio.methods_text([])


def test_repair_with_this_method_starts_a_repair_of_the_compared_model(tmp_path, studio):
    runner = JobRunner(tmp_path, pipeline="fake", speed=3000)
    with pytest.raises(ValueError, match="Run a comparison first"):
        studio.repair_from_comparison(runner, "hqq-4", "fast", 0.10)
    job = runner.start_compare("Qwen/Qwen2.5-3B-Instruct", "hqq:3,rtn:4,bnb_nf4:4", questions=100)
    wait(runner, job, {"done"})
    with pytest.raises(ValueError, match="restores parts only to plain 16-bit"):
        studio.repair_from_comparison(runner, "bnb_nf4-4", "fast", 0.10)
    with pytest.raises(ValueError, match="Pick one of the compared methods"):
        studio.repair_from_comparison(runner, "gptq-4", "fast", 0.10)
    repair = studio.repair_from_comparison(runner, "rtn-4", byom.MODES["fast"], 0.10)
    try:
        info = runner.job_info(repair)
        assert repair != job and info["model_id"] == "Qwen/Qwen2.5-3B-Instruct"
        assert info["method"] == "rtn" and info["base_nbits"] == 4 and "kind" not in info
    finally:
        runner.cancel(repair)
