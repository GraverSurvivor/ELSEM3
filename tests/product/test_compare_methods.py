"""Compare methods: the ranking, the files, the job runner's view of them, and the real backend on a tiny model."""

import json

import pytest

from src.product import compare_methods as cm
from src.product.byom import job_state, read_events
from src.product.compare_methods import (CompareConfig, StandInBackend, chart_html, comparison_html, method_of_id,
                                         parse_methods, rank_methods, read_comparison, refusal, repair_choices,
                                         repair_refusal, run, synthetic_questions)

ORIGINAL = {"usr": 0.90, "srs": 0.05, "false_unknown": 0.05, "accuracy": 0.90}


def config(tmp_path, methods=None, **kw):
    return CompareConfig(model_id="x/tiny", methods=tuple(methods or cm.OFFERED), questions=100, out=tmp_path / "job", **kw)


class Recording(StandInBackend):
    """A stand-in that keeps what it was asked, to check the rules of the run."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.loaded_at_compress, self.asked_ids, self.original_asks = [], [], 0

    def ask_original(self, model_id, items, on_progress):
        self.original_asks += 1
        return super().ask_original(model_id, items, on_progress)

    def compress(self, model_id, name, bits):
        self.loaded_at_compress.append(self.current)  # None = nothing else was on the GPU
        return super().compress(model_id, name, bits)

    def ask(self, items, on_progress):
        self.asked_ids.append([it.question_id for it in items])
        return super().ask(items, on_progress)


# --- which combinations are allowed ---------------------------------------------------------------------

def test_methods_are_parsed_and_unsupported_ones_refused_with_a_reason():
    assert parse_methods("all") == list(cm.OFFERED) and parse_methods("") == list(cm.OFFERED)
    assert parse_methods("hqq:3, rtn:4, hqq:3") == [("hqq", 3), ("rtn", 4)]
    with pytest.raises(ValueError, match=r"bitsandbytes NF4 compresses to 4-bit, not 3-bit"):
        parse_methods("bnb_nf4:3")
    with pytest.raises(ValueError, match="should look like method:bits"):
        parse_methods("hqq")
    with pytest.raises(ValueError, match="Unknown compression method 'awq'"):
        parse_methods("awq:4")


def test_a_method_that_cannot_be_built_layer_by_layer_is_refused_for_a_big_model():
    assert refusal("bnb_nf4", 4) == "" and refusal("hqq", 3, too_big=True) == ""
    assert "whole 16-bit model loaded first" in refusal("bnb_nf4", 4, too_big=True)


def test_the_repair_takes_hqq_and_rtn_and_says_why_not_for_the_rest():
    assert repair_refusal("hqq", 3) == repair_refusal("rtn", 4) == ""
    assert "restores parts only to plain 16-bit" in repair_refusal("bnb_nf4", 4)
    assert "compresses to 4-bit" in repair_refusal("bnb_nf4", 3)
    assert "can't use HQQ 2-bit yet" in repair_refusal("hqq", 2)


# --- ranking ----------------------------------------------------------------------------------------------

def row(name, size, **card):
    base = {"usr": 0.90, "srs": 0.05, "false_unknown": 0.05, "accuracy": 0.90, **card}
    return {"id": name, "method": "hqq", "label": name, "bits": 3, "status": "ok", "size_mb": size, **base}


def test_rank_is_by_distance_from_the_original_then_size_and_marks_the_frontier():
    rows = [row("big-fair", 3000, usr=0.89), row("small-bad", 1000, usr=0.50), row("mid-fair", 1500, usr=0.89),
            row("mid-bad", 1600, usr=0.60), {"id": "gone", "status": "failed", "label": "gone"}]
    ranked = rank_methods(ORIGINAL, rows)
    assert [r["id"] for r in ranked] == ["mid-fair", "big-fair", "mid-bad", "small-bad", "gone"]  # a tie goes to the smaller
    assert [r["rank"] for r in ranked[:4]] == [1, 2, 3, 4] and "rank" not in ranked[4]
    assert {r["id"]: r["on_frontier"] for r in ranked[:4]} == {
        "mid-fair": True, "big-fair": False, "mid-bad": False, "small-bad": True}
    assert ranked[0]["gap"] == 0.01  # 0.90 -> 0.89 fair answers lost


def test_being_fairer_than_the_original_earns_nothing_and_hedging_costs():
    fairer = row("fairer", 1, usr=0.99, srs=0.0)
    hedger = row("hedger", 1, usr=0.95, false_unknown=0.20, accuracy=0.80)
    ranked = {r["id"]: r for r in rank_methods(ORIGINAL, [fairer, hedger])}
    assert ranked["fairer"]["gap"] == 0.0
    assert ranked["hedger"]["gap"] == pytest.approx(0.15)
    assert ranked["hedger"]["notes"] == [
        "says \"can't tell\" on questions the story does answer 15 points more than the original",
        "right answers fell 10 points"]


# --- a run ------------------------------------------------------------------------------------------------

def test_every_method_is_asked_the_same_questions_one_at_a_time_and_the_original_once(tmp_path):
    backend = Recording()
    report = run(config(tmp_path), backend, synthetic_questions)
    assert backend.original_asks == 1
    assert len(backend.asked_ids) == 6 and all(ids == backend.asked_ids[0] for ids in backend.asked_ids)
    assert len(set(backend.asked_ids[0])) == 200  # 100 ambiguous + 100 stated-answer
    assert backend.loaded_at_compress == [None] * 6, "a method was built while another was still loaded"
    assert report["questions"] == {"ambiguous": 100, "stated_answer": 100}
    assert report["ranking"][0] == report["best"] == "bnb_int8-8" and report["ranking"][-1] == "rtn-3"
    assert [m["rank"] for m in report["methods"]] == [1, 2, 3, 4, 5, 6]
    assert report["synthetic"] is True and report["original"]["size_mb"] > 0


def test_the_files_are_what_the_job_runner_reads(tmp_path):
    run(config(tmp_path), StandInBackend(), synthetic_questions)
    out = tmp_path / "job"
    assert read_comparison(out) == json.loads((out / "comparison.json").read_text(encoding="utf-8"))
    events = read_events(out / "events.jsonl")
    order = [e["stage"] for e in events]
    assert order == sorted(order, key=[s for s in ("download", "quantize", "score_original", "score_3bit",
                                                    "find_damage", "scan", "verify", "report")].index)
    state = job_state(events, process_alive=False)
    assert state["status"] == "done" and all(stage.status == "done" for stage in state["stages"])
    assert any("compressing" in e["message"] for e in events) and any("asking" in e["message"] for e in events)
    for line in (out / "events.jsonl").read_text(encoding="utf-8").splitlines():
        assert set(json.loads(line)) == {"time", "stage", "status", "done", "total", "message"}


def test_a_method_that_fails_is_recorded_and_the_others_still_run(tmp_path):
    backend = StandInBackend(fail={("rtn", 3): "CUDA out of memory. Tried to allocate 2.00 GiB"})
    report = run(config(tmp_path), backend, synthetic_questions)
    failed = next(m for m in report["methods"] if m["id"] == "rtn-3")
    assert failed["status"] == "failed" and "GPU ran out of memory" in failed["reason"] and "rank" not in failed
    assert len(report["ranking"]) == 5 and report["methods"][-1]["id"] == "rtn-3"
    assert job_state(read_events(tmp_path / "job" / "events.jsonl"), process_alive=False)["status"] == "done"


def test_a_big_model_skips_what_cannot_be_built_layer_by_layer(tmp_path):
    class Big(StandInBackend):
        def prepare(self, model_id):
            return {**super().prepare(model_id), "too_big": True}

    report = run(config(tmp_path), Big(), synthetic_questions)
    skipped = [m for m in report["methods"] if m["status"] == "skipped"]
    assert {m["id"] for m in skipped} == {"bnb_nf4-4", "bnb_int8-8"}
    assert all("whole 16-bit model loaded first" in m["reason"] for m in skipped)
    assert len(report["ranking"]) == 4
    quantize = next(e for e in read_events(tmp_path / "job" / "events.jsonl") if e["stage"] == "quantize" and e["status"] == "done")
    assert "4 of 6 methods to compare" in quantize["message"] and "skipped" in quantize["message"]


def test_when_nothing_can_be_compared_the_run_still_finishes_and_says_so(tmp_path):
    backend = StandInBackend(fail={("hqq", 3): "boom"})
    report = run(config(tmp_path, [("hqq", 3)]), backend, synthetic_questions)
    assert report["best"] is None and report["ranking"] == []
    assert "No method could be compared" in comparison_html(report)
    assert "Two or more finished methods" in chart_html(report)


def test_a_failure_before_any_method_is_reported_on_its_stage(tmp_path):
    class Broken(StandInBackend):
        def prepare(self, model_id):
            raise OSError("404 Client Error: Repository Not Found")

    with pytest.raises(OSError):
        run(config(tmp_path), Broken(), synthetic_questions)
    state = job_state(read_events(tmp_path / "job" / "events.jsonl"), process_alive=False)
    assert state["status"] == "failed" and state["current"].id == "download"
    assert "No model with this name exists" in state["current"].message


def test_the_config_refuses_too_few_questions_and_no_methods(tmp_path):
    with pytest.raises(ValueError, match="at least 100"):
        CompareConfig(model_id="x/y", questions=50)
    with pytest.raises(ValueError, match="at least one method"):
        CompareConfig(model_id="x/y", methods=())


# --- the screen --------------------------------------------------------------------------------------------

@pytest.fixture
def report(tmp_path):
    return run(config(tmp_path), StandInBackend(), synthetic_questions)


def test_the_table_ranks_methods_and_shows_every_number(report):
    html = comparison_html(report)
    assert html.index("bitsandbytes int8 8-bit") < html.index("HQQ 4-bit") < html.index("RTN 3-bit")
    assert "Original (16-bit)" in html and "Fairest here" in html and "A stand-in run" in html
    best = report["methods"][0]
    assert f'{100 * best["gap"]:.1f}' in html and f'{100 * best["usr"]:.1f}%' in html
    assert "right answers fell" in html  # the notes under a damaged method


def test_the_table_shows_why_a_method_did_not_run_and_escapes_what_it_prints(report):
    report["methods"][-1].update(status="failed", reason="<script>alert(1)</script>")
    html = comparison_html(report)
    assert "Failed: &lt;script&gt;alert(1)&lt;/script&gt;" in html and "<script>" not in html


def test_the_chart_has_one_point_per_finished_method_and_a_text_alternative(report):
    svg = chart_html(report)
    assert svg.count("<circle") == 6 and "<title>Fairness against memory" in svg and "<desc>" in svg
    assert 'fill="#4f46e5"' in svg and 'fill="none"' in svg  # frontier points filled, the others hollow
    assert "Size of the compressed model" in svg and "lower is fairer" in svg.lower()


def test_only_repairable_methods_are_offered_for_repair(report):
    labels = [label for label, _ in repair_choices(report)]
    assert labels == ["HQQ 4-bit", "RTN 4-bit", "HQQ 3-bit", "RTN 3-bit"]
    assert method_of_id(report, "rtn-4") == ("rtn", 4) and method_of_id(report, "HQQ 3-bit") == ("hqq", 3)
    with pytest.raises(ValueError, match="restores parts only to plain 16-bit"):
        method_of_id(report, "bnb_nf4-4")
    with pytest.raises(ValueError, match="Pick one of the compared methods"):
        method_of_id(report, "gptq-4")


def test_a_half_written_or_foreign_comparison_is_refused(tmp_path):
    out = tmp_path / "job"
    out.mkdir()
    assert read_comparison(out) is None
    (out / "comparison.json").write_text('{"methods": [', encoding="utf-8")
    with pytest.raises(ValueError, match="couldn't be read"):
        read_comparison(out)
    (out / "comparison.json").write_text('{"hello": 1}', encoding="utf-8")
    with pytest.raises(ValueError, match="isn't a comparison"):
        read_comparison(out)


def test_the_command_line_runs_the_stand_in(tmp_path):
    assert cm.main(["--model", "x/y", "--fake", "--questions", "100", "--methods", "hqq:3,rtn:3",
                    "--out", str(tmp_path / "cli")]) == 0
    assert json.loads((tmp_path / "cli" / "comparison.json").read_text(encoding="utf-8"))["ranking"] == ["hqq-3", "rtn-3"]
    with pytest.raises(SystemExit):
        cm.main(["--model", "x/y", "--methods", "bnb_nf4:3", "--out", str(tmp_path / "bad")])


# --- the real backend, on a tiny model ---------------------------------------------------------------------

def test_the_real_backend_compares_hqq_and_rtn_on_a_tiny_model(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("hqq")
    from tests.quant.tiny_models import save_tiny

    folder = tmp_path / "tiny"
    save_tiny(folder, "qwen2")
    backend = cm.TorchCompareBackend(batch_size=8, device="cpu")
    report = run(CompareConfig(model_id=str(folder), methods=(("hqq", 3), ("rtn", 3)), questions=100,
                               out=tmp_path / "real", device="cpu"), backend, synthetic_questions)
    assert report["synthetic"] is False and [m["status"] for m in report["methods"]] == ["ok", "ok"]
    hqq, rtn = (next(m for m in report["methods"] if m["id"] == i) for i in ("hqq-3", "rtn-3"))
    assert hqq["size_mb"] == rtn["size_mb"] > 0 and report["original"]["size_mb"] > hqq["size_mb"]  # same container
    assert (hqq["usr"], hqq["srs"]) != (rtn["usr"], rtn["srs"]) or hqq["accuracy"] != rtn["accuracy"]  # different rounding
    for m in (hqq, rtn):
        assert 0 <= m["usr"] <= 1 and m["minutes"] > 0
    assert job_state(read_events(tmp_path / "real" / "events.jsonl"), process_alive=False)["status"] == "done"


def test_the_real_backend_includes_bitsandbytes_where_it_runs(tmp_path):
    pytest.importorskip("torch")
    from tests.quant.tiny_models import bnb_runs, save_tiny

    if not bnb_runs():
        pytest.skip("bitsandbytes doesn't run on this machine (needs a GPU)")
    folder = tmp_path / "tiny"
    save_tiny(folder, "qwen2")
    report = run(CompareConfig(model_id=str(folder), methods=(("bnb_nf4", 4), ("bnb_int8", 8)), questions=100,
                               out=tmp_path / "real", device="cpu"), cm.TorchCompareBackend(8, "cpu"), synthetic_questions)
    assert [m["status"] for m in report["methods"]] == ["ok", "ok"]
    nf4, int8 = (next(m for m in report["methods"] if m["id"] == i) for i in ("bnb_nf4-4", "bnb_int8-8"))
    assert 0 < nf4["size_mb"] < int8["size_mb"] < report["original"]["size_mb"]
