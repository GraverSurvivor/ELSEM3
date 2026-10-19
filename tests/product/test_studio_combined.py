"""One window for both: the live Phi-3.5 demo (tabs 1-4) and tab 5 on the same GPU.

While a tab-5 job runs, the demo's 16-bit model steps aside (a T4 can't hold it and the customer's
model) and the original's answers come from the saved full-benchmark run; it loads back when the job
ends. Checked here on CPU with the live backend's GPU parts replaced."""

import pytest

pytest.importorskip("gradio")

from src.data.bbq_loader import BBQItem  # noqa: E402
from src.product import studio  # noqa: E402
from src.product.studio import LiveBackend, after_job, make_room_for_job  # noqa: E402
from src.product.studio_core import Answer  # noqa: E402


def _item(i, category="Age"):
    return BBQItem(example_id=i, category=category, question_polarity="neg", context=f"story {i}", question="Who?",
                   options=("The old one", "The young one", "Unknown"), label=2, unknown_index=2,
                   stereotype_index=0)


class Live16:
    """Stands in for the 16-bit model: LiveBackend._score is replaced, so it is never called."""


def _backend(monkeypatch):
    backend = object.__new__(LiveBackend)
    backend.original, backend.wants_original, backend.model = Live16(), True, Live16()
    backend.saved_original = {_item(i).question_id: (2 if i % 2 else 0) for i in range(10)}
    backend.original_away, backend.reloading = False, False
    backend._cache, backend._plain = {}, {}
    backend.registry = type("R", (), {"active": {}})()
    monkeypatch.setattr(LiveBackend, "_score", lambda self, model, items: [Answer(2) for _ in items])
    return backend


class Runner:
    def __init__(self, running):
        self.running = running

    def current(self):
        return "job" if self.running is not None else None

    def is_running(self, job):
        return self.running


def test_the_original_steps_aside_for_a_job_and_answers_from_the_saved_run(monkeypatch):
    backend = _backend(monkeypatch)
    asked = _item(3)
    assert backend.answers(asked, False)["original"] == Answer(2)  # live, and now cached
    make_room_for_job(backend)
    assert backend.original is None and backend.original_away and backend.has_original
    assert backend.answers(asked, False)["original"] == Answer(2)  # what the live model said
    assert backend.answers(_item(4), False)["original"] == Answer(0)  # the saved run's answer
    assert backend.answers(_item(4, "Custom"), False)["original"] is None  # no saved answer for a typed one
    assert backend.answers_for([_item(4), _item(4, "Custom")], "original") is None
    assert backend.counts([_item(i) for i in range(4)], "original") is not None
    assert backend.answers(_item(4), False)["compressed"] == Answer(2)  # the 3-bit model stays live


def test_the_original_loads_back_only_when_no_job_runs(monkeypatch):
    backend = _backend(monkeypatch)
    loads = []

    import src.eval.model_loader as model_loader

    monkeypatch.setattr(model_loader, "load_model", lambda dtype: (loads.append(dtype) or Live16(), None))
    real_reload = LiveBackend.reload_original
    monkeypatch.setattr(LiveBackend, "reload_original", lambda self, wait=False: real_reload(self, wait=True))
    make_room_for_job(backend)
    after_job(backend, Runner(running=True))
    assert backend.original is None and loads == []  # still running: stays aside
    after_job(backend, Runner(running=False))
    assert loads == ["fp16"] and backend.original is not None and not backend.original_away
    after_job(backend, Runner(running=False))
    assert loads == ["fp16"]  # back already: nothing more to do


def test_a_launch_without_the_16_bit_model_never_reloads_it(monkeypatch):
    backend = _backend(monkeypatch)
    backend.original, backend.wants_original = None, False
    make_room_for_job(backend)
    assert not backend.original_away and not backend.has_original
    after_job(backend, Runner(running=False))
    assert backend.original is None


def test_the_saved_preview_is_left_alone():
    class Saved:
        live = False

    make_room_for_job(Saved())  # no error, nothing to free
    after_job(Saved(), Runner(running=False))


def test_the_command_line_offers_one_window_for_both(monkeypatch, capsys):
    import sys

    monkeypatch.setattr(sys, "argv", ["studio", "--help"])
    with pytest.raises(SystemExit):
        studio.main()
    assert "--live" in capsys.readouterr().out


def test_ship_says_there_is_nothing_to_ship_when_nothing_was_restored(tmp_path, monkeypatch):
    """Regression: with no patch.json the Ship section showed an empty file box and a raw
    "not available: [Errno 2] ..." error."""
    import json

    from src.product.studio import byom_view

    report = {"model_id": "Qwen/Qwen2.5-0.5B-Instruct", "arch": "qwen2", "mode": "fast", "kept": [], "gate": "N/A",
              "gate_failures": [], "original": {}, "compressed": {}, "repaired": {},
              "recovery": {"point": None, "lower": None, "upper": None}, "outcome": "not_applicable",
              "outcome_reason": "x", "other_damage": []}
    (tmp_path / "report.json").write_text(json.dumps(report), encoding="utf-8")

    class Done:
        def current(self):
            return tmp_path

        def state(self, job_dir):
            return {"status": "done", "stages": [], "elapsed": 0, "now": None, "current": None}

        def job_info(self, job_dir):
            return {"model_id": report["model_id"], "mode": "fast"}

        def log_tail(self, job_dir):
            return ""

    monkeypatch.setattr(studio, "progress_html", lambda state, info=None: "")  # the stages aren't checked here
    _, _, _, results, grid, patch_file, llama = byom_view(Done())
    assert patch_file is None and "Errno" not in llama and llama.startswith("# Nothing to ship")
    assert "No repair made" in results and "No parts were kept" in grid


def test_a_typed_question_is_answered_by_the_live_models_in_all_six_orders():
    """The live app's path for "write your own question", run on CPU with a tiny real model and Qwen's
    real tokenizer and chat template: the prompt is built, all 6 choice orders are scored, and each
    model gets one combined answer with a probability for every choice."""
    torch = pytest.importorskip("torch")
    from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2Config

    from src.product.studio_core import QUESTION_TEMPLATES, answer_orders, combine_orders, make_question

    try:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct", local_files_only=True)
    except OSError:
        pytest.skip("Qwen tokenizer not cached")
    torch.manual_seed(0)
    tiny = AutoModelForCausalLM.from_config(Qwen2Config(hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                                                        num_attention_heads=4, num_key_value_heads=2,
                                                        vocab_size=len(tok))).eval()
    backend = object.__new__(LiveBackend)
    backend.model = backend.original = tiny
    backend.tokenizer, backend.device, backend.batch_size = tok, "cpu", 4
    backend.registry = type("R", (), {"active": {}})()
    backend._cache, backend._plain, backend.saved_original = {}, {}, {}
    backend.original_away, backend.reloading, backend.wants_original = False, False, True

    for name, fields in QUESTION_TEMPLATES.items():
        item = make_question(*fields)
        orders = answer_orders(item)
        assert len(orders) == 6
        for view in ("original", "compressed", "repaired"):
            scored = backend.answers_for([o for o, _ in orders], view)
            assert len(scored) == 6 and all(a.probs is not None and abs(sum(a.probs) - 1) < 1e-4 for a in scored)
            answer, note = combine_orders(item, scored, [p for _, p in orders])
            assert answer.pick in (0, 1, 2) and len(answer.probs) == 3, name
