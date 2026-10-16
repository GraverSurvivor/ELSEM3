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
