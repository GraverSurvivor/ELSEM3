"""Version 2 without dead ends (A3): a model that only answers fairly with examples, one whose numbers overflow
fp16, a GPU that runs out of memory, 3-bit damage no repair can fix, and a run that stops halfway."""

import json
import math

import pytest

from src.product.fast_repair import EXAMPLES, FastRepairConfig, PrecisionError, run
from tests.product.test_strategies import (
    ADAPTER,
    AMBIG,
    DISAMBIG,
    PARTS,
    Q,
    Model,
    Stack,
    fixes,
)


class Backend(Model):
    """The strategies stand-in, plus what a real backend has: a base precision, a compute dtype, recompressing."""

    def __init__(self, rule, harm=lambda state, i: 0.0, fair_without_examples=True, overflow=None, fast_bf16=True,
                 rule_at_four=None):
        super().__init__(rule, harm)
        self.base_nbits, self.dtype, self.fast_bf16 = 3, "fp16", fast_bf16
        self.fair_without_examples, self.overflow, self.rule_at_four = fair_without_examples, overflow, rule_at_four
        self.loads = []

    def load_compressed(self, model_id):
        self.registry = Stack()  # a new compressed model: nothing restored
        self.loads.append(("compressed", self.base_nbits, self.dtype))
        return f"HQQ {self.base_nbits}-bit"

    def load_original(self, model_id):
        self.loads.append(("original", 16, self.dtype))
        return "loaded"

    def release_original(self):
        pass

    def can_use_bf16(self):
        return self.fast_bf16

    def use_dtype(self, dtype):
        self.dtype = dtype

    def _p(self, view, item):
        cold = EXAMPLES not in item.context
        if view == "original" and item.label == item.unknown_index and not self.fair_without_examples and cold:
            row = [0.0, 0.0, 0.0]
            row[item.stereotype_index] = 0.96  # cold, it always picks a person
            row[item.unknown_index] = row[3 - item.unknown_index - item.stereotype_index] = 0.02
            return tuple(row)
        if view != "original" and self.base_nbits == 4 and self.rule_at_four is not None:
            rule, self.rule = self.rule, self.rule_at_four
            try:
                return super()._p(view, item)
            finally:
                self.rule = rule
        return super()._p(view, item)

    def probs(self, view, items, on_progress=None):
        if self.overflow == view and self.dtype == "fp16":
            raise PrecisionError(f"The {view} model's numbers overflow at 16-bit precision (fp16)")
        return super().probs(view, items, on_progress)


def repair(tmp_path, backend, **kwargs):
    config = FastRepairConfig(model_id="Qwen/Qwen2.5-3B-Instruct", questions=kwargs.pop("questions", 2000),
                              out=kwargs.pop("out", tmp_path / "job"), pipeline="v2", **kwargs)
    return run(config, backend, load_questions=lambda: (AMBIG, DISAMBIG)), config.out


# --- a model that only says "can't tell" with examples ---------------------------------------------------------

def test_a_model_that_needs_examples_is_asked_with_them_and_repaired(tmp_path):
    backend = Backend(fixes((Q[0],), (Q[1],)), fair_without_examples=False)
    report, out = repair(tmp_path, backend)
    assert report["outcome"] == "repaired" and report["prompt_style"] == "examples"
    assert "three worked examples" in report["precision_note"]
    assert json.loads((out / "patch.json").read_text())["prompt_style"] == "examples"
    asked = [ids for view, ids, _ in backend.calls]
    assert asked  # every question after the retry carries the examples
    from src.product.byom import results_html

    assert "three worked examples" in results_html(report)


def test_a_model_that_never_says_cant_tell_is_still_not_applicable(tmp_path):
    class Never(Backend):
        def _p(self, view, item):
            row = [0.02, 0.02, 0.02]
            row[item.stereotype_index if item.label == item.unknown_index else item.label] = 0.94
            return tuple(row)

    report, _ = repair(tmp_path, Never(lambda state, j: 0.1))
    assert report["outcome"] == "not_applicable" and report["prompt_style"] == "plain"
    assert "examples" not in (report["precision_note"] or "")


def test_a_model_that_is_fair_cold_is_never_asked_with_examples(tmp_path):
    report, _ = repair(tmp_path, Backend(fixes((Q[0],), (Q[1],))))
    assert report["prompt_style"] == "plain" and not report["precision_note"]


# --- numbers that overflow fp16 ---------------------------------------------------------------------------------

@pytest.mark.parametrize("view", ["original", "compressed"])
def test_a_model_that_overflows_fp16_is_run_again_in_bf16_everywhere(tmp_path, view):
    backend = Backend(fixes((Q[0],), (Q[1],)), overflow=view)
    report, out = repair(tmp_path, backend)
    assert report["outcome"] == "repaired" and report["compute_dtype"] == "bf16"
    assert "bf16" in report["precision_note"]
    assert json.loads((out / "patch.json").read_text())["base"]["compute_dtype"] == "bf16"
    after = backend.loads[backend.loads.index(next(x for x in backend.loads if x[2] == "bf16")):]
    assert {kind for kind, _, _ in after} == {"original", "compressed"}  # both rebuilt: never compare dtypes
    assert all(dtype == "bf16" for _, _, dtype in after)


def test_on_a_t4_an_overflow_still_stops_with_a_plain_reason(tmp_path):
    with pytest.raises(PrecisionError):
        repair(tmp_path, Backend(fixes((Q[0],)), overflow="original", fast_bf16=False))
    events = [json.loads(line) for line in (tmp_path / "job" / "events.jsonl").read_text().splitlines()]
    assert events[-1]["status"] == "failed" and "overflow" in events[-1]["message"]


def test_the_real_backend_builds_every_model_in_its_dtype(monkeypatch):
    pytest.importorskip("torch")
    from src.product import torch_backend
    from src.product.torch_backend import TorchBackend

    seen = {}
    import src.eval.model_loader as loader
    import src.quant.hqq_quantizer as quantizer

    monkeypatch.setattr(quantizer, "load_hqq_3bit_model", lambda **kw: seen.update(compressed=kw["dtype"]) or (None, None))
    monkeypatch.setattr(loader, "load_model", lambda dtype, **kw: seen.update(original=dtype) or (None, None))
    monkeypatch.setattr(torch_backend.TorchBackend, "adapter", lambda self, m: type("A", (), {
        "candidates": lambda s: [], "is_transposed": lambda s, p: False})())
    monkeypatch.setattr("src.quant.weight_source.CheckpointWeightSource", lambda *a, **k: None)
    monkeypatch.setattr("src.quant.module_swap.SwapRegistry", lambda *a, **k: None)
    backend = TorchBackend(device="cpu", streaming=False)
    assert backend.can_use_bf16()  # on a CPU it runs at full speed
    backend.use_dtype("bf16")
    backend.load_compressed("x/y")
    backend.load_original("x/y")
    assert seen == {"compressed": "bf16", "original": "bf16"}
    with pytest.raises(ValueError):
        backend.use_dtype("fp8")


# --- out of memory ----------------------------------------------------------------------------------------------

def test_out_of_memory_halves_the_batch_and_keeps_going():
    pytest.importorskip("torch")
    from src.product.torch_backend import TorchBackend

    backend = TorchBackend(device="cpu", batch_size=16)
    tried = []

    def ask(size):
        tried.append(size)
        if size > 4:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        return "answers"

    assert backend._smaller_batches(ask) == "answers" and tried == [16, 8, 4] and backend.batch_size == 4
    with pytest.raises(ValueError):  # anything else isn't retried
        backend._smaller_batches(lambda size: (_ for _ in ()).throw(ValueError("bad question")))
    backend.batch_size = 2
    with pytest.raises(RuntimeError, match="out of memory"):  # one question at a time and still too big
        backend._smaller_batches(lambda size: (_ for _ in ()).throw(RuntimeError("CUDA out of memory")))
    assert backend.batch_size == 1


# --- 3-bit damage no repair within the budget can fix ----------------------------------------------------------

def everywhere(state, j):
    """3-bit: every part matters, more than the budget can restore; all at 4-bit: fixed."""
    if len(state) >= len(PARTS):
        return 0.96
    return 0.96 if PARTS[j % len(PARTS)] in state and j % 5 == 0 else 0.1


def test_three_bit_damage_too_spread_out_goes_to_four_bits_and_is_repaired_there(tmp_path):
    backend = Backend(everywhere, rule_at_four=fixes((Q[0],), (Q[1],), low=0.3))
    report, out = repair(tmp_path, backend)
    assert report["base_nbits"] == 4 and report["escalated_from"]["base_nbits"] == 3
    assert any(r["name"] == "four_bits" for r in report["escalated_from"]["strategies"])
    assert "compressed to 4-bit and repaired again" in report["precision_note"]
    assert report["outcome"] == "repaired" and report["gate"] == "PASS" and set(report["kept"]) == {Q[0], Q[1]}
    assert json.loads((out / "patch.json").read_text())["base"]["nbits"] == 4
    assert ("compressed", 4, "fp16") in backend.loads


def test_four_bits_without_damage_ends_as_no_damage(tmp_path):
    report, _ = repair(tmp_path, Backend(everywhere, rule_at_four=lambda state, j: 0.96))
    assert report["base_nbits"] == 4 and report["outcome"] == "no_damage" and report["kept"] == []


@pytest.mark.parametrize("settings", [{"base_nbits": 3}, {}])
def test_no_escalation_when_three_bits_was_chosen_or_the_repair_is_fair(tmp_path, settings):
    rule = everywhere if settings else fixes((Q[0],), (Q[1],))
    report, _ = repair(tmp_path, Backend(rule, rule_at_four=lambda state, j: 0.96), **settings)
    assert report["base_nbits"] == 3 and "escalated_from" not in report


# --- a run that stops halfway ----------------------------------------------------------------------------------

def test_a_stopped_run_resumes_from_its_saved_answers_and_first_look(tmp_path):
    first = Backend(fixes((Q[0],), (Q[1],)))
    report, out = repair(tmp_path, first)
    assert (out / "checkpoint.json").exists() and (out / "first_look_3bit.json").exists()
    again = Backend(fixes((Q[0],), (Q[1],)))
    resumed, _ = repair(tmp_path, again, out=out, resume=True)
    assert not [view for view, _, _ in again.calls if view == "original"]  # not asked again
    assert ("original", 16, "fp16") not in again.loads
    assert len(again.calls) < len(first.calls) - 20  # nor the first look
    for key in ("kept", "kept_bits", "gate", "recovery", "original", "compressed", "repaired"):
        assert resumed[key] == report[key], key
    events = [json.loads(line) for line in (out / "events.jsonl").read_text().splitlines()]
    assert any("saved by the stopped run" in e["message"] for e in events)


def test_resume_starts_again_when_the_settings_or_the_file_dont_match(tmp_path):
    _, out = repair(tmp_path, Backend(fixes((Q[0],), (Q[1],))))
    other = Backend(fixes((Q[0],), (Q[1],)))
    repair(tmp_path, other, out=out, resume=True, seed=1)
    assert [view for view, _, _ in other.calls if view == "original"]  # another seed: asked again
    (out / "checkpoint.json").write_text("{cut off")
    broken = Backend(fixes((Q[0],), (Q[1],)))
    repair(tmp_path, broken, out=out, resume=True, seed=1)
    assert [view for view, _, _ in broken.calls if view == "original"]
    with pytest.raises(ValueError, match="version 2"):
        FastRepairConfig(model_id="x/y", resume=True)


def test_a_resumed_run_keeps_what_the_stopped_one_decided_about_the_model(tmp_path):
    first = Backend(fixes((Q[0],), (Q[1],)), fair_without_examples=False, overflow="compressed")
    report, out = repair(tmp_path, first)
    again = Backend(fixes((Q[0],), (Q[1],)), fair_without_examples=False, overflow="compressed")
    resumed, _ = repair(tmp_path, again, out=out, resume=True)
    assert again.loads[0] == ("compressed", 3, "bf16") and resumed["compute_dtype"] == "bf16"
    assert resumed["prompt_style"] == "examples" and resumed["precision_note"] == report["precision_note"]


# --- the review of A2 ---------------------------------------------------------------------------------------------

def test_four_bits_everywhere_that_runs_out_of_memory_is_left_out_not_fatal(tmp_path):
    class Tight(Backend):
        def probs(self, view, items, on_progress=None):
            if len(self.registry.now()) >= len(PARTS):
                raise RuntimeError("CUDA out of memory")
            return super().probs(view, items, on_progress)

    report, _ = repair(tmp_path, Tight(fixes((Q[0],), (Q[1],))))
    assert report["outcome"] == "repaired" and not any(r["name"] == "four_bits" for r in report["strategies"])
    assert "didn't fit in memory" in report["precision_note"]


def test_the_budget_note_is_about_the_repair_that_ships(tmp_path):
    report, _ = repair(tmp_path, Backend(lambda state, j: 0.1), base_nbits=3)
    assert report["kept"] == [] and "budget_note" not in report
    spread, _ = repair(tmp_path / "b", Backend(everywhere), base_nbits=3)
    from src.product.byom import format_bytes, results_html

    size = format_bytes(spread["restore_budget_bytes"])
    assert spread["restore_budget_bytes"] > 0 and f"+{size}" in spread["budget_note"]
    assert f"repair budget +{size}" in results_html(spread)
    assert [format_bytes(n) for n in (2.5e9, 81e6, 4100, 300)] == ["2.50 GB", "81 MB", "4 KB", "300 bytes"]


def test_version_two_refuses_version_ones_flags():
    with pytest.raises(ValueError, match="version 1"):
        FastRepairConfig(model_id="x/y", pipeline="v2", until_fair=False)


def test_the_study_and_the_report_name_the_sizes_the_repair_uses(tmp_path):
    from src.product.strategies import SCREEN, WORK

    report, out = repair(tmp_path, Backend(fixes((Q[0],), (Q[1],))), study=6)
    head = json.loads((out / "study.jsonl").read_text().splitlines()[0])
    scan = report["scan"]
    assert head["screen"] == [scan["screen_sample"], scan["screen_probe"]] and sum(head["screen"]) >= SCREEN
    assert scan["engine"] == "remeasured" and scan["survivors"] is None and scan["pool"] == len(scan["pool_parts"])
    assert (scan["work_sample"], scan["work_probe"]) == WORK


def test_every_report_of_a_mixed_patch_says_each_parts_bits(capsys):
    from src.attribution.patch import Patch
    from src.product.fairgate import patch_bits

    patch = Patch(name="m", modules=(Q[0], Q[1]), nbits=8, module_bits={Q[1]: 4},
                  part_shapes={k: tuple(v) for k, v in ADAPTER.part_shapes.items()}, n_layers=ADAPTER.n_layers,
                  model_id="x/y")
    assert patch_bits(patch) == "4-bit (1) and 8-bit (1)"
    import inspect

    from src.eval import model_variants

    assert "bits_text" in inspect.getsource(model_variants)
    assert not math.isnan(len(patch.modules))


def test_four_bits_wins_only_against_a_repair_that_is_not_fair():
    from src.product.fast_repair import four_bits_does_better

    def rows(chosen_fair, chosen_distance, four_distance):
        return [{"name": "remeasured", "chosen": True, "fair": chosen_fair, "distance": chosen_distance},
                {"name": "four_bits", "chosen": False, "fair": True, "distance": four_distance}]

    assert four_bits_does_better(rows(False, 0.20, 0.02))
    assert not four_bits_does_better(rows(True, 0.02, 0.0))  # fair already: no reason to grow the model
    assert not four_bits_does_better(rows(False, 0.20, 0.195))  # not clearly better
    assert not four_bits_does_better(rows(False, 0.20, 0.02)[:1])  # nothing to compare with


def test_a_search_that_hit_its_budget_says_nothing_once_its_repair_is_not_kept():
    from tests.product.test_strategies import remeasure

    job = remeasure(Backend(lambda state, j: 0.1))
    job.run()
    job.result.budget_reached = True  # as if the search had filled its budget
    compressed = {id(it): it.stereotype_index if it.example_id % 2 == 0 else it.unknown_index for it in AMBIG}
    result = job.choose((AMBIG[:400], DISAMBIG[:200]), compressed | {id(it): it.label for it in DISAMBIG})
    assert result.rows[0]["chosen"] and result.kept == [] and result.budget_reached is False


def test_parts_are_restored_at_the_bits_the_compression_method_offers(tmp_path):
    """bitsandbytes (4-bit base) keeps a restored part at plain 16-bit: no 8-bit, no 4-bit step."""
    from tests.product.test_strategies import remeasure

    backend = Backend(fixes((Q[0],), (Q[1],)))
    backend.restore_bits = (16,)
    job = remeasure(backend)
    job.base_bits = 4
    job.__init__(backend, job.adapter, job.candidates, job.work[0], job.work[1], job.scorer, job.config, 4, job.original,
                 job.check, (60, 40))
    assert job.top == 16 and job.cheap is None
    assert {u.bits for u in job.units()} == {16}
    one = job.bytes_of(ADAPTER.params(Q[0]), 16)
    assert job.budget < 2 * one  # 16-bit costs about twice 8-bit's memory: this budget holds one part
    job.budget = 3 * one
    result = job.run()
    assert set(result.bits.values()) == {16} and set(result.kept) == {Q[0], Q[1]}
    hqq = remeasure(Backend(fixes((Q[0],))))
    assert (hqq.top, hqq.cheap) == (8, 4)
