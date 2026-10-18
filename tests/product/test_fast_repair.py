"""Fast Repair end to end on CPU, with a stand-in backend whose "models" are rules.

The stand-in compressed model gets every third fair question wrong (the stereotype). Two parts each
fix half of those (A: even questions, B: odd ones). A third part, H, fixes 70% of them but makes the
model answer "unknown" on half the stated-answer questions: exactly the hedging the penalty exists to
catch. So with the penalty the scan should keep A and B, and without it, H.
"""

import json
from contextlib import contextmanager

import pytest

from src.data.bbq_loader import BBQItem
from src.models.adapter import adapter_for
from src.product import fast_repair
from src.product.fast_repair import FastRepairConfig, in_search_half, run, stratified_sample

TINY = {"model_type": "qwen2", "hidden_size": 64, "num_attention_heads": 4, "num_key_value_heads": 2,
        "intermediate_size": 128, "num_hidden_layers": 4}
A = "model.layers.1.self_attn.q_proj"
B = "model.layers.2.self_attn.v_proj"
H = "model.layers.0.mlp.down_proj"


def _items(context, n=400):
    out = []
    for i in range(n):
        category = ("Age", "SES")[i % 2]
        label = 2 if context == "ambig" else i % 2
        out.append(BBQItem(example_id=i if context == "ambig" else 10_000 + i, category=category,
                           question_polarity="neg", context=f"{context} story {i}", question="Who?",
                           options=("The old one", "The young one", "Unknown"), label=label, unknown_index=2,
                           stereotype_index=0))
    return out


AMBIG, DISAMBIG = _items("ambig"), _items("disambig")


class FakeRegistry:
    def __init__(self):
        self.active, self.trying = {}, None

    @contextmanager
    def trial(self, path, nbits):
        self.trying = path
        try:
            yield
        finally:
            self.trying = None

    def commit(self, path, nbits):
        self.active[path] = nbits

    def restored(self):
        return set(self.active) | ({self.trying} if self.trying else set())


class FakeBackend:
    def __init__(self, damage=True, fail_at=None):
        self.registry, self.damage, self.fail_at = FakeRegistry(), damage, fail_at
        self.calls = []  # (view, question ids, restored set) for every scoring call

    def download(self, model_id):
        if self.fail_at == "download":
            raise OSError("404 Client Error: Repository Not Found")
        return "1.0 GB of weights"

    def adapter(self, model_id):
        return adapter_for(TINY)

    def load_compressed(self, model_id):
        if self.fail_at == "quantize":
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        return "HQQ 3-bit"

    def load_original(self, model_id):
        return "loaded"

    def _answer(self, view, item, restored):
        i = item.example_id % 10_000
        if item.label != item.unknown_index:  # stated-answer question
            hedges = view != "original" and H in restored and i % 2 == 0
            return item.unknown_index if hedges else item.label
        if view == "original":
            return item.stereotype_index if i % 10 == 0 else item.unknown_index
        broken = self.damage and i % 3 == 0 and i % 10 != 0
        fixed = (A in restored and i % 2 == 0) or (B in restored and i % 2 == 1) or (H in restored and i % 10 < 7)
        return item.stereotype_index if broken and not fixed else (item.stereotype_index if i % 10 == 0 else item.unknown_index)

    def score(self, view, items, on_progress=None):
        restored = self.registry.restored()
        self.calls.append((view, [it.question_id for it in items], frozenset(restored)))
        picks = [self._answer(view, it, restored) for it in items]
        if on_progress:
            on_progress(len(picks))
        return picks

    def memory_mb(self):
        return 2.5 * len(self.registry.active)

    def gpu_name(self):
        return "stand-in"


def _run(tmp_path, backend=None, **kwargs):
    config = FastRepairConfig(model_id="Qwen/Qwen2.5-3B-Instruct", questions=kwargs.pop("questions", 200),
                              out=tmp_path / "job", **kwargs)
    backend = backend or FakeBackend()
    return run(config, backend, load_questions=lambda: (AMBIG, DISAMBIG)), backend, config.out


def _events(out):
    return [json.loads(line) for line in (out / "events.jsonl").read_text(encoding="utf-8").splitlines()]


def test_the_scan_keeps_the_real_fixes_and_avoids_the_hedging_part(tmp_path):
    report, backend, out = _run(tmp_path)
    assert report["outcome"] == "repaired" and report["other_damage"] == []
    assert set(report["kept"]) == {A, B} and H not in report["kept"]
    assert report["recovery"]["point"] > 0.9
    assert report["repaired"]["false_unknown"] == report["original"]["false_unknown"] == 0.0
    assert report["compressed"]["usr"] < report["repaired"]["usr"]


def test_without_the_penalty_the_hedging_part_wins(tmp_path):
    report, _, _ = _run(tmp_path, penalty_weight=0.0)
    assert report["kept"][0] == H
    assert report["repaired"]["false_unknown"] > 0.3  # the hedging the penalty is there to stop


def test_files_follow_the_contract(tmp_path):
    report, _, out = _run(tmp_path, mode="exact")
    events = _events(out)
    stages = [e["stage"] for e in events if e["status"] == "started"]
    assert stages == list(fast_repair.STAGES)
    assert all(e["status"] in ("started", "progress", "done", "failed") for e in events)
    assert all(set(e) == {"time", "stage", "status", "done", "total", "message"} for e in events)
    assert events[-1]["stage"] == "report" and events[-1]["status"] == "done"
    for key in ("model_id", "arch", "mode", "questions", "gpu", "minutes", "safe", "flipped", "original",
                "compressed", "repaired", "recovery", "kept", "memory_mb", "gate", "gate_failures", "patch",
                "n_layers", "part_types"):
        assert key in report
    assert json.loads((out / "report.json").read_text(encoding="utf-8")) == report
    assert (report["arch"], report["n_layers"], len(report["part_types"])) == ("qwen2", 4, 7)

    from src.attribution.patch import load_patch

    patch = load_patch(out / "patch.json")
    assert list(patch.modules) == report["kept"] and patch.n_layers == 4
    assert patch.params == sum(adapter_for(TINY).params(p) for p in report["kept"])


def test_the_repair_is_checked_on_questions_the_scan_never_used(tmp_path):
    _, backend, _ = _run(tmp_path)
    compressed_calls = [ids for view, ids, _ in backend.calls if view == "compressed"]
    scan_ids = {q for ids in compressed_calls[1:] for q in ids}  # [0] is score_3bit, on every question
    verify_ids = {q for view, ids, _ in backend.calls if view == "repaired" for q in ids}
    assert scan_ids and verify_ids and not scan_ids & verify_ids
    assert all(in_search_half(q, 0) for q in scan_ids) and not any(in_search_half(q, 0) for q in verify_ids)


def test_no_damage_means_nothing_to_repair(tmp_path):
    report, _, out = _run(tmp_path, backend=FakeBackend(damage=False))
    assert report["kept"] == [] and report["patch"] is None and not (out / "patch.json").exists()
    assert report["gate"] == "PASS"
    assert report["outcome"] == "no_damage" and report["recovery"]["point"] is None
    assert "no fairness damage to repair" in [e for e in _events(out) if e["stage"] == "scan"][-1]["message"]


@pytest.mark.parametrize("stage,words", [("quantize", "out of memory"), ("download", "No model with this name")])
def test_a_failure_is_reported_in_plain_words_at_its_stage(tmp_path, stage, words):
    with pytest.raises(Exception):
        _run(tmp_path, backend=FakeBackend(fail_at=stage))
    last = _events(tmp_path / "job")[-1]
    assert (last["stage"], last["status"]) == (stage, "failed") and words.lower() in last["message"].lower()
    assert not (tmp_path / "job" / "report.json").exists()


def test_a_model_that_isnt_a_text_generator_fails_at_the_first_stage(tmp_path):
    class Encoder(FakeBackend):
        def adapter(self, model_id):
            return adapter_for({"model_type": "bert"})

    with pytest.raises(ValueError):
        _run(tmp_path, backend=Encoder())
    last = _events(tmp_path / "job")[-1]
    assert last["stage"] == "download" and "text-generation" in last["message"]


def test_the_studio_command_line_matches(monkeypatch, tmp_path):
    seen = {}

    def fake_run(config, backend=None, load_questions=None):
        seen.update(model=config.model_id, mode=config.mode, questions=config.questions, out=config.out)
        return {"minutes": 0.1, "kept": [], "gate": "PASS"}

    monkeypatch.setattr(fast_repair, "run", fake_run)
    code = fast_repair.main(["--model", "Qwen/Qwen2.5-3B-Instruct", "--mode", "exact", "--questions", "500",
                             "--out", str(tmp_path)])
    assert code == 0 and seen == {"model": "Qwen/Qwen2.5-3B-Instruct", "mode": "exact", "questions": 500, "out": tmp_path}


def test_sampling_is_stratified_and_seeded():
    sample = stratified_sample(AMBIG, 100, seed=0)
    assert len(sample) == 100 and sum(it.category == "Age" for it in sample) == 50
    assert sample == stratified_sample(list(reversed(AMBIG)), 100, seed=0)
    assert sample != stratified_sample(AMBIG, 100, seed=1)
    halves = [in_search_half(it.question_id, 0) for it in AMBIG]
    assert 0.4 < sum(halves) / len(halves) < 0.6


def test_config_is_checked():
    with pytest.raises(ValueError):
        FastRepairConfig(model_id="x/y", mode="turbo")
    with pytest.raises(ValueError):
        FastRepairConfig(model_id="x/y", questions=10)


def test_question_sets_that_share_ids_stay_apart(tmp_path):
    """Regression: answers were once keyed by question id, so a stated-answer question with the same
    id as an ambiguous one overwrote its answer (BBQ's ids don't overlap, but nothing should rely on it)."""
    shared = [BBQItem(example_id=it.example_id - 10_000, category=it.category, question_polarity="neg",
                      context=it.context, question=it.question, options=it.options, label=it.label,
                      unknown_index=it.unknown_index, stereotype_index=it.stereotype_index) for it in DISAMBIG]
    assert {it.question_id for it in shared} == {it.question_id for it in AMBIG}
    config = FastRepairConfig(model_id="Qwen/Qwen2.5-3B-Instruct", questions=200, out=tmp_path / "job")
    report = run(config, FakeBackend(), load_questions=lambda: (AMBIG, shared))
    assert report["safe"] > 0 and set(report["kept"]) == {A, B}


# --- the scan: Fast Scan (src/attribution/fast_scan.py) as the pipeline drives it ------------------------

def test_the_scan_tells_the_screen_what_it_is_doing_in_plain_words(tmp_path, monkeypatch):
    monkeypatch.setattr(fast_repair.Events, "min_interval", 0.0)  # every update, not two a second
    report, _, out = _run(tmp_path)
    scan = [e for e in _events(out) if e["stage"] == "scan"]
    started, progress, done = scan[0], [e for e in scan if e["status"] == "progress"], scan[-1]
    assert started["message"].startswith("28 parts · ") and started["total"] > 0
    assert all(e["total"] == started["total"] for e in progress)
    assert progress[0]["message"].startswith("Step 1 of 2 · Quick look at every part: 1 of 28")
    assert any(e["message"].startswith("Step 2 of 2 · Choosing among the best ") for e in progress)
    assert all(e["message"].endswith(" left") for e in progress if e["done"] < e["total"])
    assert [e["done"] for e in progress] == sorted(e["done"] for e in progress)
    assert done["message"] == "Kept 2 of 28 parts" and done["done"] == done["total"]
    block = report["scan"]
    assert block["engine"] == "fast_scan" and block["pool"] == 16 and set(report["kept"]) <= set(block["pool_parts"])
    assert block["trials"] >= 28 and block["stop_reason"]
    damage = [e for e in _events(out) if e["stage"] == "find_damage"][-1]["message"]
    assert "fair answers flipped to the stereotype" in damage and "kept back to check the repair" in damage


def test_exact_mode_rechecks_the_most_promising_on_every_question(tmp_path, monkeypatch):
    monkeypatch.setattr(fast_repair.Events, "min_interval", 0.0)
    report, backend, out = _run(tmp_path, mode="exact", questions=400)
    messages = [e["message"] for e in _events(out) if e["stage"] == "scan" and e["status"] == "progress"]
    assert messages[0].startswith("Step 1 of 3 · Quick look at every part")
    assert any(m.startswith("Step 2 of 3 · Re-checking the 28 most promising on every question") for m in messages)
    assert any(m.startswith("Step 3 of 3 · Choosing among the best 16") for m in messages)
    assert set(report["kept"]) == {A, B}


class _Always:
    """A registry and scorer where every part fixes its own questions, so every round keeps one and the
    scan runs all five rounds: the case the plan counts."""

    def __init__(self, parts):
        self.active, self.trying, self.parts, self.asked = {}, None, parts, 0

    @contextmanager
    def trial(self, path, nbits):
        self.trying = path
        try:
            yield
        finally:
            self.trying = None

    def commit(self, path, nbits):
        self.active[path] = nbits

    def scorer(self, sample_ids, probe_ids):
        from src.attribution.fast_scan import ScanScore

        self.asked += len(sample_ids) + len(probe_ids)
        on = set(self.active) | ({self.trying} if self.trying else set())
        fixed = sum(1 for q in sample_ids if any(int(q[1:]) % len(self.parts) == self.parts.index(p) for p in on))
        return ScanScore(fixed / len(sample_ids))


@pytest.mark.parametrize("mode,n_parts,n_sample,n_probe", [
    ("fast", 28, 300, 200), ("fast", 252, 300, 200), ("exact", 28, 300, 200), ("exact", 40, 50, 30),
    ("fast", 20, 30, 10)])
def test_the_progress_total_is_exactly_what_the_scan_asks(mode, n_parts, n_sample, n_probe):
    from src.attribution.fast_scan import fast_scan
    from src.product.fast_repair import plan_questions, scan_config

    parts = [f"model.layers.{i}.mlp.down_proj" for i in range(n_parts)]
    world = _Always(parts)
    cfg = scan_config(FastRepairConfig(model_id="x/y", mode=mode), n_parts)
    result = fast_scan(parts, world, world.scorer, [f"s{i}" for i in range(n_sample)],
                       [f"p{i}" for i in range(n_probe)], cfg)
    assert result.stop_reason == "reached max_rounds"
    assert world.asked == plan_questions(cfg, n_parts, n_sample, n_probe)


def test_fast_mode_costs_the_same_whatever_the_number_of_parts():
    from src.product.fast_repair import SCAN_BUDGET, plan_questions, scan_config

    phi = scan_config(FastRepairConfig(model_id="x/y"), 128)
    assert (phi.screen_sample, phi.screen_probe, phi.pool) == (60, 40, 16)  # Fast Scan's own defaults
    assert plan_questions(phi, 128, 300, 200) == SCAN_BUDGET
    for parts in (196, 252):
        cfg = scan_config(FastRepairConfig(model_id="x/y"), parts)
        assert cfg.screen_sample + cfg.screen_probe >= fast_repair.MIN_SCREEN
        assert plan_questions(cfg, parts, 300, 200) <= SCAN_BUDGET * 1.01
    exact = scan_config(FastRepairConfig(model_id="x/y", mode="exact"), 252)
    assert (exact.mode, exact.screen_sample, exact.screen_probe, exact.survivors) == ("exact", 60, 40, 40)


def test_the_scorer_gives_fast_scan_the_objective_and_each_answer(tmp_path):
    from src.product.fast_repair import run_scan, scan_config

    backend = FakeBackend()
    events = fast_repair.Events(tmp_path)
    events.stage = "scan"
    flipped = [it for it in AMBIG if it.example_id % 3 == 0 and it.example_id % 10 != 0][:40]
    seen = []
    real = backend.score

    def spy(view, items, on_progress=None):
        picks = real(view, items, on_progress)
        seen.append((list(items), picks))
        return picks

    backend.score = spy
    result, cfg, total = run_scan(backend, adapter_for(TINY).candidates(), flipped, DISAMBIG[:30],
                                  FastRepairConfig(model_id="x/y"), events, speed=10.0)
    assert set(result.kept) == {A, B} and cfg == scan_config(FastRepairConfig(model_id="x/y"), 28)
    assert 0 < sum(len(items) for items, _ in seen) <= total  # every question asked went through the scorer


def test_search_questions_are_a_random_sample_not_the_first_categories():
    from src.product.fast_repair import seeded_subset

    items = sorted(AMBIG, key=lambda it: it.category)  # grouped by category, like the real lists
    picked = seeded_subset(items, 100, seed=0)
    assert len(picked) == 100 and {it.category for it in picked} == {"Age", "SES"}
    assert picked == [it for it in items if it in picked]  # original order kept
    assert picked == seeded_subset(items, 100, seed=0) != seeded_subset(items, 100, seed=1)
    assert seeded_subset(items[:5], 100, seed=0) == items[:5]


# --- runs that end without a repair must say why, and never look like one (Qwen2.5-0.5B, Oct 2026) ---

class NeverUnsure(FakeBackend):
    """Like Qwen2.5-0.5B: the original almost never says "can't tell" and picks a person ~50/50; the
    compressed copy says "can't tell" at random a quarter of the time and loses accuracy."""

    def _answer(self, view, item, restored):
        i = item.example_id % 10_000
        if view == "original":
            if item.label != item.unknown_index:
                return item.label if i % 9 < 5 else 1 - item.label
            return item.unknown_index if i % 125 == 0 else (item.stereotype_index if i % 2 else 1)
        if i % 4 == 0:
            return item.unknown_index
        if item.label != item.unknown_index:
            return item.label if i % 3 == 0 else 1 - item.label
        return item.stereotype_index if i % 2 else 1


def test_a_model_that_never_says_cant_tell_is_reported_as_not_checkable(tmp_path):
    report, backend, out = _run(tmp_path, backend=NeverUnsure(), questions=400)
    assert report["outcome"] == "not_applicable" and "doesn't give the fair answer" in report["outcome_reason"]
    assert report["gate"] == "N/A" and report["gate_failures"] == []  # never a PASS nobody measured
    assert report["recovery"] == {"point": None, "lower": None, "upper": None}  # not "-0.0%"
    assert report["kept"] == [] and report["patch"] is None and report["scan"]["engine"] == "none"
    assert not any(view == "compressed" for view, _, restored in backend.calls if restored)  # no scan
    assert any("fewer questions right" in d for d in report["other_damage"])
    assert any("can't tell" in d for d in report["other_damage"])
    events = _events(out)
    damage = [e for e in events if e["stage"] == "find_damage"][-1]["message"]
    assert damage.startswith("The original model says \"can't tell\" on only")
    assert events[-1]["message"] == "This model can't be checked this way"


def test_a_scan_that_keeps_nothing_says_so(tmp_path, monkeypatch):
    import src.product.fast_repair as fr

    monkeypatch.setattr(fr, "run_scan", lambda *a, **k: (type("R", (), {"kept": [], "pool": [], "trials": [],
                                                                         "stop_reason": "x"})(),
                                                         fr.scan_config(FastRepairConfig(model_id="x/y"), 28), 10))
    report, _, out = _run(tmp_path)
    assert report["outcome"] == "no_gain" and report["patch"] is None
    assert "No single part" in [e for e in _events(out) if e["stage"] == "scan"][-1]["message"]
    assert report["recovery"]["point"] == 0.0 or report["recovery"]["point"] is None  # nothing restored


def test_other_damage_is_measured_against_the_original():
    from src.product.fast_repair import other_damage

    card = lambda acc, fu: {"usr": 0.8, "srs": 0.1, "accuracy": acc, "false_unknown": fu}  # noqa: E731
    found = other_damage({"original": card(0.9, 0.05), "compressed": card(0.8, 0.06), "repaired": card(0.89, 0.2)},
                         repaired=True)
    assert len(found) == 2 and found[0].startswith("Compression gets fewer") and "After the repair" in found[1]
    assert other_damage({"original": card(0.9, 0.05), "compressed": card(0.88, 0.06), "repaired": card(0.9, 0.05)},
                        repaired=False) == []


def test_the_original_is_released_once_it_has_answered(tmp_path):
    class Releasing(FakeBackend):
        released_after = None

        def release_original(self):
            self.released_after = len(self.calls)

    backend = Releasing()
    _run(tmp_path, backend=backend)
    assert backend.released_after == 1  # right after its one scoring pass
    assert all(view != "original" for view, _, _ in backend.calls[backend.released_after:])


def test_scores_that_overflow_stop_the_run_in_plain_words():
    from src.product.fast_repair import PrecisionError, plain_error

    message = plain_error(PrecisionError("The original model's numbers overflow at 16-bit precision (fp16)..."))
    assert message.startswith("The original model's numbers overflow") and "PrecisionError" not in message


# --- the Qwen2.5-3B flaw (Oct 2026): compression made the model hedge, and the scan "repaired" that ---

R = "model.layers.3.mlp.up_proj"  # removes the compression's hedging, and costs fair answers


class HedgingCompressed(FakeBackend):
    """Like Qwen2.5-3B at 3-bit: the compressed model says "can't tell" on a third of the questions whose
    story states the answer. Part R removes that hedging (general repair) but also turns some fair
    answers into the stereotype; parts A and B bring the flipped answers back."""

    def _answer(self, view, item, restored):
        i = item.example_id % 10_000
        if view != "original" and item.label != item.unknown_index:
            hedges = (H in restored and i % 2 == 0) or (i % 3 == 0 and R not in restored)
            return item.unknown_index if hedges else item.label
        pick = super()._answer(view, item, restored)
        if view != "original" and R in restored and item.label == item.unknown_index and i % 4 == 1:
            return item.stereotype_index
        return pick


def test_removing_the_compressions_hedging_does_not_buy_a_part_its_place(tmp_path):
    report, _, _ = _run(tmp_path, backend=HedgingCompressed())
    assert R not in report["kept"] and set(report["kept"]) == {A, B}
    assert report["outcome"] == "repaired" and report["repaired"]["usr"] > report["compressed"]["usr"]


def test_the_hedging_penalty_counts_only_hedging_a_part_adds():
    from src.product.fast_repair import excess_hedging

    stated = [it for it in DISAMBIG[:6]]
    before = {it.question_id: k < 3 for k, it in enumerate(stated)}  # the compressed model hedged on 3
    hedge_all = [it.unknown_index for it in stated]
    assert excess_hedging(stated, hedge_all, before) == 0.5  # only the 3 new ones count
    assert excess_hedging(stated, [it.label for it in stated], before) == 0.0  # removing hedging: no reward
    assert excess_hedging([], [], before) == 0.0


X = "model.layers.2.mlp.down_proj"  # a part that breaks fair answers everywhere


class BreaksFairness(FakeBackend):
    def _answer(self, view, item, restored):
        if view != "original" and X in restored and item.label == item.unknown_index:
            return item.stereotype_index
        return super()._answer(view, item, restored)


def test_a_repair_that_makes_fairness_worse_on_held_out_questions_is_rejected(tmp_path, monkeypatch):
    import src.product.fast_repair as fr

    def scan_keeping_x(backend, *args, **kwargs):
        backend.registry.commit(X, 8)
        return type("R", (), {"kept": [X], "pool": [X], "trials": [], "stop_reason": "stub"})(), \
            fr.scan_config(FastRepairConfig(model_id="x/y"), 28), 10

    monkeypatch.setattr(fr, "run_scan", scan_keeping_x)
    report, _, out = _run(tmp_path, backend=BreaksFairness())
    assert report["outcome"] == "rejected" and "made fairness worse" in report["outcome_reason"]
    assert report["patch"] is None and not (out / "patch.json").exists()  # never offered to ship
    assert report["kept"] == [X] and report["repaired"]["usr"] < report["compressed"]["usr"]
    assert _events(out)[-1]["message"].startswith("Repair rejected")

    from src.product.byom import results_html

    page = results_html(report)
    assert "Repair rejected" in page and "Rejected repair" in page and "not offered for shipping" in page


# --- the scan also guards the fair answers the compressed model still gives (Qwen2.5-3B, Oct 2026) ---

G = "model.layers.1.mlp.gate_proj"  # brings back a few flipped answers, breaks many still-fair ones


class BreaksStillFair(FakeBackend):
    def _answer(self, view, item, restored):
        pick = super()._answer(view, item, restored)
        if view == "original" or G not in restored or item.label != item.unknown_index:
            return pick
        i = item.example_id % 10_000
        broken = self.damage and i % 3 == 0 and i % 10 != 0
        if broken and i % 2 == 0:
            return item.unknown_index  # fixes these flipped answers
        if not broken and i % 10 != 0 and i % 4 != 0:
            return item.stereotype_index  # but breaks fair answers elsewhere
        return pick


def test_a_part_that_breaks_more_fair_answers_than_it_fixes_is_not_kept(tmp_path):
    report, backend, _ = _run(tmp_path, backend=BreaksStillFair())
    assert G not in report["kept"] and set(report["kept"]) == {A, B}
    scan_ids = {q for view, ids, restored in backend.calls if view == "compressed" and restored for q in ids}
    still_fair = {it.question_id for it in AMBIG
                  if it.example_id % 3 != 0 and it.example_id % 10 != 0}  # the compressed model's fair answers
    assert scan_ids & still_fair  # the scan asked some of them


def test_four_bit_compression_is_offered_end_to_end(tmp_path, monkeypatch):
    report, _, out = _run(tmp_path, base_nbits=4)
    assert report["base_nbits"] == 4
    from src.attribution.patch import load_patch

    assert load_patch(out / "patch.json").base["nbits"] == 4
    seen = {}
    monkeypatch.setattr(fast_repair, "run", lambda config, backend=None, load_questions=None:
                        seen.update(bits=config.base_nbits) or {"minutes": 0, "kept": [], "gate": "PASS"})
    assert fast_repair.main(["--model", "x/y", "--out", str(tmp_path / "cli"), "--base-nbits", "4"]) == 0
    assert seen["bits"] == 4
    with pytest.raises(ValueError):
        FastRepairConfig(model_id="x/y", base_nbits=2)

    from src.product.byom import check_bits, pipeline_command

    assert pipeline_command("real", "x/y", "fast", 1000, tmp_path, base_nbits=4)[-2:] == ["--base-nbits", "4"]
    assert check_bits("4-bit: a little bigger, much less damage") == 4
