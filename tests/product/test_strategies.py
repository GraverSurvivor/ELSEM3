"""Version 2's repair (src/product/strategies.py): stand-in models built for each way version 1 went wrong."""

import json
from contextlib import contextmanager

import pytest

from src.models.adapter import adapter_for
from src.product import strategies
from src.product.decisions import TrialScorer, WatchedRegistry, split_of
from src.product.fast_repair import REAL_BITS, FastRepairConfig, run
from src.product.strategies import Reading, Remeasure, arranged, distance, gain, trying
from tests.product import test_fast_repair as t

ADAPTER = adapter_for(t.TINY)  # 4 layers x 7 part types
PARTS = ADAPTER.candidates()
Q = [f"model.layers.{i}.self_attn.q_proj" for i in range(4)]
V = [f"model.layers.{i}.self_attn.v_proj" for i in range(4)]
UP = [f"model.layers.{i}.mlp.up_proj" for i in range(4)]
AMBIG, DISAMBIG = t._items("ambig", 2400), t._items("disambig", 2400)
RIGHT = (0.96, 0.02, 0.02)


class Stack:
    """A swap registry on paper: what is committed, and what is being tried or taken out right now."""

    def __init__(self):
        self._committed, self._layers = {}, []
        self.commits = []

    @property
    def active(self):
        return dict(self._committed)

    def now(self):
        state = dict(self._committed)
        for path, bits in self._layers:
            if bits is None:
                state.pop(path, None)
            else:
                state[path] = bits
        return state

    def restored(self):
        return set(self.now())

    @contextmanager
    def _with(self, path, bits):
        self._layers.append((path, bits))
        try:
            yield
        finally:
            self._layers.pop()

    def trial(self, path, nbits):
        return self._with(path, nbits)

    def without(self, path):
        assert path in self._committed
        return self._with(path, None)

    def commit(self, path, nbits):
        assert not self._layers, "committed in the middle of a trial"
        self._committed[path] = nbits
        self.commits.append((path, nbits))

    def revert(self, path):
        del self._committed[path]


class Model:
    """`rule(state, i) -> probability of "can't tell"` for damaged ambiguous question i (half of them);
    `harm(state, i)`: probability the part state takes away from stated answer i."""

    def __init__(self, rule, harm=lambda state, i: 0.0):
        self.rule, self.harm, self.registry, self.calls = rule, harm, Stack(), []

    def download(self, model_id):
        return "1.0 GB of weights"

    def adapter(self, model_id):
        return ADAPTER

    def load_compressed(self, model_id):
        return "HQQ 3-bit"

    def load_original(self, model_id):
        return "loaded"

    def _p(self, view, item):
        i = item.example_id % 10_000
        state = self.registry.now()
        if item.label != item.unknown_index:
            keep = 0.9 if view == "original" else 0.9 - self.harm(state, i)
            row = [0.0, 0.0, 0.0]
            row[item.label], row[item.unknown_index] = keep, 0.95 - keep
            row[3 - item.label - item.unknown_index] = 0.05
            return tuple(row)
        sure = 0.96 if view == "original" or i % 2 else self.rule(state, i // 2)
        row = [0.0, 0.0, 0.0]
        row[item.unknown_index], row[item.stereotype_index] = sure, (1 - sure) * 0.9
        row[3 - item.unknown_index - item.stereotype_index] = (1 - sure) * 0.1
        return tuple(row)

    def probs(self, view, items, on_progress=None):
        self.calls.append((view, [it.question_id for it in items], dict(self.registry.now())))
        if on_progress:
            on_progress(len(items))
        return [self._p(view, it) for it in items]

    def score(self, view, items, on_progress=None):
        return [max(range(3), key=p.__getitem__) for p in self.probs(view, items, on_progress)]

    def memory_mb(self):
        return sum(ADAPTER.params(p) * (REAL_BITS[b] - 3.75) / 8 for p, b in self.registry.active.items()) / 2**20

    def gpu_name(self):
        return "stand-in"


def repair(tmp_path, model, **kwargs):
    config = FastRepairConfig(model_id="Qwen/Qwen2.5-3B-Instruct", questions=kwargs.pop("questions", 2000),
                              out=tmp_path / "job", pipeline="v2", **kwargs)
    report = run(config, model, load_questions=lambda: (AMBIG, DISAMBIG))
    return report, config.out


def fixes(*pairs, low=0.1, high=0.96):
    """rule: kind k of the damage (question // 1 % len) is fixed by any of pairs[k]'s parts"""
    def rule(state, j):
        return high if set(pairs[j % len(pairs)]) & set(state) else low
    return rule


# --- the failures version 1 had ---------------------------------------------------------------------------

def test_a_part_that_only_repeats_another_is_not_restored(tmp_path):
    """Phi-3.5: the 2nd and 3rd best parts alone add almost nothing once the best one is restored."""
    model = Model(fixes((Q[0], Q[1], Q[2]), (Q[0], Q[1], Q[2]), (Q[3],)))  # 3 look best, and are the same
    report, _ = repair(tmp_path, model)
    assert Q[3] in report["kept"] and len(set(report["kept"]) & {Q[0], Q[1], Q[2]}) == 1
    assert report["gate"] == "PASS" and report["recovery"]["point"] > 0.95


def test_a_part_that_hurts_once_another_is_restored_is_left_out(tmp_path):
    def rule(state, j):
        kind = j % 4
        if kind < 2:
            return 0.96 if Q[0] in state or Q[1] in state else 0.1
        if Q[0] in state and Q[1] in state:
            return 0.1  # the two together break what Q[2] fixes
        return 0.96 if Q[2] in state else 0.1

    report, _ = repair(tmp_path, Model(rule))
    assert Q[2] in report["kept"] and not {Q[0], Q[1]} <= set(report["kept"])
    assert report["gate"] == "PASS"


def test_damage_spread_over_a_whole_part_type_is_found_as_one(tmp_path):
    """Each up_proj alone changes nothing you can measure; all of them together fix it."""
    model = Model(lambda state, j: 0.96 if set(UP) <= set(state) else 0.1)
    report, _ = repair(tmp_path, model)
    assert set(UP) <= set(report["kept"]) and report["gate"] == "PASS"
    assert {report["kept_bits"][p] for p in UP} == {4}  # lifted the cheap way
    trials = report["strategies"]
    assert next(r for r in trials if r["chosen"])["name"] == "remeasured"


def test_each_part_gets_the_fewest_bits_it_needs(tmp_path):
    def rule(state, j):
        part, needs = ((Q[0], 4), (Q[1], 8))[j % 2]
        return 0.96 if state.get(part, 3) >= needs else 0.1

    report, out = repair(tmp_path, Model(rule))
    assert report["kept_bits"] == {Q[0]: 4, Q[1]: 8}
    patch = json.loads((out / "patch.json").read_text())
    assert patch["module_bits"] == {Q[0]: 4} and patch["nbits"] == 8
    from src.attribution.patch import load_patch

    assert [load_patch(out / "patch.json").bits(p) for p in report["kept"]] == [report["kept_bits"][p] for p in report["kept"]]


def test_a_part_made_unnecessary_later_is_taken_out_again():
    """P looks best and is taken first; the two parts that come after cover everything it did."""
    def rule(state, j):
        fixed = {0: (V[0], Q[0]), 1: (V[0], Q[1]), 2: (Q[0],), 3: (Q[1],)}[j % 4]
        return 0.96 if set(fixed) & set(state) else 0.1

    model = Model(rule)
    result = remeasure(model).run()
    assert {Q[0], Q[1]} <= set(result.kept) and V[0] not in result.kept
    assert model.registry.active == result.bits


def test_harming_stated_answers_keeps_a_part_out(tmp_path):
    model = Model(fixes((Q[0], Q[1])), harm=lambda state, i: 0.6 if Q[0] in state else 0.0)
    report, _ = repair(tmp_path, model)
    assert report["kept"] == [Q[1]] and report["repaired"]["accuracy"] == report["compressed"]["accuracy"]


# --- limits -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("share", [0.05, 0.10, 0.15])
def test_the_memory_budget_is_never_passed(tmp_path, share):
    model = Model(lambda state, j: 0.96 if PARTS[j % len(PARTS)] in state else 0.1)  # damage everywhere
    report, _ = repair(tmp_path, model, max_restore=share)
    budget = share * ADAPTER.total_params() * (8.5 - 3.75) / 8
    used = sum(ADAPTER.params(p) * (REAL_BITS[b] - 3.75) / 8 for p, b in report["kept_bits"].items())
    assert 0 < used <= budget + 1
    for row in report["strategies"]:
        if row["ships"]:
            assert row["extra_mb"] <= budget / 2**20 + 0.1
    assert report["size"]["repaired"] < ADAPTER.total_params() * REAL_BITS[4] / 8  # smaller than 4-bit everywhere


def test_it_stops_as_soon_as_the_model_is_as_fair_as_the_original(tmp_path):
    report, _ = repair(tmp_path, Model(fixes((Q[0],), (Q[1],))))
    assert report["scan"]["stop_reason"] == strategies.STOP_FAIR
    assert set(report["kept"]) == {Q[0], Q[1]} and "budget_note" not in report


def test_nothing_that_helps_means_nothing_restored(tmp_path):
    report, out = repair(tmp_path, Model(lambda state, j: 0.1))
    assert report["outcome"] == "no_gain" and report["kept"] == [] and not (out / "patch.json").exists()


def test_the_question_limit_stops_new_rounds(monkeypatch):
    monkeypatch.setitem(strategies.QUESTION_LIMIT, "fast", 10)
    result = remeasure(Model(fixes((Q[0],), (Q[1],)))).run()
    assert result.stop_reason == strategies.STOP_LIMIT and result.kept == []


def test_four_bit_bases_restore_to_eight_bits_only(tmp_path):
    report, out = repair(tmp_path, Model(fixes((Q[0],), (Q[1],))), base_nbits=4)
    assert set(report["kept_bits"].values()) == {8} and "module_bits" not in json.loads((out / "patch.json").read_text())
    assert not any(r["name"] == "four_bits" for r in report["strategies"])


# --- choosing, on questions that chose nothing ----------------------------------------------------------------

def test_the_finished_repairs_are_compared_on_validation_and_never_on_the_final_questions(tmp_path):
    model = Model(fixes((Q[0], Q[1], Q[2]), (Q[0], Q[1], Q[2]), (Q[3],)))
    report, _ = repair(tmp_path, model)
    end = next(k for k, (view, _, _) in enumerate(model.calls) if view == "repaired")
    first_trial = next(k for k, (view, _, state) in enumerate(model.calls) if state)
    read = [{split_of(q, 0) for q in ids} for _, ids, _ in model.calls[first_trial:end]]
    assert "final" not in set().union(*read)
    assert read.count({"validation"}) == len([r for r in report["strategies"] if r["name"] != "compressed"])
    assert all(len(splits) == 1 for splits in read)  # no measurement mixes the splits
    names = [r["name"] for r in report["strategies"]]
    assert names[0] == "compressed" and "remeasured" in names and "four_bits" in names
    assert sum(r["chosen"] for r in report["strategies"]) == 1
    chosen = next(r for r in report["strategies"] if r["chosen"])
    assert chosen["ships"] and chosen["distance"] == min(r["distance"] for r in report["strategies"] if r["ships"])


def test_the_earlier_ordering_ships_when_it_is_clearly_better():
    model = Model(fixes((Q[0],), (Q[1],)))
    job = remeasure(model)
    job.run()
    for path in list(model.registry.active):  # pretend the search went wrong
        model.registry.revert(path)
    model.registry.commit(V[3], 8)
    job.result.kept, job.result.bits = [V[3]], {V[3]: 8}
    result = job.choose((AMBIG[1500:1900], DISAMBIG[1500:1700]), {id(it): it.stereotype_index if it.example_id % 2 == 0
                                                                  else it.unknown_index for it in AMBIG} |
                        {id(it): it.label for it in DISAMBIG})
    assert next(r for r in result.rows if r["chosen"])["name"] == "first_scores"
    assert {Q[0], Q[1]} <= set(result.kept) and model.registry.active == result.bits and V[3] not in result.bits


def test_four_bits_everywhere_is_reported_but_never_shipped(tmp_path):
    """Damage the budget can't cover, all gone at 4-bit: the report says so, the repair stays a repair."""
    model = Model(lambda state, j: 0.96 if len(state) >= 28 else (0.6 if Q[0] in state and j % 3 == 0 else 0.1))
    report, _ = repair(tmp_path, model)
    four = next(r for r in report["strategies"] if r["name"] == "four_bits")
    assert four["distance"] < 0.02 and not four["chosen"] and not four["ships"]
    assert "4-bit" in report["advice"] and len(report["kept"]) < 28
    from src.product.byom import results_html

    assert "compressing everything to 4-bit" in results_html(report)


def test_a_repair_no_better_than_the_compressed_model_is_not_kept():
    model = Model(lambda state, j: 0.1)
    job = remeasure(model)
    job.run()
    model.registry.commit(Q[0], 8)
    job.result.kept, job.result.bits = [Q[0]], {Q[0]: 8}
    compressed = {id(it): it.stereotype_index if it.example_id % 2 == 0 else it.unknown_index for it in AMBIG}
    result = job.choose((AMBIG[:400], DISAMBIG[:200]), compressed | {id(it): it.label for it in DISAMBIG})
    assert result.kept == [] and model.registry.active == {} and result.rows[0]["chosen"]


# --- the pieces -------------------------------------------------------------------------------------------------

def remeasure(model, objective="soft"):
    config = FastRepairConfig(model_id="x/y", pipeline="v2", objective=objective)
    flipped = [it for it in AMBIG[:1000] if it.example_id % 2 == 0][:300]
    fair = [it for it in AMBIG[:1000] if it.example_id % 2][:150]
    stated = DISAMBIG[:200]
    reference = {id(it): model._p("original", it) for it in flipped + fair + stated}
    before = {id(it): model._p("compressed", it) for it in stated}
    original = {id(it): it.unknown_index for it in AMBIG} | {id(it): it.label for it in DISAMBIG}
    return Remeasure(model, ADAPTER, PARTS, flipped + fair, stated, TrialScorer(reference, before, {}), config, 3,
                     original, (AMBIG[:600], DISAMBIG[:300]), (60, 40))


@pytest.mark.parametrize("objective", ["soft", "kl", "count"])
def test_every_trial_score_finds_the_two_parts(objective):
    result = remeasure(Model(fixes((Q[0],), (Q[1],))), objective).run()
    assert set(result.kept) == {Q[0], Q[1]}


def test_gain_is_paired_and_knows_its_own_noise():
    old = Reading([0.1] * 50 + [0.9] * 50, [0.0] * 40)
    steady, noisy = Reading([v + 0.02 for v in old.sample], [0.0] * 40), Reading(
        [v + (0.5 if i % 2 else -0.46) for i, v in enumerate(old.sample)], [0.0] * 40)
    assert gain(steady, old) == pytest.approx((0.02, 0.0))
    value, error = gain(noisy, old)
    assert value == pytest.approx(0.02) and error > value  # the same gain, but nothing to act on
    assert gain(Reading(old.sample, [0.1] * 40), old)[0] == pytest.approx(-0.1)  # harm costs
    assert gain(Reading([0.2], []), Reading([0.1], [])) == pytest.approx((0.1, 0.0))


def test_arranging_and_trying_put_everything_back_even_after_an_error():
    stack = Stack()
    stack.commit(Q[0], 8)
    stack.commit(Q[1], 4)
    registry = WatchedRegistry(stack)
    with arranged(registry, {Q[1]: 8, V[0]: 4}):
        assert stack.now() == {Q[1]: 8, V[0]: 4}
    with pytest.raises(RuntimeError), arranged(registry, {}):
        assert stack.now() == {}
        raise RuntimeError("out of memory")
    with trying(registry, {V[1]: 8, V[2]: 4}, "two"):
        assert stack.now() == {Q[0]: 8, Q[1]: 4, V[1]: 8, V[2]: 4} and registry.trying == "two"
    assert stack.now() == stack.active == {Q[0]: 8, Q[1]: 4} and registry.trying is None


def test_distance_counts_what_got_worse_only():
    goal = {"usr": 0.8, "srs": 0.1, "false_unknown": 0.05}
    assert distance(goal, goal) == 0
    assert distance({"usr": 0.9, "srs": 0.0, "false_unknown": 0.0}, goal) == 0  # fairer than the original: no credit
    assert distance({"usr": 0.7, "srs": 0.15, "false_unknown": 0.10}, goal) == pytest.approx(0.2)


def test_the_first_look_covers_every_part_and_every_part_type():
    model = Model(fixes((Q[0],), (Q[1],)))
    job = remeasure(model)
    names = [u.name for u in job.units()]
    assert names[:28] == PARTS and names[28:] == [f"every {p}" for p in ADAPTER.part_types]
    first = job.first_look()
    assert {u.name for u in first} >= set(PARTS)
    assert all(job.cost(job.todo(u)) <= job.budget for u in first if len(u.parts) > 1)  # no group past the budget
    assert sum(map(len, job.screen)) >= strategies.SCREEN


def test_a_part_is_measured_again_with_the_others_in_place():
    """The scores that decide are the ones taken now, not the first look's."""
    kinds = ((Q[0], Q[1]), (Q[0], Q[1]), (Q[3],), (V[0],), (V[1],), (V[2],))

    def rule(state, j):  # only at 8-bit, so no part type lifted to 4-bit gets in the way
        return 0.96 if 8 in [state.get(p) for p in kinds[j % 6]] else 0.1

    model = Model(rule)
    job = remeasure(model)
    result = job.run()
    taken = next(p for p in result.kept if p in (Q[0], Q[1]))
    other = Q[1] if taken == Q[0] else Q[0]
    looks = [(tr.round, tr.gain) for tr in result.trials if tr.path == other]
    assert looks[0][1] > 0.15 and looks[-1][0] > looks[0][0]  # it looked great alone, and was looked at again
    assert looks[-1][1] < 0.05  # and the second look, with its twin restored, is what counted
    from src.product.strategies import Unit

    again, _ = gain(job.try_parts({other: 8}, other, job.work), job.read(job.work))
    assert again < 0.01 and Unit(other, (other,), 8) not in [u for u in job.units() if not job.todo(u)]


def test_a_part_type_taken_whole_is_thinned_out_to_the_layers_that_need_it():
    """Taken as a whole for speed, then halved and halved again: only layer 2's part is needed."""
    model = Model(lambda state, j: 0.96 if UP[2] in state else 0.1)
    job = remeasure(model)
    start = job.read(job.work)
    for path in UP:
        model.registry.commit(path, 4)
    job.groups_taken = [("every mlp.up_proj", tuple(UP))]
    current = job.read(job.work)
    job.clean_up({p: 0.1 for p in UP}, current)
    assert model.registry.active == {UP[2]: 4}
    assert gain(job.read(job.work), current)[0] == pytest.approx(0) and gain(current, start)[0] > 0.3

    needed = Model(lambda state, j: 0.96 if set(UP) <= set(state) else 0.1)  # every layer is needed: all stay
    job = remeasure(needed)
    for path in UP:
        needed.registry.commit(path, 4)
    job.groups_taken = [("every mlp.up_proj", tuple(UP))]
    job.clean_up({}, job.read(job.work))
    assert set(needed.registry.active) == set(UP)


def test_a_part_is_dropped_only_when_it_is_hardly_missed():
    lift = {Q[0]: 0.4, Q[1]: 0.010, Q[2]: 0.0005}  # what each adds to every damaged question
    model = Model(lambda state, j: 0.5 + sum(v for p, v in lift.items() if p in state))
    job = remeasure(model)
    for path in lift:
        model.registry.commit(path, 8)
    job.clean_up(lift, job.read(job.work))
    assert set(model.registry.active) == {Q[0], Q[1]}  # a point is worth keeping, a twentieth of one isn't
    many = {p: 0.0029 for p in PARTS[:20]}  # each hardly missed, all of them together very much so
    model = Model(lambda state, j: 0.5 + sum(v for p, v in many.items() if p in state))
    job = remeasure(model)
    for path in many:
        model.registry.commit(path, 8)
    start = job.read(job.work)
    job.clean_up(many, start)
    assert gain(start, job.read(job.work))[0] <= strategies.DROP_TOTAL + 1e-9 and len(model.registry.active) > 10


def test_eight_bits_are_kept_only_where_four_lose_the_gain():
    from src.product.strategies import Unit

    model = Model(lambda state, j: 0.96 if state.get((Q[0], Q[1])[j % 2], 3) >= (4, 8)[j % 2] else 0.1)
    job = remeasure(model)
    chosen = [Unit(Q[0], (Q[0],), 8), Unit(Q[1], (Q[1],), 8)]
    start = job.read(job.work)
    both = job.try_parts({Q[0]: 8, Q[1]: 8}, "both", job.work)
    fresh = {u: (*gain(job.try_parts({u.name: 8}, u.name, job.work), start), None) for u in chosen}
    parts, reading = job.right_size({Q[0]: 8, Q[1]: 8}, chosen, fresh, both)
    assert parts == {Q[0]: 4, Q[1]: 8} and gain(both, reading)[0] == pytest.approx(0)
    job.base_bits = 4  # nothing between 4-bit and 8-bit to try
    assert job.right_size({Q[0]: 8}, chosen[:1], fresh, both) == ({Q[0]: 8}, both)


def test_harm_to_stated_answers_lowers_a_parts_gain_by_what_it_takes_away():
    model = Model(fixes((Q[0], Q[1])), harm=lambda state, i: 0.6 if Q[0] in state else 0.0)
    job = remeasure(model)
    start = job.read(job.work)
    clean, _ = gain(job.try_parts({Q[1]: 8}, Q[1], job.work), start)
    harmful, _ = gain(job.try_parts({Q[0]: 8}, Q[0], job.work), start)
    assert clean > 0.5 and harmful == pytest.approx(clean - 0.6)


def test_parts_are_taken_together_only_when_they_add_up():
    from src.product.strategies import Unit

    def job_for(rule):
        job = remeasure(Model(rule))
        start = job.read(job.work)
        units = [Unit(p, (p,), 8) for p in (Q[0], Q[1], Q[2])]
        fresh = {u: (*gain(r, start), r) for u in units for r in [job.try_parts({u.name: 8}, u.name, job.work)]}
        ranked = sorted(units, key=lambda u: -fresh[u][0])
        return job.take_together(ranked, fresh, start, 0.0)[0], job

    friends, _ = job_for(fixes((Q[0],), (Q[1],), (Q[2],)))
    assert {u.name for u in friends} == {Q[0], Q[1], Q[2]}

    def rivals(state, j):  # each fixes its own third; any two together undo everything
        if len(set(state) & {Q[0], Q[1], Q[2]}) > 1:
            return 0.1
        return 0.96 if Q[j % 3] in state else 0.1

    alone, _ = job_for(rivals)
    assert len(alone) == 1
    same, _ = job_for(fixes((Q[0], Q[1], Q[2])))  # they fix the same questions: no reason to take two
    assert len(same) == 1
    crowded, job = job_for(fixes((Q[0],), (Q[1],), (Q[2],)))
    job.budget = job.cost({Q[0]: 8}) * 1.5  # room for one
    start = job.read(job.work)
    fresh = {u: (*gain(r, start), r) for u in crowded for r in [job.try_parts({u.name: 8}, u.name, job.work)]}
    assert len(job.take_together(crowded, fresh, start, 0.0)[0]) == 1


# --- mixed bits, wherever a patch is used --------------------------------------------------------------------

MIXED = {"name": "m", "model_id": "Qwen/Qwen2.5-3B-Instruct", "modules": [Q[0], UP[1]], "nbits": 8,
         "module_bits": {UP[1]: 4}, "base": {"method": "hqq", "nbits": 3, "group_size": 64, "compute_dtype": "fp16"}}


def test_the_kit_the_card_and_llama_cpp_say_each_parts_own_bits():
    from src.product.byom import llamacpp_lines
    from src.product.hub_publish import model_card
    from src.product.repair_kit import bits_of, bits_text, readme

    assert bits_of(MIXED) == {Q[0]: 8, UP[1]: 4} and bits_text(MIXED) == "4-bit (1) and 8-bit (1)"
    assert llamacpp_lines(MIXED) == ['--tensor-type "blk\\.0\\.attn_q=q8_0"', '--tensor-type "blk\\.1\\.ffn_up=q4_K"']
    for text in (readme(MIXED, {"memory_mb": 3}), model_card(MIXED, {"memory_mb": 3}, "a/b-fairgate")):
        assert f"- `{UP[1]}` (4-bit)" in text and "4-bit (1) and 8-bit (1)" in text
    plain = {k: v for k, v in MIXED.items() if k != "module_bits"}
    assert bits_text(plain) == "8-bit" and f"- `{UP[1]}`\n" in readme(plain, {"memory_mb": 3})
    with pytest.raises(ValueError, match="no llama.cpp type"):
        llamacpp_lines({**MIXED, "module_bits": {UP[1]: 7}})
