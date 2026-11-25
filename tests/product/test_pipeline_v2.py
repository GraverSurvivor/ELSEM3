"""Version 2's decisions (src/product/decisions.py): splits, probability scores, letter-balanced answers,
and the pipeline that uses them. Version 1 staying as it was is tests/product/test_v1_unchanged.py."""

import json
import math
from contextlib import contextmanager

import pytest

from src.product import fast_repair
from src.product.decisions import (
    OBJECTIVES,
    ScanLog,
    TrialScorer,
    WatchedRegistry,
    ask,
    ask_balanced,
    combine,
    kl,
    rotate,
    split_of,
    study,
)
from src.product.fast_repair import FastRepairConfig, in_search_half, run
from tests.product import test_fast_repair as t
from tests.product.test_strategies import Stack

AMBIG, DISAMBIG = t._items("ambig", 1200), t._items("disambig", 1200)


def _v2(tmp_path, backend, **kwargs):
    backend.registry = Stack()
    config = FastRepairConfig(model_id="Qwen/Qwen2.5-3B-Instruct", questions=kwargs.pop("questions", 800),
                              out=tmp_path / "job", pipeline="v2", **kwargs)
    return run(config, backend, load_questions=lambda: (AMBIG, DISAMBIG)), config.out


# --- splits --------------------------------------------------------------------------------------------

def test_splits_are_40_20_40_and_sit_inside_version_ones_halves():
    ids = [f"Age:{i}" for i in range(20_000)]
    splits = [split_of(q, 0) for q in ids]
    for name, share in (("search", 0.4), ("validation", 0.2), ("final", 0.4)):
        assert abs(splits.count(name) / len(ids) - share) < 0.01
    assert all(in_search_half(q, 0) for q, s in zip(ids, splits) if s == "search")
    assert not any(in_search_half(q, 0) for q, s in zip(ids, splits) if s == "final")
    assert splits == [split_of(q, 0) for q in ids] != [split_of(q, 1) for q in ids]


# --- asking --------------------------------------------------------------------------------------------

def test_rotating_moves_every_choice_and_its_meaning_together():
    item = t._items("disambig", 2)[1]
    for k in range(3):
        turned = rotate(item, k)
        assert sorted(turned.options) == sorted(item.options)
        assert turned.options[turned.label] == item.options[item.label]
        assert turned.options[turned.unknown_index] == item.options[item.unknown_index]
        assert turned.options[turned.stereotype_index] == item.options[item.stereotype_index]
    assert rotate(item, 0) == item and rotate(item, 3) == item
    from dataclasses import replace

    assert rotate(replace(item, stereotype_index=-1), 1).stereotype_index == -1  # "none" stays none


class LetterLover:
    """Believes in "can't tell", but adds a strong pull towards letter A, as Phi-3.5 at 3 bits does away
    from it: its printed answer depends on where the choices sit."""

    def probs(self, view, items, on_progress=None):
        out = []
        for it in items:
            row = [0.25, 0.25, 0.25]
            row[it.unknown_index] += 0.25
            row[0] += 0.4
            out.append(tuple(v / sum(row) for v in row))
        return out


def test_letter_balanced_answers_take_the_letters_out():
    items = [it for it in AMBIG[:20]]  # "can't tell" is choice C
    picks, printed = ask(LetterLover(), "compressed", items)
    assert set(picks) == {0}  # as printed: always A
    assert ask_balanced(LetterLover(), "compressed", items, printed) == [it.unknown_index for it in items]
    assert ask_balanced(LetterLover(), "compressed", [], []) == []


def test_a_backend_without_probabilities_still_works():
    picks, probs = ask(t.FakeBackend(), "original", AMBIG[:30])
    assert [max(range(3), key=p.__getitem__) for p in probs] == picks
    assert all(math.isclose(sum(p), 1) and max(p) == 0.98 for p in probs)


# --- the trial scores ------------------------------------------------------------------------------------

ITEM, STATED = AMBIG[0], DISAMBIG[0]  # ITEM: "can't tell" is choice 2; STATED: the answer is choice 0
FAIR, SURE_RIGHT = (0.02, 0.02, 0.96), (0.9, 0.05, 0.05)


def _scores(p, stated_p=SURE_RIGHT, before=SURE_RIGHT, hedged=False, **kwargs):
    scorer = TrialScorer({id(ITEM): FAIR, id(STATED): SURE_RIGHT}, {id(STATED): before},
                         {STATED.question_id: hedged}, **kwargs)
    top = lambda row: max(range(3), key=row.__getitem__)  # noqa: E731
    return scorer.scores([ITEM], [top(p)], [p], [STATED], [top(stated_p)], [stated_p])


def test_probability_scores_see_progress_the_count_cannot():
    """"Can't tell" rises from 10% to 45% without winning: nothing for the count, real progress."""
    low, high = _scores((0.8, 0.1, 0.1)), _scores((0.53, 0.02, 0.45))
    assert low["count"] == high["count"] == 0
    assert high["soft"] - low["soft"] == pytest.approx(0.35)
    assert high["kl"] > low["kl"]


def test_harm_to_stated_answers_costs_points_in_every_score():
    unsure = (0.45, 0.05, 0.5)  # now says "can't tell" though the story says who
    clean, hedges = _scores(FAIR), _scores(FAIR, stated_p=unsure)
    for name in OBJECTIVES:
        assert hedges[name] < clean[name], name
    assert _scores(FAIR, stated_p=unsure, weight=0)["soft"] == clean["soft"]


def test_general_repair_never_buys_a_part_its_place():
    """Qwen2.5-3B's trap: parts that brought general accuracy back while fair answers fell. Improving the
    stated answers over the compressed model earns nothing; only fair answers do."""
    wrong = (0.1, 0.85, 0.05)
    same = _scores((0.8, 0.1, 0.1), stated_p=wrong, before=wrong)
    better = _scores((0.8, 0.1, 0.1), stated_p=SURE_RIGHT, before=wrong)
    for name in OBJECTIVES:
        assert better[name] == pytest.approx(same[name]), name
    assert _scores((0.8, 0.1, 0.1), stated_p=(0.05, 0.05, 0.9), before=wrong, hedged=True)["count"] == same["count"]


def test_kl_is_zero_at_the_original_and_one_extreme_question_cannot_decide():
    assert kl(FAIR, FAIR) == pytest.approx(0)
    assert kl(FAIR, (0.9998, 0.0001, 0.0001)) < 5  # clamped: not thousands
    assert kl(FAIR, (1.0, 0.0, 0.0)) == pytest.approx(kl(FAIR, (0.98, 0.01, 0.01)), abs=0.05)


def test_combine_matches_a_hand_calculation():
    assert combine([(1, 0.6, 0.2), (0, 0.4, 0.6)], [(1, 0.3, 0.1), (0, 0.1, 0.1)], weight=0.5) == pytest.approx(
        {"soft": 0.5 - 0.1, "kl": -0.4 - 0.05, "count": 0.5 - 0.25})
    assert combine([(1, 0.6, 0.2)], []) == pytest.approx({"soft": 0.6, "kl": -0.2, "count": 1.0})


def test_the_scan_log_names_the_part_being_tried(tmp_path):
    registry = WatchedRegistry(t.FakeRegistry())
    log = ScanLog(tmp_path / "scan.jsonl")
    log.write(registry, 60, 40, {"soft": 0.5})
    with registry.trial(t.A, 8):
        log.write(registry, 60, 40, {"soft": 0.6})
    registry.commit(t.A, 8)
    log.write(registry, 60, 40, {"soft": 0.7})
    lines = [json.loads(line) for line in (tmp_path / "scan.jsonl").read_text().splitlines()]
    assert [(r["path"], r["kept"]) for r in lines] == [(None, []), (t.A, []), (None, [t.A])]
    with pytest.raises(RuntimeError), registry.trial(t.B, 8):
        raise RuntimeError("out of memory")
    assert registry.trying is None  # an error mid-trial doesn't leave its name behind


# --- the pipeline ----------------------------------------------------------------------------------------

@pytest.mark.parametrize("objective", OBJECTIVES)
def test_version_two_finds_the_repair_with_every_score(tmp_path, objective):
    report, out = _v2(tmp_path, t.FakeBackend(), objective=objective)
    assert report["outcome"] == "repaired" and report["gate"] == "PASS"
    assert {t.A, t.B} <= set(report["kept"]) and t.H not in report["kept"]
    assert report["pipeline"] == "v2" and report["objective"] == objective
    assert set(report["splits"]) == {"search", "validation", "final"}
    assert sum(s["ambiguous"] for s in report["splits"].values()) == 800
    assert report["checked_on"] == {"ambiguous": report["splits"]["final"]["ambiguous"],
                                    "stated_answer": report["splits"]["final"]["stated_answer"]}
    lines = [json.loads(line) for line in (out / "scan.jsonl").read_text().splitlines()]
    assert len(lines) > 28 and all(set(OBJECTIVES) <= set(r) for r in lines)
    assert {r["path"] for r in lines} >= {None, t.A, t.B}
    answers = json.loads((out / "answers.json").read_text())
    assert {a["split"] for a in answers["ambig"].values()} == {"search", "validation", "final"}


def test_no_decision_reads_the_final_or_validation_questions(tmp_path):
    """Between the first full pass over the compressed model and the final check, every question asked to find
    the parts belongs to the search split; the finished repairs are compared on validation; none is final."""
    backend = t.FakeBackend()
    _v2(tmp_path, backend)
    views = [view for view, _, _ in backend.calls]
    first_repaired = views.index("repaired")
    deciding = [ids for view, ids, restored in backend.calls[:first_repaired] if view == "compressed" and restored]
    assert len(deciding) > 28
    assert {split_of(q, 0) for ids in deciding for q in ids} == {"search", "validation"}
    last_search = max(k for k, ids in enumerate(deciding) if {split_of(q, 0) for q in ids} == {"search"})
    first_validation = min(k for k, ids in enumerate(deciding) if "validation" in {split_of(q, 0) for q in ids})
    assert last_search < first_validation  # chosen first, compared afterwards
    final = {q for view, ids, _ in backend.calls[first_repaired:] for q in ids}
    assert {split_of(q, 0) for q in final} == {"final"}  # and the check reads only the final split


def test_letter_balanced_numbers_are_reported_for_all_three_models(tmp_path):
    report, _ = _v2(tmp_path, t.FakeBackend())
    block = report["letter_balanced"]
    assert 0 < block["questions"]["ambiguous"] <= 400 and 0 < block["questions"]["stated_answer"] <= 400
    for view in ("original", "compressed", "repaired"):  # this stand-in doesn't care about letters
        assert block[view]["usr"] == pytest.approx(report[view]["usr"], abs=0.08)
    none, _ = _v2(tmp_path / "b", t.FakeBackend(damage=False))
    assert none["letter_balanced"]["repaired"] == none["letter_balanced"]["compressed"]


@pytest.mark.parametrize("make", [t.HedgingCompressed, t.BreaksFairness, t.BreaksStillFair, t.NeverUnsure,
                                  t.DiffuseDamage, t.DiffuseWithHedger, t.BrokenAtThreeBits])
def test_version_two_runs_every_stand_in_and_never_ships_a_worse_model(tmp_path, make):
    report, _ = _v2(tmp_path, make())
    assert report["outcome"] in fast_repair.OUTCOMES
    if report["patch"]:
        assert report["repaired"]["usr"] > report["compressed"]["usr"]
        from src.product.fast_repair import REAL_BITS

        adapter, base = t.adapter_for(t.TINY), report["base_nbits"]
        used = sum(adapter.params(p) * (REAL_BITS[b] - REAL_BITS[base]) for p, b in report["kept_bits"].items())
        assert used <= 0.10 * adapter.total_params() * (REAL_BITS[8] - REAL_BITS[base]) + 1


def test_version_one_has_none_of_this(tmp_path):
    report, _, out = t._run(tmp_path)
    assert report["pipeline"] == "v1"
    assert not {"objective", "splits", "letter_balanced"} & set(report)
    assert not (out / "scan.jsonl").exists()


def test_the_command_line_chooses_the_version_and_its_question_count(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(fast_repair, "run", lambda config: seen.update(vars(config)) or
                        {"minutes": 0, "kept": [], "gate": "PASS"})
    base = ["--model", "x/y", "--out", str(tmp_path)]
    assert fast_repair.main(base) == 0 and (seen["pipeline"], seen["questions"]) == ("v1", 1000)
    fast_repair.main([*base, "--pipeline", "v2", "--objective", "kl", "--study", "60"])
    assert (seen["pipeline"], seen["questions"], seen["objective"], seen["study"]) == ("v2", 2000, "kl", 60)
    fast_repair.main([*base, "--pipeline", "v2", "--questions", "500"])
    assert seen["questions"] == 500
    for bad in ({"pipeline": "v3"}, {"objective": "luck"}):
        with pytest.raises(ValueError):
            FastRepairConfig(model_id="x/y", **bad)


# --- choosing the score ----------------------------------------------------------------------------------

class SmoothParts:
    """Parts that each raise "can't tell" a little, from far below winning: the count sees almost nothing."""

    def __init__(self, n=24):
        self.parts = [f"model.layers.{i}.mlp.up_proj" for i in range(n)]
        self.registry = self
        self.active, self.trying = {}, None

    @contextmanager
    def trial(self, path, nbits):
        self.trying = path
        try:
            yield
        finally:
            self.trying = None

    def probs(self, view, items, on_progress=None):
        lift = 0.0 if self.trying is None else 0.02 * self.parts.index(self.trying)
        out = []
        for it in items:
            if it.label != it.unknown_index:
                out.append(SURE_RIGHT)
                continue
            p = min(0.9, 0.05 + 0.45 * ((it.example_id * 37) % 100) / 100 + lift)
            out.append(((1 - p) * 0.9, (1 - p) * 0.1, p))
        return out


def test_the_study_measures_random_parts_and_compares_the_scores_fairly(tmp_path, capsys):
    from src.product import compare_objectives

    backend, sample, stated = SmoothParts(), AMBIG[:300], DISAMBIG[:150]
    reference = {id(it): FAIR for it in sample} | {id(it): SURE_RIGHT for it in stated}
    scorer = TrialScorer(reference, {id(it): SURE_RIGHT for it in stated}, {})
    parts = study(backend, backend.parts, 20, sample, stated, scorer, 8, 0, tmp_path / "study.jsonl")
    assert len(parts) == len(set(parts)) == 20 and parts != backend.parts[:20]  # random, not the first

    records = compare_objectives.load(tmp_path / "study.jsonl")
    assert records[0]["path"] is None and [r["path"] for r in records[1:]] == parts
    assert records[0]["screen"] == [60, 40] and records[0]["weight"] == 1.0 and "screen" not in records[1]
    result = compare_objectives.compare(records, repeats=30)
    assert list(result["sizes"]) == ["36+24", "60+40", "120+75"]  # never more than half of the questions
    middle = result["sizes"]["60+40"]
    assert middle["soft"]["rank_correlation"] > middle["count"]["rank_correlation"]  # it sees the small steps
    assert result["best"] in ("soft", "kl")

    assert compare_objectives.main([str(tmp_path), "--repeats", "10"]) == 0
    assert "Best at the scan's own size" in capsys.readouterr().out
    assert json.loads((tmp_path / "objectives.json").read_text())["parts"] == 20
    (tmp_path / "bad.jsonl").write_text(json.dumps({"path": "a", "sample": [], "stated": []}) + "\n")
    with pytest.raises(ValueError, match="finished study"):
        compare_objectives.load(tmp_path / "bad.jsonl")


def test_the_best_score_is_judged_at_the_scans_own_size_and_weight(tmp_path):
    """Qwen2.5-3B has 252 parts, so a trial of its scan sees 36+24 questions, not Phi-3.5's 60+40."""
    from src.product.compare_objectives import compare, gains

    def part(path, small, large):  # question i: the first 40 sample questions say one thing, the rest another
        return {"path": path, "sample": [[0, small if i < 40 else large, 1.0] for i in range(400)],
                "stated": [[int(i % 2 == 0 and path is not None), 0.0, 0.0] for i in range(200)]}

    records = [part(None, 0.1, 0.1)] + [part(f"p{k}", 0.1 + k / 50, 0.1 + k / 50) for k in range(8)]
    records[0].update(screen=[36, 24], weight=2.0)
    result = compare(records, repeats=5)
    assert result["screen"] == [36, 24] and result["weight"] == 2.0 and "36+24" in result["sizes"]
    assert gains(records, [0], [0])["count"][0] == -2.0  # hedging costs the run's weight, not 1
    odd = compare(records, repeats=5, screen=(44, 29))  # 196 parts: a size of its own
    assert odd["screen"] == [44, 29] and "44+29" in odd["sizes"]
    old = [{k: v for k, v in r.items() if k not in ("screen", "weight")} for r in records]
    assert compare(old, repeats=2)["screen"] == [60, 40]  # a study from before it was recorded


def test_settings_that_would_do_nothing_or_fail_hours_later_are_refused_at_once(tmp_path, capsys):
    for bad in ({"study": 60}, {"objective": "kl"}, {"max_rounds": 0}, {"batch_size": 0},
                {"pipeline": "v2", "study": -1}, {"pipeline": "v2", "study": 3}):
        with pytest.raises(ValueError):
            FastRepairConfig(model_id="x/y", **bad)
    with pytest.raises(SystemExit):
        fast_repair.main(["--model", "x/y", "--out", str(tmp_path), "--study", "60"])
    assert "add --pipeline v2" in capsys.readouterr().err
    from src.product import compare_objectives

    with pytest.raises(SystemExit):
        compare_objectives.main([str(tmp_path)])
    assert "--study N first" in capsys.readouterr().err


def test_the_versions_share_their_final_questions_when_asked_for_the_same_number():
    items = t._items("ambig", 3000)
    sample = {it.question_id for it in fast_repair.stratified_sample(items, 1200, 0)}
    final = {q for q in sample if split_of(q, 0) == "final"}
    assert final and final <= {q for q in sample if not in_search_half(q, 0)}


def test_a_run_with_a_study_writes_it_and_still_repairs(tmp_path):
    report, out = _v2(tmp_path, t.FakeBackend(), study=10)
    lines = (out / "study.jsonl").read_text().splitlines()
    assert len(lines) == 11 and report["outcome"] == "repaired"
    assert json.loads(lines[0])["screen"] == [60, 40]
    events = [json.loads(line) for line in (out / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any("Study for choosing" in e["message"] for e in events)
