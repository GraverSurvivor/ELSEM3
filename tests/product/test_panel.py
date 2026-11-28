"""The accuracy panel: version 1 vs version 2, from run folders (src/product/panel.py)."""

import json

import pytest

from src.product import panel

N = 400  # questions per set


def labels():
    return {"ambig": {f"a{i}": (i % 3, i % 3) for i in range(N)},
            "stated": {f"s{i}": ((i + 1) % 3, i % 3) for i in range(N)}}


def split(i, pipeline):
    """v1: halves. v2: v1's held-out half minus every tenth question (validation), as the real splits nest."""
    if pipeline == "v1":
        return "final" if i % 2 else "search"
    return "validation" if i % 10 == 1 else ("final" if i % 2 else "search")


def answers(pipeline, fix, right):
    """Picks: the original says "can't tell" on 90% of ambiguous questions, the compressed model
    on 30%, the repair brings back `fix` of what was lost; the repair is right on `right` of stated answers."""
    ambig, stated = {}, {}
    for i in range(N):
        unknown, other = i % 3, (i % 3 + 1) % 3
        orig = i % 10 != 0
        comp = i % 10 in (2, 3, 4)
        rep = comp or (orig and (i * 7) % 100 < 100 * fix)
        held = split(i, pipeline) == "final"
        ambig[f"a{i}"] = {"split": split(i, pipeline), "original": unknown if orig else other,
                          "compressed": unknown if comp else other, "repaired": (unknown if rep else other) if held else None}
        label = (i + 1) % 3
        stated[f"s{i}"] = {"split": split(i, pipeline), "original": label, "compressed": label,
                           "repaired": (label if (i * 13) % 100 < 100 * right else unknown) if held else None}
    return {"ambig": ambig, "stated": stated}


def report(model, pipeline, table, gate="PASS", recovery=None, extra=0.10, outcome="repaired", **more):
    def card(view):
        a = [(q, x) for q, x in table["ambig"].items() if x["split"] == "final"]
        s = [(q, x) for q, x in table["stated"].items() if x["split"] == "final"]
        lab = labels()
        return {"usr": sum(x[view] == lab["ambig"][q][1] for q, x in a) / len(a), "srs": 0.1,
                "false_unknown": sum(x[view] == lab["stated"][q][1] for q, x in s) / len(s),
                "accuracy": sum(x[view] == lab["stated"][q][0] for q, x in s) / len(s)}
    cards = {v: card(v) for v in ("original", "compressed", "repaired")}
    point = recovery if recovery is not None else panel.recovered(cards)
    return {"model_id": f"org/{model}", "pipeline": pipeline, "gate": gate, "outcome": outcome, **cards,
            "recovery": {"point": point, "lower": None if point is None else point - 0.05,
                         "upper": None if point is None else point + 0.05},
            "size": {"original": 400, "compressed": 100, "repaired": round(100 * (1 + extra))},
            "memory_mb": 10.0 * extra, "kept": ["p"] * 3, "base_nbits": 3, "minutes": 50.0, **more}


def make(folder, model, pipeline, fix=0.8, right=0.85, **kw):
    table = answers(pipeline, fix, right)
    run = folder / f"{model}_{pipeline}"
    run.mkdir(parents=True)
    (run / "answers.json").write_text(json.dumps(table))
    (run / "report.json").write_text(json.dumps(report(model, pipeline, table, **kw)))
    (run / "patch.json").write_text("{}")
    (run / "events.jsonl").write_text("big file, not copied\n")
    return run


def models(folder, names=("m1", "m2", "m3"), v1=None, v2=None):
    for name in names:
        make(folder, name, "v1", **(v1 or {}).get(name, {}))
        make(folder, name, "v2", **(v2 or {"m1": {}, "m2": {}, "m3": {}}).get(name, {"fix": 0.95}))


def test_the_runs_are_found_and_summarised_per_version(tmp_path):
    models(tmp_path, v2={n: {"fix": 0.95, "extra": 0.08} for n in ("m1", "m2", "m3")})
    make(tmp_path, "m4", "v1")  # only one version finished
    (tmp_path / "m5_v2").mkdir()  # no report yet: ignored
    p = panel.build(tmp_path, labels())
    assert {(r["model"], r["pipeline"]) for r in p["rows"]} == {(m, v) for m in ("m1", "m2", "m3") for v in ("v1", "v2")} | {("m4", "v1")}
    assert p["summary"]["v1"]["models"] == 4 and p["summary"]["v2"]["models"] == 3
    assert p["summary"]["v2"]["median_recovery"] > p["summary"]["v1"]["median_recovery"]
    assert p["summary"]["v2"]["passed"] == 3
    assert p["summary"]["v2"]["median_extra_size"] == pytest.approx(0.08)
    assert [m["complete"] for m in p["models"]] == [True, True, True, False]


def test_the_comparison_uses_only_questions_both_runs_held_out():
    first, second = answers("v1", 0.6, 0.8), answers("v2", 0.6, 0.8)
    # A v2 validation question answered (it never is): still not compared.
    second["ambig"]["a1"]["repaired"] = 0
    # A question version 1 searched on (runs with different --questions): not held out by both, not compared.
    first["ambig"]["a3"]["split"] = "search"
    ids = panel.shared(first, second, "ambig")
    assert ids == sorted(f"a{i}" for i in range(N) if i % 2 and i % 10 != 1 and i != 3)
    assert panel.paired(first, second, labels())["questions"] == {"ambiguous": len(ids), "stated_answer": len(ids) + 1}


def test_a_better_version_two_is_seen_question_by_question():
    better = panel.paired(answers("v1", 0.5, 0.80), answers("v2", 0.9, 0.80), labels())
    diff = better["recovery"]["diff"]
    assert diff["point"] > 0.3 and diff["lower"] > 0
    assert better["accuracy"]["diff"]["point"] == pytest.approx(0)
    same = panel.paired(answers("v1", 0.7, 0.8), answers("v2", 0.7, 0.8), labels())
    assert same["recovery"]["diff"]["point"] == pytest.approx(0)
    # Paired: the same answers resampled together differ by exactly nothing, in every resample.
    assert same["recovery"]["diff"]["lower"] == same["recovery"]["diff"]["upper"] == pytest.approx(0)
    assert same["same_original"] == 1.0
    fewer_right = panel.paired(answers("v1", 0.7, 0.9), answers("v2", 0.7, 0.6), labels())
    assert fewer_right["accuracy"]["diff"]["upper"] < 0
    assert fewer_right["false_unknown"]["diff"]["lower"] > 0  # its wrong answers are "can't tell"


def test_without_damage_there_is_no_recovery_to_compare():
    first, second = answers("v1", 0.5, 0.8), answers("v2", 0.5, 0.8)
    for table in (first, second):
        for answer in table["ambig"].values():
            answer["compressed"] = answer["original"]
    result = panel.paired(first, second, labels())
    assert result["recovery"] == {"v1": None, "v2": None, "diff": None}


def test_version_two_is_adopted_only_when_it_is_as_good_everywhere(tmp_path):
    models(tmp_path / "good")
    good = panel.build(tmp_path / "good", labels())
    assert good["verdict"]["switch_to_v2"] and "switch the app's default to version 2" in good["verdict"]["message"]

    models(tmp_path / "less_fair", v2={"m1": {"fix": 0.95}, "m2": {"fix": 0.95}, "m3": {"fix": 0.3}})
    bad = panel.build(tmp_path / "less_fair", labels())
    assert not bad["verdict"]["switch_to_v2"] and bad["verdict"]["less_fair"] == ["m3"]
    assert "m3" in bad["verdict"]["message"] and bad["verdict"]["message"].startswith("Keep version 1")

    # As many PASSes overall (m2 gains one), but m3 loses its own.
    models(tmp_path / "lost_pass", v1={"m2": {"gate": "FAIL"}},
           v2={"m1": {"fix": 0.95}, "m2": {"fix": 0.95}, "m3": {"fix": 0.95, "gate": "FAIL"}})
    lost = panel.build(tmp_path / "lost_pass", labels())
    assert not lost["verdict"]["switch_to_v2"] and lost["verdict"]["lost_pass"] == ["m3"]

    models(tmp_path / "less_right", v2={"m1": {"fix": 0.95}, "m2": {"fix": 0.95, "right": 0.5}, "m3": {"fix": 0.95}})
    wrong = panel.build(tmp_path / "less_right", labels())
    assert not wrong["verdict"]["switch_to_v2"] and wrong["verdict"]["less_right"] == ["m2"]


def test_two_models_decide_nothing(tmp_path):
    models(tmp_path, names=("m1", "m2"))
    p = panel.build(tmp_path, labels())
    assert not p["verdict"]["switch_to_v2"] and "at least 3" in p["verdict"]["message"]


def test_the_targets_are_reported_as_met_or_missed_never_adjusted(tmp_path):
    models(tmp_path, v2={"m1": {"fix": 0.85, "recovery": 0.85}, "m2": {"fix": 0.85, "recovery": 0.86},
                         "m3": {"fix": 0.85, "recovery": 0.84, "gate": "FAIL"}},
           v1={"m3": {"gate": "FAIL"}})
    targets = {t["target"]: t for t in panel.build(tmp_path, labels())["targets"]}
    assert not targets["Median fairness recovered"]["met"] and targets["Median fairness recovered"]["result"] == "85%"
    assert not targets["Models passing FairGate"]["met"] and targets["Models passing FairGate"]["result"] == "2 of 3"
    assert targets["No model worse than version 1"]["met"]


def test_the_panel_is_written_and_rebuilt_from_its_own_copy(tmp_path):
    models(tmp_path / "drive", v2={"m1": {"fix": 0.95, "prompt_style": "examples"},
                                   "m2": {"fix": 0.95, "escalated_from": {"base_nbits": 3}}, "m3": {"fix": 0.95}})
    objectives = tmp_path / "drive" / "m1_v2" / "objectives.json"
    objectives.write_text(json.dumps({"best": "soft", "screen": [36, 24]}))
    out = tmp_path / "final" / "panel"
    p = panel.build(tmp_path / "drive", labels())
    panel.copy_runs(tmp_path / "drive", out)
    panel.write(p, out)
    for name in ("panel.json", "panel.md", "panel.csv", "panel_recovery.png", "panel_memory.png"):
        assert (out / name).stat().st_size > 0
    assert sorted(f.name for f in (out / "runs" / "m1_v2").iterdir()) == ["answers.json", "objectives.json",
                                                                         "patch.json", "report.json"]
    text = (out / "panel.md").read_text(encoding="utf-8")
    assert "**Switch to version 2.**" in text and "three worked examples" in text and "moved to 4-bit" in text
    assert "`soft`" in text and "v2 (4-bit)" in text
    assert len((out / "panel.csv").read_text(encoding="utf-8").splitlines()) == 7
    again = panel.build(out / "runs", labels())
    assert json.loads(json.dumps(again)) == json.loads((out / "panel.json").read_text(encoding="utf-8"))
    panel.copy_runs(out / "runs", out)  # rebuilding from the copy doesn't copy onto itself


def test_the_final_tables_and_the_fairgate_report_show_the_panel(tmp_path):
    from src.product.fairgate import final_sections_html
    from src.stats.final_tables import collect, to_markdown

    ref = tmp_path / "results" / "reference"
    ref.mkdir(parents=True)
    pending = collect(ref, tmp_path, only={"panel"})
    assert pending["panel"]["status"] == "pending" and "panel" in pending["panel"]["waiting_for"]
    models(tmp_path / "drive")
    panel.write(panel.build(tmp_path / "drive", labels()), tmp_path / "results" / "final" / "panel", figures=False)
    done = collect(ref, tmp_path, only={"panel"})
    assert done["panel"]["status"] == "done"
    assert "Switch to version 2" in to_markdown(done)
    assert "version 1 vs version 2" in final_sections_html(done) and "Switch to version 2" in final_sections_html(done)


def test_the_command_line_refuses_a_folder_without_runs(tmp_path, capsys):
    with pytest.raises(SystemExit):
        panel.main([str(tmp_path)])
    assert "no finished runs" in capsys.readouterr().err
