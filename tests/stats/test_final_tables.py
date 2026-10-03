"""Final tables: sections fill in from whatever runs are committed, and say what they wait for."""

import json

import pytest

from src.experiments.schema import QuestionResult
from src.stats.final_tables import collect, to_markdown

UNKNOWN, STEREO = 2, 0
TOP5 = ["model.layers.1.self_attn.qkv_proj", "model.layers.14.self_attn.qkv_proj",
        "model.layers.15.self_attn.qkv_proj", "model.layers.13.self_attn.qkv_proj",
        "model.layers.10.self_attn.o_proj"]


def _write_run(folder, condition, answers, modules=None):
    folder.mkdir(parents=True)
    rows = []
    for i, a in enumerate(answers):
        category = ("Age", "SES")[i % 2]
        rows.append(QuestionResult(run_id=f"{condition}_1", condition=condition, question_id=f"{category}:{i}",
                                   category=category, predicted_index=a, label=UNKNOWN, unknown_index=UNKNOWN,
                                   is_unknown_predicted=a == UNKNOWN, is_correct=a == UNKNOWN,
                                   is_stereotyped_predicted=a == STEREO).__dict__)
    (folder / "question_results.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    if modules:
        (folder / "summary.json").write_text(json.dumps({"condition": condition, "modules": modules}), encoding="utf-8")


def _setup(tmp_path):
    ref, stats = tmp_path / "reference", tmp_path / "stats"
    stats.mkdir()
    _write_run(ref / "baseline_fp16", "fp16", [UNKNOWN] * 180 + [STEREO] * 20)
    _write_run(ref / "hqq_3bit", "hqq_3bit", [UNKNOWN] * 80 + [STEREO] * 120)
    _write_run(ref / "hqq_repaired", "hqq_repaired", [UNKNOWN] * 170 + [STEREO] * 30, TOP5)
    _write_run(ref / "repair_search_top5_top1", "hqq_repaired", [UNKNOWN] * 130 + [STEREO] * 70, TOP5[:1])
    return ref, stats


def test_sections_fill_in_or_say_what_they_wait_for(tmp_path, monkeypatch):
    ref, stats = _setup(tmp_path)
    final = collect(ref, stats)
    assert final["main"]["status"] == "done"
    assert final["main"]["usr_recovery"]["point"] == pytest.approx(0.9)  # (170 - 80) / (180 - 80)
    assert final["per_category"]["status"] == "done"

    curve = final["recovery_by_modules"]
    assert curve["status"] == "partial" and curve["missing_prefixes"] == [2, 3, 4]
    assert [r["modules"] for r in curve["rows"]] == [1, 5]
    assert curve["rows"][0]["usr_recovery"]["point"] == pytest.approx(0.5)
    assert not curve["rows"][0]["memory_measured"]

    for key in ("hedging", "crows_pairs", "open_ended", "llamacpp", "monitoring", "headline", "dissociation"):
        assert final[key]["status"] == "pending" and final[key]["waiting_for"]


def test_llamacpp_compares_only_llamacpp_runs(tmp_path):
    ref, stats = _setup(tmp_path)
    _write_run(ref / "llamacpp_f16", "llamacpp_f16", [UNKNOWN] * 180 + [STEREO] * 20)
    _write_run(ref / "llamacpp_q3km", "llamacpp_q3km", [UNKNOWN] * 100 + [STEREO] * 100)
    _write_run(ref / "llamacpp_q3km_fairgate", "llamacpp_q3km_fairgate", [UNKNOWN] * 140 + [STEREO] * 60)
    section = collect(ref, stats, only={"llamacpp"})["llamacpp"]
    assert section["status"] == "done"
    assert section["usr_recovery"]["point"] == pytest.approx(0.5)


def test_markdown_lists_what_is_pending(tmp_path):
    ref, stats = _setup(tmp_path)
    text = to_markdown(collect(ref, stats))
    assert "| Original (fp16) | 200 |" in text
    assert "## Still pending" in text and "CrowS-Pairs" in text.split("## Still pending")[1]
    assert "Not run yet: the 2, 3, 4-module versions." in text
