"""The ARC general-knowledge check (src/product/arc_check.py): questions, sample, scoring, models, result."""

import json

import numpy as np
import pytest

from src.product import arc_check, panel
from src.product.arc_check import ArcItem


def row(qid, labels=("A", "B", "C", "D"), key="B"):
    return {"id": qid, "question": f" Question {qid}? ", "answerKey": key,
            "choices": {"text": [f" choice {x} " for x in labels], "label": list(labels)}}


def items(n=40):
    return [ArcItem(qid=f"q{i:03d}", set=arc_check.SETS[i % 2], question=f"Question {i}?",
                    choices=("w", "x", "y", "z"), answer=i % 4) for i in range(n)]


# --- questions -----------------------------------------------------------------------------------

def test_rows_become_questions_with_a_position_for_the_answer():
    item = arc_check.to_item(row("a", key="C"), "ARC-Easy")
    assert item.answer == 2 and item.choices[0] == "choice A" and item.question == "Question a?"
    assert arc_check.to_item(row("b", labels=("1", "2", "3", "4"), key="4"), "ARC-Easy").answer == 3
    assert arc_check.to_item(row("c", labels=("A", "B", "C")), "ARC-Easy") is None          # three choices
    assert arc_check.to_item(row("d", labels=("A", "B", "C", "D", "E")), "ARC-Easy") is None  # five
    assert arc_check.to_item(row("e", key="Z"), "ARC-Easy") is None


def test_the_sample_is_seeded_even_and_independent_of_dataset_order():
    pool = {"ARC-Easy": [arc_check.to_item(row(f"e{i}"), "ARC-Easy") for i in range(100)],
            "ARC-Challenge": [arc_check.to_item(row(f"c{i}"), "ARC-Challenge") for i in range(100)]}
    first = arc_check.sample(pool, 51, seed=3)
    shuffled = {k: list(reversed(v)) for k, v in pool.items()}
    assert first == arc_check.sample(shuffled, 51, seed=3)
    assert first != arc_check.sample(pool, 51, seed=4)
    assert sum(it.set == "ARC-Easy" for it in first) == 26 and sum(it.set == "ARC-Challenge" for it in first) == 25
    assert len({it.qid for it in first}) == 51
    small = {"ARC-Easy": pool["ARC-Easy"][:5], "ARC-Challenge": pool["ARC-Challenge"]}
    assert sum(it.set == "ARC-Easy" for it in arc_check.sample(small, 40)) == 5  # a small set gives all it has


# --- scoring on a real tokenizer -------------------------------------------------------------------

torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct", local_files_only=True)
    except OSError:
        pytest.skip("Qwen's tokenizer isn't available offline")


@pytest.fixture(scope="module")
def tiny(tokenizer):
    from transformers import AutoModelForCausalLM, Qwen2Config

    torch.manual_seed(0)
    config = Qwen2Config(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=2, vocab_size=len(tokenizer))
    return AutoModelForCausalLM.from_config(config).to(torch.float32).eval()


def test_the_prompt_lists_four_lettered_choices_and_reads_each_letters_own_token(tokenizer):
    item = items(1)[0]
    prompt = arc_check.format_prompt(item, tokenizer)
    assert "A) w\nB) x\nC) y\nD) z" in prompt and prompt.endswith("assistant\n")
    ids = arc_check.letter_ids(prompt, tokenizer)
    assert [tokenizer.decode([i]) for i in ids] == ["A", "B", "C", "D"]

    import copy

    plain = copy.deepcopy(tokenizer)
    plain.chat_template = None
    text = arc_check.format_prompt(item, plain)
    assert text.endswith("\nAnswer:")
    assert [plain.decode([i]).strip() for i in arc_check.letter_ids(text, plain)] == ["A", "B", "C", "D"]


def test_scores_are_the_models_own_and_dont_depend_on_the_batch(tokenizer, tiny):
    qs = items(12)
    one = arc_check.score(tiny, tokenizer, qs, "cpu", batch_size=1)
    many = arc_check.score(tiny, tokenizer, qs, "cpu", batch_size=5)
    assert [p for p, _ in one] == [p for p, _ in many]
    assert np.allclose([q for _, q in one], [q for _, q in many], atol=1e-4)
    # Checked by hand for one question: the letter with the highest logit is the pick.
    from src.eval.option_scorer import last_position_logits

    prompt = arc_check.format_prompt(qs[0], tokenizer)
    ids = arc_check.letter_ids(prompt, tokenizer)
    with torch.no_grad():
        logits = last_position_logits(tiny, tokenizer([prompt], return_tensors="pt"))[0, ids]
    probs = torch.softmax(logits.float(), dim=-1)
    assert one[0][0] == int(probs.argmax()) and one[0][1] == pytest.approx(float(probs[qs[0].answer]), abs=1e-4)


def test_running_out_of_memory_halves_the_batch(tokenizer, tiny, monkeypatch):
    from src.eval import option_scorer

    real, sizes = option_scorer.last_position_logits, []

    def flaky(model, inputs):
        sizes.append(inputs["input_ids"].shape[0])
        if inputs["input_ids"].shape[0] > 2:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2 GiB")
        return real(model, inputs)

    monkeypatch.setattr(option_scorer, "last_position_logits", flaky)
    picks = arc_check.score(tiny, tokenizer, items(6), "cpu", batch_size=8)
    assert len(picks) == 6 and sizes[:3] == [6, 4, 2]


# --- a run folder ------------------------------------------------------------------------------------

def make_run(tmp_path, patch=True, reference_bits=16, **report):
    run = tmp_path / "run"
    run.mkdir()
    (run / "report.json").write_text(json.dumps({"model_id": "org/model", "base_nbits": 3, "outcome": "repaired",
                                                  "reference_bits": reference_bits, **report}))
    if patch:
        (run / "patch.json").write_text(json.dumps({
            "model_id": "org/model", "modules": ["model.layers.0.mlp.down_proj"], "nbits": 8,
            "base": {"method": "rtn", "nbits": 4, "group_size": 64, "compute_dtype": "bf16"}}))
    return run


def fake(rates: dict, qs):
    """A loader and scorer whose `view` model is right on the first rates[view] share of questions."""
    def loader(view):
        return view, None

    def score(model, tokenizer, its, device, batch_size=8):
        k = int(round(rates[model] * len(its)))
        return [(it.answer if i < k else (it.answer + 1) % 4, 0.9 if i < k else 0.1) for i, it in enumerate(its)]
    return loader, score


@pytest.mark.parametrize("rates, state", [
    ({"original": 0.80, "compressed": 0.70, "repaired": 0.80}, "as_good_as_original"),
    ({"original": 0.90, "compressed": 0.60, "repaired": 0.62}, "no_harm_from_repair"),
    ({"original": 0.80, "compressed": 0.70, "repaired": 0.50}, "harm"),
])
def test_the_result_says_whether_the_repair_did_harm(tmp_path, monkeypatch, rates, state):
    qs = items(200)
    loader, score = fake(rates, qs)
    monkeypatch.setattr(arc_check, "score", score)
    result = arc_check.run(make_run(tmp_path), items=qs, loader=loader, device="cpu", on_progress=lambda m: None)
    assert result["state"] == state
    assert result["accuracy"] == pytest.approx(rates)
    saved = json.loads((tmp_path / "run" / "arc.json").read_text())
    assert saved["state"] == state and len(saved["answers"]) == 200
    assert saved["method"] == "rtn" and saved["base_nbits"] == 4 and saved["compute_dtype"] == "bf16"
    assert set(saved["by_set"]) == set(arc_check.SETS)
    assert saved["diff"]["repaired_minus_compressed"]["point"] == pytest.approx(rates["repaired"] - rates["compressed"])
    assert arc_check.summary_line(saved) == " / ".join(f"{rates[v]:.0%}" for v in arc_check.VIEWS)


def test_a_run_without_a_repair_compares_original_and_compressed_only(tmp_path, monkeypatch):
    qs = items(100)
    asked = []
    loader, score = fake({"original": 0.8, "compressed": 0.5}, qs)
    monkeypatch.setattr(arc_check, "score", score)
    run = make_run(tmp_path, patch=False, outcome="rejected")
    result = arc_check.run(run, items=qs, loader=lambda v: (asked.append(v), loader(v))[1], device="cpu",
                           on_progress=lambda m: None)
    assert asked == ["original", "compressed"]
    assert result["state"] == "no_repair" and result["accuracy"]["repaired"] is None
    assert "lowered" in result["message"] and not result["repaired"]
    assert all(a["repaired"] is None for a in result["answers"].values())


def test_identical_models_differ_by_exactly_nothing(tmp_path, monkeypatch):
    qs = items(100)
    loader, score = fake({"original": 0.7, "compressed": 0.7, "repaired": 0.7}, qs)
    monkeypatch.setattr(arc_check, "score", score)
    d = arc_check.run(make_run(tmp_path), items=qs, loader=loader, device="cpu", on_progress=lambda m: None)["diff"]
    assert d["repaired_minus_original"] == {"point": 0.0, "lower": 0.0, "upper": 0.0}


def test_the_models_are_rebuilt_the_way_the_run_measured_them(tmp_path, monkeypatch):
    from src.eval import model_loader as ml
    from src.quant import repaired, streaming

    calls = []
    monkeypatch.setattr(repaired, "load_repaired", lambda patch, device, streaming=False: calls.append(
        ("repaired", patch, streaming)) or ("m", "t"))
    monkeypatch.setattr(ml, "load_model", lambda dtype, model_id, device_map: calls.append(
        ("original", dtype, model_id)) or ("m", "t"))
    monkeypatch.setattr(streaming, "load_streaming", lambda model_id, nbits, device, dtype: calls.append(
        ("reference", nbits)) or ("m", "t"))

    load = arc_check.model_loader(arc_check.run_settings(make_run(tmp_path)), "cpu")
    for view in arc_check.VIEWS:
        load(view)
    assert calls[0] == ("original", "bf16", "org/model")
    bare, full = calls[1][1], calls[2][1]
    assert bare["modules"] == [] and bare["base"] == {"method": "rtn", "nbits": 4, "group_size": 64,
                                                      "compute_dtype": "bf16"}
    assert full["modules"] == ["model.layers.0.mlp.down_proj"] and calls[2][2] is False

    calls.clear()
    (tmp_path / "big").mkdir()
    big = make_run(tmp_path / "big", patch=False, reference_bits=8, base_nbits=3)
    load = arc_check.model_loader(arc_check.run_settings(big), "cpu")
    load("original")
    load("compressed")
    assert calls[0] == ("reference", 8)  # a big model's run compared against an 8-bit reference: so does ARC
    assert calls[1][1]["base"]["method"] == "hqq" and calls[1][2] is True


def test_a_folder_that_isnt_a_finished_run_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError, match="report.json"):
        arc_check.run_settings(tmp_path)


def test_the_command_line_skips_folders_already_checked(tmp_path, monkeypatch, capsys):
    run = make_run(tmp_path)
    (run / "arc.json").write_text("{}")
    monkeypatch.setattr(arc_check, "load_arc", lambda n, seed: pytest.fail("nothing should be loaded"))
    assert arc_check.main([str(run)]) == 0
    assert "already there" in capsys.readouterr().out


# --- the panel shows it ------------------------------------------------------------------------------

def test_the_panel_shows_general_knowledge_when_the_check_ran(tmp_path):
    from tests.product.test_panel import labels, models

    models(tmp_path / "drive")
    arc = {"accuracy": {"original": 0.8, "compressed": 0.7, "repaired": 0.6}, "state": "harm",
           "sample": {"n": 500}, "diff": {"repaired_minus_compressed": {"point": -0.1, "lower": -0.15, "upper": -0.05}}}
    (tmp_path / "drive" / "m1_v2" / "arc.json").write_text(json.dumps(arc))
    p = panel.build(tmp_path / "drive", labels())
    text = panel.to_markdown(p)
    assert "80% / 70% / 60%" in text and "not run" in text
    assert "m1 v2: the repair lowered general knowledge" in text
    out = tmp_path / "final"
    panel.copy_runs(tmp_path / "drive", out)
    panel.write(p, out, figures=False)
    assert (out / "runs" / "m1_v2" / "arc.json").exists()
    assert "arc_repaired" in (out / "panel.csv").read_text().splitlines()[0]


# --- end to end on a tiny saved model ----------------------------------------------------------------

def test_a_finished_run_is_checked_end_to_end_with_its_rebuilt_models(tmp_path):
    """The real rebuilding (load_model, load_repaired with HQQ 3-bit and an 8-bit part) and the real scorer,
    on a tiny random model saved on the spot: the three models are really different models."""
    pytest.importorskip("hqq")
    from tests.quant.tiny_models import save_tiny

    folder = tmp_path / "tiny"
    save_tiny(folder, "qwen2")
    run = tmp_path / "run"
    run.mkdir()
    (run / "report.json").write_text(json.dumps({"model_id": str(folder), "base_nbits": 3, "outcome": "repaired",
                                                  "reference_bits": 16}))
    (run / "patch.json").write_text(json.dumps({
        "name": "t", "model_id": str(folder), "modules": ["model.layers.0.self_attn.q_proj"], "nbits": 8,
        "base": {"method": "hqq", "nbits": 3, "group_size": 64, "compute_dtype": "fp16"}}))
    result = arc_check.run(run, items=items(16), device="cpu", on_progress=lambda m: None)
    assert result["state"] in {"as_good_as_original", "no_harm_from_repair", "harm"}
    probs = {v: [a[f"p_right_{v}"] for a in result["answers"].values()] for v in arc_check.VIEWS}
    assert probs["original"] != probs["compressed"] != probs["repaired"]
    assert json.loads((run / "arc.json").read_text())["sample"]["n"] == 16
