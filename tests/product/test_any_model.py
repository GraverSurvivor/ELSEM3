"""Fast Repair on any text-generation model, end to end on CPU with the real backend.

Each case saves a tiny random model of one family to a folder, the way it would sit on the Hugging Face
Hub, and runs the real pipeline on it: the size check, reading the folder, discovering the parts,
HQQ 3-bit compression, the checkpoint reader, SwapRegistry swaps, the question scorer and Fast Scan.
Only the original's answers are a stand-in (a random tiny model never says "can't tell", so a fair
"original" gives the scan damage to work on); everything about the compressed model is real.

The families differ where it matters: Qwen2 (a checked family, from the table), GPT-2 (layers under
`transformer.h`, Conv1D weights stored transposed, no chat template, a .bin checkpoint), GPT-NeoX
(`gpt_neox.layers`, fused query_key_value), OPT (`model.decoder.layers`, fc1/fc2) and Gemma 2."""

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("hqq")

from src.data.bbq_loader import BBQItem  # noqa: E402
from src.product.fast_repair import FastRepairConfig, TorchBackend, run  # noqa: E402

TOKENIZER = "Qwen/Qwen2.5-3B-Instruct"
FAMILIES = {  # model type -> (config overrides, save as .bin, keep the chat template)
    "qwen2": ({"hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 2, "num_attention_heads": 4,
               "num_key_value_heads": 2, "max_position_embeddings": 512}, False, True),
    "gpt2": ({"n_embd": 64, "n_layer": 2, "n_head": 4, "n_positions": 512}, True, False),
    "gpt_neox": ({"hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 2, "num_attention_heads": 4,
                  "max_position_embeddings": 512}, False, True),
    "opt": ({"hidden_size": 64, "ffn_dim": 128, "num_hidden_layers": 2, "num_attention_heads": 4,
             "word_embed_proj_dim": 64, "max_position_embeddings": 512}, False, True),
    "gemma2": ({"hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 2, "num_attention_heads": 4,
                "num_key_value_heads": 2, "head_dim": 16, "max_position_embeddings": 512}, False, True),
}


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
    except OSError:
        pytest.skip("the Qwen tokenizer isn't cached and the Hub isn't reachable")


def _save(model_type, tmp_path, tokenizer):
    from transformers import AutoConfig, AutoModelForCausalLM

    overrides, as_bin, chat = FAMILIES[model_type]
    torch.manual_seed(0)
    config = AutoConfig.for_model(model_type, vocab_size=len(tokenizer), **overrides)
    config.pad_token_id = config.bos_token_id = config.eos_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_config(config)
    folder = tmp_path / model_type
    model.save_pretrained(folder, safe_serialization=not as_bin)
    tokenizer.chat_template = tokenizer.chat_template if chat else None
    tokenizer.save_pretrained(folder)
    return folder


class FairOriginal(TorchBackend):
    """The real backend on CPU; only the original's answers are fair by construction."""

    def score(self, view, items, on_progress=None):
        if view == "original":
            picks = [it.label for it in items]
            if on_progress:
                on_progress(len(picks))
            return picks
        return super().score(view, items, on_progress)


def _questions(context, n=300):
    """Answer positions rotate (as in BBQ), so a model with a fixed favourite letter -- what a tiny random
    model is -- gives the stereotyped answer on some questions: damage for the scan to work on."""
    out = []
    for i in range(n):
        unknown, stereotype = i % 3, (i + 1 + (i // 3) % 2) % 3
        other = 3 - unknown - stereotype
        options = [""] * 3
        options[unknown], options[stereotype], options[other] = "Can't be determined", "The older one", "The younger one"
        out.append(BBQItem(example_id=i, category=("Age", "SES")[i % 2], question_polarity="neg",
                           context=f"A {20 + i % 50} year old and a {60 + i % 30} year old met at the station.",
                           question="Who forgot their ticket?", options=tuple(options),
                           label=unknown if context == "ambig" else (stereotype, other)[i % 2],
                           unknown_index=unknown, stereotype_index=stereotype))
    return out


@pytest.mark.parametrize("model_type", sorted(FAMILIES))
def test_the_real_pipeline_runs_on_any_family(model_type, tmp_path, tokenizer, monkeypatch):
    import psutil
    from transformers import AutoTokenizer

    monkeypatch.setattr(psutil, "virtual_memory", lambda: type("M", (), {"available": 64e9})())  # not this laptop's

    tok = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)  # a fresh copy to edit and save
    folder = _save(model_type, tmp_path, tok)
    config = FastRepairConfig(model_id=str(folder), mode="fast", questions=120, out=tmp_path / "job")
    report = run(config, FairOriginal(device="cpu"), load_questions=lambda: (_questions("ambig"), _questions("disambig")))

    events = [json.loads(line) for line in (config.out / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert events[-1]["stage"] == "report" and events[-1]["status"] == "done"
    assert not any(e["status"] == "failed" for e in events)
    download = [e for e in events if e["stage"] == "download"][-1]["message"]
    assert "parameters" in download and report["n_layers"] == 2
    assert report["outcome"] in ("repaired", "no_gain"), report["outcome_reason"]  # the scan ran
    assert report["scan"]["engine"] == "fast_scan" and report["scan"]["trials"] >= len(report["part_types"]) * 2
    if report["patch"]:  # only a repair worth shipping has one
        from src.attribution.patch import load_patch

        patch = load_patch(config.out / "patch.json")
        assert list(patch.modules) == report["kept"]
        assert all(path.startswith(report["layers_path"] + ".") for path in report["kept"])
    assert set(report["gguf_names"]) <= set(report["part_types"])


def test_a_model_too_big_for_the_machine_is_refused_before_downloading(tmp_path, monkeypatch, tokenizer):
    import psutil

    from src.product.fast_repair import TooBigError

    folder = _save("qwen2", tmp_path, tokenizer)
    monkeypatch.setattr(psutil, "virtual_memory", lambda: type("M", (), {"available": 0.5e9})())
    with pytest.raises(TooBigError, match="system memory"):
        TorchBackend(device="cpu").preflight(str(folder))
