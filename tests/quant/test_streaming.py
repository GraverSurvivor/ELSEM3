"""The layer-by-layer loader builds the same model as loading whole and then quantizing."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("hqq")

FAMILIES = {
    "qwen2": {"hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 2, "num_attention_heads": 4,
              "num_key_value_heads": 2, "tie_word_embeddings": False},
    "gpt2": {"n_embd": 64, "n_layer": 2, "n_head": 4, "n_positions": 128},  # Conv1D, tied output head
}


def _save(model_type, folder):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    try:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct", local_files_only=True)
    except OSError:
        pytest.skip("Qwen tokenizer not cached")
    torch.manual_seed(0)
    config = AutoConfig.for_model(model_type, vocab_size=len(tok), **FAMILIES[model_type])
    AutoModelForCausalLM.from_config(config).save_pretrained(folder)
    tok.save_pretrained(folder)
    return tok


@pytest.mark.parametrize("model_type", sorted(FAMILIES))
def test_streaming_equals_load_then_quantize(tmp_path, model_type):
    from src.quant.hqq_quantizer import load_hqq_3bit_model
    from src.quant.streaming import load_streaming

    folder = tmp_path / model_type
    tok = _save(model_type, folder)
    inputs = tok(["A nurse and a mechanic met.", "Who was late?"], return_tensors="pt", padding=True)
    whole, _ = load_hqq_3bit_model(device="cpu", model_id=str(folder), nbits=4)
    streamed, streamed_tok = load_streaming(str(folder), nbits=4, device="cpu")
    with torch.no_grad():
        a = whole(**inputs).logits.float()
        b = streamed(**inputs).logits.float()
    assert torch.equal(a, b)  # bit-identical: same tensors, same casts, same HQQ settings
    assert streamed_tok.pad_token is not None


def test_the_eight_bit_reference_answers_like_the_original(tmp_path):
    from transformers import AutoModelForCausalLM

    from src.quant.streaming import load_streaming

    folder = tmp_path / "qwen2"
    tok = _save("qwen2", folder)
    inputs = tok(["A grandmother and her grandson set up a phone. Who struggled?"], return_tensors="pt")
    original = AutoModelForCausalLM.from_pretrained(folder, dtype=torch.float16).eval()
    reference, _ = load_streaming(str(folder), nbits=8, device="cpu")
    with torch.no_grad():
        a = original(**inputs).logits[0, -1].float()
        b = reference(**inputs).logits[0, -1].float()
    assert torch.nn.functional.cosine_similarity(a, b, dim=0) > 0.999
