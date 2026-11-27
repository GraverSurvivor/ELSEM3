"""Tiny random models and a tiny tokenizer, written to a folder like a Hub checkpoint, with no network:
so the method / patch tests run on any machine instead of skipping when a real tokenizer isn't cached."""

from __future__ import annotations

import pytest

TEXTS = ["A nurse and a mechanic met. Who was late?", "The grandmother and her grandson set up a phone.",
         "Who was not comfortable using the phone? Answer briefly.", "A. unknown B. the nurse C. the mechanic"] * 4
FAMILIES = {
    "qwen2": {"hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 2, "num_attention_heads": 4,
              "num_key_value_heads": 2, "max_position_embeddings": 128, "tie_word_embeddings": False},
    "gpt2": {"n_embd": 64, "n_layer": 2, "n_head": 4, "n_positions": 128},  # Conv1D weights, tied head
}


def tiny_tokenizer():
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(TEXTS, trainers.BpeTrainer(
        vocab_size=300, special_tokens=["<|endoftext|>"], initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    return PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="<|endoftext|>")


def save_tiny(folder, model_type: str = "qwen2", seed: int = 0):
    """A random model of one family and the tokenizer, saved in `folder`; returns the tokenizer."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    tok = tiny_tokenizer()
    torch.manual_seed(seed)
    config = AutoConfig.for_model(model_type, vocab_size=len(tok), **FAMILIES[model_type])
    config.pad_token_id = config.bos_token_id = config.eos_token_id = tok.eos_token_id
    AutoModelForCausalLM.from_config(config).save_pretrained(folder)
    tok.save_pretrained(folder)
    return tok


def bnb_runs(device: str = "cpu") -> bool:
    """Does bitsandbytes quantize and run a tiny model on this machine (some laptops: no)?"""
    try:
        import tempfile

        import torch
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig

        folder = tempfile.mkdtemp()
        save_tiny(folder)
        model = AutoModelForCausalLM.from_pretrained(
            folder, quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                           bnb_4bit_compute_dtype=torch.float16),
            dtype=torch.float16, device_map={"": device})
        with torch.no_grad():
            model(input_ids=torch.zeros(1, 4, dtype=torch.long, device=device))
        return True
    except Exception:  # noqa: BLE001 -- any failure means "not usable here"
        return False


needs_bnb = pytest.mark.skipif(not bnb_runs(), reason="bitsandbytes doesn't run on this machine (needs a GPU)")
