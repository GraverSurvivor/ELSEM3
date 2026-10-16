"""The model adapter, checked against real (tiny, randomly initialised) transformers models of each
architecture, so its part names and shapes are the ones transformers actually builds."""

import pytest
import torch.nn as nn
from transformers import AutoModelForCausalLM, LlamaConfig, MistralConfig, Phi3Config, Qwen2Config

from src.attribution.patch import MODULE_SHAPES, Patch, load_patch, save_patch
from src.models.adapter import adapter_for
from src.quant.candidates import candidate_paths, candidate_paths_for_model, filter_candidates
from src.quant.module_swap import get_module_by_path

SMALL = dict(hidden_size=64, intermediate_size=96, num_hidden_layers=3, num_attention_heads=4, vocab_size=128)
CONFIGS = {
    "phi3": Phi3Config(**SMALL, num_key_value_heads=4, pad_token_id=0),
    "qwen2": Qwen2Config(**SMALL, num_key_value_heads=2),
    "llama": LlamaConfig(**SMALL, num_key_value_heads=2),
    "mistral": MistralConfig(**SMALL, num_key_value_heads=2),
}
PHI35 = {"model_type": "phi3", "hidden_size": 3072, "num_attention_heads": 32, "num_key_value_heads": 32,
         "intermediate_size": 8192, "num_hidden_layers": 32}


@pytest.mark.parametrize("arch", sorted(CONFIGS))
def test_every_candidate_is_a_real_linear_of_the_stated_shape(arch):
    model = AutoModelForCausalLM.from_config(CONFIGS[arch])
    adapter = adapter_for(model.config)
    assert adapter.arch == arch and adapter.n_layers == 3
    candidates = adapter.candidates()
    assert len(candidates) == 3 * len(adapter.part_types) and len(set(candidates)) == len(candidates)
    for path in candidates:
        module = get_module_by_path(model, path)
        assert isinstance(module, nn.Linear), path
        assert tuple(module.weight.shape) == adapter.part_shapes[adapter.split(path)[1]], path
        assert adapter.params(path) == module.weight.numel()
    linear_weights = sum(m.weight.numel() for name, m in model.named_modules()
                         if isinstance(m, nn.Linear) and ".layers." in name)
    assert adapter.total_params() == linear_weights  # the candidates are every decoder linear layer
    assert candidate_paths_for_model(model) == candidates


def test_phi35_is_exactly_todays_candidate_set():
    adapter = adapter_for(PHI35)
    assert adapter.candidates() == candidate_paths(32)
    assert {k: tuple(v) for k, v in adapter.part_shapes.items()} == MODULE_SHAPES
    assert adapter.total_params() == 3_623_878_656  # the number in CLAUDE.md
    assert adapter.gguf_name("model.layers.13.self_attn.qkv_proj") == "blk.13.attn_qkv"
    assert adapter.gguf_name("model.layers.2.mlp.gate_up_proj") == "blk.2.ffn_up"


def test_attention_parts_come_first_across_every_layer():
    adapter = adapter_for(CONFIGS["qwen2"])
    candidates = adapter.candidates()
    first_mlp = next(i for i, p in enumerate(candidates) if ".mlp." in p)
    assert all(".self_attn." in p for p in candidates[:first_mlp])
    assert candidates[:3] == [f"model.layers.{i}.self_attn.q_proj" for i in range(3)]
    assert filter_candidates(candidates, "mlp") == candidates[first_mlp:]
    assert filter_candidates(candidates, "attention") == candidates[:first_mlp]
    assert adapter.gguf_name("model.layers.2.self_attn.k_proj") == "blk.2.attn_k"
    assert adapter.gguf_name("model.layers.0.mlp.gate_proj") == "blk.0.ffn_gate"


def test_bad_paths_and_architectures_are_refused():
    adapter = adapter_for(CONFIGS["llama"])
    for bad in ("model.layers.3.self_attn.q_proj", "model.layers.0.self_attn.qkv_proj", "lm_head"):
        with pytest.raises(ValueError):
            adapter.split(bad)
    with pytest.raises(ValueError, match="supported"):
        adapter_for({"model_type": "gpt2"})


def test_a_non_phi_patch_carries_its_shapes_and_round_trips(tmp_path):
    adapter = adapter_for(CONFIGS["qwen2"])
    patch = Patch(name="fast_repair_fast", modules=("model.layers.1.self_attn.q_proj", "model.layers.2.mlp.down_proj"),
                  nbits=8, source={"kind": "search"}, model_id="Qwen/Qwen2.5-3B-Instruct",
                  part_shapes={k: tuple(v) for k, v in adapter.part_shapes.items()}, n_layers=adapter.n_layers)
    assert patch.params == 64 * 64 + 64 * 96
    assert patch.pct_params == pytest.approx(100 * patch.params / adapter.total_params())
    path = save_patch(patch, tmp_path / "p.json")
    assert load_patch(path) == patch
    with pytest.raises(ValueError):  # a Phi-3.5 part isn't valid for this model
        Patch(name="x", modules=("model.layers.1.self_attn.qkv_proj",), nbits=8, part_shapes=patch.part_shapes,
              n_layers=3)


def test_a_qwen_part_can_be_restored_to_8bit_and_put_back():
    """The scan's core move on a split-attention model (Qwen2's q/k/v have biases, unlike Phi-3)."""
    pytest.importorskip("hqq")
    import copy

    import torch
    from hqq.core.quantize import HQQLinear
    from hqq.models.hf.base import AutoHQQHFModel

    from src.quant.hqq_quantizer import build_quant_config
    from src.quant.module_swap import SwapRegistry

    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(CONFIGS["qwen2"]).to(torch.float32)
    original = copy.deepcopy(model)

    class TwinWeights:  # stands in for CheckpointWeightSource: the unquantized weights, by path
        def get_linear(self, path, dtype):
            source = get_module_by_path(original, path)
            linear = nn.Linear(source.in_features, source.out_features, bias=source.bias is not None)
            linear.weight.data = source.weight.data.clone().to(dtype)
            if source.bias is not None:
                linear.bias.data = source.bias.data.clone().to(dtype)
            return linear

    AutoHQQHFModel.quantize_model(model, quant_config=build_quant_config(3, 64), compute_dtype=torch.float32,
                                  device="cpu")
    tokens = torch.randint(0, 128, (2, 6))
    compressed = model(tokens).logits
    registry = SwapRegistry(model, TwinWeights(), device="cpu")
    path = "model.layers.1.self_attn.q_proj"
    assert get_module_by_path(original, path).bias is not None
    with registry.trial(path, 8):
        assert isinstance(get_module_by_path(model, path), HQQLinear)
        assert get_module_by_path(model, path).meta["nbits"] == 8
        assert not torch.equal(model(tokens).logits, compressed)  # restoring changed the answers
    assert torch.equal(model(tokens).logits, compressed)  # and putting it back restored them exactly
