"""CheckpointWeightSource: reads the right tensor out of a sharded safetensors checkpoint.

Built against real safetensors files written to a temp dir rather than mocks, so the sharding
lookup, the dtype cast and the shape convention are exercised for real. The actual Phi-3.5
shards are 7.6 GB, which is not something a test suite should depend on.
"""

import json

import pytest
import torch
from safetensors.torch import save_file

from src.quant.weight_source import CheckpointWeightSource

OUT_FEATURES, IN_FEATURES = 8, 4


@pytest.fixture
def sharded_checkpoint(tmp_path, monkeypatch):
    """A two-shard checkpoint, so the weight-map lookup has to pick the right file."""
    torch.manual_seed(0)
    tensors = {
        "model.layers.0.self_attn.qkv_proj.weight": torch.randn(OUT_FEATURES, IN_FEATURES, dtype=torch.bfloat16),
        "model.layers.1.mlp.down_proj.weight": torch.randn(OUT_FEATURES, IN_FEATURES, dtype=torch.bfloat16),
    }
    layout = {
        "model.layers.0.self_attn.qkv_proj.weight": "model-00001-of-00002.safetensors",
        "model.layers.1.mlp.down_proj.weight": "model-00002-of-00002.safetensors",
    }

    for shard_name in set(layout.values()):
        save_file(
            {key: value for key, value in tensors.items() if layout[key] == shard_name},
            str(tmp_path / shard_name),
        )
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": layout}))

    def fake_download(model_id, filename, revision=None):
        path = tmp_path / filename
        if not path.exists():
            raise FileNotFoundError(filename)
        return str(path)

    monkeypatch.setattr("src.quant.weight_source.hf_hub_download", fake_download)
    return tensors


def test_reads_a_tensor_from_the_correct_shard(sharded_checkpoint):
    source = CheckpointWeightSource("fake/model")
    for key, expected in sharded_checkpoint.items():
        assert torch.equal(source.get_tensor(key), expected)


def test_unknown_key_raises(sharded_checkpoint):
    source = CheckpointWeightSource("fake/model")
    with pytest.raises(KeyError):
        source.get_tensor("model.layers.99.self_attn.qkv_proj.weight")


def test_get_linear_matches_the_checkpoint_after_casting(sharded_checkpoint):
    # The equivalence the whole lazy-load design rests on: reading a tensor and casting it is
    # the same thing from_pretrained(torch_dtype=...) does, so this must equal what
    # snapshot_linear would have captured off the loaded model.
    source = CheckpointWeightSource("fake/model")
    path = "model.layers.0.self_attn.qkv_proj"

    linear = source.get_linear(path, dtype=torch.float16)
    expected = sharded_checkpoint[f"{path}.weight"].to(torch.float16)

    assert torch.equal(linear.weight.data, expected)
    assert linear.weight.dtype == torch.float16


def test_get_linear_uses_the_nn_linear_shape_convention(sharded_checkpoint):
    # nn.Linear stores (out_features, in_features); getting this backwards would build a layer
    # that still runs on square inputs and silently computes the wrong thing.
    source = CheckpointWeightSource("fake/model")
    linear = source.get_linear("model.layers.0.self_attn.qkv_proj", dtype=torch.float32)

    assert (linear.in_features, linear.out_features) == (IN_FEATURES, OUT_FEATURES)
    assert linear.weight.shape == (OUT_FEATURES, IN_FEATURES)


def test_no_bias_when_the_checkpoint_has_none(sharded_checkpoint):
    # Phi-3.5 sets attention_bias=False, so no .bias keys exist at all.
    source = CheckpointWeightSource("fake/model")
    assert source.get_linear("model.layers.0.self_attn.qkv_proj", dtype=torch.float32).bias is None


def test_returns_an_independent_object_each_call(sharded_checkpoint):
    # HQQLinear(del_orig=True) destroys whatever linear it's handed, so a shared or cached
    # instance would be consumed by the first promotion and corrupt every later one.
    source = CheckpointWeightSource("fake/model")
    path = "model.layers.0.self_attn.qkv_proj"

    first = source.get_linear(path, dtype=torch.float32)
    second = source.get_linear(path, dtype=torch.float32)

    assert first is not second
    first.weight.data.fill_(0.0)
    assert not torch.equal(first.weight.data, second.weight.data)
    assert torch.equal(second.weight.data, sharded_checkpoint[f"{path}.weight"].to(torch.float32))


def test_unsharded_checkpoint_without_an_index(tmp_path, monkeypatch):
    torch.manual_seed(0)
    tensors = {"model.layers.0.mlp.down_proj.weight": torch.randn(OUT_FEATURES, IN_FEATURES)}
    save_file(tensors, str(tmp_path / "model.safetensors"))

    def fake_download(model_id, filename, revision=None):
        path = tmp_path / filename
        if not path.exists():
            raise FileNotFoundError(filename)  # no index.json -> falls back to the single file
        return str(path)

    monkeypatch.setattr("src.quant.weight_source.hf_hub_download", fake_download)

    source = CheckpointWeightSource("fake/model")
    assert source.has("model.layers.0.mlp.down_proj.weight")
    assert torch.equal(
        source.get_tensor("model.layers.0.mlp.down_proj.weight"),
        tensors["model.layers.0.mlp.down_proj.weight"],
    )


def test_any_checkpoint_layout_is_read(tmp_path):
    """A local folder, a .bin checkpoint, names saved without the model's prefix (GPT-2), and
    Conv1D weights stored (in, out): all read back as the module's nn.Linear."""
    torch.manual_seed(0)
    w = torch.randn(12, 8)  # Conv1D: (in=12, out=8)
    b = torch.randn(8)
    torch.save({"h.0.attn.c_attn.weight": w, "h.0.attn.c_attn.bias": b, "h.1.attn.c_attn.weight": w + 1},
               tmp_path / "pytorch_model.bin")
    source = CheckpointWeightSource(str(tmp_path), transposed=lambda path: path.endswith("c_attn"))
    linear = source.get_linear("transformer.h.0.attn.c_attn", dtype=torch.float32)
    assert (linear.out_features, linear.in_features) == (8, 12)
    assert torch.equal(linear.weight.data, w.t()) and torch.equal(linear.bias.data, b)
    assert not source.has("transformer.h.0.mlp.c_fc.weight")
    assert not source.has("transformer.h.2.attn.c_attn.weight")  # never another layer's weight
    with pytest.raises(FileNotFoundError):
        CheckpointWeightSource(str(tmp_path / "empty_dir_that_does_not_exist_as_model"))
