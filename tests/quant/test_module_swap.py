import torch
import torch.nn as nn

from src.quant.module_swap import get_module_by_path, set_module_by_path, snapshot_linear


class _Inner(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4)


class _Outer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_Inner(), _Inner()])


def test_get_module_by_path_resolves_nested_path():
    model = _Outer()
    resolved = get_module_by_path(model, "layers.1.proj")
    assert resolved is model.layers[1].proj


def test_set_module_by_path_replaces_in_place():
    model = _Outer()
    replacement = nn.Linear(4, 4)
    set_module_by_path(model, "layers.0.proj", replacement)
    assert model.layers[0].proj is replacement
    assert model.layers[1].proj is not replacement  # untouched


def test_snapshot_linear_clones_weights_independently():
    model = _Outer()
    original_weight = model.layers[0].proj.weight.data.clone()

    snapshot = snapshot_linear(model, "layers.0.proj")
    assert torch.equal(snapshot.weight.data, original_weight)

    # Mutating the live model's weight afterward must not affect the snapshot -- it has to be
    # a real clone, since HQQLinear deletes the original in place once it's quantized.
    model.layers[0].proj.weight.data.fill_(0.0)
    assert torch.equal(snapshot.weight.data, original_weight)
    assert not torch.equal(model.layers[0].proj.weight.data, original_weight)


def test_snapshot_linear_rejects_non_linear_module():
    model = _Outer()
    try:
        snapshot_linear(model, "layers")  # a ModuleList, not an nn.Linear
        assert False, "expected TypeError"
    except TypeError:
        pass
