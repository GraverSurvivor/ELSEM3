"""The 128-candidate list Stage 4 searches over."""

import torch.nn as nn

from src.quant.candidates import candidate_paths, candidate_paths_for_model


def test_four_candidates_per_layer():
    paths = candidate_paths(n_layers=32)
    assert len(paths) == 128
    assert len(set(paths)) == 128


def test_uses_phi3_fused_projection_names():
    # Phi-3.5 has no q_proj/k_proj/v_proj; building the search around Llama-style names would
    # fail to resolve any module at all.
    suffixes = {path.split(".", 3)[3] for path in candidate_paths(n_layers=2)}
    assert suffixes == {"self_attn.qkv_proj", "self_attn.o_proj", "mlp.gate_up_proj", "mlp.down_proj"}


def test_attention_comes_before_mlp():
    paths = candidate_paths(n_layers=32)
    last_attention = max(i for i, path in enumerate(paths) if "self_attn" in path)
    first_mlp = min(i for i, path in enumerate(paths) if "mlp" in path)
    assert last_attention < first_mlp


def test_each_module_type_spans_every_layer_before_the_next_begins():
    # Grouped by type, not by layer: a search cut short by a time budget should have covered all
    # 32 layers of the cheap attention modules, not exhausted layer 0 and never reached layer 31.
    paths = candidate_paths(n_layers=32)
    qkv = [path for path in paths[:32]]
    assert all(path.endswith("self_attn.qkv_proj") for path in qkv)
    assert {int(path.split(".")[2]) for path in qkv} == set(range(32))


def test_paths_resolve_against_a_phi3_shaped_model():
    class _Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv_proj = nn.Linear(4, 12, bias=False)
            self.o_proj = nn.Linear(4, 4, bias=False)

    class _Mlp(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_up_proj = nn.Linear(4, 16, bias=False)
            self.down_proj = nn.Linear(8, 4, bias=False)

    class _Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = _Attn()
            self.mlp = _Mlp()

    class _Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([_Layer(), _Layer()])

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = _Inner()

    from src.quant.module_swap import get_module_by_path

    model = _Model()
    paths = candidate_paths_for_model(model)
    assert len(paths) == 8
    for path in paths:
        assert isinstance(get_module_by_path(model, path), nn.Linear)
