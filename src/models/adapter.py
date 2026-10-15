"""What the repair needs to know about a model's architecture: its restorable parts, their sizes,
and their llama.cpp names (sprint contract C3, docs/CONTRACTS.md).

The search, the patch format and the exports were written for Phi-3.5, whose attention is a fused
`qkv_proj` + `o_proj` and whose MLP is a fused `gate_up_proj` + `down_proj`. Qwen2 and Llama-family
models keep those projections separate: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`,
`up_proj`, `down_proj`. An adapter turns a model's config into the per-layer candidate list
everything else uses, so the rest of the code needs no per-architecture branches.

Order rule (same as src/quant/candidates.py): grouped by part type, attention types first, each
type across every layer before the next, so a search cut short still covers every layer of the
cheaper attention parts. For Phi-3.5 the candidates equal today's 128 exactly.

    adapter_for(config).candidates()      # ["model.layers.0.self_attn.qkv_proj", ...]
    adapter_for(config).params(path)      # weights in that part, from the config's dimensions
    adapter_for(config).gguf_name(path)   # "blk.13.attn_qkv"
"""

from __future__ import annotations

from dataclasses import dataclass

LAYER_PREFIX = "model.layers"

# Part types per architecture family, in display/search order, with their llama.cpp tensor names.
FUSED = (  # Phi-3 / Phi-3.5
    ("self_attn.qkv_proj", "attn_qkv"),
    ("self_attn.o_proj", "attn_output"),
    ("mlp.gate_up_proj", "ffn_up"),
    ("mlp.down_proj", "ffn_down"),
)
SPLIT = (  # Qwen2, Llama, Mistral
    ("self_attn.q_proj", "attn_q"),
    ("self_attn.k_proj", "attn_k"),
    ("self_attn.v_proj", "attn_v"),
    ("self_attn.o_proj", "attn_output"),
    ("mlp.gate_proj", "ffn_gate"),
    ("mlp.up_proj", "ffn_up"),
    ("mlp.down_proj", "ffn_down"),
)
ARCHS = {"phi3": FUSED, "qwen2": SPLIT, "llama": SPLIT, "mistral": SPLIT}


@dataclass(frozen=True)
class ModelAdapter:
    arch: str
    n_layers: int
    part_shapes: dict          # part type -> (out_features, in_features), in display order
    gguf_names: dict           # part type -> llama.cpp tensor name

    @property
    def part_types(self) -> tuple[str, ...]:
        return tuple(self.part_shapes)

    def candidates(self) -> list[str]:
        return [f"{LAYER_PREFIX}.{layer}.{part}" for part in self.part_types for layer in range(self.n_layers)]

    def split(self, path: str) -> tuple[int, str]:
        """'model.layers.13.self_attn.qkv_proj' -> (13, 'self_attn.qkv_proj'), checked against this model."""
        parts = path.split(".")
        if len(parts) != 5 or parts[0] != "model" or parts[1] != "layers" or not parts[2].isdigit():
            raise ValueError(f"not a decoder-layer module path: {path!r}")
        layer, part = int(parts[2]), ".".join(parts[3:])
        if part not in self.part_shapes:
            raise ValueError(f"{part!r} is not a {self.arch} part; expected one of {self.part_types}")
        if not 0 <= layer < self.n_layers:
            raise ValueError(f"{path!r} names a layer outside 0-{self.n_layers - 1}")
        return layer, part

    def params(self, path: str) -> int:
        out_features, in_features = self.part_shapes[self.split(path)[1]]
        return out_features * in_features

    def total_params(self) -> int:
        return self.n_layers * sum(o * i for o, i in self.part_shapes.values())

    def gguf_name(self, path: str) -> str:
        layer, part = self.split(path)
        return f"blk.{layer}.{self.gguf_names[part]}"


def _get(config, *names, default=None):
    for name in names:
        value = config.get(name) if isinstance(config, dict) else getattr(config, name, None)
        if value is not None:
            return value
    return default


def adapter_for(config) -> ModelAdapter:
    """The adapter for a transformers config (or its dict). Raises ValueError for an architecture
    the repair doesn't support yet, naming the ones it does."""
    arch = str(_get(config, "model_type", default="")).lower()
    if arch not in ARCHS:
        raise ValueError(f"model type {arch or '(unknown)'!r} isn't supported yet; supported: {', '.join(sorted(ARCHS))}")
    hidden = int(_get(config, "hidden_size"))
    heads = int(_get(config, "num_attention_heads"))
    kv_heads = int(_get(config, "num_key_value_heads", default=heads))
    head_dim = int(_get(config, "head_dim", default=hidden // heads))
    inter = int(_get(config, "intermediate_size"))
    n_layers = int(_get(config, "num_hidden_layers"))
    q, kv = heads * head_dim, kv_heads * head_dim
    if ARCHS[arch] is FUSED:
        shapes = {
            "self_attn.qkv_proj": (q + 2 * kv, hidden),
            "self_attn.o_proj": (hidden, q),
            "mlp.gate_up_proj": (2 * inter, hidden),
            "mlp.down_proj": (hidden, inter),
        }
    else:
        shapes = {
            "self_attn.q_proj": (q, hidden),
            "self_attn.k_proj": (kv, hidden),
            "self_attn.v_proj": (kv, hidden),
            "self_attn.o_proj": (hidden, q),
            "mlp.gate_proj": (inter, hidden),
            "mlp.up_proj": (inter, hidden),
            "mlp.down_proj": (hidden, inter),
        }
    return ModelAdapter(arch=arch, n_layers=n_layers, part_shapes=shapes, gguf_names=dict(ARCHS[arch]))


def adapter_for_model(model) -> ModelAdapter:
    return adapter_for(model.config)


def adapter_for_id(model_id: str) -> ModelAdapter:
    """Reads only the model's config.json from the Hugging Face Hub (or the local cache)."""
    from transformers import AutoConfig

    return adapter_for(AutoConfig.from_pretrained(model_id))
