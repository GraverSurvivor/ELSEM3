"""What the repair needs to know about a model's architecture: its restorable parts, their sizes,
and their llama.cpp names (sprint contract C3, docs/CONTRACTS.md).

The search, the patch format and the exports were written for Phi-3.5, whose attention is a fused
`qkv_proj` + `o_proj` and whose MLP is a fused `gate_up_proj` + `down_proj`. Other models name and
split those projections differently (`q_proj`/`k_proj`/`v_proj`, `query_key_value`, `c_attn`,
`fc1`/`fc2`, ...) and keep their layers under different paths (`model.layers`, `transformer.h`,
`gpt_neox.layers`, `model.decoder.layers`). An adapter turns a model into the per-layer candidate
list everything else uses, so the rest of the code needs no per-architecture branches.

Two ways to get one:
* the tables below, for the families the project was built and checked on (Phi-3, Qwen2, Llama,
  Mistral) -- for Phi-3.5 the candidates equal src/quant/candidates.py's 128 exactly;
* discovery, for any other text-generation model in transformers: build the model on the "meta"
  device from its config (no weights, no memory), find its stack of identical decoder blocks, and
  take every linear layer inside a block as a part type. GPT-2-style `Conv1D` layers count too;
  they store the weight transposed, which `transposed` records for the weight reader.

Order rule (same as src/quant/candidates.py): grouped by part type, attention types first, each
type across every layer before the next, so a search cut short still covers every layer of the
cheaper attention parts.

    adapter_for(config).candidates()      # ["model.layers.0.self_attn.qkv_proj", ...]
    adapter_for(config).params(path)      # weights in that part
    adapter_for(config).gguf_name(path)   # "blk.13.attn_qkv", or None when llama.cpp has no name for it
"""

from __future__ import annotations

from dataclasses import dataclass, field

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

# llama.cpp names for discovered parts, by the part's last name and whether it is in attention.
_GGUF_ATTENTION = {"q_proj": "attn_q", "k_proj": "attn_k", "v_proj": "attn_v", "o_proj": "attn_output",
                   "qkv_proj": "attn_qkv", "query_key_value": "attn_qkv", "c_attn": "attn_qkv", "Wqkv": "attn_qkv",
                   "dense": "attn_output", "out_proj": "attn_output", "c_proj": "attn_output"}
_GGUF_MLP = {"gate_proj": "ffn_gate", "up_proj": "ffn_up", "down_proj": "ffn_down", "gate_up_proj": "ffn_up",
             "fc1": "ffn_up", "fc2": "ffn_down", "fc_in": "ffn_up", "fc_out": "ffn_down", "c_fc": "ffn_up",
             "c_proj": "ffn_down", "dense_h_to_4h": "ffn_up", "dense_4h_to_h": "ffn_down"}
_ATTENTION_WORDS = ("attn", "attention")


def is_attention(part: str) -> bool:
    return any(word in part.split(".")[0].lower() for word in _ATTENTION_WORDS)


def _gguf_for(part: str) -> str | None:
    name = part.split(".")[-1]
    return (_GGUF_ATTENTION if is_attention(part) else _GGUF_MLP).get(name)


@dataclass(frozen=True)
class ModelAdapter:
    arch: str
    n_layers: int
    part_shapes: dict          # part type -> (out_features, in_features), in display order
    gguf_names: dict           # part type -> llama.cpp tensor name (missing: llama.cpp has none)
    layers_path: str = LAYER_PREFIX                      # where the decoder blocks live
    transposed: frozenset = field(default_factory=frozenset)  # part types stored as Conv1D (in, out)

    @property
    def part_types(self) -> tuple[str, ...]:
        return tuple(self.part_shapes)

    def candidates(self) -> list[str]:
        return [f"{self.layers_path}.{layer}.{part}" for part in self.part_types for layer in range(self.n_layers)]

    def split(self, path: str) -> tuple[int, str]:
        """'model.layers.13.self_attn.qkv_proj' -> (13, 'self_attn.qkv_proj'), checked against this model."""
        prefix = self.layers_path + "."
        rest = path[len(prefix):] if path.startswith(prefix) else ""
        layer, _, part = rest.partition(".")
        if not layer.isdigit() or not part:
            raise ValueError(f"not a decoder-layer module path of this model ({self.layers_path}.<n>.<part>): {path!r}")
        if part not in self.part_shapes:
            raise ValueError(f"{part!r} is not a {self.arch} part; expected one of {self.part_types}")
        if not 0 <= int(layer) < self.n_layers:
            raise ValueError(f"{path!r} names a layer outside 0-{self.n_layers - 1}")
        return int(layer), part

    def params(self, path: str) -> int:
        out_features, in_features = self.part_shapes[self.split(path)[1]]
        return out_features * in_features

    def total_params(self) -> int:
        return self.n_layers * sum(o * i for o, i in self.part_shapes.values())

    def gguf_name(self, path: str) -> str | None:
        layer, part = self.split(path)
        name = self.gguf_names.get(part)
        return None if name is None else f"blk.{layer}.{name}"

    def is_transposed(self, path: str) -> bool:
        return self.split(path)[1] in self.transposed


def _get(config, *names, default=None):
    for name in names:
        value = config.get(name) if isinstance(config, dict) else getattr(config, name, None)
        if value is not None:
            return value
    return default


def _from_table(config) -> ModelAdapter:
    arch = str(_get(config, "model_type", default="")).lower()
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


def _linear_shape(module) -> tuple[tuple[int, int], bool] | None:
    """(out_features, in_features) and whether the weight is stored transposed, for a linear layer."""
    import torch.nn as nn

    if isinstance(module, nn.Linear):
        return (module.out_features, module.in_features), False
    if type(module).__name__ == "Conv1D" and getattr(module, "weight", None) is not None:  # GPT-2 family
        in_features, out_features = module.weight.shape
        return (int(out_features), int(in_features)), True
    return None


def discover(model) -> ModelAdapter:
    """The adapter for any model object (real or on the meta device): its stack of decoder blocks
    and the linear layers every block has, with the same shape in every block."""
    import torch.nn as nn

    config = getattr(model, "config", None)
    arch = str(getattr(config, "model_type", "") or type(model).__name__).lower()
    n_expected = _get(config, "num_hidden_layers", "n_layer", "num_layers", "n_layers") if config is not None else None
    stacks = [(name, mods) for name, mods in model.named_modules()
              if isinstance(mods, nn.ModuleList) and len(mods) > 0 and len({type(m) for m in mods}) == 1]
    if not stacks:
        raise ValueError(f"can't find the decoder layers of this {arch} model")
    matching = [s for s in stacks if n_expected is not None and len(s[1]) == int(n_expected)]
    layers_path, blocks = (matching or sorted(stacks, key=lambda s: -len(s[1])))[0]

    def linears(block) -> dict:
        found = {}
        for name, module in block.named_modules():
            shape = _linear_shape(module)
            if shape is not None and name:
                found[name] = shape
        return found

    first = linears(blocks[0])
    common = {name: shape for name, shape in first.items() if all(linears(b).get(name) == shape for b in blocks[1:])}
    if not common:
        raise ValueError(f"this {arch} model has no linear layers shared by all its decoder layers")
    order = list(common)
    parts = [p for p in order if is_attention(p)] + [p for p in order if not is_attention(p)]
    shapes = {p: common[p][0] for p in parts}
    gguf = {p: name for p in parts if (name := _gguf_for(p)) is not None}
    return ModelAdapter(arch=arch, n_layers=len(blocks), part_shapes=shapes, gguf_names=gguf,
                        layers_path=layers_path, transposed=frozenset(p for p in parts if common[p][1]))


# Encoder models transformers also registers for causal LM (with is_decoder): not chat/text models.
ENCODERS = {"bert", "roberta", "xlm-roberta", "xlm-roberta-xl", "camembert", "electra", "ernie", "data2vec-text",
            "megatron-bert", "rembert", "roc_bert", "xlm", "xmod", "bert-generation", "big_bird",
            "roberta-prelayernorm", "deberta", "deberta-v2", "distilbert", "albert", "mpnet", "modernbert"}
GENERATIVE_HEADS = ("ForCausalLM", "LMHeadModel", "ForConditionalGeneration")


def check_generative(config) -> None:
    """Refuses encoders and seq2seq models: the repair is for text-generation (decoder) models.
    Uses the architecture the model's repo declares when there is one."""
    arch = str(getattr(config, "model_type", "") or "").lower()
    declared = list(getattr(config, "architectures", None) or [])
    reason = None
    if getattr(config, "is_encoder_decoder", False):
        reason = "an encoder-decoder (seq2seq) model"
    elif arch in ENCODERS:
        reason = "an encoder (embedding/classification) model"
    elif declared and not any(head in name for name in declared for head in GENERATIVE_HEADS[:2]):
        reason = f"declared as {', '.join(declared)}, not a text-generation model"
    if reason:
        raise ValueError(f"model type {arch!r} is {reason}; FairGate repairs chat and text-generation models")


def meta_model(config):
    """The model built from its config on the meta device: every module and shape, no weights.
    Raises ValueError for a model that isn't a text-generation (causal) language model."""
    from accelerate import init_empty_weights
    from transformers import AutoModelForCausalLM

    check_generative(config)
    try:
        with init_empty_weights():
            return AutoModelForCausalLM.from_config(config)
    except ValueError as error:
        if "Unrecognized configuration class" in str(error):
            raise ValueError(f"model type {getattr(config, 'model_type', '?')!r} isn't a text-generation "
                             "(causal) language model; FairGate repairs chat and text-generation models") from None
        raise


def adapter_for(config) -> ModelAdapter:
    """The adapter for a transformers config (or a dict of one). The checked families use the
    tables; any other model type is discovered from its structure. Raises ValueError for a model
    that isn't a text-generation language model, or whose layers can't be found."""
    arch = str(_get(config, "model_type", default="")).lower()
    if arch in ARCHS:
        return _from_table(config)
    if isinstance(config, dict):
        from transformers import AutoConfig

        try:
            config = AutoConfig.for_model(**config)
        except (ValueError, KeyError, TypeError) as error:
            raise ValueError(f"model type {arch or '(unknown)'!r} isn't one transformers knows: {error}") from None
    return discover(meta_model(config))


def adapter_for_model(model) -> ModelAdapter:
    arch = str(getattr(model.config, "model_type", "")).lower()
    return _from_table(model.config) if arch in ARCHS else discover(model)


def adapter_for_id(model_id: str) -> ModelAdapter:
    """Reads only the model's config.json from the Hugging Face Hub (or the local cache/folder)."""
    from transformers import AutoConfig

    return adapter_for(AutoConfig.from_pretrained(model_id))
