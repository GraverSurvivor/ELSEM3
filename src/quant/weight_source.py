"""On-demand access to the original (pre-quantization) weights of any linear module.

Why this exists instead of snapshotting before quantization
-----------------------------------------------------------
`snapshot_linear` has to run BEFORE the model is quantized, because `HQQLinear` deletes the
original weights as it packs them. For the single-module spike that is fine. For Stage 4's
search it is not.

The candidate set is every {qkv_proj, o_proj, gate_up_proj, down_proj} across 32 layers — 128
modules, 3.62B parameters, which is essentially every linear weight Phi-3.5-mini has. Holding
fp16 snapshots of all of them costs ~7.25 GB, against a 3-bit model of ~1.70 GB. Free Colab has
~12.7 GB of system RAM, so that is at best precarious and at worst an OOM before the search
starts. It also quietly wrecks the number this project reports: we would be holding 4.3x the
compressed model in order to claim a small "memory overhead vs plain 3-bit".

The weights are already on disk. `from_pretrained` leaves the safetensors shards in the HF
cache, and safetensors exists for exactly this access pattern — mmap a shard, read one tensor,
ignore the rest. So the search reads a candidate's weights when it tries that candidate, and
holds nothing in between. Peak cost is one module (~17-50 MB), not 128.

This is not an approximation of the snapshot path. `from_pretrained(torch_dtype=...)` casts
these same checkpoint tensors on load, so a tensor read here and cast the same way is
bit-identical to what `snapshot_linear` would have captured off the live model.

On caching: there deliberately isn't one. The dominant access pattern is a sequential scan over
all 128 candidates per greedy round, which is the worst case for LRU — any cache small enough to
be worth having thrashes completely. The read itself is ~50 ms of mmap'd disk against a
flipped-set evaluation measured in seconds, so it is not the bottleneck. If profiling ever says
otherwise, the fix is pinning a specific shortlist of candidates, not a general cache.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

from src.eval.model_loader import MODEL_ID

INDEX_FILE = "model.safetensors.index.json"
SINGLE_FILE = "model.safetensors"
BIN_INDEX_FILE = "pytorch_model.bin.index.json"   # older checkpoints, read only when there's no safetensors
BIN_SINGLE_FILE = "pytorch_model.bin"


class CheckpointWeightSource:
    """Reads individual pre-quantization weights out of a model's checkpoint.

    Resolves module paths the same way the model does, so `"model.layers.0.self_attn.qkv_proj"`
    reads the checkpoint's `"model.layers.0.self_attn.qkv_proj.weight"`. For other models it also
    finds names saved without the model's top-level prefix (GPT-2's checkpoints say `h.0.attn...`
    for `transformer.h.0.attn...`), reads `.bin` checkpoints when there are no safetensors, reads
    a local folder as well as a Hub id, and turns GPT-2-style `Conv1D` weights (stored (in, out))
    the right way round when `transposed(path)` says so.

    `revision` is worth pinning for a final run: this reads the checkpoint independently of
    whatever `from_pretrained` loaded, so if the Hub repo moved in between, the restored weights
    and the quantized model would come from different commits. In practice both hit the same
    local cache, which is why it defaults to None rather than being required.
    """

    def __init__(self, model_id: str = MODEL_ID, revision: str | None = None,
                 transposed: Callable[[str], bool] | None = None) -> None:
        self.model_id = model_id
        self.revision = revision
        self.transposed = transposed or (lambda path: False)
        self._local = Path(model_id) if Path(model_id).is_dir() else None
        self._shard_paths: dict[str, str] = {}
        self._bin_shards: dict[str, dict] = {}
        self._format = "safetensors"
        self._weight_map = self._load_weight_map()

    def _download(self, filename: str) -> str:
        if self._local is not None:
            path = self._local / filename
            if not path.exists():
                raise FileNotFoundError(f"{filename} is not in {self._local}")
            return str(path)
        return hf_hub_download(self.model_id, filename, revision=self.revision)

    def _load_weight_map(self) -> dict[str, str]:
        """Maps tensor name -> shard filename, from the index a sharded checkpoint ships."""
        try:
            index_path = self._download(INDEX_FILE)
        except Exception:
            # No index means the checkpoint isn't sharded, so every tensor is in one file.
            from safetensors import safe_open

            try:
                path = self._download(SINGLE_FILE)
            except Exception:
                return self._load_bin_weight_map()
            self._shard_paths = {SINGLE_FILE: path}
            with safe_open(path, framework="pt") as shard:
                return {key: SINGLE_FILE for key in shard.keys()}
        with open(index_path, encoding="utf-8") as handle:
            return json.load(handle)["weight_map"]

    def _load_bin_weight_map(self) -> dict[str, str]:
        self._format = "bin"
        try:
            with open(self._download(BIN_INDEX_FILE), encoding="utf-8") as handle:
                return json.load(handle)["weight_map"]
        except Exception:
            pass
        try:
            return {key: BIN_SINGLE_FILE for key in self._bin_shard(BIN_SINGLE_FILE)}
        except Exception as error:
            raise FileNotFoundError(f"{self.model_id} has no safetensors or pytorch_model.bin checkpoint") from error

    def _bin_shard(self, name: str) -> dict:
        if name not in self._bin_shards:
            # Memory-mapped and weights only: nothing is executed, and only the tensors read are touched.
            self._bin_shards[name] = torch.load(self._download(name), map_location="cpu", mmap=True,
                                                weights_only=True)
        return self._bin_shards[name]

    def _resolve(self, key: str) -> str | None:
        """The checkpoint's name for a model tensor name, or None."""
        if key in self._weight_map:
            return key
        parts = key.split(".")
        # Saved without the model's top-level prefix(es): drop leading names, but never past the
        # layer number, or one layer's weight could be read for another's.
        first_number = next((i for i, part in enumerate(parts) if part.isdigit()), len(parts) - 1)
        for i in range(1, first_number + 1):
            if ".".join(parts[i:]) in self._weight_map:
                return ".".join(parts[i:])
        for prefix in ("model.", "transformer."):  # or with one the model object doesn't have
            if prefix + key in self._weight_map:
                return prefix + key
        return None

    def has(self, key: str) -> bool:
        return self._resolve(key) is not None

    def get_tensor(self, key: str) -> torch.Tensor:
        """Reads one tensor. Only that tensor's bytes are touched, not the whole shard."""
        from safetensors import safe_open

        name = self._resolve(key)
        if name is None:
            raise KeyError(f"{key!r} is not in {self.model_id}'s checkpoint")
        shard_name = self._weight_map[name]
        if self._format == "bin":
            return self._bin_shard(shard_name)[name]
        if shard_name not in self._shard_paths:
            self._shard_paths[shard_name] = self._download(shard_name)
        with safe_open(self._shard_paths[shard_name], framework="pt") as shard:
            return shard.get_tensor(name)

    def get_linear(self, path: str, dtype: torch.dtype) -> nn.Linear:
        """Rebuilds the module at `path` as a plain `nn.Linear` at `dtype`, from the checkpoint.

        Returns a fresh object every call, on CPU. That matters because the caller hands it to
        `HQQLinear(..., del_orig=True)`, which destroys whatever linear it is given — a shared
        or cached instance would be consumed by the first promotion and unusable afterward.
        """
        weight_key = f"{path}.weight"
        weight = self.get_tensor(weight_key).to(dtype=dtype, copy=True)
        if weight.ndim != 2:
            raise ValueError(f"{weight_key!r} has shape {tuple(weight.shape)}, expected 2-D")
        if self.transposed(path):  # a Conv1D's (in, out) weight, as nn.Linear's (out, in)
            weight = weight.t().contiguous()
        out_features, in_features = weight.shape
        bias_key = f"{path}.bias"
        has_bias = self.has(bias_key)
        linear = nn.Linear(in_features, out_features, bias=has_bias)
        linear.weight.data = weight
        if has_bias:
            linear.bias.data = self.get_tensor(bias_key).to(dtype=dtype, copy=True)
        return linear
