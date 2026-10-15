"""The patch file: one small JSON that says which modules to restore, and where that choice came from.

Before this, the list of modules to restore only existed inside a search's `summary.json`, so the
only repair `run_repair` could verify was "whatever a search kept". The revised plan needs it to
verify other choices too -- every baseline in the fairness controls is just a different set of
modules, and so is every prefix of the search's own list (1 module, 2 modules, ...). A patch is
that choice written down once, so the same verification and the same export code handle all of
them.

    {
      "name": "search_top5",
      "model_id": "microsoft/Phi-3.5-mini-instruct",
      "base": {"method": "hqq", "nbits": 3, "group_size": 64, "compute_dtype": "fp16"},
      "modules": ["model.layers.1.self_attn.qkv_proj", ...],   # restored in this order
      "nbits": 8,                                                # precision they're restored to
      "params": 122683392,                                       # weights restored
      "source": {"kind": "search", "search_run_id": "search_20260926T155230Z", ...}
    }

`params` is computed from Phi-3.5-mini's fixed module shapes rather than a loaded model, so
patches -- including budget-matched baselines -- can be built and checked on a CPU.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from src.eval.model_loader import MODEL_ID

# Phi-3.5-mini-instruct linear shapes (out_features, in_features), per decoder layer. Read from the
# model's config (hidden 3072, intermediate 8192, 32 heads x 96, fused qkv / gate_up).
MODULE_SHAPES: dict[str, tuple[int, int]] = {
    "self_attn.qkv_proj": (9216, 3072),
    "self_attn.o_proj": (3072, 3072),
    "mlp.gate_up_proj": (16384, 3072),
    "mlp.down_proj": (3072, 8192),
}
N_LAYERS = 32

DEFAULT_BASE = {"method": "hqq", "nbits": 3, "group_size": 64, "compute_dtype": "fp16"}
KINDS = ("search", "baseline", "prefix", "manual")


def module_suffix(path: str) -> str:
    """'model.layers.13.self_attn.qkv_proj' -> 'self_attn.qkv_proj'."""
    parts = path.split(".")
    if len(parts) != 5 or parts[0] != "model" or parts[1] != "layers" or not parts[2].isdigit():
        raise ValueError(f"not a decoder-layer module path: {path!r}")
    return ".".join(parts[3:])


def module_layer(path: str) -> int:
    module_suffix(path)
    return int(path.split(".")[2])


def module_params(path: str, part_shapes: dict | None = None, n_layers: int | None = None) -> int:
    """Weights in one candidate module: from Phi-3.5-mini's fixed shapes by default, or from the
    shapes a non-Phi patch carries (written by src/models/adapter.py)."""
    shapes = MODULE_SHAPES if part_shapes is None else part_shapes
    layers = N_LAYERS if n_layers is None else n_layers
    suffix = module_suffix(path)
    if suffix not in shapes:
        raise ValueError(f"{path!r} is not one of the candidate module types ({', '.join(shapes)})")
    if not 0 <= module_layer(path) < layers:
        raise ValueError(f"{path!r} names a layer outside 0-{layers - 1}")
    out_features, in_features = shapes[suffix]
    return out_features * in_features


def total_candidate_params(part_shapes: dict | None = None, n_layers: int | None = None) -> int:
    shapes = MODULE_SHAPES if part_shapes is None else part_shapes
    layers = N_LAYERS if n_layers is None else n_layers
    return layers * sum(o * i for o, i in shapes.values())


@dataclass(frozen=True)
class Patch:
    name: str
    modules: tuple[str, ...]
    nbits: int
    source: dict = field(default_factory=dict)
    base: dict = field(default_factory=lambda: dict(DEFAULT_BASE))
    model_id: str = MODEL_ID
    # For models other than Phi-3.5 (fast_repair): part type -> [out, in], and the layer count.
    # None means Phi-3.5's fixed shapes, so every existing patch file reads and writes unchanged.
    part_shapes: dict | None = None
    n_layers: int | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("a patch needs a name")
        if not self.modules:
            raise ValueError(f"patch {self.name!r} restores no modules")
        if len(set(self.modules)) != len(self.modules):
            raise ValueError(f"patch {self.name!r} lists a module twice: {self.modules}")
        for path in self.modules:
            module_params(path, self.part_shapes, self.n_layers)  # raises on anything that isn't a candidate
        if self.nbits <= self.base.get("nbits", 0):
            raise ValueError(
                f"patch {self.name!r} restores to {self.nbits}-bit, which isn't above the base "
                f"model's {self.base.get('nbits')}-bit -- that would not restore anything"
            )
        kind = self.source.get("kind")
        if kind is not None and kind not in KINDS:
            raise ValueError(f"unknown patch source kind {kind!r}; expected one of {KINDS}")

    @property
    def params(self) -> int:
        return sum(module_params(path, self.part_shapes, self.n_layers) for path in self.modules)

    @property
    def pct_params(self) -> float:
        return 100 * self.params / total_candidate_params(self.part_shapes, self.n_layers)

    def prefix(self, k: int, name: str | None = None) -> "Patch":
        """The first `k` modules, in restore order -- a search's state after round k."""
        if not 1 <= k <= len(self.modules):
            raise ValueError(f"prefix length must be 1-{len(self.modules)}, got {k}")
        return Patch(
            name=name or f"{self.name}_top{k}",
            modules=self.modules[:k],
            nbits=self.nbits,
            source={**self.source, "kind": "prefix", "prefix_of": self.name, "k": k},
            base=dict(self.base),
            model_id=self.model_id,
            part_shapes=self.part_shapes,
            n_layers=self.n_layers,
        )

    def to_dict(self) -> dict:
        data = asdict(self)
        data["modules"] = list(self.modules)
        for key in ("part_shapes", "n_layers"):  # Phi-3.5 patches keep their original format
            if data[key] is None:
                del data[key]
        if self.part_shapes is not None:
            data["part_shapes"] = {part: list(shape) for part, shape in self.part_shapes.items()}
        data["params"] = self.params
        return data


def patch_from_dict(data: dict) -> Patch:
    patch = Patch(
        name=data["name"],
        modules=tuple(data["modules"]),
        nbits=int(data["nbits"]),
        source=dict(data.get("source", {})),
        base=dict(data.get("base", DEFAULT_BASE)),
        model_id=data.get("model_id", MODEL_ID),
        part_shapes={k: tuple(v) for k, v in data["part_shapes"].items()} if data.get("part_shapes") else None,
        n_layers=int(data["n_layers"]) if data.get("n_layers") else None,
    )
    if "params" in data and int(data["params"]) != patch.params:
        raise ValueError(
            f"patch {patch.name!r} says it restores {data['params']} weights but its modules "
            f"add up to {patch.params} -- the file was edited by hand or built for another model"
        )
    return patch


def save_patch(patch: Patch, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(patch.to_dict(), indent=2) + "\n", encoding="utf-8")
    return path


def load_patch(path: Path) -> Patch:
    return patch_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def patch_from_search_summary(summary: dict, name: str = "search") -> Patch:
    """The modules a `run_search` (or `run_repair`) summary kept, as a patch, in keep order."""
    modules = summary.get("kept") or summary.get("modules")
    if not modules:
        raise ValueError("summary lists no kept modules -- is this a run_search or run_repair summary.json?")
    nbits = summary.get("nbits") or summary.get("config", {}).get("nbits")
    search_run_id = summary.get("search_run_id") or summary.get("run_id")
    return Patch(
        name=name,
        modules=tuple(modules),
        nbits=int(nbits),
        source={"kind": "search", "search_run_id": search_run_id},
    )
