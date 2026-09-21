"""Runtime module swap: change one HQQ-quantized linear layer's bit-width in the live model,
without reloading it — and put it back exactly.

Two things live here. `promote_module` / `snapshot_linear` are the raw mechanism, de-risked as a
Week 2 spike on a single module. `SwapRegistry` is what Stage 4's greedy search actually drives:
it owns the swap state so a trial can be undone, and reports what the kept promotions cost.

Together with `src/quant/weight_source.py` they answer the two questions the search poses that a
one-off swap does not. Where do the original weights come from, when snapshotting all 128
candidates would cost 7.25 GB against a 1.70 GB model (answer: read them from the checkpoint on
demand). And how does a trial get undone, when requantizing back is slow and assumes HQQ is
bit-deterministic (answer: keep the displaced module object and put it back).

Phi-3.5 fuses projections Llama-family models keep separate — attention is `qkv_proj` + `o_proj`,
MLP is `gate_up_proj` + `down_proj`, so 4 candidates per layer and 128 in total. See
`src/quant/candidates.py`, which builds that list in the order the search walks it.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from hqq.core.quantize import HQQLinear

from src.quant.hqq_quantizer import DEFAULT_GROUP_SIZE, build_quant_config

if TYPE_CHECKING:
    from src.quant.weight_source import CheckpointWeightSource


def infer_compute_dtype(model: nn.Module) -> torch.dtype:
    """Reads the compute dtype off an already-quantized layer in `model`.

    A promoted module that dequantizes into a different dtype than the rest of the model is a
    silent correctness bug, not a crash: torch will happily upcast across the boundary and the
    model keeps answering, just at a precision nothing recorded. Since the project's baseline
    dtype is now a measured choice rather than a constant (see CLAUDE.md > Precision), taking
    the dtype from the live model is safer than any default this module could hardcode.
    """
    for module in model.modules():
        if isinstance(module, HQQLinear):
            return module.compute_dtype
    raise ValueError(
        "No HQQLinear found in model - promote_module expects an already-quantized model. "
        "Pass compute_dtype explicitly if swapping into an unquantized one."
    )


def get_module_by_path(model: nn.Module, path: str) -> nn.Module:
    """Resolves a dotted module path (e.g. "model.layers.0.self_attn.qkv_proj") to the module."""
    module = model
    for part in path.split("."):
        module = getattr(module, part)
    return module


def set_module_by_path(model: nn.Module, path: str, new_module: nn.Module) -> None:
    """Replaces the module at `path` with `new_module`, in place — no reload."""
    *parent_parts, leaf = path.split(".")
    parent = model
    for part in parent_parts:
        parent = getattr(parent, part)
    setattr(parent, leaf, new_module)


def snapshot_linear(model: nn.Module, path: str) -> nn.Linear:
    """Clones a not-yet-quantized nn.Linear's weight/bias, to requantize at a different
    bit-width later without needing the original checkpoint again.

    Must be called BEFORE the module at `path` is quantized: `HQQLinear` deletes the original
    linear layer's weights in place by default (`del_orig=True`, to free memory as it packs the
    quantized weights), so there's no way to recover them from the model after quantization.
    """
    original = get_module_by_path(model, path)
    if not isinstance(original, nn.Linear):
        raise TypeError(f"Expected nn.Linear at {path!r}, got {type(original).__name__}")

    clone = nn.Linear(original.in_features, original.out_features, bias=original.bias is not None)
    clone.weight.data = original.weight.data.clone()
    if original.bias is not None:
        clone.bias.data = original.bias.data.clone()
    return clone


def promote_module(
    model: nn.Module,
    path: str,
    snapshot: nn.Linear,
    nbits: int,
    group_size: int = DEFAULT_GROUP_SIZE,
    compute_dtype: torch.dtype | None = None,
    device: str = "cuda",
) -> None:
    """Swaps the quantized module at `path` for a fresh HQQLinear at `nbits`, requantized from
    `snapshot` (the pre-quantization weights — see `snapshot_linear`). Modifies `model` in
    place; no reload of the model, no re-running `from_pretrained`.

    `snapshot` itself is never mutated: a fresh nn.Linear is built from its weights each call,
    since `HQQLinear(..., del_orig=True)` deletes whatever linear layer it's handed, and the
    snapshot needs to survive being promoted more than once (e.g. demoted back to 3-bit later).

    `compute_dtype=None` (the default) reads the dtype off the model's existing quantized
    layers, so the promoted module can't end up at a different precision than everything around
    it. Pass it explicitly only to override that deliberately.
    """
    if compute_dtype is None:
        compute_dtype = infer_compute_dtype(model)

    fresh = nn.Linear(snapshot.in_features, snapshot.out_features, bias=snapshot.bias is not None)
    fresh.weight.data = snapshot.weight.data.clone()
    if snapshot.bias is not None:
        fresh.bias.data = snapshot.bias.data.clone()

    quant_config = build_quant_config(nbits=nbits, group_size=group_size)
    promoted = HQQLinear(fresh, quant_config=quant_config, compute_dtype=compute_dtype, device=device)
    set_module_by_path(model, path, promoted)


def module_nbytes(module: nn.Module) -> int:
    """Actual bytes of tensor storage a module holds.

    Measured rather than computed from a bits-per-weight formula, because HQQ's real footprint
    isn't what the nbits alone implies: packing overhead makes 3-bit cost 3.75 bits/weight and
    8-bit cost 8.50, once the per-group scale and zero are counted. A formula in this file would
    be a second, quietly drifting model of HQQ's internals.

    `HQQLinear` keeps its scale and zero inside a plain `meta` dict, which `named_parameters()`
    and `named_buffers()` do not see -- only `W_q` shows up there. Missing them would undercount
    every quantized module by the exact amount that distinguishes bit-widths, so dict-valued
    attributes are walked too. Shared storage is counted once.
    """
    seen: set[tuple[int, int]] = set()
    total = 0

    def add(tensor: torch.Tensor) -> None:
        """Counts the tensor's underlying storage, once per distinct allocation.

        Keyed on storage rather than on the tensor object because HQQ hands out views: several
        tensors can share one allocation, and counting each view's `numel` would inflate the
        total. Counting the storage is also what actually occupies memory -- a small view into a
        large allocation keeps the whole thing alive.
        """
        nonlocal total
        try:
            storage = tensor.untyped_storage()
        except (RuntimeError, NotImplementedError):
            total += tensor.numel() * tensor.element_size()  # meta//fake tensors have no storage
            return

        key = (storage.data_ptr(), storage.nbytes())
        if key in seen:
            return
        seen.add(key)
        total += storage.nbytes()

    for submodule in module.modules():
        for tensor in list(submodule.parameters(recurse=False)) + list(submodule.buffers(recurse=False)):
            add(tensor)
        for value in vars(submodule).values():
            if torch.is_tensor(value):
                add(value)
            elif isinstance(value, dict):
                for nested in value.values():
                    if torch.is_tensor(nested):
                        add(nested)

    return total


class SwapRegistry:
    """Owns which modules are currently promoted, and how to put them back.

    Stage 4's greedy search is promote -> measure -> put it back, 128 times per round. Getting
    "put it back" wrong is the failure mode that would be hardest to notice: the model keeps
    answering either way, so a broken revert doesn't crash, it just means round 2 measures a
    different model than round 1 thought it was scoring, and the attribution result is built on
    drift rather than on the modules it names.

    So revert restores the displaced module object itself rather than requantizing from the
    original weights. Requantizing would be slower and would assume HQQ's optimizer is
    bit-deterministic; even a tiny non-determinism compounds over hundreds of trials. Putting
    back the exact object is O(1), allocation-free, and exact by construction.

    Holding one displaced 3-bit module per touched path costs ~7-22 MB each -- negligible
    against the 7.25 GB that holding fp16 snapshots of every candidate would have cost, which is
    what `CheckpointWeightSource` exists to avoid.
    """

    def __init__(
        self,
        model: nn.Module,
        weight_source: "CheckpointWeightSource",
        device: str = "cuda",
        group_size: int = DEFAULT_GROUP_SIZE,
    ) -> None:
        self.model = model
        self.weight_source = weight_source
        self.device = device
        self.group_size = group_size
        self.compute_dtype = infer_compute_dtype(model)

        # The original quantized module for every path ever touched, so revert is exact.
        # Keyed by path and never overwritten: a path promoted, reverted and promoted again must
        # still fall back to the model's real starting state, not to an intermediate promotion.
        self._baseline: dict[str, nn.Module] = {}
        # Baseline sizes are kept as plain ints alongside the modules, so the reported overhead
        # survives `release_baselines()` dropping the modules themselves.
        self._baseline_nbytes: dict[str, int] = {}
        self._committed: dict[str, int] = {}

    @property
    def active(self) -> dict[str, int]:
        """Currently committed promotions, as path -> nbits."""
        return dict(self._committed)

    def _install(self, path: str, nbits: int) -> None:
        current = get_module_by_path(self.model, path)
        if path not in self._baseline_nbytes:
            self._baseline[path] = current
            self._baseline_nbytes[path] = module_nbytes(current)

        linear = self.weight_source.get_linear(path, dtype=self.compute_dtype)
        quant_config = build_quant_config(nbits=nbits, group_size=self.group_size)
        set_module_by_path(
            self.model,
            path,
            HQQLinear(linear, quant_config=quant_config, compute_dtype=self.compute_dtype, device=self.device),
        )

    def _restore(self, path: str, previous: nn.Module) -> None:
        set_module_by_path(self.model, path, previous)

    @contextmanager
    def trial(self, path: str, nbits: int) -> Iterator[nn.Module]:
        """Promotes `path` for the duration of the block, then puts back exactly what was there.

        The revert runs in a `finally`, so a failure inside the block -- an OOM partway through
        scoring the flipped set is the realistic one -- still leaves the model in the state the
        next candidate expects. Without that, one bad trial silently contaminates every
        measurement after it.
        """
        displaced = get_module_by_path(self.model, path)
        self._install(path, nbits)
        try:
            yield self.model
        finally:
            self._restore(path, displaced)

    def commit(self, path: str, nbits: int) -> None:
        """Promotes `path` and keeps it. This is the search deciding a module is part of the repair."""
        self._install(path, nbits)
        self._committed[path] = nbits

    def revert(self, path: str) -> None:
        """Drops a committed promotion, restoring the module the model started with."""
        if path not in self._committed:
            raise KeyError(f"{path!r} is not committed; nothing to revert")
        if path not in self._baseline:
            raise RuntimeError(
                f"Cannot revert {path!r}: baselines were released. release_baselines() is a "
                "one-way finalization, called once the search is over."
            )
        self._restore(path, self._baseline[path])
        del self._committed[path]

    def revert_all(self) -> None:
        for path in list(self._committed):
            self.revert(path)

    def deployed_overhead_bytes(self) -> int:
        """Extra bytes the repaired model costs over plain uniform quantization.

        This is the project's reported "memory overhead vs plain 3-bit": for each committed
        path, the promoted module's size minus the size of the quantized module it replaced.
        Promoting one `gate_up_proj` to 8-bit costs ~30 MB, about +1.8% of the 3-bit model.

        Deliberately excludes the baseline modules this registry is holding onto -- those exist
        only so the search can undo itself, and a deployed model would not carry them. See
        `search_overhead_bytes` for that cost, and `release_baselines` to drop it.
        """
        return sum(
            module_nbytes(get_module_by_path(self.model, path)) - self._baseline_nbytes[path]
            for path in self._committed
        )

    def search_overhead_bytes(self) -> int:
        """Transient bytes held only to make revert possible -- the displaced modules.

        Reported separately from `deployed_overhead_bytes` because it is a cost of running the
        search, not of shipping its result, and conflating the two would overstate what the
        repair actually costs to deploy.
        """
        return sum(module_nbytes(module) for module in self._baseline.values())

    def release_baselines(self) -> None:
        """Frees the retained originals once the search is done and nothing will be reverted.

        After this, `revert`/`revert_all` raise rather than silently doing nothing, so it is a
        one-way finalization -- call it when handing the repaired model to the verification run,
        not partway through a search. The recorded baseline *sizes* are kept, so
        `deployed_overhead_bytes` still reports correctly afterward.
        """
        self._baseline.clear()
