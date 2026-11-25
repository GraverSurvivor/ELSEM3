"""The real models behind Fast Repair (`src/product/fast_repair.py`): the original, the compressed copy
whose parts the scan swaps, and the scoring of both. The pipeline itself knows only the small backend
interface (`fast_repair.Backend`), so it is tested on a CPU with stand-ins.
"""

from __future__ import annotations

from pathlib import Path

CHUNK = 128               # questions per progress update while scoring


def option_probs(model, tokenizer, items: list, device: str, batch_size: int = 8) -> list[tuple[float, ...]]:
    """The model's probability for each choice of each question (over the choice letters only), in the
    items' order: the numbers `score_items_with_confidence` takes its pick from (same prompts, same
    batches, so the most likely choice is its pick)."""
    import torch

    from src.eval.option_scorer import _resolve_label_token_ids, format_prompt, last_position_logits

    if not items:
        return []
    tokenizer.padding_side = "left"
    prompts = [format_prompt(item, tokenizer) for item in items]
    label_ids = _resolve_label_token_ids(prompts, tokenizer)
    by_length = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    out: list = [None] * len(items)
    with torch.no_grad():
        for start in range(0, len(by_length), batch_size):
            positions = by_length[start:start + batch_size]
            inputs = tokenizer([prompts[i] for i in positions], return_tensors="pt", padding=True,
                               add_special_tokens=True).to(device)
            probs = torch.softmax(last_position_logits(model, inputs)[:, label_ids].float(), dim=-1)
            for position, row in zip(positions, probs.tolist()):
                out[position] = tuple(row)
    return out


class PrecisionError(RuntimeError):
    """A model whose scores overflow at fp16 (NaN): its answers would be meaningless."""


class TooBigError(RuntimeError):
    """The model can't be compressed and scanned with this machine's memory; said before downloading."""


def memory_plan(n_params: int, n_linear: int, base_nbits: int = 3) -> dict:
    """Bytes the pipeline needs for a model of `n_params` weights, `n_linear` of them in linear
    layers (the ones HQQ compresses):
    * system memory while compressing: the 16-bit model is loaded on the CPU first (2 bytes a weight);
    * GPU memory while scanning: the 3-bit model (HQQ's real cost is ~3.75 bits a weight with its
      scales, ~0.5 bytes with packing; the rest stays 16-bit) plus working room for the questions.
    The 16-bit original also goes to the GPU once, to answer the questions; if it doesn't fit it
    spills to system memory (slower, still exact) and is freed before the scan."""
    # Worst case: repair-until-fair can end with every part at 8-bit (~1.06 bytes a weight), the
    # compressed copies it displaced still held for reverting until the scan ends.
    compressed = (1.06 + (0.5 if base_nbits == 3 else 0.62)) * n_linear + 2 * (n_params - n_linear)
    return {"ram": 2.2 * n_params + 0.5e9, "gpu": 1.1 * compressed + 0.8e9, "original": 2.2 * n_params + 0.5e9}


def format_params(n: int) -> str:
    return f"{n / 1e9:.1f}B parameters" if n >= 1e9 else f"{n / 1e6:.0f}M parameters"


class TorchBackend:
    """The real models: the original at fp16 and the HQQ 3-bit copy whose parts the scan swaps."""

    def __init__(self, nbits: int = 8, batch_size: int = 16, device: str = "cuda", base_nbits: int = 3,
                 streaming: bool | None = None):
        base_nbits = base_nbits or 3  # automatic starts at 3-bit
        import torch

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("No GPU found. Fast Repair needs a CUDA GPU (e.g. a Colab T4).")
        self.device, self.nbits, self.batch_size, self.base_nbits = device, nbits, batch_size, base_nbits
        self.original = self.compressed = self.tokenizer = self.registry = self._adapter = None
        # Big models (the 16-bit copy doesn't fit the GPU or system memory): built layer by layer
        # (src/quant/streaming.py), with an 8-bit reference standing in for the 16-bit original.
        # None = decided by preflight from the free memory; True / False force it (tests).
        self.streaming = streaming
        self.reference_bits = 16
        self._model_id = None
        self.revision = None  # the model's commit on Hugging Face, once downloaded

    def preflight(self, model_id: str) -> str:
        """Before downloading anything but config.json: does the model fit this machine? Counts the
        model's weights on the meta device (no memory) and compares memory_plan with what is free
        now (so a GPU shared with the live Phi-3.5 demo is judged by what is left of it)."""
        import psutil
        import torch
        from transformers import AutoConfig

        from src.models.adapter import meta_model

        config = AutoConfig.from_pretrained(model_id)
        model = meta_model(config)
        n_params = sum(p.numel() for p in {id(p): p for p in model.parameters()}.values())
        n_linear = sum(m.weight.numel() for m in model.modules()
                       if type(m).__name__ in ("Linear", "Conv1D") and getattr(m, "weight", None) is not None)
        plan = memory_plan(n_params, n_linear, 4)  # sized for 4-bit, so an automatic switch fits too
        free_ram = psutil.virtual_memory().available
        free_gpu = torch.cuda.mem_get_info()[0] if self.device == "cuda" else free_ram
        size = format_params(n_params)
        if self.streaming is None:
            self.streaming = plan["ram"] > free_ram or plan["original"] > free_gpu
        if self.streaming:
            # Layer by layer: system memory holds one layer; the GPU holds one copy at a time, the
            # largest being the 8-bit reference (and the repair, at most every part at 8-bit).
            reference = 1.06 * n_linear + 2 * (n_params - n_linear) + 1.0e9
            if reference > free_gpu:
                largest = (free_gpu - 1.0e9) / 1.3 / 1e9
                raise TooBigError(
                    f"{model_id} has {size}. Even built layer by layer, its 8-bit reference needs about "
                    f"{reference / 1e9:.0f} GB of GPU memory and this GPU has {free_gpu / 1e9:.0f} GB free. Models up "
                    f"to about {max(largest, 0):.0f}B parameters fit here; a bigger GPU (L4, A100) runs this one.")
            self.reference_bits = 8
            return (size + " · too big for a 16-bit copy here, so it is built layer by layer and compared with "
                    "an 8-bit reference (its answers match 16-bit's almost exactly)")
        fits = plan["ram"] <= free_ram and plan["gpu"] <= free_gpu
        if not fits:
            largest = min((free_ram - 0.5e9) / 2.2, (free_gpu - 0.8e9) / 1.9) / 1e9
            raise TooBigError(
                f"{model_id} has {size}. Compressing and scanning it needs about {plan['ram'] / 1e9:.0f} GB of "
                f"system memory and {plan['gpu'] / 1e9:.0f} GB of GPU memory; this machine has "
                f"{free_ram / 1e9:.0f} GB and {free_gpu / 1e9:.0f} GB free. Models up to about "
                f"{max(largest, 0):.1f}B parameters fit here; a bigger GPU (L4, A100) runs this one.")
        spill = "" if plan["original"] <= free_gpu else " (the original will partly run from system memory: slower)"
        return size + spill

    def download(self, model_id: str) -> str:
        from huggingface_hub import snapshot_download

        if Path(model_id).is_dir():  # a local folder (tests, or a model saved on Drive)
            path = Path(model_id)
        else:
            files = ["*.json", "*.model", "*.txt", "tokenizer*", "*.tiktoken"]
            path = Path(snapshot_download(model_id, allow_patterns=files + ["*.safetensors"]))
            if not any(path.rglob("*.safetensors")):  # an older checkpoint: weights only as .bin
                path = Path(snapshot_download(model_id, allow_patterns=files + ["*.bin"]))
        if not Path(model_id).is_dir():
            self.revision = path.name  # the cache keeps each commit in its own folder
        size = sum(f.stat().st_size for pattern in ("*.safetensors", "*.bin") for f in path.rglob(pattern)) / 1e9
        if size == 0:
            raise RuntimeError("The model has no weights in safetensors or pytorch_model.bin form.")
        return f"{size:.1f} GB of weights"

    def adapter(self, model_id: str):
        from src.models.adapter import adapter_for_id

        self._adapter = adapter_for_id(model_id)
        return self._adapter

    def load_compressed(self, model_id: str) -> str:
        import gc

        import torch

        self.compressed = self.registry = None  # a second call (3-bit -> 4-bit) frees the first copy first
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._model_id = model_id
        if self.streaming and self.original is None and self.tokenizer is None:
            # Big model: the reference must answer first, alone on the GPU; this copy is built after it.
            return f"HQQ {self.base_nbits}-bit, built layer by layer once the reference has answered"
        if self.streaming:
            self._build_streamed()
            return f"HQQ {self.base_nbits}-bit, built layer by layer"

        from src.quant.hqq_quantizer import load_hqq_3bit_model
        from src.quant.module_swap import SwapRegistry
        from src.quant.weight_source import CheckpointWeightSource

        self.compressed, self.tokenizer = load_hqq_3bit_model(device=self.device, dtype="fp16", model_id=model_id,
                                                              nbits=self.base_nbits)
        adapter = self._adapter or self.adapter(model_id)
        # GPT-2-style parts are stored (in, out) in the checkpoint; the 3-bit model holds them as nn.Linear.
        transposed_paths = {p for p in adapter.candidates() if adapter.is_transposed(p)}
        weights = CheckpointWeightSource(model_id, transposed=lambda path: path in transposed_paths)
        self.registry = SwapRegistry(self.compressed, weights, device=self.device)
        gc.collect()
        torch.cuda.empty_cache()
        return f"HQQ {self.base_nbits}-bit, group 64, fp16 compute"

    def _build_streamed(self) -> None:
        from src.quant.module_swap import SwapRegistry
        from src.quant.streaming import load_streaming
        from src.quant.weight_source import CheckpointWeightSource

        self.compressed, tokenizer = load_streaming(self._model_id, self.base_nbits, device=self.device)
        self.tokenizer = self.tokenizer or tokenizer
        adapter = self._adapter or self.adapter(self._model_id)
        transposed_paths = {p for p in adapter.candidates() if adapter.is_transposed(p)}
        self.registry = SwapRegistry(self.compressed, CheckpointWeightSource(
            self._model_id, transposed=lambda path: path in transposed_paths), device=self.device)

    def load_original(self, model_id: str) -> str:
        from src.eval.model_loader import load_model

        if self.streaming:
            from src.quant.streaming import load_streaming

            self.original, self.tokenizer = load_streaming(model_id, 8, device=self.device)
            return "8-bit reference, built layer by layer (the 16-bit original doesn't fit this GPU)"

        self.original, _ = load_model("fp16", model_id=model_id,
                                      device_map="auto" if self.device == "cuda" else self.device)
        return "original model loaded at fp16"

    def release_original(self) -> None:
        """The original is only asked once, before the scan: free its GPU memory (2 bytes a weight,
        ~6 GB for a 3B model) for the scan, the check and anything else sharing the GPU."""
        import gc

        import torch

        self.original = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _model_for(self, view: str):
        if view != "original" and self.compressed is None and self.streaming and self._model_id:
            self._build_streamed()  # big model: built now that the reference has answered and been freed
        return self.original if view == "original" else self.compressed

    @staticmethod
    def _check_precision(view: str, values) -> None:
        import math

        if any(math.isnan(v) for v in values):
            # NaN scores would make every answer "A" without any error: stop and say why instead.
            raise PrecisionError(f"The {view} model's numbers overflow at 16-bit precision (fp16), so its "
                                 "answers can't be trusted. Some model families (Gemma, for example) need "
                                 "bf16, which a T4 can't run fast; try an L4 or A100, or another model.")

    def probs(self, view: str, items: list, on_progress=None) -> list[tuple[float, ...]]:
        """Each choice's probability, question by question (`score` returns only the most likely one)."""
        model, out = self._model_for(view), []
        for start in range(0, len(items), CHUNK):
            chunk = option_probs(model, self.tokenizer, items[start:start + CHUNK], self.device, self.batch_size)
            self._check_precision(view, (p for row in chunk for p in row))
            out += chunk
            if on_progress:
                on_progress(len(out))
        return out

    def score(self, view: str, items: list, on_progress=None) -> list[int]:
        from src.eval.option_scorer import score_items_with_confidence

        model = self._model_for(view)
        picks = []
        for start in range(0, len(items), CHUNK):
            chunk, confidence = score_items_with_confidence(model, self.tokenizer, items[start:start + CHUNK],
                                                            self.device, batch_size=self.batch_size)
            self._check_precision(view, confidence)
            picks += chunk
            if on_progress:
                on_progress(len(picks))
        return picks

    def memory_mb(self) -> float:
        return self.registry.deployed_overhead_bytes() / 2**20

    def gpu_name(self) -> str:
        import torch

        return torch.cuda.get_device_name(0) if self.device == "cuda" else self.device
