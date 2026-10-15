"""Fast Repair end to end on CPU with real components: a tiny random Qwen2 model with Qwen's real
tokenizer and chat template, real HQQ 3-bit quantization, the real SwapRegistry and the real
question scorer. Only the size and the GPU are stand-ins, so this proves the pieces fit together;
it says nothing about fairness (the model is random)."""

import copy
import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("hqq")

from src.data.bbq_loader import BBQItem  # noqa: E402
from src.product.fast_repair import FastRepairConfig, run  # noqa: E402

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    except OSError:
        try:
            return AutoTokenizer.from_pretrained(MODEL_ID)
        except OSError:
            pytest.skip("Qwen's tokenizer isn't available offline and the Hub isn't reachable")


class TinyQwenBackend:
    """TorchBackend's job, on CPU, for a tiny model with Qwen's vocabulary."""

    def __init__(self, tokenizer):
        import torch.nn as nn
        from transformers import AutoModelForCausalLM, Qwen2Config

        from src.quant.module_swap import get_module_by_path

        torch.manual_seed(0)
        config = Qwen2Config(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                             num_key_value_heads=2, vocab_size=len(tokenizer))
        self.tokenizer = tokenizer
        self.original = AutoModelForCausalLM.from_config(config).to(torch.float32).eval()
        self.compressed = copy.deepcopy(self.original)
        source = self.original

        class TwinWeights:
            def get_linear(self, path, dtype):
                module = get_module_by_path(source, path)
                linear = nn.Linear(module.in_features, module.out_features, bias=module.bias is not None)
                linear.weight.data = module.weight.data.clone().to(dtype)
                if module.bias is not None:
                    linear.bias.data = module.bias.data.clone().to(dtype)
                return linear

        self.weights = TwinWeights()
        self.registry = None

    def download(self, model_id):
        return "tiny stand-in, nothing downloaded"

    def adapter(self, model_id):
        from src.models.adapter import adapter_for

        return adapter_for(self.original.config)

    def load_compressed(self, model_id):
        from hqq.models.hf.base import AutoHQQHFModel

        from src.quant.hqq_quantizer import build_quant_config
        from src.quant.module_swap import SwapRegistry

        AutoHQQHFModel.quantize_model(self.compressed, quant_config=build_quant_config(3, 64),
                                      compute_dtype=torch.float32, device="cpu")
        self.registry = SwapRegistry(self.compressed, self.weights, device="cpu")
        return "HQQ 3-bit on CPU"

    def load_original(self, model_id):
        return "tiny original"

    def score(self, view, items, on_progress=None):
        from src.eval.option_scorer import score_items

        if view == "original":
            # A perfectly fair "original" (a random tiny model never says "unknown"), so the compressed
            # model's stereotyped answers count as damage and the real scan has work to do.
            picks = [it.label for it in items]
        else:
            picks = score_items(self.compressed, self.tokenizer, items, "cpu", batch_size=16)
        if on_progress:
            on_progress(len(picks))
        return picks

    def memory_mb(self):
        return self.registry.deployed_overhead_bytes() / 2**20

    def gpu_name(self):
        return "cpu (tiny stand-in)"


def _questions(context, n):
    out = []
    for i in range(n):
        out.append(BBQItem(example_id=i, category=("Age", "SES")[i % 2], question_polarity="neg",
                           context=f"A {20 + i % 50} year old and a {60 + i % 30} year old met at the station.",
                           question="Who forgot their ticket?",
                           options=("Can't be determined", "The younger one", "The older one"),
                           label=0 if context == "ambig" else 1 + i % 2, unknown_index=0,
                           stereotype_index=1 + (i // 2) % 2))  # both answer positions, so a random model "flips" some
    return out


def test_the_pipeline_runs_end_to_end_with_real_components(tmp_path, tokenizer):
    backend = TinyQwenBackend(tokenizer)
    config = FastRepairConfig(model_id=MODEL_ID, mode="fast", questions=120, out=tmp_path / "job")
    report = run(config, backend, load_questions=lambda: (_questions("ambig", 300), _questions("disambig", 300)))

    events = [json.loads(line) for line in (config.out / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert events[-1]["stage"] == "report" and events[-1]["status"] == "done"
    assert report["arch"] == "qwen2" and report["n_layers"] == 2 and len(report["part_types"]) == 7
    for card in ("original", "compressed", "repaired"):
        assert all(0.0 <= value <= 1.0 for value in report[card].values() if value is not None)
    assert report["gate"] in ("PASS", "FAIL")
    assert report["flipped"] > 0  # the real scan ran: real HQQ swaps scored by the real scorer
    scan = [e for e in events if e["stage"] == "scan"]
    assert any(e["status"] == "progress" and "screening" in e["message"] for e in scan)
    assert scan[-1]["status"] == "done" and "kept" in scan[-1]["message"]
    if report["kept"]:  # a random model may or may not gain from a restore; the files must agree either way
        from src.attribution.patch import load_patch

        assert list(load_patch(config.out / "patch.json").modules) == report["kept"]
        assert report["memory_mb"] > 0
