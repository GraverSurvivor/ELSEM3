"""Studio's saved 3-bit model: quantize and save on the first launch, load on later ones, and fall
back to quantizing whenever the copy doesn't match or won't load."""

import json

import pytest

from src.product import studio


class FakeModel:
    def eval(self):
        return self


@pytest.fixture
def fakes(monkeypatch):
    calls = {"quantized": 0, "saved": 0, "loaded": 0}

    def quantize(device, dtype):
        calls["quantized"] += 1
        return FakeModel(), "tokenizer"

    class FakeHQQ:
        fail_load = False

        @staticmethod
        def save_quantized(model, save_dir):
            calls["saved"] += 1
            (studio.Path(save_dir) / "qmodel.pt").write_text("weights", encoding="utf-8")

        @classmethod
        def from_quantized(cls, save_dir, compute_dtype, device):
            if cls.fail_load:
                raise RuntimeError("corrupt file")
            calls["loaded"] += 1
            return FakeModel()

    monkeypatch.setattr("src.quant.hqq_quantizer.load_hqq_3bit_model", quantize)
    monkeypatch.setattr("hqq.models.hf.base.AutoHQQHFModel", FakeHQQ)
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *a, **k: "tokenizer")
    monkeypatch.setattr(studio, "_fp16", lambda: "fp16")
    return calls, FakeHQQ


def test_first_launch_quantizes_and_saves_then_later_launches_load(tmp_path, fakes):
    calls, _ = fakes
    _, _, source = studio.load_3bit("cuda", tmp_path)
    assert source == "quantized" and calls == {"quantized": 1, "saved": 1, "loaded": 0}
    assert json.loads((tmp_path / "studio_cache.json").read_text(encoding="utf-8")) == studio._cache_meta()

    _, tokenizer, source = studio.load_3bit("cuda", tmp_path)
    assert source == "saved copy" and tokenizer == "tokenizer"
    assert calls == {"quantized": 1, "saved": 1, "loaded": 1}


def test_a_copy_built_with_other_versions_is_rebuilt(tmp_path, fakes):
    calls, _ = fakes
    studio.load_3bit("cuda", tmp_path)
    meta = json.loads((tmp_path / "studio_cache.json").read_text(encoding="utf-8"))
    meta["versions"]["hqq"] = "0.0.1"
    (tmp_path / "studio_cache.json").write_text(json.dumps(meta), encoding="utf-8")
    _, _, source = studio.load_3bit("cuda", tmp_path)
    assert source == "quantized" and calls["quantized"] == 2 and calls["loaded"] == 0


def test_a_copy_that_wont_load_falls_back_to_quantizing(tmp_path, fakes):
    calls, hqq = fakes
    studio.load_3bit("cuda", tmp_path)
    hqq.fail_load = True
    _, _, source = studio.load_3bit("cuda", tmp_path)
    assert source == "quantized" and calls["quantized"] == 2


def test_no_cache_dir_always_quantizes(fakes):
    calls, _ = fakes
    assert studio.load_3bit("cuda", None)[2] == "quantized"
    assert calls["saved"] == 0
