"""Ship screen, "Get the fixed model": the repaired model published to Hugging Face, loaded back exactly."""

import shutil

import pytest

from src.product.hub_publish import PublishError, default_repo_name, full_repo_id, load_code, model_card

TOKEN = "hf_test_not_a_real_token_123"
PATCH = {"name": "t", "modules": ["model.layers.0.self_attn.q_proj", "model.layers.1.mlp.down_proj"], "nbits": 8,
         "base": {"method": "hqq", "nbits": 3, "group_size": 64, "compute_dtype": "fp16"}}


class FakeHub:
    """Stands in for huggingface_hub.HfApi: records the calls and keeps a copy of the uploaded folder."""

    def __init__(self, tmp_path, fail=None):
        self.tmp_path, self.fail, self.calls = tmp_path, fail, []

    def whoami(self):
        if self.fail == "whoami":
            raise _http_error(401)
        return {"name": "student"}

    def create_repo(self, repo_id, private, exist_ok, repo_type):
        if self.fail == "create":
            raise _http_error(403)
        self.calls.append(("create", repo_id, private))

    def upload_folder(self, folder_path, repo_id, repo_type, commit_message):
        self.calls.append(("upload", repo_id, commit_message))
        shutil.copytree(folder_path, self.tmp_path / "uploaded")


def _http_error(status):
    error = RuntimeError(f"HTTP {status}")
    error.response = type("R", (), {"status_code": status})()
    return error


def test_repo_names():
    assert default_repo_name("microsoft/Phi-3.5-mini-instruct") == "Phi-3.5-mini-instruct-fairgate"
    assert full_repo_id("student", "", "Qwen/Qwen2.5-3B-Instruct") == "student/Qwen2.5-3B-Instruct-fairgate"
    assert full_repo_id("student", "my-model", "x/y") == "student/my-model"
    assert full_repo_id("student", "team-org/my-model", "x/y") == "team-org/my-model"
    for bad in ("bad name", "a/b/c", "-x", "x..y", "x."):
        with pytest.raises(PublishError, match="valid repo name"):
            full_repo_id("student", bad, "x/y")


def test_the_model_card_says_what_was_measured_and_how_to_load_it():
    measured = model_card({**PATCH, "model_id": "Qwen/Qwen2.5-3B-Instruct"},
                          {"memory_mb": 132, "compressed": {"usr": 0.11}, "repaired": {"usr": 0.27},
                           "recovery": {"point": 0.858, "lower": 0.752, "upper": 0.971}}, "student/q-fairgate")
    assert measured.startswith("---\nbase_model: Qwen/Qwen2.5-3B-Instruct\nlibrary_name: hqq")
    assert "85.8%" in measured and "AutoHQQHFModel.from_quantized(\"student/q-fairgate\"" in measured
    assert "Picks the stereotyped person" not in measured  # only rows the run measured
    by_hand = model_card({**PATCH, "model_id": "m/x"}, {"memory_mb": 30}, "student/x-fairgate")
    assert "weren't measured" in by_hand and "Fairness recovered" not in by_hand
    assert "AutoHQQHFModel.from_quantized" in load_code("a/b")


@pytest.fixture
def tiny(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("hqq")
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    try:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct", local_files_only=True)
    except OSError:
        pytest.skip("Qwen tokenizer not cached")
    folder = tmp_path / "base"
    torch.manual_seed(0)
    config = AutoConfig.for_model("qwen2", vocab_size=len(tok), hidden_size=64, intermediate_size=128,
                                  num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                                  tie_word_embeddings=True)
    AutoModelForCausalLM.from_config(config).save_pretrained(folder)
    tok.save_pretrained(folder)
    return folder


def test_publish_uploads_a_model_that_loads_back_bit_identical(tmp_path, tiny):
    import torch
    from hqq.models.hf.base import AutoHQQHFModel

    from src.product.hub_publish import publish
    from src.quant.streaming import load_streaming

    patch = {**PATCH, "model_id": str(tiny)}
    model, tok = load_streaming(str(tiny), 3, device="cpu", keep={path: 8 for path in patch["modules"]})
    hub = FakeHub(tmp_path)
    result = publish(model, tok, patch, {"memory_mb": 1.0}, TOKEN, api=hub)

    assert result == {"repo_id": "student/base-fairgate", "url": "https://huggingface.co/student/base-fairgate",
                      "private": True}  # private unless unticked
    assert hub.calls[0] == ("create", "student/base-fairgate", True)
    uploaded = tmp_path / "uploaded"
    names = {p.name for p in uploaded.iterdir()}
    assert {"config.json", "qmodel.pt", "README.md", "patch.json", "tokenizer_config.json"} <= names
    for file in uploaded.iterdir():  # the token is never written anywhere
        assert TOKEN.encode() not in file.read_bytes()

    loaded = AutoHQQHFModel.from_quantized(str(uploaded), device="cpu", compute_dtype=torch.float16)
    assert loaded.model.layers[0].self_attn.q_proj.meta["nbits"] == 8  # the repaired parts stay 8-bit
    assert loaded.model.layers[0].self_attn.k_proj.meta["nbits"] == 3
    inputs = tok(["A nurse and a mechanic met. Who was late?"], return_tensors="pt")
    with torch.no_grad():
        assert torch.equal(model(**inputs).logits, loaded(**inputs).logits)

    # The model card's code, run as it is (on the local copy instead of the Hub, and on CPU).
    code = load_code(str(uploaded).replace("\\", "/"), device="cpu")
    exec(compile(code, "model_card_snippet", "exec"), {})


def test_publish_explains_a_bad_token_without_showing_it(tmp_path, tiny):
    from src.product.hub_publish import publish
    from src.quant.streaming import load_streaming

    model, tok = load_streaming(str(tiny), 3, device="cpu")
    patch = {**PATCH, "model_id": str(tiny)}
    with pytest.raises(PublishError, match="Write") as error:
        publish(model, tok, patch, {}, TOKEN, api=FakeHub(tmp_path, fail="whoami"))
    assert TOKEN not in str(error.value)
    with pytest.raises(PublishError, match="can.t create or write that repo"):
        publish(model, tok, patch, {}, TOKEN, api=FakeHub(tmp_path, fail="create"))
    with pytest.raises(PublishError, match="Paste a Hugging Face token"):
        publish(model, tok, patch, {}, "  ", api=FakeHub(tmp_path))


def test_gpt2_style_models_are_refused_before_anything_is_uploaded(tmp_path):
    torch = pytest.importorskip("torch")
    from transformers import GPT2Config, GPT2LMHeadModel

    from src.product.hub_publish import publish

    model = GPT2LMHeadModel(GPT2Config(n_embd=32, n_layer=1, n_head=2, vocab_size=50))
    hub = FakeHub(tmp_path)
    with pytest.raises(PublishError, match="GPT-2 style"):
        publish(model, None, {**PATCH, "model_id": "gpt2"}, {}, TOKEN, api=hub)
    assert hub.calls == [] and torch is not None
