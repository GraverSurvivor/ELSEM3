"""The Ship screen's own handlers, run on a real tiny model: a step-5 repair is served behind a key and
published, one model held at a time; preview mode explains instead of failing."""

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("hqq")
pytest.importorskip("gradio")


@pytest.fixture(scope="module")
def data():
    from src.product.studio import StudioData

    try:
        return StudioData()
    except Exception as error:  # noqa: BLE001 -- needs the BBQ questions cached
        pytest.skip(f"Studio data not available: {error}")


@pytest.fixture
def screen(tmp_path, data, monkeypatch):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    from src.product import studio
    from src.product.api_server import ApiService
    from src.product.byom import JobRunner
    from src.product.studio_core import SavedBackend

    try:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct", local_files_only=True)
    except OSError:
        pytest.skip("Qwen tokenizer not cached")
    folder = tmp_path / "tiny"
    torch.manual_seed(0)
    AutoModelForCausalLM.from_config(AutoConfig.for_model(
        "qwen2", vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2)).save_pretrained(folder)
    tok.save_pretrained(folder)

    job = tmp_path / "jobs" / "20261020T000000Z_tiny"
    job.mkdir(parents=True)
    (job / "job.json").write_text(json.dumps({"model_id": str(folder), "mode": "fast", "ended": True}))
    (job / "patch.json").write_text(json.dumps({
        "name": "t", "model_id": str(folder), "modules": ["model.layers.0.self_attn.q_proj"], "nbits": 8,
        "base": {"method": "hqq", "nbits": 3, "group_size": 64, "compute_dtype": "fp16"}}))
    (job / "report.json").write_text(json.dumps({
        "model_id": str(folder), "arch": "qwen2", "mode": "fast", "original": {}, "compressed": {}, "repaired": {},
        "recovery": {"point": 0.8, "lower": 0.7, "upper": 0.9}, "kept": ["model.layers.0.self_attn.q_proj"],
        "gate": "PASS", "gate_failures": [], "outcome": "repaired", "memory_mb": 0.1, "restored_share": 0.02}))

    uploads = []

    def fake_publish(model, tokenizer, patch, summary, token, repo_name="", private=True):
        uploads.append({"model": model, "patch": patch, "summary": summary, "private": private})
        return {"repo_id": "student/tiny-fairgate", "url": "https://huggingface.co/student/tiny-fairgate",
                "private": private}

    monkeypatch.setattr(studio, "publish", fake_publish)
    backend = SavedBackend(data.runs, list(data.patch.modules), repairs=data.saved_repairs)
    api = ApiService()
    app, _ = studio.build_app(backend, data, studio.ByomSetup(JobRunner(tmp_path / "jobs", pipeline="real")), api)
    handlers = {f.fn.__name__: f.fn for f in app.fns.values() if getattr(f, "fn", None) is not None}
    return handlers, api, uploads


def test_a_step5_repair_is_served_and_published_from_one_held_model(screen):
    from src.product.studio import SHIP_YOURS

    handlers, api, uploads = screen
    verdict = handlers["ship_switch"](SHIP_YOURS)[0]
    assert "PASS" in verdict and "80%" in verdict and "2.0% of the weights" in verdict

    message, code, token_box = handlers["ship_publish"](SHIP_YOURS, "hf_x", "", True)
    assert "Published" in message and "student/tiny-fairgate" in message and token_box == ""  # token cleared
    assert code["visible"] is True and "from_quantized" in code["value"]
    assert uploads[0]["private"] is True and uploads[0]["summary"]["gate"] == "PASS"
    assert uploads[0]["model"].model.layers[0].self_attn.q_proj.meta["nbits"] == 8  # the repaired model itself
    assert not api.running  # built only for the upload, then freed

    status = handlers["api_start"](SHIP_YOURS)
    assert api.running and api.name == "tiny-fairgate" and "Serving" in status[0] and status[2] == api.key
    reply = api.reply([{"role": "user", "content": "Who was late?"}], 4, 0.0)
    assert reply["object"] == "chat.completion" and reply["usage"]["completion_tokens"] <= 4

    handlers["ship_publish"](SHIP_YOURS, "hf_x", "my-model", False)
    assert uploads[1]["model"] is api.model and uploads[1]["private"] is False  # the served model, not a 2nd copy
    assert api.running  # publishing doesn't stop the API

    handlers["api_stop"]()
    assert not api.running


def test_end_to_end_from_diagnosis_to_a_published_model(tmp_path, data, monkeypatch):
    """The whole chain on a real tiny model: the fast-repair pipeline (answers from the rule-based stand-in,
    everything else real) writes its report, patch and kit; the kit's script builds and answers; the Ship
    screen shows the verdict, serves the repair behind a key and publishes it (the real publish code, a fake
    Hub); and the model card's code loads the published model and answers."""
    import functools
    import subprocess
    import sys
    import zipfile

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    from src.product import studio
    from src.product.api_server import ApiService
    from src.product.byom import JobRunner
    from src.product.fast_repair import FastRepairConfig, run
    from src.product.hub_publish import load_code, publish
    from src.product.studio_core import SavedBackend
    from tests.product.test_fast_repair import AMBIG, DISAMBIG, TINY, FakeBackend
    from tests.product.test_hub_publish import FakeHub

    try:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct", local_files_only=True)
    except OSError:
        pytest.skip("Qwen tokenizer not cached")
    folder = tmp_path / "tiny-chat"
    torch.manual_seed(0)
    AutoModelForCausalLM.from_config(AutoConfig.for_model(vocab_size=len(tok), **TINY)).save_pretrained(folder)
    tok.save_pretrained(folder)

    # 1. Diagnose: the pipeline's own files, as tab 5 reads them.
    job = tmp_path / "jobs" / "20261020T000000Z_tiny-chat"
    report = run(FastRepairConfig(model_id=str(folder), questions=200, out=job), FakeBackend(),
                 load_questions=lambda: (AMBIG, DISAMBIG))
    (job / "job.json").write_text(json.dumps({"model_id": str(folder), "mode": "fast", "ended": True}))
    assert report["outcome"] == "repaired" and report["gate"] == "PASS" and report["kept"]

    # 2. The kit, run as a customer would.
    with zipfile.ZipFile(job / "repair_kit.zip") as kit:
        assert sorted(kit.namelist()) == ["README.md", "fairgate.py", "patch.json"]
        kit.extractall(tmp_path / "kit")
    ran = subprocess.run([sys.executable, "fairgate.py", "chat", "Who was late?"], cwd=tmp_path / "kit",
                         capture_output=True, text=True, timeout=600)
    assert ran.returncode == 0 and "Answer:" in ran.stdout, ran.stdout[-1000:] + ran.stderr[-1000:]

    # 3. The Ship screen.
    hub = FakeHub(tmp_path)
    monkeypatch.setattr(studio, "publish", functools.partial(publish, api=hub))
    api = ApiService()
    backend = SavedBackend(data.runs, list(data.patch.modules), repairs=data.saved_repairs)
    app, _ = studio.build_app(backend, data, studio.ByomSetup(JobRunner(tmp_path / "jobs", pipeline="real")), api)
    handlers = {f.fn.__name__: f.fn for f in app.fns.values() if getattr(f, "fn", None) is not None}
    verdict = handlers["ship_switch"](studio.SHIP_YOURS)[0]
    assert "✅ PASS" in verdict and f"{len(report['kept'])} parts kept at 8-bit" in verdict
    handlers["api_start"](studio.SHIP_YOURS)
    assert api.running and api.reply([{"role": "user", "content": "Hi"}], 4, 0.0)["choices"]
    message, code, token_box = handlers["ship_publish"](studio.SHIP_YOURS, "hf_x", "", True)
    assert message.startswith("✅ **Published:** [student/tiny-chat-fairgate]") and token_box == ""
    card = (tmp_path / "uploaded" / "README.md").read_text(encoding="utf-8")
    assert "Fairness recovered" in card and f"base_model: {folder}" in card
    note, kit_path = handlers["make_kit"](studio.SHIP_YOURS)
    assert kit_path == str(job / "repair_kit.zip")
    handlers["api_stop"]()
    assert not api.running

    # 4. The published model, loaded with the model card's own code.
    exec(compile(load_code(str(tmp_path / "uploaded").replace("\\", "/"), device="cpu"), "card", "exec"), {})


def test_preview_mode_explains_what_needs_the_gpu(screen):
    from src.product.studio import SHIP_PHI

    handlers, api, uploads = screen
    message, code, token_box = handlers["ship_publish"](SHIP_PHI, "hf_x", "", True)
    assert "Not published" in message and "live app" in message and token_box == "hf_x" and not uploads
    note, kit = handlers["make_kit"](SHIP_PHI)
    assert "5 parts" in note and kit.endswith(".zip")
    verdict = handlers["ship_switch"](SHIP_PHI)[0]
    assert "is the repair above" in verdict  # the opening repair is the main one: its numbers are measured


def test_a_cached_model_is_rebuilt_from_its_local_copy(tmp_path):
    """A gated model's token went to the diagnosis only: shipping it reads the download already cached."""
    from src.product.studio import cached_copy

    assert cached_copy(str(tmp_path)) == str(tmp_path)  # a local folder is used as it is
    assert cached_copy("nobody/not-a-cached-model-xyz") == "nobody/not-a-cached-model-xyz"  # not cached: the Hub
