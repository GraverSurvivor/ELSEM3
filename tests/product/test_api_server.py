"""Studio tab 6: a repaired model served as an OpenAI-compatible API behind a key, on a tiny real model."""

import json
import socket
import threading
import time

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("fastapi")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from src.product.api_server import ApiService, latest_repaired_job, register_routes, snippets, try_it  # noqa: E402

BODY = {"messages": [{"role": "user", "content": "Who was late, the nurse or the mechanic?"}], "max_tokens": 6}


@pytest.fixture(scope="module")
def tiny():
    from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2Config

    try:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct", local_files_only=True)
    except OSError:
        pytest.skip("Qwen tokenizer not cached")
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(Qwen2Config(hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                                                         num_attention_heads=4, num_key_value_heads=2,
                                                         vocab_size=len(tok))).eval()
    return model, tok


def _client(service):
    app = FastAPI()
    register_routes(app, service)
    return TestClient(app)


def auth(key):
    return {"Authorization": f"Bearer {key}"}


def test_the_key_lifecycle(tiny):
    service = ApiService()
    client = _client(service)
    assert client.get("/fairgate/health").json() == {"status": "stopped", "model": None}
    assert client.post("/fairgate/v1/chat/completions", json=BODY).status_code == 503  # nothing served

    key = service.start(*tiny, "cpu", "tiny-fairgate", "a test")
    assert key.startswith("fg-") and len(key) > 40
    assert client.post("/fairgate/v1/chat/completions", json=BODY).status_code == 401
    assert client.post("/fairgate/v1/chat/completions", json=BODY, headers=auth("fg-wrong")).status_code == 401
    ok = client.post("/fairgate/v1/chat/completions", json=BODY, headers=auth(key))
    assert ok.status_code == 200 and ok.json()["model"] == "tiny-fairgate"
    assert ok.json()["choices"][0]["message"]["role"] == "assistant" and 0 < ok.json()["usage"]["completion_tokens"] <= 6
    assert client.get("/fairgate/v1/models", headers=auth(key)).json()["data"][0]["id"] == "tiny-fairgate"
    assert service.requests == 1

    new = service.rotate()
    assert new != key
    assert client.post("/fairgate/v1/chat/completions", json=BODY, headers=auth(key)).status_code == 401  # old key dead
    assert client.post("/fairgate/v1/chat/completions", json=BODY, headers=auth(new)).status_code == 200
    assert client.post("/fairgate/v1/chat/completions", json={**BODY, "max_tokens": 5000},
                       headers=auth(new)).status_code == 400

    service.stop()
    assert client.post("/fairgate/v1/chat/completions", json=BODY, headers=auth(new)).status_code == 503
    with pytest.raises(RuntimeError):
        service.rotate()


def test_an_answer_waits_while_the_studio_has_parts_swapped(tiny):
    """The lock is the live backend's: while the Studio computes a "compressed" answer (parts swapped
    out), an API request must wait, or it would answer with the wrong model."""
    lock = threading.RLock()
    service = ApiService(lock=lock)
    key = service.start(*tiny, "cpu", "tiny-fairgate", "a test")
    client = _client(service)
    done = {}

    def call():
        done["status"] = client.post("/fairgate/v1/chat/completions", json=BODY, headers=auth(key)).status_code
        done["at"] = time.monotonic()

    with lock:  # the Studio is mid-swap
        worker = threading.Thread(target=call)
        worker.start()
        time.sleep(0.5)
        released = time.monotonic()
        assert "status" not in done  # still waiting
    worker.join(30)
    assert done["status"] == 200 and done["at"] >= released


def test_tab_six_serves_the_latest_real_repair_only(tmp_path):
    def job(name, report, patch=True, age=0):
        folder = tmp_path / name
        folder.mkdir()
        (folder / "report.json").write_text(json.dumps(report), encoding="utf-8")
        if patch:
            (folder / "patch.json").write_text("{}", encoding="utf-8")
        stamp = time.time() - 100 + age
        import os

        os.utime(folder, (stamp, stamp))
        return folder

    job("a_old", {"outcome": "repaired"}, age=1)
    newest = job("b_new", {"outcome": "repaired"}, age=5)
    job("c_rejected", {"outcome": "rejected"}, age=9)
    job("d_nopatch", {"outcome": "repaired"}, patch=False, age=9)
    job("e_demo", {"outcome": "repaired", "synthetic": True}, age=9)
    job("f_replay", {"outcome": "repaired", "replayed_from": "x"}, age=9)
    assert latest_repaired_job(tmp_path) == newest
    assert latest_repaired_job(tmp_path / "nothing") is None


def test_the_send_button_goes_through_real_http(tiny):
    """try_it, as tab 6's Send button uses it, against a real server on a free port."""
    import uvicorn

    service = ApiService()
    key = service.start(*tiny, "cpu", "tiny-fairgate", "a test")
    app = FastAPI()
    register_routes(app, service)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        for _ in range(100):
            if server.started:
                break
            time.sleep(0.05)
        base = f"http://127.0.0.1:{port}/fairgate/v1"
        result = try_it(base, key, "Hello")
        assert result["status"] == 200 and result["body"]["object"] == "chat.completion"
        assert try_it(base, "fg-wrong", "Hello")["status"] == 401
        code = snippets(base, key, "tiny-fairgate")
        assert f'base_url="{base}"' in code and key in code and "Authorization: Bearer" in code
        compile(code.split("# curl")[0], "snippet", "exec")  # the Python part is valid Python
    finally:
        server.should_exit = True
        thread.join(10)
