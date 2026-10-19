"""Studio tab 6, "API": serve a repaired model as an OpenAI-compatible API, behind an API key, from
the Studio's own server (so on Colab it is reachable at the Studio's public link too).

    POST <studio>/fairgate/v1/chat/completions   OpenAI's chat format; needs Authorization: Bearer <key>
    GET  <studio>/fairgate/v1/models              the model being served (needs the key)
    GET  <studio>/fairgate/health                 is anything being served (no key)

One model at a time: the Phi-3.5 demo model with the repair built in tab 3 (the live model itself, no
extra memory), or the latest repaired model from tab 5 (rebuilt from its patch, as the repair kit
does). Each start makes a new key (fg- and 256 random bits); keys are compared in constant time. The
same code as the kit's serve_repair.py answers, so the dashboard and the download behave alike.
"""

from __future__ import annotations

import json
import secrets
import threading
import time
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel

PREFIX = "/fairgate"


def new_key() -> str:
    return "fg-" + secrets.token_urlsafe(32)


class ApiService:
    """What is being served, with which key. `lock` is shared with the Studio's live models, so an
    API answer never runs while the Studio has swapped parts out for a "compressed" comparison."""

    def __init__(self, lock: threading.RLock | None = None):
        self.lock = lock or threading.RLock()
        self.model = self.tokenizer = self.device = None
        self.key: str | None = None
        self.name = self.source = ""
        self.started: float | None = None
        self.requests = 0

    @property
    def running(self) -> bool:
        return self.model is not None and self.key is not None

    def start(self, model, tokenizer, device: str, name: str, source: str) -> str:
        with self.lock:
            self.model, self.tokenizer, self.device = model, tokenizer, device
            self.name, self.source = name, source
            self.key, self.started, self.requests = new_key(), time.time(), 0
        return self.key

    def rotate(self) -> str:
        if not self.running:
            raise RuntimeError("Nothing is being served: start the API first.")
        self.key = new_key()  # the old key stops working at once
        return self.key

    def stop(self) -> None:
        with self.lock:
            self.model = self.tokenizer = self.device = self.key = None
            self.name = self.source = ""
            self.started = None

    def reply(self, messages: list[dict], max_tokens: int, temperature: float) -> dict:
        from src.product.kit_files.fairgate_repair import chat

        with self.lock:
            if not self.running:
                raise RuntimeError("stopped")
            text, prompt_tokens, reply_tokens = chat(self.model, self.tokenizer, self.device, messages,
                                                     max_tokens, temperature)
            self.requests += 1
        return {"id": "chatcmpl-" + secrets.token_hex(8), "object": "chat.completion", "created": int(time.time()),
                "model": self.name,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": reply_tokens,
                          "total_tokens": prompt_tokens + reply_tokens}}


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    # Module level, not inside register_routes: with postponed annotations FastAPI can't resolve a
    # class defined in a function, and would reject every request body (422).
    model: Optional[str] = None
    messages: List[Message]
    max_tokens: int = 200
    temperature: float = 0.0


def register_routes(app, service: ApiService) -> None:
    """Adds the API to a FastAPI app (the Studio's own)."""
    from fastapi import Header, HTTPException

    def check(authorization: str | None) -> None:
        if not service.running:
            raise HTTPException(status_code=503, detail="No model is being served: start it in the Studio's API tab.")
        given = (authorization or "").removeprefix("Bearer ").strip()
        if not secrets.compare_digest(given.encode(), (service.key or "").encode()):
            raise HTTPException(status_code=401, detail="Missing or wrong API key (Authorization: Bearer <key>)")

    def health():
        return {"status": "serving" if service.running else "stopped", "model": service.name or None}

    def models(authorization: str | None = Header(default=None)):
        check(authorization)
        return {"object": "list", "data": [{"id": service.name, "object": "model", "owned_by": "fairgate"}]}

    def completions(request: ChatRequest, authorization: str | None = Header(default=None)):
        check(authorization)
        if not request.messages:
            raise HTTPException(status_code=400, detail="messages is empty")
        if not 1 <= request.max_tokens <= 2048:
            raise HTTPException(status_code=400, detail="max_tokens must be 1-2048")
        try:
            return service.reply([m.model_dump() for m in request.messages], request.max_tokens,
                                 max(0.0, request.temperature))
        except RuntimeError as error:
            raise HTTPException(status_code=503, detail=f"The model stopped being served ({error}).") from None

    app.add_api_route(f"{PREFIX}/health", health, methods=["GET"])
    app.add_api_route(f"{PREFIX}/v1/models", models, methods=["GET"])
    app.add_api_route(f"{PREFIX}/v1/chat/completions", completions, methods=["POST"])


def latest_repaired_job(jobs_dir: Path) -> Path | None:
    """The newest tab-5 diagnosis that ended with a repair worth shipping (it has a patch)."""
    jobs = [d for d in Path(jobs_dir).glob("*") if (d / "patch.json").exists() and (d / "report.json").exists()]
    good = []
    for job in jobs:
        try:
            report = json.loads((job / "report.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if report.get("outcome", "repaired") == "repaired" and not report.get("synthetic") \
                and not report.get("replayed_from"):
            good.append(job)
    return max(good, key=lambda d: d.stat().st_mtime) if good else None


def snippets(base_url: str, key: str, name: str) -> str:
    """Ready-to-copy code for the screen."""
    return f'''# Python (pip install openai)
from openai import OpenAI
client = OpenAI(base_url="{base_url}", api_key="{key}")
reply = client.chat.completions.create(
    model="{name}",
    messages=[{{"role": "user", "content": "Who was late, the nurse or the mechanic?"}}],
)
print(reply.choices[0].message.content)

# curl
curl {base_url}/chat/completions \\
  -H "Authorization: Bearer {key}" -H "Content-Type: application/json" \\
  -d '{{"messages": [{{"role": "user", "content": "Hello"}}]}}'
'''


def try_it(base_url: str, key: str, message: str, timeout: float = 120.0) -> dict:
    """One real request through the API, as a customer's app would send it."""
    import httpx

    response = httpx.post(f"{base_url}/chat/completions", timeout=timeout,
                          headers={"Authorization": f"Bearer {key}"},
                          json={"messages": [{"role": "user", "content": message}], "max_tokens": 120})
    return {"status": response.status_code, "body": response.json()}
