"""Serve the repaired model as an OpenAI-compatible API, protected by an API key.

    python serve_repair.py                    # on http://localhost:8000, with a new API key printed once
    python serve_repair.py --port 9000 --api-key YOUR_KEY

Then call it like OpenAI's API:

    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:8000/v1", api_key="YOUR_KEY")
    client.chat.completions.create(model="repaired", messages=[{"role": "user", "content": "Hi"}])

It runs on your own machine: your model, your key, nothing sent anywhere else. The key is also read
from the FAIRGATE_API_KEY environment variable; without one, a new key is made and saved to api_key.txt.
Needs:  pip install transformers hqq accelerate fastapi uvicorn
"""

import argparse
import os
import secrets
import time
from pathlib import Path

from fairgate_repair import PATCH, chat, load_repaired, read_patch


def make_key(path: Path) -> str:
    key = "fg-" + secrets.token_urlsafe(32)
    path.write_text(key + "\n", encoding="utf-8")
    try:
        os.chmod(path, 0o600)  # readable by its owner only, where the system allows
    except OSError:
        pass
    return key


def make_app(model, tokenizer, device, api_key: str, model_name: str):
    """The API: GET /v1/models, POST /v1/chat/completions (OpenAI's format), GET /health (no key)."""
    from fastapi import FastAPI, Header, HTTPException
    from pydantic import BaseModel

    class Message(BaseModel):
        role: str
        content: str

    class ChatRequest(BaseModel):
        model: str = model_name
        messages: list[Message]
        max_tokens: int = 200
        temperature: float = 0.0

    app = FastAPI(title="FairGate repaired model", version="1.0")

    def check(authorization: str | None) -> None:
        given = (authorization or "").removeprefix("Bearer ").strip()
        if not secrets.compare_digest(given, api_key):  # constant-time, so the key can't be guessed by timing
            raise HTTPException(status_code=401, detail="Missing or wrong API key (Authorization: Bearer <key>)")

    @app.get("/health")
    def health():
        return {"status": "ok", "model": model_name}

    @app.get("/v1/models")
    def models(authorization: str | None = Header(default=None)):
        check(authorization)
        return {"object": "list", "data": [{"id": model_name, "object": "model", "owned_by": "fairgate"}]}

    @app.post("/v1/chat/completions")
    def completions(request: ChatRequest, authorization: str | None = Header(default=None)):
        check(authorization)
        if not request.messages:
            raise HTTPException(status_code=400, detail="messages is empty")
        if not 1 <= request.max_tokens <= 2048:
            raise HTTPException(status_code=400, detail="max_tokens must be 1-2048")
        reply, prompt_tokens, reply_tokens = chat(model, tokenizer, device,
                                                  [m.model_dump() for m in request.messages],
                                                  request.max_tokens, max(0.0, request.temperature))
        return {"id": "chatcmpl-" + secrets.token_hex(8), "object": "chat.completion", "created": int(time.time()),
                "model": model_name,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": reply}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": reply_tokens,
                          "total_tokens": prompt_tokens + reply_tokens}}

    return app


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="Serve the repaired model as an OpenAI-compatible API.")
    parser.add_argument("--patch", default=str(PATCH))
    parser.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to accept other machines (keep the key secret)")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--api-key", default=os.environ.get("FAIRGATE_API_KEY"))
    args = parser.parse_args()

    key_file = Path(__file__).with_name("api_key.txt")
    key = args.api_key or (key_file.read_text(encoding="utf-8").strip() if key_file.exists() else make_key(key_file))
    patch = read_patch(args.patch)
    model, tokenizer, device = load_repaired(patch)
    name = patch["model_id"].split("/")[-1] + "-fairgate"
    print(f"\nServing {name} on http://{args.host}:{args.port}/v1  (OpenAI-compatible)")
    print(f"API key: {key}   (also in {key_file.name}; send it as  Authorization: Bearer <key>)\n")
    uvicorn.run(make_app(model, tokenizer, device, key, name), host=args.host, port=args.port)
