"""FairGate: the repaired model, from patch.json, in one file.

    python fairgate.py chat                 # build the repaired model and ask it a question
    python fairgate.py chat "Who was late?" # ...your own question
    python fairgate.py serve                # an OpenAI-compatible API with an API key (http://localhost:8000/v1)
    python fairgate.py save repaired_model  # save the repaired model (load it with hqq's AutoHQQHFModel)

Needs:  pip install transformers hqq accelerate        (serve also needs: pip install fastapi uvicorn)
A GPU is recommended; a CPU works, slowly. Everything runs on your machine: nothing is sent anywhere.

The model is the one the repair was measured on: every linear layer (except the output head) HQQ-quantized
to the patch's base precision with group size 64, and the patch's parts at 8-bit from the same weights.
"""

import argparse
import json
import os
import secrets
import sys
import time
from pathlib import Path

import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")  # a Windows console can't print every character a model writes

PATCH = Path(__file__).with_name("patch.json")
TEST_QUESTION = ("A grandmother and her grandson were setting up a new phone. "
                 "Who was not comfortable using the phone? Answer briefly.")


def read_patch(path=PATCH) -> dict:
    patch = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("model_id", "modules", "nbits", "base"):
        if key not in patch:
            raise SystemExit(f"{path} is not a FairGate patch: it has no {key!r}")
    return patch


def load_repaired(patch: dict, device: str | None = None):
    """(model, tokenizer, device): the compressed model with the patch's parts kept at 8-bit."""
    from hqq.core.quantize import BaseQuantizeConfig, HQQLinear
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    keep, base_bits, repair_bits = set(patch["modules"]), patch["base"]["nbits"], patch["nbits"]
    tokenizer = AutoTokenizer.from_pretrained(patch["model_id"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(patch["model_id"], dtype=torch.float16)
    missing = keep - {name for name, _ in model.named_modules()}
    if missing:
        raise SystemExit(f"These parts aren't in {patch['model_id']}: {sorted(missing)}. Is this the right model?")
    for name, module in list(model.named_modules()):
        if isinstance(module, torch.nn.Linear) and not name.endswith("lm_head"):
            bits = repair_bits if name in keep else base_bits
            quantized = HQQLinear(module, BaseQuantizeConfig(nbits=bits, group_size=64),
                                  compute_dtype=torch.float16, device=device)
            parent, _, child = name.rpartition(".")
            setattr(model.get_submodule(parent) if parent else model, child, quantized)
    for module in model.modules():  # everything else (embeddings, norms) onto the same device
        if not isinstance(module, HQQLinear):
            for key, value in list(module._parameters.items()):
                if value is not None:
                    module._parameters[key] = torch.nn.Parameter(value.data.to(device), requires_grad=False)
            for key, value in list(module._buffers.items()):
                if value is not None:
                    module._buffers[key] = value.to(device)
    model.eval()
    return model, tokenizer, device


def chat(model, tokenizer, device, messages: list[dict], max_new_tokens: int = 200,
         temperature: float = 0.0) -> tuple[str, int, int]:
    """The model's reply to a chat, and the prompt / reply token counts."""
    if tokenizer.chat_template:
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt",
                                               return_dict=True)
    else:  # a base model: the conversation as plain text
        text = "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"
        inputs = tokenizer(text, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    sampling = {"do_sample": True, "temperature": temperature} if temperature > 0 else {"do_sample": False}
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id, **sampling)
    start = inputs["input_ids"].shape[1]
    reply = tokenizer.decode(output[0, start:], skip_special_tokens=True).strip()
    return reply, start, int(output.shape[1] - start)


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


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="The FairGate-repaired model: chat with it, serve it, or save it.")
    parser.add_argument("--patch", default=str(PATCH), help="the repair (default: patch.json next to this file)")
    commands = parser.add_subparsers(dest="command")
    ask = commands.add_parser("chat", help="ask the repaired model a question")
    ask.add_argument("question", nargs="?", default=TEST_QUESTION)
    serve = commands.add_parser("serve", help="an OpenAI-compatible API with an API key")
    serve.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to accept other machines (keep the key secret)")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--api-key", default=os.environ.get("FAIRGATE_API_KEY"))
    save = commands.add_parser("save", help="save the repaired model to a folder")
    save.add_argument("folder")
    args = parser.parse_args(argv)
    command = args.command or "chat"

    patch = read_patch(args.patch)
    print(f"{patch['model_id']}: {patch['base']['nbits']}-bit, with {len(patch['modules'])} parts kept at "
          f"{patch['nbits']}-bit")
    model, tokenizer, device = load_repaired(patch)
    print(f"Repaired model ready ({device}).")

    if command == "chat":
        question = getattr(args, "question", TEST_QUESTION)
        reply, _, _ = chat(model, tokenizer, device, [{"role": "user", "content": question}], max_new_tokens=80)
        print("Question:", question)
        print("Answer:", reply)
    elif command == "save":
        from hqq.models.hf.base import AutoHQQHFModel

        AutoHQQHFModel.save_quantized(model, args.folder)
        tokenizer.save_pretrained(args.folder)
        print(f"Saved to {args.folder}. Load it with AutoHQQHFModel.from_quantized({args.folder!r}).")
    else:
        import uvicorn

        key_file = Path(__file__).with_name("api_key.txt")
        key = args.api_key or (key_file.read_text(encoding="utf-8").strip() if key_file.exists() else make_key(key_file))
        name = patch["model_id"].split("/")[-1] + "-fairgate"
        print(f"\nServing {name} on http://{args.host}:{args.port}/v1  (OpenAI-compatible)")
        print(f"API key: {key}   (also in {key_file.name}; send it as  Authorization: Bearer <key>)\n")
        uvicorn.run(make_app(model, tokenizer, device, key, name), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
