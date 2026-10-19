"""FairGate repair: rebuild the repaired model from patch.json (shared by apply_repair.py and serve_repair.py).

Every linear layer (except the output head) is HQQ-quantized to the patch's base precision with group
size 64, and the patch's parts to 8-bit from the same original weights: exactly the model the repair
was measured on. Needs:  pip install transformers hqq accelerate
"""

import json
import sys
from pathlib import Path

import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")  # a Windows console can't print every character a model writes

PATCH = Path(__file__).with_name("patch.json")


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
