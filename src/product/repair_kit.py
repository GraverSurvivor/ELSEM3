"""The download a customer gets: one zip that applies the repair with a single command.

    repair_kit.zip
      README.md         what was found, what the repair does, how to use it (3 steps, plain words)
      apply_repair.py   loads the model, compresses it, keeps the repaired parts at 8-bit, saves it
      patch.json        the repair itself (which parts, at which precision)
      llama_cpp.txt     the same repair for llama.cpp users (llama-quantize overrides)

No server or API key: everything runs on the customer's machine from the public model and this
file, so it works offline, can't expire, and nothing of theirs is sent anywhere.

apply_repair.py needs only `transformers` and `hqq`. It rebuilds exactly the model the repair was
measured on: every linear layer (except the output head) HQQ-quantized to the base precision with
group size 64, and the patch's parts quantized to 8-bit from the same original weights -- the
pipeline reads those from the checkpoint, which is bit-identical to the loaded weights it uses here.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

APPLY_SCRIPT = '''"""Apply a FairGate repair. Usage:  python apply_repair.py  [--out repaired_model]

Needs:  pip install transformers hqq accelerate   (a GPU is recommended; a CPU works, slowly)
"""
import argparse
import json
from pathlib import Path

import torch
from hqq.core.quantize import BaseQuantizeConfig, HQQLinear
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys

sys.stdout.reconfigure(errors="replace")  # a Windows console can't print every character a model writes

parser = argparse.ArgumentParser(description="Compress the model and apply the FairGate repair.")
parser.add_argument("--patch", default=str(Path(__file__).with_name("patch.json")))
parser.add_argument("--out", help="folder to save the repaired model in (optional)")
args = parser.parse_args()

patch = json.loads(Path(args.patch).read_text(encoding="utf-8"))
model_id, keep = patch["model_id"], set(patch["modules"])
base_bits, repair_bits = patch["base"]["nbits"], patch["nbits"]
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Model {model_id}: {base_bits}-bit, with {len(keep)} parts kept at {repair_bits}-bit ({device})")

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16)
missing = keep - {name for name, _ in model.named_modules()}
if missing:
    raise SystemExit(f"These parts aren't in {model_id}: {sorted(missing)}. Is this the right model?")

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
print("Repaired model ready.")

question = "A grandmother and her grandson were setting up a new phone. Who was not comfortable using the phone?"
messages = [{"role": "user", "content": question + " Answer briefly."}]
if tokenizer.chat_template:
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt",
                                           return_dict=True)
else:
    inputs = tokenizer(question + "\\nAnswer:", return_tensors="pt")
inputs = {key: value.to(device) for key, value in inputs.items()}
with torch.no_grad():
    answer = model.generate(**inputs, max_new_tokens=40, do_sample=False,
                            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
print("Test question:", question)
start = inputs["input_ids"].shape[1]
print("Answer:", tokenizer.decode(answer[0, start:], skip_special_tokens=True).strip())

if args.out:
    from hqq.models.hf.base import AutoHQQHFModel

    AutoHQQHFModel.save_quantized(model, args.out)
    tokenizer.save_pretrained(args.out)
    print(f"Saved to {args.out}. Load it with AutoHQQHFModel.from_quantized({args.out!r}).")
'''


def _pct(value) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"


def readme(patch: dict, summary: dict) -> str:
    """Plain-language instructions and what the repair is, from the patch and the run's numbers."""
    model_id = patch.get("model_id", "the model")
    base, parts = patch.get("base", {}).get("nbits", 3), patch.get("modules", [])
    before, after = summary.get("compressed") or {}, summary.get("repaired") or {}
    recovery = summary.get("recovery") or {}
    lines = [
        f"# FairGate repair kit: {model_id}",
        "",
        f"Compressing {model_id} to {base}-bit made it answer more questions with a stereotype. "
        f"This kit keeps **{len(parts)} of its parts** at 8-bit (about +{summary.get('memory_mb', 0):.0f} MB), "
        "which brings fair answers back.",
        "",
        *([
            "| On questions the repair never saw | Compressed | Repaired |",
            "|---|---|---|",
            f"| Says \"can't tell\" when the story doesn't say (fair) | {_pct(before.get('usr'))} | {_pct(after.get('usr'))} |",
            f"| Picks the stereotyped person | {_pct(before.get('srs'))} | {_pct(after.get('srs'))} |",
            f"| Right answer when the story does say | {_pct(before.get('accuracy'))} | {_pct(after.get('accuracy'))} |",
            "",
            f"Fairness recovered: **{_pct(recovery.get('point'))}** (95% interval {_pct(recovery.get('lower'))} "
            f"to {_pct(recovery.get('upper'))}).",
            "",
        ] if before and after else []),
        "## Use it (3 steps)",
        "",
        "1. `pip install transformers hqq accelerate`",
        "2. `python apply_repair.py` (downloads the model, compresses it, applies the repair, answers a test question)",
        "3. To keep the repaired model: `python apply_repair.py --out repaired_model`",
        "",
        "Using llama.cpp instead? Add the lines in `llama_cpp.txt` to your `llama-quantize` command.",
        "",
        "## What's inside",
        "",
        "- `patch.json`: the repair (which parts, at which precision). Nothing else is needed.",
        "- `apply_repair.py`: applies it; runs on your machine, nothing is sent anywhere.",
        "- `llama_cpp.txt`: the same repair for llama.cpp.",
        "",
        "Parts kept at 8-bit, in the order the scan chose them:",
        "",
        *[f"- `{path}`" for path in parts],
    ]
    return "\n".join(lines) + "\n"


def build_kit(patch: dict, summary: dict, llama_lines: list[str], out_zip: Path) -> Path:
    """Writes the zip and returns its path."""
    out_zip = Path(out_zip)
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    llama = ("# Add to llama-quantize, e.g.:\n# llama-quantize \\\n#   <these lines> \\\n"
             "#   model-f16.gguf model-repaired.gguf Q3_K_M\n" + "\n".join(llama_lines) + "\n")
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as kit:
        kit.writestr("README.md", readme(patch, summary))
        kit.writestr("apply_repair.py", APPLY_SCRIPT)
        kit.writestr("patch.json", json.dumps(patch, indent=2) + "\n")
        kit.writestr("llama_cpp.txt", llama)
    return out_zip
