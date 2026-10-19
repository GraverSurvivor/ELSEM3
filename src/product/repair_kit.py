"""The download a customer gets: one zip that applies the repair with a single command.

    repair_kit.zip
      README.md         what was found, what the repair does, how to use it (3 steps, plain words)
      apply_repair.py   loads the model, compresses it, keeps the repaired parts at 8-bit, saves it
      serve_repair.py   serves the repaired model as an OpenAI-compatible API, protected by an API key
      fairgate_repair.py  the loading code both share; requirements.txt  what to install
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

KIT_FILES = Path(__file__).with_name("kit_files")  # shipped into the zip as they are
SCRIPTS = ("fairgate_repair.py", "apply_repair.py", "serve_repair.py")
REQUIREMENTS = """transformers>=4.44
hqq>=0.2.8
accelerate
torch
# for serve_repair.py only:
fastapi
uvicorn
"""


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
        "1. `pip install -r requirements.txt`",
        "2. `python apply_repair.py` (downloads the model, compresses it, applies the repair, answers a test question)",
        "3. To keep the repaired model: `python apply_repair.py --out repaired_model`",
        "",
        "## Use it as an API (with an API key)",
        "",
        "`python serve_repair.py` serves the repaired model on your machine as an **OpenAI-compatible API**. It",
        "prints a new API key once (also saved in `api_key.txt`; or choose your own with `--api-key` or the",
        "`FAIRGATE_API_KEY` environment variable). Call it like OpenAI:",
        "",
        "```python",
        "from openai import OpenAI",
        'client = OpenAI(base_url="http://localhost:8000/v1", api_key="<your key>")',
        f'reply = client.chat.completions.create(model="{model_id.split("/")[-1]}-fairgate",',
        '                                       messages=[{"role": "user", "content": "Hello"}])',
        "print(reply.choices[0].message.content)",
        "```",
        "",
        "or `curl http://localhost:8000/v1/chat/completions -H \"Authorization: Bearer <your key>\" "
        "-H \"Content-Type: application/json\" -d '{\"messages\": [{\"role\": \"user\", \"content\": \"Hello\"}]}'`.",
        "Requests without the right key are refused (401). By default it only listens on this machine; add",
        "`--host 0.0.0.0` to accept others on your network, and keep the key secret.",
        "",
        "Using llama.cpp instead? Add the lines in `llama_cpp.txt` to your `llama-quantize` command.",
        "",
        "## What's inside",
        "",
        "- `patch.json`: the repair (which parts, at which precision). Nothing else is needed.",
        "- `apply_repair.py`: applies it; runs on your machine, nothing is sent anywhere.",
        "- `serve_repair.py`: the repaired model as an API with a key. `fairgate_repair.py`: the code both share.",
        "- `requirements.txt`: what to install.",
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
        for name in SCRIPTS:
            kit.writestr(name, (KIT_FILES / name).read_text(encoding="utf-8"))
        kit.writestr("requirements.txt", REQUIREMENTS)
        kit.writestr("patch.json", json.dumps(patch, indent=2) + "\n")
        kit.writestr("llama_cpp.txt", llama)
    return out_zip
