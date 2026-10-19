"""The developer download: one zip, three files, one script.

    repair_kit.zip
      README.md     what the repair does, its numbers, and the three commands (plus llama.cpp settings)
      fairgate.py   chat with the repaired model, serve it as an OpenAI-compatible API with a key, or save it
      patch.json    the repair itself (which parts, at which precision)

Most people don't need it: the Studio's Ship screen serves the repaired model as an API ("Use it now") or
publishes it to Hugging Face ("Get the fixed model"). The kit is for running it on your own machine, offline:
it rebuilds the model from the public checkpoint and this patch, nothing of yours is sent anywhere.

fairgate.py needs only `transformers` and `hqq` (and fastapi/uvicorn to serve). It rebuilds exactly the
model the repair was measured on: every linear layer (except the output head) HQQ-quantized to the base
precision with group size 64, and the patch's parts quantized to 8-bit from the same original weights.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

KIT_FILES = Path(__file__).with_name("kit_files")  # shipped into the zip as it is
SCRIPT = "fairgate.py"
INSTALL = "pip install transformers hqq accelerate"


def _pct(value) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"


def results_lines(summary: dict) -> list[str]:
    """The measured before/after numbers, as Markdown; only the rows the run measured."""
    before, after = summary.get("compressed") or {}, summary.get("repaired") or {}
    rows = [(key, label) for key, label in (
        ("usr", "Says \"can't tell\" when the story doesn't say (fair)"),
        ("srs", "Picks the stereotyped person"),
        ("accuracy", "Right answer when the story does say"))
        if before.get(key) is not None and after.get(key) is not None]
    if not rows:
        return []
    scope = summary.get("scope", "On questions the repair never saw")
    lines = [f"| {scope} | Compressed | Repaired |", "|---|---|---|",
             *[f"| {label} | {_pct(before[key])} | {_pct(after[key])} |" for key, label in rows], ""]
    recovery = summary.get("recovery") or {}
    if recovery.get("point") is not None:
        lines += [f"Fairness recovered: **{_pct(recovery['point'])}** (95% interval {_pct(recovery.get('lower'))} "
                  f"to {_pct(recovery.get('upper'))}).", ""]
    return lines


def headline(patch: dict, summary: dict) -> str:
    model_id = patch.get("model_id", "the model")
    base, parts = patch.get("base", {}).get("nbits", 3), patch.get("modules", [])
    return (f"Compressing {model_id} to {base}-bit made it answer more questions with a stereotype. "
            f"This repair keeps **{len(parts)} of its parts** at 8-bit (about +{summary.get('memory_mb', 0):.0f} MB), "
            "which brings fair answers back.")


def readme(patch: dict, summary: dict, llama_lines: list[str] | None = None) -> str:
    """Plain-language instructions and what the repair is, from the patch and the run's numbers."""
    model_id = patch.get("model_id", "the model")
    lines = [
        f"# FairGate repair: {model_id}", "",
        headline(patch, summary), "",
        *results_lines(summary),
        "## Use it", "",
        "```bash",
        INSTALL,
        "python fairgate.py chat \"Who was late, the nurse or the mechanic?\"   # ask it something",
        "python fairgate.py serve                                             # an API with a key",
        "python fairgate.py save repaired_model                               # keep the repaired model",
        "```", "",
        "Building it holds the 16-bit model in memory while it loads (about 2 GB per billion parameters);",
        "a GPU is recommended, a CPU works slowly.", "",
        "`serve` runs the repaired model on your machine as an **OpenAI-compatible API** at",
        "`http://localhost:8000/v1` (`pip install fastapi uvicorn` first). It prints a new API key once and keeps it in",
        "`api_key.txt` (or pass `--api-key`). Use it from any OpenAI client:",
        "",
        "```python",
        "from openai import OpenAI",
        'client = OpenAI(base_url="http://localhost:8000/v1", api_key="<your key>")',
        'print(client.chat.completions.create(model="repaired", messages=[{"role": "user", "content": "Hi"}])',
        "      .choices[0].message.content)",
        "```", "",
    ]
    if llama_lines:
        lines += ["## llama.cpp (optional)", "",
                  "Add these to `llama-quantize` to keep the same parts at 8-bit in a GGUF model:", "",
                  "```bash", "llama-quantize \\",
                  *[f"  {line} \\" for line in llama_lines],
                  "  model-f16.gguf model-fairgate.gguf Q3_K_M", "```", ""]
    lines += ["## Parts kept at 8-bit", "", "In the order the scan chose them:", "",
              *[f"- `{path}`" for path in patch.get("modules", [])]]
    return "\n".join(lines) + "\n"


def build_kit(patch: dict, summary: dict, llama_lines: list[str], out_zip: Path) -> Path:
    """Writes the zip and returns its path."""
    out_zip = Path(out_zip)
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as kit:
        kit.writestr("README.md", readme(patch, summary, [line for line in llama_lines if not line.startswith("#")]))
        kit.writestr(SCRIPT, (KIT_FILES / SCRIPT).read_text(encoding="utf-8"))
        kit.writestr("patch.json", json.dumps(patch, indent=2) + "\n")
    return out_zip
