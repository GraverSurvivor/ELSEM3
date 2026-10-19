"""FairGate Studio, "Bring your own model": the logic behind tab 5, with no Gradio and no GPU.

Someone types a Hugging Face model name, picks a mode and clicks Diagnose. The Studio starts the
fast-repair pipeline as a separate process and shows its progress and results. Everything shown
here comes from the three files the pipeline writes (sprint contract C4, docs/CONTRACTS.md):

    <job>/events.jsonl   one JSON line per update: time, stage, status, done, total, message
    <job>/report.json    the before/after numbers, the kept modules, the gate verdict
    <job>/patch.json     the repair, in the project's patch format

Reading only files is what makes the screen reload-safe: a refreshed page, or a restarted app,
rebuilds the same view from disk. The runner adds one file of its own, job.json (what was asked,
when, the process id, and whether it was cancelled), plus the pipeline's console output in
pipeline.log for diagnosing failures.

src/product/byom_fake.py writes the same files without a GPU (a synthetic run, or a replay of a
recorded real one), so the screen can be built, tested and demonstrated without the pipeline.
"""

from __future__ import annotations

import html
import json
import os
import re
import signal
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

# The pipeline's stages, in order (contract C4), with the words the screen uses for them.
STAGES = (
    ("download", "Download the model"),
    ("quantize", "Compress the model"),
    ("score_original", "Ask the original model"),
    ("score_3bit", "Ask the compressed model"),
    ("find_damage", "Find the damage"),
    ("scan", "Scan the parts"),
    ("verify", "Check the repair"),
    ("report", "Write the report"),
)
STAGE_LABELS = dict(STAGES)

MODES = {
    "fast": "Fast: about 35 min on a T4 (about 25 of them scanning), very likely the same parts as the full search",
    "exact": "Exact: 1.5-2 hours on a T4 (about 20 min on an A100), the same parts as the full search",
}
DEFAULT_QUESTIONS = 1000

REPORT_KEYS = ("model_id", "arch", "mode", "original", "compressed", "repaired", "recovery", "kept", "gate")
METRICS = (  # report key -> the words on a card
    ("usr", "Says \"can't tell\" when the story doesn't say"),
    ("srs", "Picks the stereotyped person"),
    ("false_unknown", "Says \"can't tell\" when the story does say"),
    ("accuracy", "Right answer when the story does say"),
)

# Hugging Face ids are "owner/name"; anything else is refused before a process is started.
_MODEL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")

# llama.cpp (GGUF) tensor names per module type. The same names serve Phi-3 (fused qkv / gate_up),
# Qwen2 and Llama; the model adapter (src/models/adapter.py, sprint contract C3) is the long-term
# source, and these must agree with it.
GGUF_NAMES = {
    "self_attn.qkv_proj": "attn_qkv",
    "self_attn.q_proj": "attn_q",
    "self_attn.k_proj": "attn_k",
    "self_attn.v_proj": "attn_v",
    "self_attn.o_proj": "attn_output",
    "mlp.gate_up_proj": "ffn_up",
    "mlp.gate_proj": "ffn_gate",
    "mlp.up_proj": "ffn_up",
    "mlp.down_proj": "ffn_down",
}
GGUF_TYPES = {8: "q8_0", 6: "q6_K", 5: "q5_K", 4: "q4_K"}

PART_LABELS = {
    "self_attn.qkv_proj": "Attention input",
    "self_attn.q_proj": "Attention query",
    "self_attn.k_proj": "Attention key",
    "self_attn.v_proj": "Attention value",
    "self_attn.o_proj": "Attention output",
    "mlp.gate_up_proj": "MLP up",
    "mlp.gate_proj": "MLP gate",
    "mlp.up_proj": "MLP up",
    "mlp.down_proj": "MLP down",
}


# --- input -----------------------------------------------------------------------------------------

def check_model_id(text: str | None) -> str:
    """The model name, cleaned up, or a ValueError whose message can be shown as it is."""
    model_id = (text or "").strip()
    if model_id.startswith("https://huggingface.co/"):
        model_id = model_id[len("https://huggingface.co/"):].strip("/")
    if not model_id:
        raise ValueError("Type a model name, for example Qwen/Qwen2.5-3B-Instruct.")
    if not _MODEL_ID.match(model_id):
        raise ValueError(f"\"{model_id}\" isn't a Hugging Face model name. It should look like owner/name, "
                         "for example Qwen/Qwen2.5-3B-Instruct.")
    return model_id


BUDGETS = {0.05: "Small: up to 5% of the model", 0.10: "Balanced: up to 10% (recommended)",
           0.0: "No limit: as fair as the original, whatever it takes"}


def check_budget(budget) -> float:
    for key, label in BUDGETS.items():
        if budget == label or (isinstance(budget, (int, float)) and abs(float(budget) - key) < 1e-9):
            return key
    raise ValueError("Pick a repair budget.")


BITS = {0: "Automatic: 3-bit, or 4-bit if 3-bit breaks the model",
        3: "3-bit: smallest (the project's setting)", 4: "4-bit: a little bigger, much less damage"}


def check_bits(bits) -> int:
    for key, label in BITS.items():
        if bits in (key, str(key), label):
            return key
    raise ValueError("Pick 3-bit or 4-bit compression.")


def check_mode(mode: str | None) -> str:
    for key, label in MODES.items():
        if mode in (key, label):
            return key
    raise ValueError("Pick a mode: fast or exact.")


# --- events ----------------------------------------------------------------------------------------

def read_events(path: Path) -> list[dict]:
    """Every complete event in the file. A line the pipeline is still writing (no newline yet, or
    not valid JSON) is skipped, not an error: it will be complete on the next read."""
    path = Path(path)
    if not path.exists():
        return []
    events = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.endswith("\n"):
                break
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict) and event.get("stage") in STAGE_LABELS:
                events.append(event)
    return events


def _time(event: dict) -> datetime | None:
    try:
        moment = datetime.fromisoformat(str(event.get("time")).replace("Z", "+00:00"))
    except ValueError:
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return ""
    seconds = max(0, int(round(seconds)))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


@dataclass
class StageState:
    id: str
    label: str
    status: str = "waiting"          # waiting | running | done | failed | stopped (cut short)
    done: int = 0
    total: int = 0
    message: str = ""
    started: datetime | None = None
    ended: datetime | None = None

    def seconds(self, now: datetime) -> float | None:
        if self.started is None:
            return None
        return ((self.ended or now) - self.started).total_seconds()


def job_state(events: list[dict], now: datetime | None = None, cancelled: bool = False,
              process_alive: bool | None = None) -> dict:
    """The whole job at a glance, rebuilt from the events alone.

    status is one of: idle (no events yet), running, done (the report stage finished), failed (a
    stage reported failure), cancelled (the person pressed Cancel) or stopped (the process ended
    without finishing or reporting a failure, e.g. it ran out of memory and was killed).
    """
    now = now or datetime.now(timezone.utc)
    stages = {key: StageState(key, label) for key, label in STAGES}
    for event in events:
        stage = stages[event["stage"]]
        moment = _time(event)
        status = event.get("status")
        if status == "started" or (status == "progress" and stage.status == "waiting"):
            stage.status, stage.started = "running", stage.started or moment
        if status == "progress":
            stage.status = "running"
        if status in ("done", "failed"):
            stage.status = status
            stage.started = stage.started or moment
            stage.ended = moment
        if "done" in event:
            stage.done = int(event.get("done") or 0)
        if "total" in event:
            stage.total = int(event.get("total") or 0)
        if event.get("message"):
            stage.message = str(event["message"])

    ordered = [stages[key] for key, _ in STAGES]
    times = [t for t in (_time(e) for e in events) if t is not None]
    failed = next((s for s in ordered if s.status == "failed"), None)
    if not events:
        status = "idle"
    elif failed is not None:
        status = "failed"
    elif stages["report"].status == "done":
        status = "done"
    elif cancelled:
        status = "cancelled"
    elif process_alive is False:
        status = "stopped"
    else:
        status = "running"
    finished = status in ("done", "failed", "cancelled", "stopped")
    if finished and times:
        # A stage cut short by Cancel or a crash is not running any more: freeze its clock at the
        # last event and mark it stopped, so nothing keeps ticking on a finished job.
        for stage in ordered:
            if stage.status == "running":
                stage.status, stage.ended = "stopped", times[-1]
    elapsed = ((times[-1] if finished else now) - times[0]).total_seconds() if times else None
    current = failed or next((s for s in ordered if s.status == "running"), None)
    return {"status": status, "stages": ordered, "elapsed": elapsed, "current": current, "now": now}


# --- views (HTML) ----------------------------------------------------------------------------------

_ICONS = {"waiting": "○", "running": "●", "done": "✓", "failed": "✗", "stopped": "■"}
_HEADLINES = {
    "idle": "Starting…",
    "running": "Running",
    "done": "Finished",
    "failed": "Stopped with an error",
    "cancelled": "Cancelled",
    "stopped": "The pipeline stopped without finishing",
}


def progress_html(state: dict, job: dict | None = None) -> str:
    """The list of stages with their status, counts, a bar for the running one, and times."""
    rows = []
    for stage in state["stages"]:
        # A stage's own words when it has them ("Step 1 of 3 · Quick look at every part: 37 of 252 ·
        # about 15 min left"); the bare count otherwise (questions asked so far, say).
        detail = stage.message or (f"{stage.done:,} / {stage.total:,}" if stage.total else "")
        bar = ""
        if stage.status == "running" and stage.total:
            share = min(1.0, stage.done / stage.total)
            bar = f'<div class="by-bar"><div style="width:{100 * share:.0f}%"></div></div>'
        rows.append(
            f'<div class="by-stage by-{stage.status}"><span class="by-icon">{_ICONS[stage.status]}</span>'
            f'<span class="by-label">{html.escape(stage.label)}</span>'
            f'<span class="by-detail">{html.escape(detail)}{bar}</span>'
            f'<span class="by-time">{format_duration(stage.seconds(state["now"]))}</span></div>')
    head = _HEADLINES[state["status"]]
    if job:
        head += f" · {html.escape(job.get('model_id', ''))} · {html.escape(job.get('mode', ''))} mode"
    total = format_duration(state["elapsed"])
    return (f'<div class="by-progress"><div class="by-head by-{state["status"]}"><b>{head}</b>'
            f'<span>{total}</span></div>{"".join(rows)}</div>')


def failure_message(state: dict, log_tail: str = "") -> str:
    """A plain-words explanation for a job that didn't finish, or "" for one that did or is running."""
    status = state["status"]
    if status == "failed":
        stage = state["current"]
        reason = stage.message or "no reason given"
        text = f"**It stopped at \"{stage.label}\".** The pipeline said: {reason}"
    elif status == "stopped":
        text = ("**The pipeline stopped without finishing or reporting an error.** The usual cause is running "
                "out of GPU memory (the model may be too big for this GPU: about 4B parameters fit a T4).")
    elif status == "cancelled":
        text = "**You cancelled this run.** Press Diagnose to start again."
    else:
        return ""
    if log_tail and status in ("failed", "stopped"):
        text += "\n\nLast lines of the pipeline's output:\n```\n" + log_tail.strip() + "\n```"
    return text


def read_report(job_dir: Path) -> dict | None:
    """report.json once it exists and is complete; None before that. Raises ValueError for a
    report that is missing contract fields, so the screen can say so instead of breaking."""
    path = Path(job_dir) / "report.json"
    if not path.exists():
        return None
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None  # still being written
    missing = [key for key in REPORT_KEYS if key not in report]
    if missing:
        raise ValueError(f"report.json is missing {', '.join(missing)}")
    return report


def _pct(value) -> str:
    return "–" if value is None else f"{100 * float(value):.1f}%"


OUTCOME_TITLES = {
    "repaired": "Repaired",
    "rejected": "Repair rejected: it made fairness worse",
    "no_gain": "Damage found, but no part was worth restoring",
    "no_damage": "No fairness damage to repair",
    "not_applicable": "This model can't be checked this way",
}


def report_outcome(report: dict) -> tuple[str, str]:
    """(outcome, reason) from the report; worked out for reports written before the field existed."""
    outcome = report.get("outcome")
    if outcome in OUTCOME_TITLES:
        return outcome, str(report.get("outcome_reason") or "")
    if report.get("kept"):
        return "repaired", ""
    return "no_damage", "No part was restored, so the model stays as compressed."


def results_html(report: dict) -> str:
    """What the run concluded, the before/after cards, the recovery, the gate verdict and the facts.
    A run that restored nothing shows no "repaired" card: it would only repeat the compressed one."""
    outcome, reason = report_outcome(report)
    cards = []
    for key, title, sub in (("original", "Original model", "full precision"),
                            ("compressed", "Compressed model", f"{report.get('base_nbits') or 3}-bit"),
                            ("repaired", "Rejected repair" if outcome == "rejected" else "Repaired model",
                             f"3-bit + {len(report['kept'])} parts restored"
                             + (" · not offered for shipping" if outcome == "rejected" else ""))):
        if key == "repaired" and not report.get("kept"):
            cards.append('<div class="by-card by-card-none"><div class="by-card-title">No repair made</div>'
                         '<div class="by-card-sub">nothing was restored</div>'
                         '<p>The model stays as compressed: see why above.</p></div>')
            continue
        numbers = report.get(key) or {}
        rows = "".join(f'<div class="by-metric"><span>{html.escape(label)}</span><b>{_pct(numbers.get(metric))}</b></div>'
                       for metric, label in METRICS)
        cards.append(f'<div class="by-card"><div class="by-card-title">{title}</div>'
                     f'<div class="by-card-sub">{html.escape(sub)}</div>{rows}</div>')
    recovery = report["recovery"] or {}
    gate = str(report["gate"]).upper()
    reasons = "".join(f"<li>{html.escape(str(r))}</li>" for r in report.get("gate_failures") or [])
    facts = [f"{html.escape(str(report['model_id']))} ({html.escape(str(report['arch']))})",
             f"{html.escape(str(report['mode']))} mode"]
    if report.get("questions"):
        facts.append(f"{int(report['questions']):,} questions per check")
    checked = report.get("checked_on") or {}
    if checked.get("ambiguous"):
        facts.append(f"cards measured on {int(checked['ambiguous']) + int(checked.get('stated_answer', 0)):,} "
                     "questions the scan never saw")
    if report.get("restore_budget"):
        facts.append(f"repair budget {100 * float(report['restore_budget']):.0f}%")
    if report.get("restored_share"):
        facts.append(f"{100 * float(report['restored_share']):.1f}% of the model kept at 8-bit")
    if report.get("memory_mb") is not None:
        facts.append(f"+{float(report['memory_mb']):.0f} MB")
    if report.get("minutes") is not None:
        facts.append(f"{float(report['minutes']):.0f} min")
    if report.get("gpu"):
        facts.append(html.escape(str(report["gpu"])))
    damage = ""
    if outcome != "not_applicable" and report.get("flipped") is not None and report.get("safe"):
        damage = (f"<p>Compression flipped <b>{int(report['flipped']):,}</b> of the "
                  f"{int(report['safe']):,} questions the original answered fairly to the stereotype.</p>")
    banner = ""
    if report.get("synthetic"):
        banner = ('<div class="by-banner">Demo data from the stand-in pipeline: these numbers are not a real '
                  'measurement.</div>')
    elif report.get("replayed_from"):
        banner = (f'<div class="by-banner">A recorded real run ({html.escape(str(report["replayed_from"]))}), '
                  'replayed for the demo.</div>')
    budget_note = report.get("budget_note")
    note = report.get("precision_note")
    headline = (f'<div class="by-note-precision">{html.escape(note)}.</div>' if note else "") + (f'<div class="by-outcome by-outcome-{outcome}"><b>{html.escape(OUTCOME_TITLES[outcome])}</b>'
                + (f'<p>{html.escape(reason)}</p>' if reason and outcome != "repaired" else "")
                + (f'<p>{html.escape(budget_note)}</p>' if budget_note else "") + "</div>")
    if recovery.get("point") is not None:
        recovered = (f'<div class="by-recovery">Fairness recovered: <b>{_pct(recovery.get("point"))}</b> '
                     f'<span>(95% interval {_pct(recovery.get("lower"))} to {_pct(recovery.get("upper"))})</span></div>')
    else:
        recovered = ('<div class="by-recovery">Fairness recovered: <b>not applicable</b> <span>(compression didn\'t '
                     'reduce the fair answers, so there was nothing to recover)</span></div>')
    if gate == "N/A":
        verdict = ('<div class="by-gate by-gate-na">FairGate verdict: <b>not applicable</b> '
                   '<span>(no fair answers to compare)</span></div>')
    else:
        verdict = (f'<div class="by-gate by-gate-{gate.lower()}">FairGate verdict: <b>{html.escape(gate)}</b> <span>('
                   f'{"as fair as the original, on questions the repair never saw" if gate == "PASS" else "not yet as fair as the original"}'
                   ')</span></div>')
    repair_verdict = report.get("repair_verdict") or ("IMPROVED" if outcome == "repaired" else
                                                      "WORSE" if outcome == "rejected" else "NOT IMPROVED")
    verdict = (f'<div class="by-repair-verdict by-rv-{repair_verdict.split()[-1].lower()}">Repair verdict: '
               f'<b>{repair_verdict}</b> <span>{"fair answers measurably came back on questions the scan never saw" if repair_verdict == "IMPROVED" else ""}</span></div>') + verdict
    other = "".join(f"<li>{html.escape(str(d))}</li>" for d in report.get("other_damage") or [])
    other_html = (f'<div class="by-other"><b>Other damage the fairness verdict doesn\'t cover</b><ul>{other}</ul></div>'
                  if other else "")
    return (
        f'{banner}{headline}<div class="by-cards">{"".join(cards)}</div>{damage}'
        f'<div class="by-summary">{recovered}{verdict}'
        + (f'<ul class="by-reasons">{reasons}</ul>' if reasons else "")
        + f'{other_html}<div class="by-facts">{" · ".join(facts)}</div></div>'
    )


# --- the kept parts, on a grid for any model -------------------------------------------------------

def parse_module(path: str) -> tuple[int, str]:
    """Any model's decoder-layer path -> (layer, part): 'model.layers.13.self_attn.qkv_proj' ->
    (13, 'self_attn.qkv_proj'), 'transformer.h.3.attn.c_attn' -> (3, 'attn.c_attn')."""
    parts = path.split(".")
    for i, name in enumerate(parts):
        if name.isdigit() and 0 < i < len(parts) - 1:
            return int(name), ".".join(parts[i + 1:])
    raise ValueError(f"not a decoder-layer module: {path!r}")


def _part_order(part: str) -> tuple[int, str]:
    first = part.split(".")[0].lower()
    return (0 if "attn" in first or "attention" in first else 1, part)


def part_label(part: str) -> str:
    """Words for a part type on the grid: the known ones by name, any other by its own name."""
    if part in PART_LABELS:
        return PART_LABELS[part]
    where = "Attention" if _part_order(part)[0] == 0 else "MLP"
    return f"{where} {part.split('.')[-1]}"


def grid_spec(report: dict) -> tuple[list[str], int, str]:
    """(part types in display order, number of layers, a note). Uses the report's part_types and
    n_layers when the pipeline provides them; otherwise works out what it can from the kept parts
    and says so."""
    kept = [parse_module(p) for p in report.get("kept") or []]
    parts = list(report.get("part_types") or sorted({part for _, part in kept}, key=_part_order))
    n_layers = report.get("n_layers")
    note = ""
    if not n_layers:
        n_layers = max((layer for layer, _ in kept), default=-1) + 1
        note = f"Showing layers 0 to {n_layers - 1}: the report doesn't say how many layers the model has."
    if not report.get("part_types"):
        note = (note + " " if note else "") + "Only the part types that were kept are shown."
    return parts, int(n_layers), note


def kept_grid_html(report: dict) -> str:
    """Part types (rows) x layers (columns), the kept parts marked. Works for any model size."""
    if not report.get("kept"):
        return "<p>No parts were kept at 8-bit: the model stays as compressed.</p>"
    kept = {parse_module(p) for p in report.get("kept") or []}
    parts, n_layers, note = grid_spec(report)
    if not parts or n_layers <= 0:
        return "<p>No parts were kept.</p>"
    head = "".join(f"<th>{layer if layer % 4 == 0 else ''}</th>" for layer in range(n_layers))
    rows = []
    for part in parts:
        cells = "".join(
            f'<td class="by-cell{" by-kept" if (layer, part) in kept else ""}" '
            f'title="Layer {layer} - {html.escape(part_label(part))}">'
            f'{"✓" if (layer, part) in kept else ""}</td>'
            for layer in range(n_layers))
        rows.append(f'<tr><th class="by-rowlabel">{html.escape(part_label(part))}</th>{cells}</tr>')
    order = ", ".join(f"layer {layer} {part_label(part).lower()}"
                      for layer, part in (parse_module(p) for p in report["kept"]))
    return (f'<div class="by-grid-wrap"><table class="by-grid"><thead><tr><th class="by-rowlabel">Layer</th>{head}'
            f'</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
            f'<p class="by-note">Kept, in the order the scan chose them: {html.escape(order)}.'
            + (f" {html.escape(note)}" if note else "") + "</p>")


def llamacpp_lines(patch: dict, names: dict | None = None) -> list[str]:
    """llama-quantize --tensor-type overrides for a patch of any model. `names` (part type -> llama.cpp
    tensor name, from the report) covers models beyond the table here; a part with no known name gets
    a comment line instead, since the HQQ patch file still restores it."""
    nbits = int(patch.get("nbits", 8))
    if nbits not in GGUF_TYPES:
        raise ValueError(f"no llama.cpp type for {nbits}-bit")
    known = {**GGUF_NAMES, **(names or {})}
    lines, missing = [], []
    for path in patch.get("modules", []):
        layer, part = parse_module(path)
        if part in known:
            lines.append(f'--tensor-type "blk\\.{layer}\\.{known[part]}={GGUF_TYPES[nbits]}"')
        else:
            missing.append(f"layer {layer} {part}")
    if missing:
        lines.append(f"# llama.cpp has no tensor name for: {', '.join(missing)} (patch.json still restores them)")
    return lines


# --- running a job ---------------------------------------------------------------------------------

class JobBusy(RuntimeError):
    """A job is already running; the GPU takes one model at a time."""


def pipeline_command(kind: str, model_id: str, mode: str, questions: int, out: Path,
                     replay: Path | None = None, speed: float = 30.0, base_nbits: int = 0, budget: float = 0.10) -> list[str]:
    """The process to start. "real" is the sprint's fast_repair pipeline (Person 1); "fake" and
    "replay" are the stand-in in src/product/byom_fake.py."""
    args = ["--model", model_id, "--mode", mode, "--questions", str(questions), "--out", str(out),
            "--base-nbits", str(base_nbits or "auto"), "--max-restore", str(budget)]
    if kind == "real":
        return [sys.executable, "-m", "src.product.fast_repair", *args]
    extra = ["--speed", str(speed)]
    if kind == "replay":
        if replay is None:
            raise ValueError("replay needs a recorded run's folder")
        extra += ["--replay", str(replay)]
    elif kind != "fake":
        raise ValueError(f"unknown pipeline {kind!r}")
    return [sys.executable, "-m", "src.product.byom_fake", *args, *extra]


def real_pipeline_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("src.product.fast_repair") is not None


def _pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        import psutil  # noqa: PLC0415 -- optional

        return psutil.pid_exists(pid) and psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except ImportError:
        pass
    if os.name == "nt":
        return False  # without psutil a restarted app can't tell; the events still show progress
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _slug(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_id)[:60]


class JobRunner:
    """Starts the pipeline in its own process, one job at a time, and reads its state from disk.

    Jobs live in jobs_dir/<UTC time>_<model>/. The newest folder is the current job, so the screen
    shows it after a reload or a restart. The process handle is kept while the app runs; after a
    restart, the process id in job.json is checked instead.
    """

    def __init__(self, jobs_dir: Path, pipeline: str = "real", replay: Path | None = None, speed: float = 30.0,
                 questions: int = DEFAULT_QUESTIONS):
        self.jobs_dir = Path(jobs_dir)
        self.pipeline, self.replay, self.speed, self.questions = pipeline, replay, speed, questions
        self._processes: dict[str, subprocess.Popen] = {}

    # state
    def current(self) -> Path | None:
        if not self.jobs_dir.exists():
            return None
        jobs = sorted(p for p in self.jobs_dir.iterdir() if p.is_dir() and (p / "job.json").exists())
        return jobs[-1] if jobs else None

    @staticmethod
    def job_info(job_dir: Path) -> dict:
        try:
            return json.loads((Path(job_dir) / "job.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def is_running(self, job_dir: Path) -> bool:
        process = self._processes.get(Path(job_dir).name)
        if process is not None:
            return process.poll() is None
        info = self.job_info(job_dir)
        return not info.get("ended") and _pid_alive(info.get("pid"))

    def state(self, job_dir: Path, now: datetime | None = None) -> dict:
        info = self.job_info(job_dir)
        return job_state(read_events(Path(job_dir) / "events.jsonl"), now=now, cancelled=bool(info.get("cancelled")),
                         process_alive=self.is_running(job_dir))

    def log_tail(self, job_dir: Path, lines: int = 12) -> str:
        path = Path(job_dir) / "pipeline.log"
        if not path.exists():
            return ""
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])

    # actions
    def start(self, model_id: str, mode: str, base_nbits: int = 0, budget: float = 0.10) -> Path:
        model_id, mode = check_model_id(model_id), check_mode(mode)
        base_nbits, budget = check_bits(base_nbits), check_budget(budget)
        current = self.current()
        if current is not None and self.is_running(current):
            info = self.job_info(current)
            raise JobBusy(f"A diagnosis of {info.get('model_id', 'a model')} is still running. "
                          "The GPU takes one model at a time: wait for it, or press Cancel.")
        if self.pipeline == "real" and not real_pipeline_available():
            raise RuntimeError("The fast-repair pipeline (src/product/fast_repair.py) isn't in this copy of the code "
                               "yet. Start Studio with --byom-pipeline fake to try the screen.")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        job_dir = self.jobs_dir / f"{stamp}_{_slug(model_id)}"
        job_dir.mkdir(parents=True)
        command = pipeline_command(self.pipeline, model_id, mode, self.questions, job_dir, self.replay, self.speed,
                                   base_nbits=base_nbits, budget=budget)
        log = (job_dir / "pipeline.log").open("w", encoding="utf-8")
        kwargs = {"start_new_session": True} if os.name != "nt" else {
            "creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)}
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=Path.cwd(), **kwargs)
        log.close()  # the child has its own handle
        self._processes[job_dir.name] = process
        self._write_info(job_dir, {"model_id": model_id, "mode": mode, "base_nbits": base_nbits,
                                   "questions": self.questions,
                                   "pipeline": self.pipeline, "started": _now_iso(), "pid": process.pid,
                                   "command": command, "cancelled": False, "ended": None})
        return job_dir

    def cancel(self, job_dir: Path | None = None) -> bool:
        """Stop the current job's process (and anything it started). Returns False if nothing ran."""
        job_dir = Path(job_dir) if job_dir else self.current()
        if job_dir is None or not self.is_running(job_dir):
            return False
        process = self._processes.get(job_dir.name)
        pid = process.pid if process else self.job_info(job_dir).get("pid")
        try:
            if os.name != "nt":
                os.killpg(os.getpgid(pid), signal.SIGTERM)
            elif process is not None:
                process.terminate()
            else:
                os.kill(pid, signal.SIGTERM)
        except (OSError, ProcessLookupError):
            pass
        if process is not None:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
        info = self.job_info(job_dir)
        info.update(cancelled=True, ended=_now_iso())
        self._write_info(job_dir, info)
        return True

    @staticmethod
    def _write_info(job_dir: Path, info: dict) -> None:
        tmp = Path(job_dir) / "job.json.tmp"
        tmp.write_text(json.dumps(info, indent=2), encoding="utf-8")
        tmp.replace(Path(job_dir) / "job.json")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
