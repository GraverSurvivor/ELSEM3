"""Tab 6, "My models": every run in the database as a table, a page, a comparison and a printable report.
No Gradio and no GPU here: rows in, HTML out. Everything shown comes from the local copy (contract C9).
"""

from __future__ import annotations

import html
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.product.byom import METRICS, OUTCOME_TITLES, REPORT_KEYS, kept_grid_html, llamacpp_lines, results_html

HEADERS = ["Model", "Method", "Bits", "Verdict", "Fairness recovered", "Extra MB", "Date", "Who ran it", "Outcome"]
QUIET = timedelta(minutes=30)  # a "running" run that has said nothing for this long has lost its machine
STATUS_WORDS = {"running": "Running", "failed": "Stopped with an error", "cancelled": "Cancelled",
                "stopped": "Stopped without finishing", "done": "Finished", "interrupted": "Interrupted (its session ended)"}
ANY = "All"
OUTCOMES = {ANY: None, "Repaired": "repaired", "Rejected": "rejected", "No gain": "no_gain", "No damage": "no_damage",
            "Not applicable": "not_applicable", "Running": "running", "Failed": "failed", "Cancelled": "cancelled"}


def esc(value) -> str:
    return html.escape("" if value is None else str(value))


def pct(value, digits: int = 1) -> str:
    return "–" if value is None else f"{100 * float(value):.{digits}f}%"


def day(moment: datetime | None) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC") if moment else "–"


def bits_text(run: dict) -> str:
    base, repair = run.get("base_bits"), run.get("repair_bits")
    if not base:
        return "–"
    return f"{base}-bit + {repair}-bit parts" if repair else f"{base}-bit"


def status_of(run: dict, last_news: datetime | None = None, now: datetime | None = None) -> str:
    """The run's status, with one addition: a run still marked running whose machine has gone quiet."""
    status = run.get("status") or "running"
    if status == "running":
        latest = max(filter(None, (last_news, run.get("updated_at"), run.get("started_at"))), default=None)
        if latest is not None and (now or datetime.now(timezone.utc)) - latest > QUIET:
            return "interrupted"
    return status


def outcome_text(run: dict, last_news: datetime | None = None) -> str:
    if run.get("source") == "reference":
        return "Research run"
    stand_in = " (stand-in)" if run.get("pipeline") in ("fake", "replay") else ""
    if run.get("outcome"):
        return {"repaired": "Repaired", "rejected": "Rejected", "no_gain": "No gain", "no_damage": "No damage",
                "not_applicable": "Not applicable"}.get(run["outcome"], run["outcome"]) + stand_in
    return STATUS_WORDS.get(status_of(run, last_news), "–") + stand_in


def title(run: dict) -> str:
    """What to call a run: its model, and for a research run which one it is."""
    return f"{run.get('hub_id') or 'unknown model'}" + (f" · {run['job']}" if run.get("source") == "reference" else "")


def table(rows: list[dict], news: dict | None = None) -> list[list[str]]:
    news = news or {}
    return [[title(r), (r.get("method") or "–").upper(), bits_text(r), r.get("verdict") or "–", pct(r.get("recovery")),
             "–" if r.get("extra_mb") is None else f"{r['extra_mb']:+.0f}", day(r.get("started_at")),
             r.get("ran_by") or "–", outcome_text(r, news.get(r["id"]))] for r in rows]


def choices(rows: list[dict]) -> list[tuple[str, str]]:
    """(label, run id) for the pickers."""
    return [(f"{title(r)} · {day(r.get('started_at'))} · {outcome_text(r)}", r["id"]) for r in rows]


# --- one run's page --------------------------------------------------------------------------------

def _facts(run: dict, last_news: datetime | None) -> list[tuple[str, str]]:
    artifact = run.get("artifact") or {}
    link = artifact.get("weights_url")
    recipe = artifact.get("recipe") or {}
    facts = [
        ("Model", title(run)), ("Commit", recipe.get("revision") or run.get("revision") or "not saved"),
        ("Status", STATUS_WORDS.get(status_of(run, last_news), "–")), ("Compressed with", (run.get("method") or "–").upper()),
        ("Bits", bits_text(run)),
        ("Repair budget", "–" if run.get("budget") is None else "no limit" if not run["budget"] else pct(run["budget"], 0)),
        ("Mode", run.get("mode") or "–"), ("Questions per check", f"{run['questions']:,}" if run.get("questions") else "–"),
        ("GPU", run.get("gpu") or "–"), ("Minutes", "–" if run.get("minutes") is None else f"{run['minutes']:.0f}"),
        ("Started", day(run.get("started_at"))), ("Ended", day(run.get("ended_at"))), ("Who ran it", run.get("ran_by") or "–"),
        ("Saved weights", link or "not kept: rebuilt from the recipe when needed"),
        ("Run id", run["id"]),
    ]
    return [(k, str(v)) for k, v in facts]


def facts_html(run: dict, last_news: datetime | None = None) -> str:
    cells = []
    for name, value in _facts(run, last_news):
        shown = esc(value)
        if name == "Saved weights" and value.startswith("https://huggingface.co/"):
            shown = f'<a href="{esc(value)}" target="_blank" rel="noopener">{esc(value)}</a>'
        cells.append(f"<tr><th>{esc(name)}</th><td>{shown}</td></tr>")
    return f'<table class="mm-facts"><tbody>{"".join(cells)}</tbody></table>'


def versions_html(run: dict) -> str:
    saved = (run.get("artifact") or {}).get("versions") or run.get("versions") or {}
    if not saved:
        return "<p>Library versions weren't saved with this run.</p>"
    return "<p>" + " · ".join(f"{esc(k)} {esc(v or 'not installed')}" for k, v in saved.items()) + "</p>"


def strategies_html(run: dict) -> str:
    """The compared repairs, as bars of what each recovered against what it costs (Person 1's data)."""
    rows = run.get("strategies") or []
    if not rows:
        return "<p>Only one repair was tried in this run.</p>"

    def recovered(row):
        value = (row.get("validation") or {}).get("recovery")
        value = value.get("point") if isinstance(value, dict) else value
        return float(value) if isinstance(value, (int, float)) else None

    best = max([v for v in map(recovered, rows) if v is not None] or [1.0])
    lines = []
    for row in rows:
        value = recovered(row)
        width = 0 if value is None or best <= 0 else max(0.0, min(1.0, value / best)) * 100
        cost = [f"{row['n_parts']} parts"] if row.get("n_parts") is not None else []
        cost += [f"{row['extra_mb']:+.0f} MB"] if row.get("extra_mb") is not None else []
        lines.append(
            f'<div class="mm-strategy{" mm-chosen" if row.get("chosen") else ""}"><span>{esc(row["name"])}'
            f'{" ✓ chosen" if row.get("chosen") else ""}</span><div class="mm-bar"><div style="width:{width:.0f}%"></div></div>'
            f'<b>{pct(value, 0)}</b><em>{", ".join(cost)}</em></div>')
    return f'<div class="mm-strategies">{"".join(lines)}</div>'


METRIC_WORDS = {**dict(METRICS), "recovery": "Fairness recovered"}


def metrics_html(run: dict) -> str:
    """The stored numbers as a plain table: for runs with no report (the research ones)."""
    rows = run.get("metrics") or []
    if not rows:
        return "<p>No numbers were recorded.</p>"
    body = "".join(
        f"<tr><td>{esc(m['view'])}</td><td>{esc(METRIC_WORDS.get(m['metric'], m['metric']))}</td>"
        f"<td>{esc(m['name'] or 'overall')}</td><td><b>{pct(m['value'])}</b>"
        + (f" [{pct(m['lower'])}, {pct(m['upper'])}]" if m.get("lower") is not None else "") + "</td></tr>" for m in rows)
    return ('<table class="mm-metrics"><thead><tr><th>Model</th><th>What</th><th>Scope</th><th>Value</th></tr></thead>'
            f"<tbody>{body}</tbody></table>")


def report_html(run: dict) -> str:
    """The results as tab 5 shows them, or the stored numbers when there is no (whole) report."""
    report = run.get("report")
    if report and all(key in report for key in REPORT_KEYS):
        try:
            return results_html(report) + "<h4>Parts kept</h4>" + kept_grid_html(report)
        except Exception:  # noqa: BLE001 -- a report this version can't draw: its numbers still show
            pass
    parts = "".join(f"<li><code>{esc(p['path'])}</code>" + (f" at {p['bits']}-bit" if p.get("bits") else "") + "</li>"
                    for p in run.get("parts") or [])
    return metrics_html(run) + (f"<h4>Parts kept</h4><ol>{parts}</ol>" if parts else "")


def progress_text(run: dict) -> str:
    events = run.get("events") or []
    if not events:
        return ""
    last = events[-1]
    return f"{len(events)} progress lines; the last: {last.get('stage')} {last.get('status')}" + (
        f" ({last['message']})" if last.get("message") else "")


def run_page(run: dict | None, last_news: datetime | None = None) -> str:
    if run is None:
        return "<p>Pick a run above.</p>"
    parts = [f"<h3>{esc(title(run))}</h3>"]
    if run.get("message"):
        parts.append(f'<div class="mm-problem">{esc(run["message"])}</div>')
    if run.get("report") or run.get("metrics"):
        parts.append(report_html(run))
    elif progress_text(run):
        parts.append(f"<p>{esc(progress_text(run))}</p>")
    if run.get("strategies"):
        parts.append("<h4>Repairs compared</h4>" + strategies_html(run))
    parts += ["<h4>About this run</h4>", facts_html(run, last_news), "<h4>Library versions</h4>", versions_html(run)]
    return f'<div class="mm-page">{"".join(parts)}</div>'


# --- two runs side by side -------------------------------------------------------------------------

def compare_html(a: dict | None, b: dict | None) -> str:
    if a is None or b is None:
        return "<p>Pick two runs to compare.</p>"
    if a["id"] == b["id"]:
        return "<p>Pick two different runs.</p>"

    def numbers(run):
        return {(m["view"], m["metric"]): m for m in run.get("metrics") or [] if m["scope"] == "overall"}

    na, nb = numbers(a), numbers(b)
    rows = [("Model", title(a), title(b)), ("Started", day(a.get("started_at")), day(b.get("started_at"))),
            ("Bits", bits_text(a), bits_text(b)), ("Outcome", outcome_text(a), outcome_text(b)),
            ("FairGate verdict", a.get("verdict") or "–", b.get("verdict") or "–"),
            ("Parts kept", str(len(a.get("parts") or [])), str(len(b.get("parts") or []))),
            ("Extra MB", *("–" if r.get("extra_mb") is None else f"{r['extra_mb']:+.0f}" for r in (a, b))),
            ("Minutes", *("–" if r.get("minutes") is None else f"{r['minutes']:.0f}" for r in (a, b)))]
    for view in ("original", "compressed", "repaired"):
        for metric in (*dict(METRICS), "recovery"):
            left, right = na.get((view, metric)), nb.get((view, metric))
            if left is not None or right is not None:
                rows.append((f"{view.capitalize()}: {METRIC_WORDS[metric].lower()}",
                             *(pct(m["value"]) if m else "–" for m in (left, right))))
    body = "".join(f"<tr><th>{esc(n)}</th><td>{esc(x)}</td><td>{esc(y)}</td></tr>" for n, x, y in rows)
    pa, pb = ({p["path"] for p in r.get("parts") or []} for r in (a, b))
    shared = ""
    if a.get("hub_id") == b.get("hub_id") and (pa or pb):
        shared = (f"<p>Parts in both: <b>{len(pa & pb)}</b> · only in the first: {len(pa - pb)} · "
                  f"only in the second: {len(pb - pa)}</p>")
    return (f'<table class="mm-compare"><thead><tr><th></th><th>First</th><th>Second</th></tr></thead>'
            f"<tbody>{body}</tbody></table>{shared}")


# --- files -----------------------------------------------------------------------------------------

def build_run_kit(run: dict, folder: Path) -> Path:
    """The repair kit, from the recipe and the report (it was three small files all along)."""
    from src.product.repair_kit import build_kit
    from src.product.reopen import CannotReopen

    recipe, report = (run.get("artifact") or {}).get("recipe"), run.get("report") or {}
    if not recipe:
        raise CannotReopen("This run ended without a repair, so there is no kit.")
    try:
        lines = llamacpp_lines(recipe, report.get("gguf_names"))
    except ValueError:  # a bit-width llama.cpp has no type for: the kit still works
        lines = []
    return build_kit(recipe, report or {"memory_mb": run.get("extra_mb") or 0}, lines,
                     Path(folder) / f"repair_kit_{run['id'][:8]}.zip")


REPORT_CSS = """
body {font: 14px/1.5 system-ui, sans-serif; color: #111827; max-width: 900px; margin: 24px auto; padding: 0 16px}
h1 {font-size: 22px; margin: 0} h2 {font-size: 16px; margin: 20px 0 6px} h4 {margin: 14px 0 4px}
table {border-collapse: collapse; width: 100%} th, td {text-align: left; padding: 4px 8px; border-bottom: 1px solid #e5e7eb;
  vertical-align: top; overflow-wrap: anywhere} .sub {color: #6b7280}
.by-cards {display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px} .by-card {border: 1px solid #d1d5db;
  border-radius: 8px; padding: 10px} .by-card-title {font-weight: 700} .by-card-sub {font-size: 12px; color: #6b7280}
.by-metric {display: flex; justify-content: space-between; gap: 8px; font-size: 12px}
.by-outcome, .by-banner, .by-other, .mm-problem {border-left: 4px solid #6366f1; padding: 6px 10px; background: #f3f4f6; margin: 8px 0}
.by-gate {font-weight: 700} .by-facts, .by-note {font-size: 12px; color: #6b7280}
.by-grid th, .by-grid td {border: none; padding: 0 1px; font-size: 9px} .by-grid td.by-cell {width: 12px; height: 12px;
  background: #e5e7eb; text-align: center; border: 1px solid #fff} .by-grid td.by-kept {background: #16a34a; color: #fff}
.by-grid-wrap {overflow: hidden} .mm-strategy {display: grid; grid-template-columns: 2fr 3fr 60px 2fr; gap: 8px; align-items: center}
.mm-bar {height: 10px; background: #e5e7eb} .mm-bar div {height: 100%; background: #6366f1}
@media print {body {margin: 0; max-width: none} .by-card, table, .by-grid-wrap {break-inside: avoid}
  * {-webkit-print-color-adjust: exact; print-color-adjust: exact}}
"""


def audit_report(run: dict, last_news: datetime | None = None, fingerprint: str = "") -> str:
    """One self-contained page about one run, made to be printed to PDF."""
    artifact = run.get("artifact") or {}
    recipe_hash = artifact.get("recipe_sha256")
    checks = [f"<li>{esc(fingerprint)}</li>"] if fingerprint else []
    if artifact.get("fingerprint"):
        checks.append(f"<li>{len(artifact['fingerprint'])} fingerprint questions are saved: a rebuilt copy can be "
                      "checked against them.</li>")
    if recipe_hash:
        checks.append(f"<li>Recipe SHA-256: <code>{esc(recipe_hash)}</code></li>")
    if artifact.get("kit_sha256"):
        checks.append(f"<li>Repair kit SHA-256: <code>{esc(artifact['kit_sha256'])}</code></li>")
    body = [f"<h1>FairGate audit report</h1><p class='sub'>{esc(title(run))} · {day(run.get('started_at'))} · made "
            f"{day(datetime.now(timezone.utc))}</p>"]
    if run.get("message"):
        body.append(f'<div class="mm-problem">{esc(run["message"])}</div>')
    body += ["<h2>Result</h2>", report_html(run) if run.get("report") or run.get("metrics") else
             f"<p>{esc(progress_text(run)) or 'Nothing was measured.'}</p>"]
    if run.get("strategies"):
        body += ["<h2>Repairs compared</h2>", strategies_html(run)]
    body += ["<h2>The run</h2>", facts_html(run, last_news), "<h2>Library versions</h2>", versions_html(run)]
    if checks:
        body += ["<h2>Reproducing it</h2><ul>", *checks, "</ul>"]
    body.append("<p class='sub'>The fairness check is BBQ's ambiguous questions: a fair model says \"can't tell\" when "
                "the story doesn't say who did it. Measured on questions the repair never saw.</p>")
    return (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><title>FairGate audit: {esc(title(run))}</title>"
            f"<style>{REPORT_CSS}</style></head><body>{''.join(body)}</body></html>")


def write_audit_report(run: dict, folder: Path, **kwargs) -> Path:
    path = Path(folder) / f"fairgate_audit_{run['id'][:8]}.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(audit_report(run, **kwargs), encoding="utf-8")
    return path


CSS = """
.mm-facts th, .mm-compare th, .mm-metrics th {text-align: left; font-weight: 600; padding: 3px 10px; white-space: nowrap}
.mm-facts td, .mm-compare td, .mm-metrics td {padding: 3px 10px; overflow-wrap: anywhere}
.mm-problem {border-left: 4px solid #dc2626; background: #dc26261a; padding: 8px 12px; border-radius: 6px; margin: 6px 0}
.mm-strategy {display: grid; grid-template-columns: 2fr 3fr 60px 2fr; gap: 8px; align-items: center; font-size: 14px}
.mm-bar {height: 10px; border-radius: 5px; background: var(--background-fill-secondary); overflow: hidden}
.mm-bar div {height: 100%; background: var(--color-accent)} .mm-chosen span {font-weight: 700}
.mm-status {font-size: 13px; opacity: .8}
"""

__all__ = ["HEADERS", "OUTCOMES", "ANY", "OUTCOME_TITLES", "table", "choices", "run_page", "compare_html", "audit_report",
           "write_audit_report", "build_run_kit", "status_of", "CSS"]
