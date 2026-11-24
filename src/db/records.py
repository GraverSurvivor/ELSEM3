"""The pipeline's files as database rows: pure functions, no database.

Every id is worked out from what the row is (a run's 12th progress line is always the same id), so
syncing a job twice, importing it on another machine, or retrying after a dropped connection writes
the same rows again instead of new ones.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone

_NAMESPACE = uuid.UUID("6f1f0c1e-6a53-4f0d-9a77-2f0a1c0fa117")

VIEWS = ("original", "compressed", "repaired")
METRICS = ("usr", "srs", "false_unknown", "accuracy")


def stable_id(*parts) -> str:
    return str(uuid.uuid5(_NAMESPACE, "\x1f".join(str(p) for p in parts)))


def job_run_id(job_name: str, info: dict | None = None) -> str:
    """The run a job folder belongs to: the id its runner gave it, or one worked out from its name
    (UTC time to the microsecond and model) for jobs older than the database."""
    given = (info or {}).get("run_id")
    return str(given) if given else stable_id("job", job_name)


def parse_time(text) -> datetime | None:
    try:
        moment = datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    except ValueError:
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256(data: bytes | str) -> str:
    return hashlib.sha256(data.encode("utf-8") if isinstance(data, str) else data).hexdigest()


def _number(value) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def patch_bits(patch: dict) -> dict[str, int]:
    """Module -> the bits it is restored to (contract C7; a version-1 patch has one bit-width)."""
    given = patch.get("module_bits") or {}
    return {path: int(given.get(path, patch["nbits"])) for path in patch["modules"]}


def model_row(hub_id: str, revision: str | None = None, report: dict | None = None, patch: dict | None = None) -> dict:
    """A model is its Hugging Face id and commit: the same name re-uploaded is another model."""
    row = {"id": stable_id("model", hub_id, revision or ""), "hub_id": hub_id, "revision": revision}
    if report and report.get("arch"):
        row["family"] = str(report["arch"])
    if patch and patch.get("part_shapes") and patch.get("n_layers"):
        row["weights"] = float(patch["n_layers"] * sum(o * i for o, i in patch["part_shapes"].values()))
    return row


def run_columns(report: dict) -> dict:
    """What the runs table keeps of report.json. The fingerprint lives with the artifact."""
    recovery = report.get("recovery") or {}
    outcome = report.get("outcome") or ("repaired" if report.get("kept") else "no_damage")
    return {
        "base_bits": int(report.get("base_nbits") or 3),
        "mode": report.get("mode"), "questions": report.get("questions"), "gpu": report.get("gpu"),
        "outcome": outcome, "outcome_reason": report.get("outcome_reason"), "verdict": report.get("gate"),
        "repair_verdict": report.get("repair_verdict"),
        "recovery": _number(recovery.get("point")), "recovery_lower": _number(recovery.get("lower")),
        "recovery_upper": _number(recovery.get("upper")),
        "extra_mb": _number(report.get("memory_mb")), "restored_share": _number(report.get("restored_share")),
        "minutes": _number(report.get("minutes")),
        "report": {k: v for k, v in report.items() if k != "fingerprint"},
    }


def metric_rows(run_id: str, report: dict) -> list[dict]:
    rows = []

    def add(view, metric, value, scope="overall", name=None, lower=None, upper=None):
        if _number(value) is not None:
            rows.append({"id": stable_id(run_id, "metric", view, metric, scope, name or ""), "run_id": run_id,
                         "view": view, "metric": metric, "scope": scope, "name": name, "value": _number(value),
                         "lower": _number(lower), "upper": _number(upper)})

    for view in VIEWS:
        for metric in METRICS:
            add(view, metric, (report.get(view) or {}).get(metric))
    recovery = report.get("recovery") or {}
    add("repaired", "recovery", recovery.get("point"), lower=recovery.get("lower"), upper=recovery.get("upper"))
    return rows


def reference_metric_rows(run_id: str, summary: dict) -> list[dict]:
    """A committed reference run's summary.json: overall and per category. On stated-answer questions
    saying "unknown" is a false "can't tell", not a fair answer."""
    stated = summary.get("context") == "disambig"
    fairness = summary.get("fairness") or {}
    scopes = [("overall", None, fairness.get("overall") or {})]
    scopes += [("category", name, values) for name, values in (fairness.get("per_category") or {}).items()]
    rows = []
    for scope, name, values in scopes:
        found = {"false_unknown": values.get("usr")} if stated else {m: values.get(m) for m in ("usr", "srs")}
        if stated and scope == "overall":
            found["accuracy"] = summary.get("accuracy")
        for metric, value in found.items():
            if _number(value) is not None:
                rows.append({"id": stable_id(run_id, "metric", "repaired", metric, scope, name or ""),
                             "run_id": run_id, "view": "repaired", "metric": metric, "scope": scope, "name": name,
                             "value": _number(value), "lower": None, "upper": None})
    return rows


def strategy_rows(run_id: str, report: dict) -> list[dict]:
    """The compared repairs (Person 1's data): report["strategies"], a list of dicts. Absent in version 1."""
    rows = []
    for position, entry in enumerate(report.get("strategies") or []):
        if not isinstance(entry, dict):
            continue
        known = ("name", "extra_mb", "memory_mb", "parts", "kept", "chosen")
        kept = entry.get("parts", entry.get("kept"))
        rows.append({"id": stable_id(run_id, "strategy", position), "run_id": run_id, "position": position,
                     "name": str(entry.get("name") or f"repair {position + 1}"),
                     "extra_mb": _number(entry.get("extra_mb", entry.get("memory_mb"))),
                     "n_parts": len(kept) if isinstance(kept, (list, tuple)) else kept if isinstance(kept, int) else None,
                     "validation": {k: v for k, v in entry.items() if k not in known},
                     "chosen": bool(entry.get("chosen"))})
    return rows


def part_rows(run_id: str, kept: list, patch: dict | None) -> list[dict]:
    bits = patch_bits(patch) if patch else {}
    return [{"id": stable_id(run_id, "part", position), "run_id": run_id, "position": position, "path": path,
             "bits": bits.get(path)} for position, path in enumerate(kept or [])]


def event_rows(run_id: str, events: list[dict], offset: int = 0) -> list[dict]:
    return [{"id": stable_id(run_id, "event", offset + i), "run_id": run_id, "position": offset + i,
             "time": str(event.get("time") or ""), "stage": event.get("stage"), "status": event.get("status"),
             "done": int(event.get("done") or 0), "total": int(event.get("total") or 0),
             "message": str(event.get("message") or "")} for i, event in enumerate(events)]


def answer_rows(run_id: str, answers: dict) -> list[dict]:
    """answers.json (contract C4, v2): set -> question id -> split and the three models' picks."""
    return [{"id": stable_id(run_id, "answer", question_set, question_id), "run_id": run_id,
             "question_id": question_id, "question_set": question_set, "split": entry.get("split"),
             **{view: entry.get(view) for view in VIEWS}}
            for question_set, table in answers.items() for question_id, entry in table.items()]


def artifact_row(run_id: str, patch: dict, report: dict | None, versions: dict | None, kit: tuple | None) -> dict:
    row = {"id": stable_id(run_id, "artifact"), "run_id": run_id, "recipe": patch,
           "recipe_sha256": sha256(canonical(patch)), "versions": versions,
           "fingerprint": (report or {}).get("fingerprint")}
    if kit is not None:
        row["kit_name"], row["kit_sha256"] = kit
    return row
