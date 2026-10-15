"""A stand-in for the fast-repair pipeline: writes the same files (sprint contract C4) without a GPU.

Two uses:

* building and testing the Studio's "Bring your own model" screen before the real pipeline
  (src/product/fast_repair.py, Person 1) exists, or on a laptop;
* a presentation backup: --replay plays back a recorded real run's events.jsonl, report.json and
  patch.json, faster, so the screen can be demonstrated without a GPU.

A synthetic run's report says "synthetic": true and a replayed one says "replayed_from", and the
screen shows a banner for both, so made-up numbers can never pass for a measurement.

    python -m src.product.byom_fake --model Qwen/Qwen2.5-3B-Instruct --mode fast --out results/byom/x
    python -m src.product.byom_fake ... --replay results/reference/fast_repair_qwen --speed 20
    python -m src.product.byom_fake ... --fail-at scan          # test the error display
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

from src.product.byom import STAGES, check_mode, check_model_id, format_duration

# Seconds each stage takes on a T4 for a ~3-4B model (the sprint's estimates), per mode.
DURATIONS = {
    "fast": {"download": 60, "quantize": 130, "score_original": 90, "score_3bit": 160, "find_damage": 5,
             "scan": 420, "verify": 240, "report": 5},
    "exact": {"download": 60, "quantize": 130, "score_original": 90, "score_3bit": 160, "find_damage": 5,
              "scan": 1500, "verify": 480, "report": 5},
}
PROGRESS_STEPS = 12

# Architecture guesses for synthetic runs: (arch, layers, part types in display order).
ARCHS = {
    "phi": ("phi3", 32, ["self_attn.qkv_proj", "self_attn.o_proj", "mlp.gate_up_proj", "mlp.down_proj"]),
    "qwen": ("qwen2", 36, ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                           "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]),
    "llama": ("llama", 28, ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                            "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]),
}


def guess_arch(model_id: str) -> tuple[str, int, list[str]]:
    name = model_id.lower()
    for key, spec in ARCHS.items():
        if key in name:
            return spec
    return ARCHS["llama"]


class Writer:
    def __init__(self, out: Path, speed: float):
        self.out, self.speed = out, speed
        self.out.mkdir(parents=True, exist_ok=True)
        self.path = out / "events.jsonl"
        self.path.write_text("", encoding="utf-8")
        self.real_start = time.monotonic()
        self.offset = 0.0  # seconds of pipeline time elapsed

    def wait_until(self, offset: float) -> None:
        """Advance pipeline time to `offset`, sleeping offset/speed in real time."""
        self.offset = offset
        delay = self.real_start + offset / self.speed - time.monotonic()
        if delay > 0:
            time.sleep(delay)

    def event(self, stage: str, status: str, done: int = 0, total: int = 0, message: str = "") -> None:
        # Real wall-clock time, so the screen's running timers agree with the event times. A sped-up
        # run therefore shows short stage times; a replay says what the recorded run took.
        record = {"time": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                  "stage": stage, "status": status, "done": done, "total": total, "message": message}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")


def synthetic_report(model_id: str, mode: str, questions: int) -> tuple[dict, dict]:
    arch, n_layers, parts = guess_arch(model_id)
    rng = random.Random(int(hashlib.sha256(model_id.encode()).hexdigest(), 16) % 2**32)
    attention = [p for p in parts if p.startswith("self_attn")]
    kept = []
    while len(kept) < 5:
        path = f"model.layers.{rng.randrange(n_layers)}.{rng.choice(attention)}"
        if path not in kept:
            kept.append(path)
    original = {"usr": 0.84, "srs": 0.117, "false_unknown": 0.073, "accuracy": 0.915}
    compressed = {"usr": 0.365, "srs": 0.373, "false_unknown": 0.038, "accuracy": 0.808}
    repaired = {"usr": 0.773, "srs": 0.142, "false_unknown": 0.109, "accuracy": 0.837}
    report = {
        "synthetic": True, "model_id": model_id, "arch": arch, "mode": mode, "questions": questions,
        "gpu": "none (stand-in pipeline)", "minutes": sum(DURATIONS[mode].values()) / 60,
        "safe": int(questions * 0.84), "flipped": int(questions * 0.28),
        "original": original, "compressed": compressed, "repaired": repaired,
        "recovery": {"point": 0.859, "lower": 0.844, "upper": 0.873}, "kept": kept, "memory_mb": 81.0,
        "gate": "FAIL", "gate_failures": ["'unknown' answers fell by 6.7 points (limit 2.0)",
                                          "stereotyped answers rose by 2.5 points (limit 1.0)"],
        "patch": "patch.json", "n_layers": n_layers, "part_types": parts,
    }
    patch = {"name": f"fast_repair_{arch}", "modules": kept, "nbits": 8,
             "source": {"kind": "search", "synthetic": True, "mode": mode},
             "base": {"method": "hqq", "nbits": 3, "group_size": 64, "compute_dtype": "fp16"},
             "model_id": model_id}
    return report, patch


def run_synthetic(args, writer: Writer) -> None:
    totals = {"score_original": args.questions * 2, "score_3bit": args.questions * 2, "verify": args.questions * 2,
              "scan": 70 if args.mode == "fast" else 140}
    messages = {"download": "fetching the weights", "quantize": "HQQ 3-bit, group 64",
                "find_damage": f"{int(args.questions * 0.28)} biased answers found"}
    offset = 0.0
    for stage, _ in STAGES:
        writer.wait_until(offset)
        writer.event(stage, "started", 0, totals.get(stage, 0), messages.get(stage, ""))
        duration = DURATIONS[args.mode][stage]
        steps = PROGRESS_STEPS if stage in totals else 1
        for step in range(1, steps + 1):
            writer.wait_until(offset + duration * step / steps)
            if args.fail_at == stage and step >= steps // 2:
                writer.event(stage, "failed", 0, totals.get(stage, 0), "simulated failure (--fail-at)")
                raise SystemExit(1)
            if stage in totals and step < steps:
                total = totals[stage]
                message = f"round {1 + step * 5 // steps} of 5" if stage == "scan" else ""
                writer.event(stage, "progress", total * step // steps, total, message)
        offset += duration
        if stage == "report":
            report, patch = synthetic_report(args.model, args.mode, args.questions)
            (writer.out / "patch.json").write_text(json.dumps(patch, indent=2), encoding="utf-8")
            (writer.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        writer.event(stage, "done", totals.get(stage, 0), totals.get(stage, 0), messages.get(stage, ""))


def run_replay(args, writer: Writer) -> None:
    source = Path(args.replay)
    events = [json.loads(line) for line in (source / "events.jsonl").read_text(encoding="utf-8").splitlines()
              if line.strip()]
    if not events:
        raise SystemExit(f"{source}/events.jsonl has no events")
    times = [datetime.fromisoformat(e["time"].replace("Z", "+00:00")) for e in events]
    started = {}
    for event, moment in zip(events, times):
        started.setdefault(event["stage"], moment)
        message = event.get("message", "")
        if event["status"] == "done":
            took = format_duration((moment - started[event["stage"]]).total_seconds())
            message = f"{message} · took {took} in the recorded run" if message else f"took {took} in the recorded run"
        writer.wait_until((moment - times[0]).total_seconds())
        if event["stage"] == "report" and event["status"] == "done":
            report = json.loads((source / "report.json").read_text(encoding="utf-8"))
            report["replayed_from"] = f"{source.name}, recorded {times[0]:%d %b %Y}"
            shutil.copy2(source / "patch.json", writer.out / "patch.json")
            (writer.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        writer.event(event["stage"], event["status"], event.get("done", 0), event.get("total", 0), message)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", default="fast")
    parser.add_argument("--questions", type=int, default=1000)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--speed", type=float, default=30.0, help="how many times faster than real time")
    parser.add_argument("--replay", type=Path, help="a recorded run's folder to play back")
    parser.add_argument("--fail-at", choices=[s for s, _ in STAGES], help="simulate a failure in this stage")
    args = parser.parse_args()
    args.model, args.mode = check_model_id(args.model), check_mode(args.mode)
    writer = Writer(args.out, args.speed)
    print(f"Stand-in pipeline for {args.model} ({args.mode} mode), {args.speed:g}x speed", flush=True)
    (run_replay if args.replay else run_synthetic)(args, writer)
    print("Done", flush=True)


if __name__ == "__main__":
    main()
