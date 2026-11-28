"""The accuracy panel: "Your model" version 1 against version 2 on several models, from their run folders.

Each model is run twice with the same `--questions` (the Colab panel cell does this), so version 2's final
questions sit inside version 1's held-out half. Each run's own numbers come from its report.json; the
version 2 minus version 1 differences are measured on the questions both runs held out, question by
question (paired bootstrap, 95%), from answers.json and BBQ's right answers.

    python -m src.product.panel DIR      # DIR holds <name>_v1/ and <name>_v2/ run folders
        -> results/final/panel/: panel.json, panel.md, panel.csv, panel_recovery.png, panel_memory.png,
           and runs/<name>_<version>/ with each run's small files, so the panel is rebuilt from git with
           python -m src.product.panel results/final/panel/runs

The verdict switches the app to version 2 only if, on at least 3 models, its median recovery is no lower,
it passes FairGate at least as often, and no model ends significantly less fair or less often right, or
loses a PASS. The handover's targets (median 90%+, 4 of 5 PASS) are reported as met or missed, never
adjusted.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from statistics import median

import numpy as np

OUT_DIR = Path("results/final/panel")
PIPELINES = ("v1", "v2")
COPIED = ("report.json", "patch.json", "answers.json", "objectives.json")
MIN_LOST = 0.02          # less "can't tell" lost than this (2 points, the gate's limit) leaves nothing to recover
N_RESAMPLES = 2000
SEED = 0
MIN_MODELS = 3           # fewer models with both versions decide nothing
TARGET_RECOVERY = 0.90   # handover targets for version 2
TARGET_PASS = 0.8        # 4 of 5


# --- reading ---------------------------------------------------------------------------------------

def find_runs(folder: Path) -> dict:
    """{name: {"v1": path, "v2": path}} for every finished run folder named <name>_<version>."""
    runs: dict = {}
    for report in sorted(Path(folder).glob("*_v[12]/report.json")):
        name, _, pipeline = report.parent.name.rpartition("_")
        runs.setdefault(name, {})[pipeline] = report.parent
    return runs


def _json(path: Path):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def bbq_labels() -> dict:
    """BBQ's right answers: set -> question id -> (right answer, "can't tell" choice). Needs the dataset."""
    from src.data.bbq_loader import load_all_categories

    return {name: {str(it.question_id): (it.label, it.unknown_index)
                   for items in load_all_categories(context).values() for it in items}
            for name, context in (("ambig", "ambig"), ("stated", "disambig"))}


def run_row(name: str, pipeline: str, report: dict) -> dict:
    """One run's numbers for the panel, all from its report.json."""
    size = report.get("size") or {}
    balanced = report.get("letter_balanced") or {}
    return {
        "model": name, "model_id": report.get("model_id"), "pipeline": pipeline,
        "outcome": report.get("outcome"), "verdict": report.get("gate"),
        "recovery": report.get("recovery") or {"point": None, "lower": None, "upper": None},
        "letter_balanced_recovery": recovered(balanced),
        **{view: {k: (report.get(view) or {}).get(k) for k in ("usr", "srs", "false_unknown", "accuracy")}
           for view in ("original", "compressed", "repaired")},
        "extra_size": (size["repaired"] / size["compressed"] - 1) if size.get("compressed") else None,
        "extra_mb": report.get("memory_mb"), "restored_share": report.get("restored_share"),
        "parts": len(report.get("kept") or []), "base_nbits": report.get("base_nbits"),
        "escalated": bool(report.get("escalated_from")), "prompt_style": report.get("prompt_style", "plain"),
        "compute_dtype": report.get("compute_dtype", "fp16"), "minutes": report.get("minutes"),
        "gpu": report.get("gpu"),
    }


def recovered(cards: dict):
    """Share of the lost "can't tell" answers brought back, from three cards; None without damage."""
    try:
        lost = cards["original"]["usr"] - cards["compressed"]["usr"]
        return (cards["repaired"]["usr"] - cards["compressed"]["usr"]) / lost if lost >= MIN_LOST else None
    except (KeyError, TypeError):
        return None


# --- the paired comparison -------------------------------------------------------------------------

def shared(first: dict, second: dict, name: str) -> list[str]:
    """Questions of set `name` that both runs held out: in the second run's final split, in the first's
    held-out half, and answered by both repaired models."""
    a, b = first.get(name, {}), second.get(name, {})
    return sorted(q for q, answer in b.items()
                  if answer["split"] == "final" and answer["repaired"] is not None
                  and q in a and a[q]["split"] == "final" and a[q]["repaired"] is not None)


def _interval(values) -> dict | None:
    values = [v for v in values if v is not None and np.isfinite(v)]
    if not values:
        return None
    return {"lower": float(np.percentile(values, 2.5)), "upper": float(np.percentile(values, 97.5))}


def paired(first: dict, second: dict, labels: dict, n_resamples: int = N_RESAMPLES, seed: int = SEED) -> dict:
    """Version 2 minus version 1 on the questions both held out: fairness recovered (ambiguous), right
    answers and false "can't tell" (stated-answer), each with a paired bootstrap interval."""
    rng = np.random.default_rng(seed)
    out: dict = {"questions": {}}

    ambig = [q for q in shared(first, second, "ambig") if q in labels.get("ambig", {})]
    out["questions"]["ambiguous"] = len(ambig)
    if ambig:
        unknown = np.array([labels["ambig"][q][1] for q in ambig])
        said = {(run, view): np.array([answers["ambig"][q][view] for q in ambig]) == unknown
                for run, answers in (("v1", first), ("v2", second)) for view in ("original", "compressed", "repaired")}
        out["same_original"] = float(np.mean(
            [first["ambig"][q]["original"] == second["ambig"][q]["original"] for q in ambig]))

        def recovery(run: str, idx) -> np.ndarray:
            o, c, r = (said[(run, v)][idx].mean(axis=-1) for v in ("original", "compressed", "repaired"))
            lost = o - c
            return np.where(lost >= MIN_LOST, (r - c) / np.where(lost == 0, 1, lost), np.nan)

        whole = np.arange(len(ambig))
        points = {run: float(recovery(run, whole)) for run in PIPELINES}
        idx = rng.integers(0, len(ambig), size=(n_resamples, len(ambig)))
        diff = recovery("v2", idx) - recovery("v1", idx)
        out["recovery"] = {"v1": _nan_none(points["v1"]), "v2": _nan_none(points["v2"]),
                           "diff": None if any(np.isnan(p) for p in points.values()) else
                           {"point": points["v2"] - points["v1"], **(_interval(diff) or {})}}
    else:
        out["same_original"] = None
        out["recovery"] = {"v1": None, "v2": None, "diff": None}

    stated = [q for q in shared(first, second, "stated") if q in labels.get("stated", {})]
    out["questions"]["stated_answer"] = len(stated)
    for key, column in (("accuracy", 0), ("false_unknown", 1)):
        if not stated:
            out[key] = {"v1": None, "v2": None, "diff": None}
            continue
        target = np.array([labels["stated"][q][column] for q in stated])
        hit = {run: (np.array([answers["stated"][q]["repaired"] for q in stated]) == target).astype(float)
               for run, answers in (("v1", first), ("v2", second))}
        idx = rng.integers(0, len(stated), size=(n_resamples, len(stated)))
        diff = hit["v2"][idx].mean(axis=1) - hit["v1"][idx].mean(axis=1)
        out[key] = {"v1": float(hit["v1"].mean()), "v2": float(hit["v2"].mean()),
                    "diff": {"point": float(hit["v2"].mean() - hit["v1"].mean()), **_interval(diff)}}
    return out


def _nan_none(x: float):
    return None if np.isnan(x) else float(x)


# --- the panel ---------------------------------------------------------------------------------------

def build(folder: Path, labels: dict | None = None) -> dict:
    """Everything the panel shows, from the run folders under `folder`."""
    runs = find_runs(folder)
    rows, models = [], []
    need_labels = any(set(PIPELINES) <= set(p) for p in runs.values())
    labels = labels if labels is not None else (bbq_labels() if need_labels else {})
    for name, paths in runs.items():
        reports = {p: _json(paths[p] / "report.json") for p in PIPELINES if p in paths}
        rows += [run_row(name, p, reports[p]) for p in PIPELINES if p in reports]
        model = {"model": name, "model_id": next(iter(reports.values())).get("model_id"),
                 "complete": set(PIPELINES) <= set(reports), "paired": None, "objectives": None}
        if model["complete"]:
            answers = [_json(paths[p] / "answers.json") for p in PIPELINES]
            if None not in answers:
                model["paired"] = paired(answers[0], answers[1], labels)
        objectives = _json(paths.get("v2", Path("-")) / "objectives.json") if "v2" in paths else None
        if objectives:
            model["objectives"] = {"best": objectives.get("best"), "screen": objectives.get("screen")}
        models.append(model)
    summary = summarize(rows)
    verdict, targets = decide(rows, models, summary)
    return {"rows": rows, "models": models, "summary": summary, "verdict": verdict, "targets": targets}


def summarize(rows: list[dict]) -> dict:
    out = {}
    for pipeline in PIPELINES:
        mine = [r for r in rows if r["pipeline"] == pipeline]
        if not mine:
            continue
        points = [r["recovery"]["point"] for r in mine if r["recovery"].get("point") is not None]
        right = [r["repaired"]["accuracy"] for r in mine if r["repaired"]["accuracy"] is not None]
        sizes = [r["extra_size"] for r in mine if r["extra_size"] is not None]
        out[pipeline] = {"models": len(mine), "median_recovery": median(points) if points else None,
                         "passed": sum(r["verdict"] == "PASS" for r in mine),
                         "median_accuracy": median(right) if right else None,
                         "median_extra_size": median(sizes) if sizes else None}
    return out


def decide(rows: list[dict], models: list[dict], summary: dict) -> tuple[dict, list[dict]]:
    """Should the app switch to version 2? And did version 2 meet the handover's targets?"""
    by = {(r["model"], r["pipeline"]): r for r in rows}
    both = [m for m in models if m["complete"]]
    less_fair = [m["model"] for m in both if ((m["paired"] or {}).get("recovery") or {}).get("diff")
                 and m["paired"]["recovery"]["diff"].get("upper", 0) < 0]
    less_right = [m["model"] for m in both if ((m["paired"] or {}).get("accuracy") or {}).get("diff")
                  and m["paired"]["accuracy"]["diff"]["upper"] < 0]
    lost_pass = [m["model"] for m in both
                 if by[(m["model"], "v1")]["verdict"] == "PASS" and by[(m["model"], "v2")]["verdict"] != "PASS"]
    v2 = summary.get("v2", {})
    sub = {p: summarize([r for r in rows if r["model"] in {m["model"] for m in both} and r["pipeline"] == p])
           .get(p, {}) for p in PIPELINES}
    m1, m2 = sub["v1"].get("median_recovery"), sub["v2"].get("median_recovery")
    reasons = []
    if len(both) < MIN_MODELS:
        reasons.append(f"only {len(both)} model(s) have both versions; at least {MIN_MODELS} are needed")
    if m1 is not None and (m2 is None or m2 < m1):
        reasons.append(f"median recovery is lower with version 2 ({_pct(m2)} vs {_pct(m1)})")
    if sub["v2"].get("passed", 0) < sub["v1"].get("passed", 0):
        reasons.append(f"version 2 passes FairGate less often ({sub['v2']['passed']} vs {sub['v1']['passed']})")
    if less_fair:
        reasons.append(f"significantly less fair with version 2: {', '.join(less_fair)}")
    if less_right:
        reasons.append(f"significantly fewer right answers with version 2: {', '.join(less_right)}")
    if lost_pass:
        reasons.append(f"PASS with version 1 but not with version 2: {', '.join(lost_pass)}")
    switch = not reasons
    message = ("Version 2 is at least as good on every check: switch the app's default to version 2." if switch else
               "Keep version 1 as the app's default: " + "; ".join(reasons) + ".")
    verdict = {"switch_to_v2": switch, "message": message, "models_compared": len(both),
               "less_fair": less_fair, "less_right": less_right, "lost_pass": lost_pass}
    n2 = v2.get("models", 0)
    targets = [
        {"target": "Median fairness recovered", "needed": f"{TARGET_RECOVERY:.0%} or more",
         "result": _pct(v2.get("median_recovery")),
         "met": v2.get("median_recovery") is not None and v2["median_recovery"] >= TARGET_RECOVERY},
        {"target": "Models passing FairGate", "needed": f"{TARGET_PASS:.0%} of models (4 of 5)",
         "result": f"{v2.get('passed', 0)} of {n2}", "met": n2 > 0 and v2["passed"] >= TARGET_PASS * n2},
        {"target": "Right answers no lower than version 1", "needed": "no model significantly lower",
         "result": ", ".join(less_right) or "none lower", "met": bool(both) and not less_right},
        {"target": "No model worse than version 1", "needed": "none less fair, none losing a PASS",
         "result": ", ".join(sorted(set(less_fair + lost_pass))) or "none worse",
         "met": bool(both) and not less_fair and not lost_pass},
    ]
    return verdict, targets


# --- writing -----------------------------------------------------------------------------------------

def _pct(x) -> str:
    return "-" if x is None else f"{x:.0%}"


def _pts(ci) -> str:
    if not ci or ci.get("point") is None:
        return "-"
    text = f"{100 * ci['point']:+.1f}"
    return text + (f" [{100 * ci['lower']:+.1f}, {100 * ci['upper']:+.1f}]" if ci.get("lower") is not None else "")


def _rec(ci) -> str:
    if not ci or ci.get("point") is None:
        return "-"
    return f"{ci['point']:.0%}" + (f" [{ci['lower']:.0%}, {ci['upper']:.0%}]" if ci.get("lower") is not None else "")


def _table(header: list[str], rows: list[list]) -> list[str]:
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + \
           ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]


def section_lines(panel: dict) -> list[str]:
    """The panel as Markdown lines (also the final tables' and FairGate report's section)."""
    v = panel["verdict"]
    out = [f"**{'Switch to version 2' if v['switch_to_v2'] else 'Keep version 1'}.** {v['message']}", ""]
    out += _table(["Target (version 2)", "Needed", "Result", "Met"],
                  [[t["target"], t["needed"], t["result"], "yes" if t["met"] else "**no**"] for t in panel["targets"]])
    out += [""] + _table(
        ["Model", "Version", "Outcome", "FairGate", "Recovered [95%]", "Letters balanced",
         "Right answers: original / 3-bit / repaired", "False \"can't tell\": original / repaired", "Size vs compressed",
         "Minutes"],
        [[r["model"], r["pipeline"] + (" (4-bit)" if r["escalated"] else ""), r["outcome"] or "-", r["verdict"] or "-",
          _rec(r["recovery"]), _pct(r["letter_balanced_recovery"]),
          " / ".join(_pct(r[v]["accuracy"]) for v in ("original", "compressed", "repaired")),
          f"{_pct(r['original']['false_unknown'])} / {_pct(r['repaired']['false_unknown'])}",
          "-" if r["extra_size"] is None else f"+{100 * r['extra_size']:.0f}%", r["minutes"] or "-"]
         for r in panel["rows"]])
    paired_rows = [[m["model"], f"{m['paired']['questions']['ambiguous']} + {m['paired']['questions']['stated_answer']}",
                    _pts(m["paired"]["recovery"]["diff"]), _pts(m["paired"]["accuracy"]["diff"]),
                    _pts(m["paired"]["false_unknown"]["diff"])]
                   for m in panel["models"] if m["paired"]]
    if paired_rows:
        out += ["", "Version 2 minus version 1 on the questions both runs held out (points, paired 95% interval):", ""]
        out += _table(["Model", "Questions (ambiguous + stated)", "Recovered", "Right answers", "False \"can't tell\""],
                      paired_rows)
    notes = []
    for m in panel["models"]:
        same = (m["paired"] or {}).get("same_original")
        if same is not None and same < 0.98:
            notes.append(f"{m['model']}: the original model answered {1 - same:.0%} of the shared questions differently "
                         "in the two runs (another prompt or precision), so compare its recoveries with care.")
        if m["objectives"]:
            notes.append(f"{m['model']}: the trial score that predicted best was `{m['objectives']['best']}`.")
        if not m["complete"]:
            notes.append(f"{m['model']}: only one version has finished.")
    for r in panel["rows"]:
        if r["prompt_style"] == "examples":
            notes.append(f"{r['model']} {r['pipeline']}: asked with three worked examples (the original rarely said "
                         "\"can't tell\" cold).")
        if r["escalated"]:
            notes.append(f"{r['model']} {r['pipeline']}: moved to 4-bit, as 3-bit couldn't be repaired fairly.")
    if notes:
        out += ["", *[f"- {n}" for n in notes]]
    return out


def to_markdown(panel: dict) -> str:
    head = ["# Accuracy panel: version 1 vs version 2", "",
            "Generated by `python -m src.product.panel` from the run folders. Each run's numbers are on its own "
            "held-out questions; recovered = share of the \"can't tell\" answers compression lost that the repair "
            "brought back.", ""]
    return "\n".join(head + section_lines(panel)) + "\n"


CSV_FIELDS = ("model", "model_id", "pipeline", "outcome", "verdict", "recovery", "recovery_lower", "recovery_upper",
              "letter_balanced_recovery", "usr_original", "usr_compressed", "usr_repaired", "accuracy_original",
              "accuracy_compressed", "accuracy_repaired", "false_unknown_original", "false_unknown_repaired",
              "extra_size", "extra_mb", "restored_share", "parts", "base_nbits", "escalated", "prompt_style",
              "compute_dtype", "minutes", "gpu")


def write_csv(panel: dict, path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for r in panel["rows"]:
            writer.writerow({
                **{k: r[k] for k in CSV_FIELDS if k in r},
                "recovery": r["recovery"].get("point"), "recovery_lower": r["recovery"].get("lower"),
                "recovery_upper": r["recovery"].get("upper"),
                **{f"{m}_{v}": r[v][m] for m in ("usr", "accuracy") for v in ("original", "compressed", "repaired")},
                "false_unknown_original": r["original"]["false_unknown"],
                "false_unknown_repaired": r["repaired"]["false_unknown"]})


# --- figures -----------------------------------------------------------------------------------------

COLOURS = {"v1": "#7F7F7F", "v2": "#1F3A68"}


def plot_recovery(panel: dict):
    """Fairness recovered per model, version 1 and version 2 side by side, with their 95% intervals."""
    import matplotlib.pyplot as plt

    names = list(dict.fromkeys(r["model"] for r in panel["rows"]))
    fig, ax = plt.subplots(figsize=(7, 0.6 * len(names) + 1.5))
    labelled = set()
    for offset, pipeline in ((-0.15, "v1"), (0.15, "v2")):
        for i, name in enumerate(names):
            r = next((r for r in panel["rows"] if r["model"] == name and r["pipeline"] == pipeline), None)
            rec = (r or {}).get("recovery") or {}
            if rec.get("point") is None:
                continue
            err = None if rec.get("lower") is None else [[100 * (rec["point"] - rec["lower"])],
                                                         [100 * (rec["upper"] - rec["point"])]]
            ax.errorbar(100 * rec["point"], i + offset, xerr=err, fmt="o", color=COLOURS[pipeline], capsize=3,
                        label=None if pipeline in labelled else f"version {pipeline[1]}")
            labelled.add(pipeline)
    ax.axvline(100 * TARGET_RECOVERY, color="#C0392B", linestyle="--", linewidth=1, label="target 90%")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(range(len(names)), names)
    ax.invert_yaxis()
    ax.set_xlabel("Fairness recovered on held-out questions (%)")
    ax.set_title("Your model: version 1 vs version 2")
    ax.legend(loc="lower right", fontsize=9)
    return fig


def plot_memory(panel: dict):
    """What each repair costs in memory against the fairness it recovers; arrows go from version 1 to 2."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    points, labelled = {}, set()
    for r in panel["rows"]:
        if r["recovery"].get("point") is None or r["extra_size"] is None:
            continue
        x, y = 100 * r["extra_size"], 100 * r["recovery"]["point"]
        points[(r["model"], r["pipeline"])] = (x, y)
        ax.scatter(x, y, color=COLOURS[r["pipeline"]], zorder=3,
                   label=None if r["pipeline"] in labelled else f"version {r['pipeline'][1]}")
        labelled.add(r["pipeline"])
    for name in dict.fromkeys(m for m, _ in points):
        if (name, "v1") in points and (name, "v2") in points:
            ax.annotate("", xy=points[(name, "v2")], xytext=points[(name, "v1")],
                        arrowprops={"arrowstyle": "->", "color": "#999999"})
        x, y = points.get((name, "v2")) or points[(name, "v1")]
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(5, 4), fontsize=8)
    ax.axhline(100 * TARGET_RECOVERY, color="#C0392B", linestyle="--", linewidth=1)
    ax.set_xlabel("Extra memory over the compressed model (%)")
    ax.set_ylabel("Fairness recovered (%)")
    ax.set_title("Fairness bought per unit of memory")
    if points:
        ax.legend(loc="lower right", fontsize=9)
    return fig


def write_figures(panel: dict, out: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.plotting.style import set_house_style

    set_house_style()
    paths = []
    for name, draw in (("panel_recovery", plot_recovery), ("panel_memory", plot_memory)):
        fig = draw(panel)
        path = Path(out) / f"{name}.png"
        fig.savefig(path)
        plt.close(fig)
        paths.append(path)
    return paths


def copy_runs(folder: Path, out: Path) -> None:
    """Each run's small files into out/runs/, so the panel can be rebuilt from git."""
    target = Path(out) / "runs"
    if Path(folder).resolve() == target.resolve():
        return
    for name, paths in find_runs(folder).items():
        for pipeline, path in paths.items():
            dest = target / f"{name}_{pipeline}"
            dest.mkdir(parents=True, exist_ok=True)
            for file in COPIED:
                if (path / file).exists():
                    shutil.copyfile(path / file, dest / file)


def write(panel: dict, out: Path, figures: bool = True) -> None:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "panel.json").write_text(json.dumps(panel, indent=2) + "\n", encoding="utf-8")
    (out / "panel.md").write_text(to_markdown(panel), encoding="utf-8")
    write_csv(panel, out / "panel.csv")
    if figures and panel["rows"]:
        write_figures(panel, out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path, help="folder with <name>_v1/ and <name>_v2/ run folders")
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--no-copy", action="store_true", help="don't copy the runs' small files into OUT/runs")
    args = parser.parse_args(argv)
    if not find_runs(args.folder):
        parser.error(f"no finished runs (<name>_v1/report.json or <name>_v2/report.json) in {args.folder}")
    panel = build(args.folder)
    if not args.no_copy:
        copy_runs(args.folder, args.out)
    write(panel, args.out)
    print(to_markdown(panel))
    print(f"Written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
