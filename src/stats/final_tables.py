"""Week 6: every final number in one place, computed from the committed result files.

Each section either has its numbers or says which runbook step it is waiting for, so this is also
the checklist of what is left before results are frozen. The FairGate audit report and the deck
read the same collection, so a number can't differ between them.

    python -m src.stats.final_tables      # -> results/final/tables.md and results/final/final_results.json

Sections: main result, per category, recovery by number of modules, fairness controls, hedging,
concentrated vs spread out, beyond BBQ multiple choice (4-bit, CrowS-Pairs, open-ended), the
llama.cpp transfer test, label-free monitoring, robustness, per-group, headline windows.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from src.data.safe_set import read_results_file

from .bbq_bias import accuracy, bias_score_ambig, false_unknown_rate
from .metrics import is_unknown, srs, usr
from .recovery import recovery_fraction

REFERENCE_DIR = Path("results/reference")
STATS_DIR = Path("results/stats")
PATCH_PATH = Path("results/patches/search_top5.json")
OUT_DIR = Path("results/final")
CATEGORIES = ("Age", "Gender_identity", "Race_ethnicity", "Religion", "SES")


# --- helpers --------------------------------------------------------------------------------------

def _rows(reference_dir: Path, name: str) -> list | None:
    path = Path(reference_dir) / name / "question_results.jsonl"
    return read_results_file(path) if path.exists() else None


def _json(path: Path) -> dict | list | None:
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _ci(result) -> dict:
    return {"point": result.point, "lower": result.lower, "upper": result.upper}


def _recovery(reference, degraded, repaired) -> dict | None:
    try:
        return _ci(recovery_fraction(reference, degraded, repaired, is_unknown))
    except ValueError:
        return None


def _pending(step: str) -> dict:
    return {"status": "pending", "waiting_for": step}


def _done(**content) -> dict:
    return {"status": "done", **content}


def describe(rows: list) -> dict:
    return {"n": len(rows), "usr": usr(rows), "srs": srs(rows), "bias_score_ambig": bias_score_ambig(rows)}


# --- sections -------------------------------------------------------------------------------------

def main_result(ref: Path) -> dict:
    base, q3, rep = _rows(ref, "baseline_fp16"), _rows(ref, "hqq_3bit"), _rows(ref, "hqq_repaired")
    if not (base and q3 and rep):
        return _pending("Stages 1-5 reference runs")
    rows = {"Original (fp16)": describe(base), "HQQ 3-bit": describe(q3), "3-bit + search patch": describe(rep)}
    out = {"rows": rows, "usr_recovery": _recovery(base, q3, rep)}
    q4, rep4 = _rows(ref, "hqq_4bit"), _rows(ref, "repair_4bit")
    if q4:
        rows["HQQ 4-bit"] = describe(q4)
    if q4 and rep4:
        rows["4-bit + search patch"] = describe(rep4)
        out["usr_recovery_4bit"] = _recovery(base, q4, rep4)
    return _done(**out)


def per_category(ref: Path) -> dict:
    base, q3, rep = _rows(ref, "baseline_fp16"), _rows(ref, "hqq_3bit"), _rows(ref, "hqq_repaired")
    if not (base and q3 and rep):
        return _pending("Stages 1-5 reference runs")
    out = {}
    for category in CATEGORIES:
        pick = [[r for r in rows if r.category == category] for rows in (base, q3, rep)]
        if not all(pick):
            continue
        out[category] = {"usr": [usr(p) for p in pick], "srs": [srs(p) for p in pick],
                         "usr_recovery": _recovery(*pick)}
    return _done(rows=out)


def recovery_by_modules(ref: Path, patch_path: Path = PATCH_PATH) -> dict:
    """The search's first k modules, k = 1..5, each verified on the full benchmark (the prefix runs)."""
    from src.attribution.patch import load_patch
    from src.product.studio_core import overhead_mb, saved_repairs

    if not Path(patch_path).exists():
        return _pending("results/patches/search_top5.json")
    patch = load_patch(patch_path)
    base, q3 = _rows(ref, "baseline_fp16"), _rows(ref, "hqq_3bit")
    saved = saved_repairs(ref)
    rows, missing = [], []
    for k in range(1, len(patch.modules) + 1):
        modules = patch.modules[:k]
        path = saved.get(frozenset(modules))
        if path is None or not (base and q3):
            missing.append(k)
            continue
        summary = _json(path.parent / "summary.json") or {}
        measured = (summary.get("memory") or {}).get("deployed_overhead_bytes")
        rows.append({
            "modules": k,
            "added": modules[-1],
            "pct_weights": patch.prefix(k).pct_params if k < len(patch.modules) else patch.pct_params,
            "memory_mb": measured / 2**20 if measured else overhead_mb(modules),
            "memory_measured": bool(measured),
            "usr_recovery": _recovery(base, q3, read_results_file(path)),
        })
    if not rows:
        return _pending("RUNBOOK Phase 2, Account 3 (prefix runs)")
    return {"status": "done" if not missing else "partial", "rows": rows,
            "missing_prefixes": missing, "waiting_for": "RUNBOOK Phase 2, Account 3" if missing else None}


def controls_section(ref: Path) -> dict:
    from .controls import baselines_block, discover_repairs, hedging_block, verdict, SEARCH_NAME

    base, q3 = _rows(ref, "baseline_fp16"), _rows(ref, "hqq_3bit")
    repairs = discover_repairs(ref)
    if not (base and q3) or len(repairs) < 2:
        return _pending("RUNBOOK Phase 2, Account 2 (baseline repairs)")
    block = {"baselines": baselines_block(base, q3, {n: read_results_file(p / "question_results.jsonl")
                                                     for n, p in repairs.items()})}
    fp_dis, q3_dis, rep_dis = (_rows(ref, n) for n in ("baseline_fp16_disambig", "hqq_3bit_disambig",
                                                        "hqq_repaired_disambig"))
    if fp_dis and q3_dis and rep_dis:
        block["hedging"] = hedging_block(fp_dis, q3_dis, {SEARCH_NAME: rep_dis})
    block["verdict"] = verdict(block)
    return _done(**block)


def hedging_section(ref: Path) -> dict:
    names = ("baseline_fp16_disambig", "hqq_3bit_disambig", "hqq_repaired_disambig")
    runs = {n: _rows(ref, n) for n in names}
    if not all(runs.values()):
        return _pending("RUNBOOK Phase 2, Account 1 (disambiguated runs)")
    labels = {"baseline_fp16_disambig": "Original (fp16)", "hqq_3bit_disambig": "HQQ 3-bit",
              "hqq_repaired_disambig": "3-bit + search patch"}
    return _done(rows={labels[n]: {"n": len(r), "accuracy": accuracy(r), "false_unknown_rate": false_unknown_rate(r)}
                       for n, r in runs.items()})


def dissociation_section(stats: Path) -> dict:
    data = _json(Path(stats) / "dissociation.json")
    return _done(**data) if data else _pending("RUNBOOK Phase 3, step 8")


def crows_section(ref: Path) -> dict:
    rows = {}
    for condition in ("fp16", "hqq_3bit", "hqq_repaired"):
        summary = _json(Path(ref) / f"crows_{condition}" / "summary.json")
        if summary:
            rows[condition] = {"n": summary["n"], "bias_score": summary["bias_score"],
                               "by_bias_type": summary.get("by_bias_type", {})}
    return _done(rows=rows) if len(rows) == 3 else _pending("RUNBOOK Phase 4 (CrowS-Pairs runs)")


def open_ended_section(ref: Path) -> dict:
    rows = {}
    for condition in ("fp16", "hqq_3bit", "hqq_repaired"):
        run_dir = Path(ref) / f"open_ended_{condition}"
        summary = _json(run_dir / "summary.json")
        if not summary:
            continue
        row = {"n": summary["n"], "rates": summary["rates"]}
        check = run_dir / "handcheck.csv"
        if check.exists():
            with check.open(encoding="utf-8") as handle:
                labelled = [r for r in csv.DictReader(handle) if r["human_label"].strip()]
            if labelled:
                row["handcheck"] = {"n": len(labelled),
                                    "agreement": sum(r["auto_label"] == r["human_label"].strip() for r in labelled) / len(labelled)}
        rows[condition] = row
    return _done(rows=rows) if len(rows) == 3 else _pending("RUNBOOK Phase 4 (open-ended runs)")


def llamacpp_section(ref: Path) -> dict:
    f16, q3, fix = (_rows(ref, f"llamacpp_{v}") for v in ("f16", "q3km", "q3km_fairgate"))
    if not (f16 and q3 and fix):
        return _pending("RUNBOOK Week 6 (llama.cpp transfer test, stretch)")
    return _done(rows={"llama.cpp f16": describe(f16), "llama.cpp Q3_K_M": describe(q3),
                       "Q3_K_M + patch at q8_0": describe(fix)},
                 usr_recovery=_recovery(f16, q3, fix))


def monitoring_section(stats: Path) -> dict:
    data = _json(Path(stats) / "flip_predictability.json")
    if not data:
        return _pending("RUNBOOK Phase 2 (flip_predictability, after baseline_fp16_confidence)")
    at_20 = next((c["flips_caught"] for c in data["coverage"] if abs(c["probed"] - 0.2) < 1e-9), None)
    return _done(auroc=data["auroc"], flips_caught_probing_20pct=at_20,
                 low_vs_high_confidence_ratio=data["flip_rate_by_quartile"]["low_vs_high_ratio"])


def robustness_section(stats: Path) -> dict:
    data = _json(Path(stats) / "sensitivity.json")
    if not data:
        return _pending("python -m src.stats.sensitivity")
    return _done(ci_lower_range=data["seeds"]["lower_range"], ci_upper_range=data["seeds"]["upper_range"],
                 search_sample=data["search_sample"])


def groups_section(stats: Path) -> dict:
    data = _json(Path(stats) / "groups.json")
    if not data:
        return _pending("python -m src.stats.groups")
    worse = sorted((g for g in data if g["worse_than_baseline"]),
                   key=lambda g: -g["repaired_minus_baseline_srs"]["point"])
    return _done(n_groups=len(data), n_worse=len(worse),
                 worse=[{"category": g["category"], "group": g["group"], "srs": g["srs"]} for g in worse])


def headline_section(ref: Path) -> dict:
    data = _json(Path(ref) / "headline" / "headline.json")
    if not data:
        return _pending("RUNBOOK Phase 4 (headline windows), then copy headline.json to results/reference/headline/")
    windows = {name: {"kept": w["kept"], "usr_recovery": w["bundle"]["fairness_recovered"]["usr"]["recovery"]}
               for name, w in data.get("windows", {}).items()}
    return _done(windows=windows, overlap=data.get("overlap"), skipped=data.get("skipped", {}))


SECTIONS = {
    "main": ("Main result", main_result, "ref"),
    "per_category": ("Per category", per_category, "ref"),
    "recovery_by_modules": ("Recovery by number of modules", recovery_by_modules, "ref"),
    "controls": ("Is it really fairness? Matched-weight baselines", controls_section, "ref"),
    "hedging": ("Hedging check (disambiguated questions)", hedging_section, "ref"),
    "dissociation": ("Concentrated vs spread out", dissociation_section, "stats"),
    "crows_pairs": ("CrowS-Pairs", crows_section, "ref"),
    "open_ended": ("Open-ended answers", open_ended_section, "ref"),
    "llamacpp": ("llama.cpp transfer", llamacpp_section, "ref"),
    "monitoring": ("Label-free monitoring", monitoring_section, "stats"),
    "robustness": ("Robustness of the 92%", robustness_section, "stats"),
    "groups": ("Per targeted group", groups_section, "stats"),
    "headline": ("Headline: three failure windows", headline_section, "ref"),
}


def collect(reference_dir: Path = REFERENCE_DIR, stats_dir: Path = STATS_DIR, only=None) -> dict:
    out = {}
    for key, (_, fn, source) in SECTIONS.items():
        if only and key not in only:
            continue
        out[key] = fn(reference_dir if source == "ref" else stats_dir)
    return out


# --- markdown -------------------------------------------------------------------------------------

def _p(x) -> str:
    return "-" if x is None else f"{x:.1%}"


def _r(ci) -> str:
    return "-" if not ci else f"{ci['point']:.0%} [{ci['lower']:.0%}, {ci['upper']:.0%}]"


def _table(header: list[str], rows: list[list]) -> list[str]:
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + \
           ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]


def _metric_rows(rows: dict) -> list[list]:
    return [[name, f"{r['n']:,}", _p(r["usr"]), _p(r["srs"]), f"{r['bias_score_ambig']:+.3f}"] for name, r in rows.items()]


def to_markdown(final: dict) -> str:
    lines = ["# Final results", "", "Generated by `python -m src.stats.final_tables` from the committed result files.",
             "Recovery = share of the fairness (\"unknown\" answers) that compression lost and the repair brought "
             "back, with 95% bootstrap CIs.", ""]
    pending = []
    for key, section in final.items():
        title = SECTIONS[key][0]
        lines += [f"## {title}", ""]
        if section["status"] == "pending":
            lines += [f"*Pending: {section['waiting_for']}.*", ""]
            pending.append((title, section["waiting_for"]))
            continue
        lines += _section_md(key, section) + [""]
        if section["status"] == "partial":
            pending.append((title, section["waiting_for"]))
    lines += ["## Still pending", ""]
    lines += [f"- **{t}**: {w}" for t, w in pending] if pending else ["Nothing. Every section has its numbers."]
    return "\n".join(lines) + "\n"


def _section_md(key: str, s: dict) -> list[str]:
    header = ["Model", "n", "USR", "SRS", "BBQ bias score"]
    if key == "main":
        out = _table(header, _metric_rows(s["rows"]))
        out += ["", f"USR recovered by the search patch: **{_r(s['usr_recovery'])}**"]
        if s.get("usr_recovery_4bit"):
            out.append(f"Same patch on the 4-bit model: **{_r(s['usr_recovery_4bit'])}**")
        return out
    if key == "per_category":
        return _table(["Category", "USR fp16 / 3-bit / repaired", "SRS fp16 / 3-bit / repaired", "USR recovered"],
                      [[c, " / ".join(_p(x) for x in r["usr"]), " / ".join(_p(x) for x in r["srs"]), _r(r["usr_recovery"])]
                       for c, r in s["rows"].items()])
    if key == "recovery_by_modules":
        out = _table(["Modules", "Added", "% of weights", "Memory added", "USR recovered"],
                     [[r["modules"], r["added"].replace("model.layers.", "L"), f"{r['pct_weights']:.2f}%",
                       f"{r['memory_mb']:.0f} MB" + ("" if r["memory_measured"] else " (est.)"), _r(r["usr_recovery"])]
                      for r in s["rows"]])
        if s.get("missing_prefixes"):
            out += ["", f"Not run yet: the {', '.join(map(str, s['missing_prefixes']))}-module versions."]
        return out
    if key == "controls":
        out = _table(["Patch", "USR recovered", "Search minus this"],
                     [[n, _r(r["usr_recovery"]), _r(r.get("search_minus_this"))] for n, r in s["baselines"].items()])
        return out + ["", f"**Verdict: {s['verdict']['outcome']}.** {s['verdict']['message']}"]
    if key == "hedging":
        return _table(["Model", "n", "Accuracy", "False \"unknown\""],
                      [[n, f"{r['n']:,}", _p(r["accuracy"]), _p(r["false_unknown_rate"])] for n, r in s["rows"].items()])
    if key == "dissociation":
        conc = s.get("concentration", {})
        out = _table(["Search", "Parts that help", "Top-5 share of the gain"],
                     [[n, f"{c['n_helpful']} / {c['n_candidates']}", _p(c.get("top5_share"))] for n, c in conc.items()])
        cross = s.get("cross_test", {})
        if cross:
            out += [""] + _table(["Patch", "Fairness (USR) recovered", "Accuracy recovered"],
                                 [[n, _r(r.get("usr_recovery")), _r(r.get("accuracy_recovery"))] for n, r in cross.items()])
        return out
    if key == "crows_pairs":
        return _table(["Model", "Pairs", "Prefers the stereotyped sentence"],
                      [[n, r["n"], _p(r["bias_score"])] for n, r in s["rows"].items()]) + ["", "50% = no preference."]
    if key == "open_ended":
        return _table(["Model", "n", "Unknown", "Stereotype", "Anti-stereotype", "Unclear", "Hand-check agreement"],
                      [[n, r["n"], _p(r["rates"].get("unknown")), _p(r["rates"].get("stereotype")),
                        _p(r["rates"].get("anti_stereotype")), _p(r["rates"].get("unclear")),
                        _p(r["handcheck"]["agreement"]) + f" (n={r['handcheck']['n']})" if "handcheck" in r else "-"]
                       for n, r in s["rows"].items()])
    if key == "llamacpp":
        return _table(header, _metric_rows(s["rows"])) + ["", f"USR recovered inside llama.cpp: **{_r(s['usr_recovery'])}**"]
    if key == "monitoring":
        return [f"- Confidence predicts flips: AUROC **{s['auroc']:.2f}**",
                f"- Probing the 20% least-confident questions catches **{_p(s['flips_caught_probing_20pct'])}** of all flips",
                f"- Least-confident quarter flips **{s['low_vs_high_confidence_ratio']:.1f}x** as often as the most confident"]
    if key == "robustness":
        ss = s["search_sample"]
        return [f"- 95% CI bounds across bootstrap seeds: lower {s['ci_lower_range'][0]:.1%}-{s['ci_lower_range'][1]:.1%}, "
                f"upper {s['ci_upper_range'][0]:.1%}-{s['ci_upper_range'][1]:.1%}",
                f"- Flipped questions the search saw: {_p(ss['recovered_in_sample'])} fixed; never saw: "
                f"{_p(ss['recovered_outside_sample'])} fixed (gap {100 * ss['gap']:+.1f} points)"]
    if key == "groups":
        out = [f"**{s['n_worse']} of {s['n_groups']}** targeted groups end up significantly more stereotyped than with "
               "the original model after the repair.", ""]
        return out + _table(["Group", "Category", "SRS fp16", "3-bit", "Repaired"],
                            [[g["group"], g["category"], _p(g["srs"]["baseline"]), _p(g["srs"]["hqq_3bit"]),
                              _p(g["srs"]["repaired"])] for g in s["worse"]])
    if key == "headline":
        out = _table(["Window", "Modules kept", "USR recovered"],
                     [[n, ", ".join(m.replace("model.layers.", "L") for m in w["kept"]), _r(w["usr_recovery"])]
                      for n, w in s["windows"].items()])
        if s.get("overlap"):
            out += ["", f"Overlap verdict: **{s['overlap'].get('verdict', '-')}**"]
        return out
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    final = collect()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "final_results.json").write_text(json.dumps(final, indent=2) + "\n", encoding="utf-8")
    (args.out_dir / "tables.md").write_text(to_markdown(final), encoding="utf-8")
    done = sum(s["status"] == "done" for s in final.values())
    print(f"{done} of {len(final)} sections complete. Written to {args.out_dir / 'tables.md'}")
    for key, section in final.items():
        if section["status"] != "done":
            print(f"  {section['status']:<8} {SECTIONS[key][0]}: {section['waiting_for']}")


if __name__ == "__main__":
    main()
