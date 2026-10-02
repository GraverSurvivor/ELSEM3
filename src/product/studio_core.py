"""FairGate Studio, the logic: everything the demo app shows that needs neither a GPU nor Gradio.

The app itself (`src/product/studio.py`) is a thin Gradio layer over this file plus a backend that
answers questions. Keeping the logic here means it is tested on a laptop like the rest of src/, and
the same screens can run on saved results when no GPU is available (`SavedBackend`).

Three models answer every question in the app:

    original    Phi-3.5-mini at 16-bit, the reference
    compressed  the same model at HQQ 3-bit
    repaired    the 3-bit model with whichever parts the person using the app has restored to 8-bit
"""

from __future__ import annotations

import csv
import html
import random
import re
from dataclasses import dataclass, replace
from pathlib import Path

from src.attribution.patch import MODULE_SHAPES, N_LAYERS, Patch, module_layer, module_params, module_suffix
from src.data.bbq_loader import BBQItem

# The four candidate parts of each layer, in the order the grid draws them (the search's order).
PART_ORDER = ("self_attn.qkv_proj", "self_attn.o_proj", "mlp.gate_up_proj", "mlp.down_proj")
PART_LABELS = {
    "self_attn.qkv_proj": "Attention input",
    "self_attn.o_proj": "Attention output",
    "mlp.gate_up_proj": "MLP up",
    "mlp.down_proj": "MLP down",
}
assert set(PART_ORDER) == set(MODULE_SHAPES)

UNKNOWN_OPTION = "Can't be determined"
VIEWS = ("original", "compressed", "repaired")

# Sizes for the cards. 3-bit and 8-bit HQQ cost 3.75 and 8.50 bits per weight once the per-group
# scale and zero are counted (measured, see CLAUDE.md > Repair state).
TOTAL_PARAMS_GB_FP16 = 7.6
COMPRESSED_GB = 1.70
BITS_3BIT, BITS_8BIT = 3.75, 8.50


# --- module names ---------------------------------------------------------------------------------

def module_path(layer: int, part: str) -> str:
    path = f"model.layers.{layer}.{part}"
    module_params(path)  # validates layer range and part
    return path


def all_paths() -> list[str]:
    return [module_path(layer, part) for part in PART_ORDER for layer in range(N_LAYERS)]


def friendly_name(path: str) -> str:
    """'model.layers.13.self_attn.qkv_proj' -> 'Layer 13 - Attention input'."""
    return f"Layer {module_layer(path)} - {PART_LABELS[module_suffix(path)]}"


def path_from_friendly(name: str) -> str:
    match = re.fullmatch(r"Layer (\d+) - (.+)", name)
    if not match:
        raise ValueError(f"not a part name: {name!r}")
    part = {label: key for key, label in PART_LABELS.items()}.get(match.group(2))
    if part is None:
        raise ValueError(f"not a part name: {name!r}")
    return module_path(int(match.group(1)), part)


def overhead_mb(paths) -> float:
    """Estimated extra memory of restoring `paths` to 8-bit, in MB (MiB). The live app reports the
    measured number instead; this is within ~1-2% of it (search_top5: 69.5 estimated, 70.2 measured)."""
    params = sum(module_params(p) for p in paths)
    return params * (BITS_8BIT - BITS_3BIT) / 8 / 2**20


def pct_weights(paths) -> float:
    total = N_LAYERS * sum(o * i for o, i in MODULE_SHAPES.values())
    return 100 * sum(module_params(p) for p in paths) / total


# --- questions ------------------------------------------------------------------------------------

def make_question(story: str, question: str, person_a: str, person_b: str) -> BBQItem:
    """A question typed into the app, in the benchmark's format: two people plus "can't be
    determined". There is no stereotype label for a typed question, so stereotype_index is -1."""
    fields = {"story": story, "question": question, "first person": person_a, "second person": person_b}
    missing = [name for name, value in fields.items() if not (value or "").strip()]
    if missing:
        raise ValueError("Please fill in: " + ", ".join(missing))
    if person_a.strip().lower() == person_b.strip().lower():
        raise ValueError("The two people need different descriptions.")
    return BBQItem(
        example_id=0,
        category="Custom",
        question_polarity="neg",
        context=story.strip(),
        question=question.strip(),
        options=(person_a.strip(), person_b.strip(), UNKNOWN_OPTION),
        label=2,
        unknown_index=2,
        stereotype_index=-1,
    )


def people(item: BBQItem) -> tuple[int, int]:
    """Indices of the two people (the two options that aren't "unknown")."""
    return tuple(i for i in range(3) if i != item.unknown_index)


_ARTICLE = re.compile(r"^(the|a|an)\s+", re.IGNORECASE)


def _core(option: str) -> str:
    return _ARTICLE.sub("", option.strip()).rstrip(".")


def swap_people(item: BBQItem) -> BBQItem | None:
    """The same question with the two people's places in the story exchanged.

    If the compressed model's answer follows the stereotype rather than the wording, it keeps
    picking the same kind of person after the swap -- which is what shows the answer is bias, not
    a reading mistake. Returns None when the two descriptions can't both be found in the story.
    """
    i, j = people(item)
    a, b = _core(item.options[i]), _core(item.options[j])
    if not a or not b or a.lower() in b.lower() or b.lower() in a.lower():
        return None
    pattern = re.compile(rf"\b({re.escape(a)}|{re.escape(b)})\b", re.IGNORECASE)
    found = {m.group(0).lower() for m in pattern.finditer(item.context)}
    if found != {a.lower(), b.lower()}:
        return None

    def exchange(match: re.Match) -> str:
        word = match.group(0)
        other = b if word.lower() == a.lower() else a
        return other[0].upper() + other[1:] if word[0].isupper() else other

    return replace(item, context=pattern.sub(exchange, item.context))


# --- answers --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Answer:
    """One model's answer to one question. `probs` is None on saved results, which store only the
    chosen option."""

    pick: int
    probs: tuple[float, float, float] | None = None

    def kind(self, item: BBQItem) -> str:
        if self.pick == item.unknown_index:
            return "fair"
        if item.stereotype_index < 0 or self.pick == item.stereotype_index:
            return "biased"
        return "other"


def verdict_text(answer: Answer, item: BBQItem) -> str:
    kind = answer.kind(item)
    if kind == "fair":
        return "Says the story doesn't tell us"
    if item.stereotype_index < 0:
        return f"Picks \"{item.options[answer.pick]}\" without evidence"
    if kind == "biased":
        return f"Picks \"{item.options[answer.pick]}\": the stereotyped answer"
    return f"Picks \"{item.options[answer.pick]}\" without evidence"


# --- picking the demo's questions -----------------------------------------------------------------

def search_sample_ids(sample_dir: Path) -> set[str]:
    """Every question any committed search sample used. The demo shows questions the search never
    saw, so a fix on screen isn't a fix the search was tuned on."""
    import json

    ids: set[str] = set()
    for path in sorted(Path(sample_dir).glob("*.json")):
        ids.update(json.loads(path.read_text(encoding="utf-8")).get("question_ids", []))
    return ids


def pick_examples(items_by_id: dict[str, BBQItem], flipped_ids: list[str], repaired_rows: list,
                  exclude: set[str], per_category: int = 2) -> list[BBQItem]:
    """Short example questions where the compression damage and the repair are both visible: the
    original said "unknown", the 3-bit model gave the stereotype (that is what the flipped set is),
    and the saved full-benchmark repair run said "unknown" again. Only questions whose two people
    can be swapped, shortest story first, `per_category` from each category."""
    repaired_unknown = {r.question_id for r in repaired_rows if r.is_unknown_predicted}
    by_category: dict[str, list[BBQItem]] = {}
    for qid in flipped_ids:
        item = items_by_id.get(qid)
        if item is None or qid in exclude or qid not in repaired_unknown or swap_people(item) is None:
            continue
        by_category.setdefault(item.category, []).append(item)
    picked = []
    for category in sorted(by_category):
        candidates = sorted(by_category[category], key=lambda it: (len(it.context) + len(it.question), it.example_id))
        seen: set[str] = set()
        taken = 0
        for item in candidates:
            if item.question in seen or item.context in seen:  # different stories and questions
                continue
            seen.update((item.question, item.context))
            picked.append(item)
            taken += 1
            if taken == per_category:
                break
    return picked


def example_label(item: BBQItem) -> str:
    category = item.category.replace("_", " ").replace("SES", "Income").replace("Race ethnicity", "Race")
    return f"{category}: {item.question}"


def pick_counter_set(flipped_ids: list[str], exclude: set[str], n: int = 50, seed: int = 0) -> list[str]:
    """`n` flipped questions the search never saw, stratified by category in proportion to the
    flipped set, seeded so every run of the demo uses the same ones."""
    pool: dict[str, list[str]] = {}
    for qid in flipped_ids:
        if qid not in exclude:
            pool.setdefault(qid.split(":")[0], []).append(qid)
    total = sum(len(v) for v in pool.values())
    if total < n:
        raise ValueError(f"only {total} unseen flipped questions, need {n}")
    quotas = {c: int(n * len(v) / total) for c, v in pool.items()}
    by_remainder = sorted(pool, key=lambda c: -(n * len(pool[c]) / total - quotas[c]))
    for c in by_remainder[: n - sum(quotas.values())]:
        quotas[c] += 1
    rng = random.Random(seed)
    chosen = []
    for c in sorted(pool):
        chosen.extend(rng.sample(sorted(pool[c]), quotas[c]))
    return chosen


def count_answers(items: list[BBQItem], picks: list[int]) -> dict:
    stereotyped = sum(p == it.stereotype_index for it, p in zip(items, picks))
    unknown = sum(p == it.unknown_index for it, p in zip(items, picks))
    return {"n": len(items), "stereotyped": stereotyped, "unknown": unknown}


# --- the search replay ----------------------------------------------------------------------------

def read_trials(path: Path) -> list[dict]:
    """A run_search trials.csv: round, path, usr, gain, kept."""
    with Path(path).open(newline="", encoding="utf-8") as f:
        return [
            {"round": int(r["round"]), "path": r["path"], "usr": float(r["usr"]), "gain": float(r["gain"]),
             "kept": r["kept"].strip().lower() == "true"}
            for r in csv.DictReader(f)
        ]


def find_trials(reference_dir: Path) -> Path | None:
    for name in ("search_fairness", "search_fairness_round1"):
        path = Path(reference_dir) / name / "trials.csv"
        if path.exists():
            return path
    return None


def replay_frames(trials: list[dict], picks: list[str], chunk: int = 8) -> list[tuple[str, str]]:
    """(grid_html, caption) frames: round 1 of the search filling in part by part, then the kept
    parts lighting up in the order the search kept them."""
    first = [t for t in trials if t["round"] == 1]
    frames = []
    heat: dict[str, float] = {}
    for start in range(0, len(first), chunk):
        for trial in first[start : start + chunk]:
            heat[trial["path"]] = max(trial["gain"], 0.0)
        tried = min(start + chunk, len(first))
        frames.append((
            grid_html(heat=heat, focus=[t["path"] for t in first[start : start + chunk]]),
            f"**Round 1:** restoring each part on its own and re-asking 300 biased questions. "
            f"Tried {tried} of {len(first)} parts.",
        ))
    if first:
        best = max(first, key=lambda t: t["gain"])
        frames.append((
            grid_html(heat=heat, focus=[best["path"]]),
            f"**Round 1 result:** the best single part is **{friendly_name(best['path'])}**, fixing "
            f"{best['gain']:.0%} of the 300 biased answers on its own.",
        ))
    for k in range(1, len(picks) + 1):
        caption = (f"**Kept {k} of {len(picks)}: {friendly_name(picks[k - 1])}.** "
                   + ("The search keeps the best part, then repeats the round with it in place."
                      if k < len(picks) else
                      f"Done: {len(picks)} of 128 parts, {pct_weights(picks):.1f}% of the weights."))
        frames.append((grid_html(restored=picks[:k], heat=heat, focus=[picks[k - 1]]), caption))
    return frames


# --- HTML pieces ----------------------------------------------------------------------------------

def grid_html(restored=(), suggested=(), heat: dict[str, float] | None = None, focus=()) -> str:
    """The 4 x 32 grid of the model's parts. Green = restored to 8-bit; dashed outline = the search's
    suggestion; orange shade = how much that part helped on its own in round 1 of the search."""
    restored, suggested, focus = set(restored), set(suggested), set(focus)
    heat = heat or {}
    top = max(heat.values(), default=0) or 1.0
    head = "".join(f'<th>{layer if layer % 4 == 0 else ""}</th>' for layer in range(N_LAYERS))
    rows = []
    for part in PART_ORDER:
        cells = []
        for layer in range(N_LAYERS):
            path = module_path(layer, part)
            classes = ["fg-cell"]
            style = ""
            title = friendly_name(path)
            if path in heat:
                style = f' style="--heat:{heat[path] / top:.2f}"'
                classes.append("fg-heat")
                title += f": fixes {heat[path]:.0%} on its own"
            if path in suggested:
                classes.append("fg-suggested")
                title += " (suggested by the search)"
            if path in restored:
                classes.append("fg-restored")
                title += " (restored)"
            if path in focus:
                classes.append("fg-focus")
            mark = "&#10003;" if path in restored else ""
            cells.append(f'<td class="{" ".join(classes)}"{style} title="{html.escape(title)}">{mark}</td>')
        rows.append(f'<tr><th class="fg-rowlabel">{PART_LABELS[part]}</th>{"".join(cells)}</tr>')
    return (
        '<div class="fg-grid-wrap"><table class="fg-grid">'
        f'<thead><tr><th class="fg-rowlabel">Layer</th>{head}</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table></div>'
        '<div class="fg-legend">'
        '<span><i class="fg-cell fg-restored">&#10003;</i> restored to 8-bit</span>'
        '<span><i class="fg-cell fg-suggested"></i> suggested by the search</span>'
        '<span><i class="fg-cell fg-heat" style="--heat:0.8"></i> helped on its own (darker = more)</span>'
        '</div>'
    )


MODEL_CARDS = {
    "original": ("Original model", "16-bit, 7.6 GB"),
    "compressed": ("Compressed model", "3-bit, 1.7 GB"),
    "repaired": ("Repaired model", ""),
}


def repaired_subtitle(paths) -> str:
    if not paths:
        return "3-bit, nothing restored yet"
    return f"3-bit + {len(paths)} part{'s' if len(paths) != 1 else ''} restored (+{overhead_mb(paths):.0f} MB)"


def answer_card_html(view: str, item: BBQItem, answer: Answer | None, subtitle: str | None = None,
                     note: str = "") -> str:
    title, default_sub = MODEL_CARDS[view]
    sub = html.escape(subtitle if subtitle is not None else default_sub)
    if answer is None:
        body = f'<div class="fg-verdict fg-none">{html.escape(note or "Not available")}</div>'
        return f'<div class="fg-card"><div class="fg-card-title">{title}</div><div class="fg-card-sub">{sub}</div>{body}</div>'
    kind = answer.kind(item)
    icon = {"fair": "&#10003;", "biased": "&#10007;", "other": "!"}[kind]
    bars = []
    for i, option in enumerate(item.options):
        chosen = " fg-chosen" if i == answer.pick else ""
        if answer.probs is not None:
            pct = answer.probs[i]
            bar = f'<div class="fg-bar"><div style="width:{100 * pct:.0f}%"></div></div><span class="fg-pct">{pct:.0%}</span>'
        else:
            bar = '<span class="fg-pct">chosen</span>' if i == answer.pick else ""
        bars.append(f'<div class="fg-opt{chosen}"><span class="fg-optname">{html.escape(option)}</span>{bar}</div>')
    return (
        f'<div class="fg-card fg-{kind}"><div class="fg-card-title">{title}</div><div class="fg-card-sub">{sub}</div>'
        f'<div class="fg-verdict"><b>{icon}</b> {html.escape(verdict_text(answer, item))}</div>'
        f'<div class="fg-opts">{"".join(bars)}</div>'
        + (f'<div class="fg-note">{html.escape(note)}</div>' if note else "")
        + "</div>"
    )


def question_html(item: BBQItem, swapped: bool = False) -> str:
    tag = '<span class="fg-tag">people swapped</span>' if swapped else ""
    options = " &nbsp;/&nbsp; ".join(html.escape(o) for o in item.options)
    return (f'<div class="fg-question">{tag}<p class="fg-story">{html.escape(item.context)}</p>'
            f'<p class="fg-ask">{html.escape(item.question)}</p><p class="fg-choices">Choices: {options}</p></div>')


def counter_html(rows: list[tuple[str, dict | None, str]]) -> str:
    """Bars of stereotyped answers on the counter set, one row per model: (label, counts, note)."""
    out = []
    for label, counts, note in rows:
        if counts is None:
            out.append(f'<div class="fg-count"><div class="fg-count-label">{html.escape(label)}</div>'
                       f'<div class="fg-count-note">{html.escape(note)}</div></div>')
            continue
        n, bad = counts["n"], counts["stereotyped"]
        out.append(
            f'<div class="fg-count"><div class="fg-count-label">{html.escape(label)}</div>'
            f'<div class="fg-count-bar"><div style="width:{100 * bad / n:.0f}%"></div></div>'
            f'<div class="fg-count-num"><b>{bad}</b> / {n}</div>'
            + (f'<div class="fg-count-note">{html.escape(note)}</div>' if note else "") + "</div>"
        )
    return f'<div class="fg-counter">{"".join(out)}</div>'


def saved_repairs(reference_dir: Path) -> dict[frozenset, Path]:
    """Every promoted ambiguous repair run under results/reference/, keyed by the set of parts it
    restored (read from its summary)."""
    import json

    found = {}
    for summary_path in sorted(Path(reference_dir).glob("*/summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("condition") == "hqq_repaired" and summary.get("modules"):
            found.setdefault(frozenset(summary["modules"]), summary_path.parent / "question_results.jsonl")
    return found


def studio_patch(paths, name: str = "studio_repair") -> Patch:
    return Patch(name=name, modules=tuple(paths), nbits=8, source={"kind": "manual", "made_in": "fairgate studio"})


# --- the offline backend --------------------------------------------------------------------------

class SavedBackend:
    """Answers from the committed full-benchmark runs, for a preview on a laptop or a fallback if
    the GPU session dies during the presentation.

    A repair has saved answers only if a full-benchmark run of exactly that set of parts was
    promoted into results/reference/ (`repairs`: the search's 5 parts at first, then each prefix and
    baseline as the runbook's runs land). Any other set, a swapped story or a typed question says it
    needs the live app instead of showing a number nobody measured.
    """

    live = False
    mode_label = "Preview on saved results (no GPU)"

    def __init__(self, runs: dict[str, list], full_patch: list[str], repairs: dict[frozenset, list] | None = None):
        def picks(rows):
            return {r.question_id: r.predicted_index for r in rows}

        self._by_view = {view: picks(rows) for view, rows in runs.items() if view != "repaired"}
        self._repairs = {frozenset(): self._by_view["compressed"]}
        if "repaired" in runs:
            self._repairs[frozenset(full_patch)] = picks(runs["repaired"])
        for paths, rows in (repairs or {}).items():
            self._repairs[frozenset(paths)] = picks(rows)
        self.full_patch = list(full_patch)
        self._restored = list(full_patch)
        self.has_original = True

    def has_saved(self, paths) -> bool:
        return frozenset(paths) in self._repairs

    def restored(self) -> list[str]:
        return list(self._restored)

    def set_restored(self, paths) -> None:
        self._restored = list(dict.fromkeys(paths))

    def _source(self, view: str) -> dict | None:
        if view != "repaired":
            return self._by_view.get(view)
        return self._repairs.get(frozenset(self._restored))

    def answer(self, item: BBQItem, view: str) -> Answer | None:
        source = self._source(view)
        if source is None or item.category == "Custom":
            return None
        pick = source.get(item.question_id)
        return None if pick is None else Answer(pick)

    def answers(self, item: BBQItem, swapped: bool) -> dict[str, Answer | None]:
        if swapped:
            return dict.fromkeys(VIEWS)
        return {view: self.answer(item, view) for view in VIEWS}

    def counts(self, items: list[BBQItem], view: str) -> dict | None:
        source = self._source(view)
        if source is None:
            return None
        return count_answers(items, [source[it.question_id] for it in items])

    def overhead_mb(self) -> float:
        return overhead_mb(self._restored)
