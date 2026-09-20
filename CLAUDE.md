# ELSEM3 — Symptom-Conditioned Fairness Repair for Quantized LLMs

## Project summary

We build a live, symptom-conditioned fairness repair system for quantized LLMs. Prior work either
proves quantization breaks fairness but doesn't fix it, or protects fairness-sensitive weights
statically before deployment. This project adds the missing third piece: when a compressed model
produces biased answers on live inputs, diagnose which specific transformer components are
responsible using the actual failing examples as evidence, and restore precision only there.

- **Model**: Phi-3.5-mini-Instruct
- **Benchmark**: BBQ, ambiguous condition only, 5 categories (Age, Gender Identity,
  Race/Ethnicity, Religion, Socioeconomic Status)
- **Quantization**: HQQ 3-bit (chosen because it supports per-module precision restoration)
- **Baseline**: BF16
- **Scoring**: deterministic option-scoring (log-likelihood over the 3 answer choices —
  unknown / stereotype / anti-stereotype). No free-text generation for the benchmark loop.

## Pipeline stages

1. **Baseline** — BF16 eval on BBQ ambiguous condition, all 5 categories. Build the safe set
   (question IDs answered "unknown" at BF16 — ground truth reference).
2. **Compress** — HQQ 3-bit quantize Phi-3.5-mini. Must support runtime mixed-precision module
   swaps (promote individual `q_proj`/`k_proj`/`v_proj`/`o_proj`/MLP modules to 8-bit without a
   full reload).
3. **Detect** — Flipped-set = safe-set items that flip to a stereotyped answer post-compression.
   Statistical layer: chi-squared (stereotype vs non-stereotype, before/after) + bootstrap CIs +
   effect size.
4. **Localize** — Greedy symptom-conditioned attribution search. Candidates: attention modules
   first, then MLP. Restore one at a time, re-run only the flipped set (not the full benchmark),
   keep if USR recovers, stop at threshold.
5. **Repair** — Selective repair: keep higher-precision copies of chosen blocks in memory, swap
   into the live model. Post-repair verification re-runs the FULL benchmark to confirm the fix
   generalizes (not just the flipped set).

Headline experiment: run detect → localize → repair on 3 failure windows (Age-heavy,
Socioeconomic-heavy, balanced). Report which blocks get restored each time — same blocks across
windows means localization is category-invariant; different blocks each time means static
protection can't be sufficient. Both are publishable.

Per-run reporting bundle: fairness recovered (USR/SRS), % of model restored to higher precision,
memory overhead vs plain 3-bit, wall-clock time of the cycle.

## Team / file ownership

- **Person A** (Data & Detection Lead) — `src/data/`, `src/eval/`, `src/stats/`. Stages 1-3:
  baseline harness, safe/flipped-set logic, statistical tests.
- **Person B** (Attribution & Repair Lead) — `src/quant/`, `src/attribution/`. Stages 4-5: HQQ
  mixed-precision plumbing, greedy search, module swap.
- **Person C** (Experiments & Reporting Lead) — `src/experiments/`, `src/plotting/`. Headline
  experiment, sweeps, plots, results tables.

## Repo layout

```
src/
  data/          # BBQ loader, safe-set / flipped-set construction (Person A)
  eval/          # deterministic option-scoring evaluator, USR/SRS metrics (Person A)
  stats/         # chi-squared, bootstrap CI, effect size (Person A)
  quant/         # HQQ wrappers, module swap / mixed-precision plumbing (Person B)
  attribution/   # greedy attribution search (Person B)
  experiments/   # headline experiment, sweeps, config-driven runs (Person C)
  plotting/      # plots and results tables (Person C)
results/         # gitignored — run outputs (JSON/CSV), never committed
notebooks/       # exploration only — final results never live here
tests/           # pytest, mirrors src/ layout
```

## Environment

`requirements.txt` uses floor pins (`transformers>=4.44`, `hqq>=0.2.8`, `torch>=2.4`, ...), not
exact pins — exact mid-2024 pins have no wheels on current Colab/Kaggle images (Python 3.13) and
force slow/failing source builds. Always `pip install -r requirements.txt` in a fresh runtime
rather than relying on whatever happens to be preinstalled, since HQQ hooks into transformer
internals that shift across `transformers` versions and Stage 2+ can break in ways that look like
HQQ bugs but are actually version drift. Once Person B's HQQ module-swap code is verified working
end-to-end, freeze this file to exact versions for reproducibility of the final results.

## Conventions

- Branch per person/feature (`person-a/...`, `person-b/...`, `person-c/...`), PR into `main`.
- Every result-writing script logs structured output (JSON/CSV) under `results/` via
  `src/eval` or `src/experiments` — never hand-copy numbers into the deck.
- `notebooks/` is for exploration only; final numbers come from `src/` scripts so they're
  reproducible.
- Write pytest tests alongside new functions in `src/data`, `src/eval`, `src/stats`.
