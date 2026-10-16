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
- **Baseline**: uncompressed FP16 (see Environment — the model is BF16-trained, but Colab's
  Turing-class T4 has no native BF16 tensor cores; FP16 was verified equivalent and is 6.5x
  faster). Whatever dtype is used, EVERY condition in a comparison must use the same one.
- **Scoring**: deterministic option-scoring (log-likelihood over the 3 answer choices —
  unknown / stereotype / anti-stereotype). No free-text generation for the benchmark loop.

## Pipeline stages

1. **Baseline** — uncompressed eval on BBQ ambiguous condition, all 5 categories. Build the
   safe set (question IDs answered "unknown" uncompressed — ground truth reference).
2. **Compress** — HQQ 3-bit quantize Phi-3.5-mini. Must support runtime mixed-precision module
   swaps (promote individual attention/MLP modules to 8-bit without a full reload). Phi-3.5 does
   NOT have separate `q_proj`/`k_proj`/`v_proj` like Llama-family models — checked directly
   against `transformers`' own `modeling_phi3.py`: attention is a fused `qkv_proj` + `o_proj`,
   MLP is a fused `gate_up_proj` + `down_proj`. Real per-layer candidate set is
   `{qkv_proj, o_proj, gate_up_proj, down_proj}` — 4 modules × 32 layers = 128 candidates total
   (not "4 attention projections" as earlier drafts assumed — build Stage 4's candidate list
   around these actual names).
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

Implemented in `src/experiments/`: `windows.py` defines the windows as category weights on the
fixed search sample (Age-heavy / SES-heavy: that category 60%, others 10% each; balanced: 20%
each -- equal, not proportional, since the proportional sample is the existing Stage 4 run).
`headline.py` runs Person B's `run_search` / `run_repair` per window as subprocesses and records
every run id in `results/headline_<ts>/manifest.json` so `--resume` survives dead sessions;
`headline_report.py` (CPU) writes the comparison table, per-category recovery, the overlap verdict
and the restored-blocks figure. The three window samples are committed under
`results/search_samples/` like the Stage 4 one.

Per-run reporting bundle: fairness recovered (USR/SRS), % of model restored to higher precision,
memory overhead vs plain 3-bit, wall-clock time of the cycle.

## Team / file ownership

- **Person A** (Data & Detection Lead) — `src/data/`, `src/eval/`, `src/stats/`. Stages 1-3:
  baseline harness, safe/flipped-set logic, statistical tests.
- **Person B** (Attribution & Repair Lead) — `src/quant/`, `src/attribution/`. Stages 4-5: HQQ
  mixed-precision plumbing, greedy search, module swap.
- **Person C** (Experiments & Reporting Lead) — `src/experiments/`, `src/plotting/`,
  `src/product/`. Headline experiment, sweeps, plots, results tables, FairGate.

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

## Precision (dtype)

`src/eval/model_loader.py` defaults to **fp16**, overridable with `--dtype` on the eval scripts.

Why not bf16, which the model was trained in: Colab's T4 is Turing (compute capability 7.5) and
native BF16 tensor cores only arrived with Ampere (8.0), so bf16 runs there *unaccelerated* —
measured 3.88 items/s vs fp16's 25.39 (`src/eval/benchmark_throughput.py`). That 6.5x decides
whether Stage 4's search (128 candidates x the flipped set, per round) finishes at all.

Verified equivalent before switching: `validate_scorer --dtype fp16` reproduces the bf16 run's
disambiguated accuracy exactly (0.908, identical per category and per answer position). On
*ambiguous* items ~1.7% of predictions differ between dtypes — near-ties flipping, since that's
where the model is genuinely uncertain. Harmless for comparing conditions, fatal if conditions
disagree: **never compare runs taken at different dtypes**, or precision gets confounded with
quantization. `run_baseline` labels each run's condition with its dtype so a mismatch is visible
in the results files.

Beware `torch.cuda.is_bf16_supported()` — it returns True on a T4. It means bf16 runs, not that
it runs fast. On an Ampere+ GPU (L4/A100), bf16 is the better choice and `--dtype bf16` restores it.

## Repair state (Stage 4/5 memory + revert)

Two constraints shape how the attribution search is allowed to work. Both are implemented in
`src/quant/weight_source.py` and `src/quant/module_swap.py`, and verified end-to-end on the real
model by `src/quant/verify_repair_state.py` (`make repair-state`).

**Never snapshot the whole candidate set.** `snapshot_linear` must run before quantization
(HQQLinear deletes the originals as it packs them), and the candidate set is all 128
`{qkv_proj, o_proj, gate_up_proj, down_proj}` x 32 layers — 3.62B parameters, essentially every
linear weight in the model. Snapshotting them all at fp16 is ~7.25 GB against a 3-bit model of
~1.70 GB: likely an OOM on free Colab's 12.7 GB of system RAM, and it would make the reported
"memory overhead vs plain 3-bit" meaningless, since we'd be holding 4.3x the compressed model to
claim a small overhead. Instead `CheckpointWeightSource` reads one module's weights from the
safetensors shards on demand and holds nothing in between. This is exact, not approximate:
`from_pretrained(torch_dtype=...)` casts those same tensors on load, so the read is bit-identical
to the eager snapshot — which is check [1/4] of the verify script, and is load-bearing. If it
ever fails, promotions inject weights the baseline never had and the search attributes recovery
to a difference it created itself.

**Revert by restoring the displaced module object, never by requantizing.** The search is
promote → measure → put back, 128 times per round. Requantizing back would be slower and would
assume HQQ's optimizer is bit-deterministic; any non-determinism compounds over hundreds of
trials, and it fails silently — the model keeps answering, it just isn't the model the previous
round thought it scored. `SwapRegistry.trial()` keeps the displaced module and restores it in a
`finally`, so even an OOM mid-evaluation can't contaminate later candidates.

Two memory numbers, reported separately and not to be conflated: `deployed_overhead_bytes()` is
what the repair costs to ship (promoted modules minus the quantized ones they replaced — one
`gate_up_proj` at 8-bit is ~+30 MB, about +1.8%), while `search_overhead_bytes()` is the
transient cost of the retained originals, freed by `release_baselines()`. Sizes are measured off
the live tensors rather than computed from nbits, because HQQ's real cost includes packing and
per-group scale/zero: 3.75 bits/weight at 3-bit, 8.50 at 8-bit, not 3 and 8.

## Revised plan additions (October 2026)

A literature review (see the revised-plan PDF) found that detecting quantization bias and
statically protecting fairness-critical weights are both published; what stays distinctive here is
repair after compression, without retraining, at module level, from the model's own failures. The
additions below make that claim defensible. **RUNBOOK.md lists every GPU run, in order.**

- **Disambiguated control set.** `run_baseline`, `run_quantized` and `run_repair` take
  `--context disambig`: BBQ's questions where the text states the answer, filed as
  `<condition>_disambig`. Ambiguous and disambiguated question ids never overlap, and every
  ambiguous label is "unknown", so all scoring is "correct = label" for both sets. One shared loop
  (`src/eval/scoring_run.py`) serves all three scripts; results now record each answer's
  confidence.
- **Patch files** (`src/attribution/patch.py`, tracked under `results/patches/`): which modules
  to restore and where the choice came from. `run_repair --patch` verifies any patch; `run_search`
  writes its kept modules as `patch.json`. Sizes come from Phi-3.5's fixed shapes.
- **Fairness controls** (Improvement #1). `src/attribution/baselines.py` builds patches at the
  search patch's weight budget (+-5%): random x5, positional (same module types, earliest layers),
  largest quantization error (`src/quant/quant_error.py`), and the search's prefixes top1-top4.
  `src/stats/controls.py` reports the hedging check (false-"unknown" rate on the disambiguated set)
  and search-vs-baseline recovery differences, ending in decision point 1's verdict. Prefix runs
  are excluded from that verdict -- they are the recovery-by-modules curve, not alternatives.
- **Search options.** `run_search` loads ids from the question set its sample names (so the same
  search locates general-ability damage), `--candidates mlp` for the ablation, and an optional
  hedging penalty (`--penalty-sample`, `--penalty-weight`), off by default.
- **Analyses (CPU).** `src/stats/`: `bbq_bias` (BBQ's own s_DIS / s_AMB), `flip_predictability`,
  `sensitivity`, `groups` (per targeted group), `dissociation` (concentrated vs diffuse).
- **Beyond BBQ multiple choice.** `src/eval/run_crows_pairs.py`, `src/eval/run_open_ended.py`, and
  `run_quantized --nbits 4` / `run_repair --base-nbits 4`. `src/eval/model_variants.py` loads fp16,
  3-bit, or 3-bit with a patch, always at fp16 compute.
- **FairGate** (`src/product/fairgate.py`, Person C): `check` (PASS/FAIL gate on paired CIs),
  `repair` (summary -> patch), `export` (llama.cpp `--tensor-type` overrides using Phi-3's GGUF
  names attn_qkv / attn_output / ffn_up / ffn_down, or an HQQ per-module config), `report` (one
  self-contained HTML audit report). CPU only; it reads result files.
- **Week 6 wrap-up.** `src/stats/final_tables.py` (Person A) collects every final number into
  `results/final/tables.md`, each section marked done or with the runbook step it waits for; the
  FairGate report renders the same collection, so the two can't disagree. `src/eval/run_llamacpp.py`
  (Person B, stretch) scores BBQ through llama.cpp: conditions `llamacpp_*` map to compute type
  "llamacpp" in `metrics.py`, so they compare only with each other. `src/experiments/freeze.py`
  (Person C) refuses to freeze until every required run is committed and complete, then writes
  `results/final/manifest.json` (SHA-256 per tracked result file); `--verify` checks it later.
- **FairGate Studio** (`src/product/studio.py`, logic in `studio_core.py`, Person C): the Gradio
  presentation app. A guide, then Ask / Diagnose / Repair / Ship. Live mode holds the 16-bit model
  and one 3-bit model; "compressed" answers swap the restored modules back to their original 3-bit
  objects for the duration (instant and exact, like SwapRegistry's revert). Demo questions are
  flipped-set items no search sample contains. `--offline` runs on saved results and shows only
  states a promoted run measured. The replay reads `results/reference/search_fairness*/trials.csv`.
- **Studio tab 5, "Your model"** (`src/product/byom.py`, Person 3, sprint Oct 2026): type a Hugging Face
  model, Diagnose, watch the 8 stages, get the before/after cards, the kept parts on a grid for any
  model, patch.json and llama.cpp settings. It starts the fast-repair pipeline as a subprocess (one job
  at a time, Cancel) and reads only its files (contract C4: events.jsonl, report.json, patch.json), so
  it is reload- and restart-safe. `--byom` keeps the GPU for that model (tabs 1-4 use saved results);
  `--byom-pipeline fake` and `--byom-replay <run>` use `src/product/byom_fake.py`, a stand-in that
  writes the same files without a GPU; its reports carry `synthetic`/`replayed_from`, shown as a banner.

**Fast Repair (sprint, Oct 2026; contracts in docs/CONTRACTS.md).** `src/models/adapter.py`
(Person 1) gives any supported model's parts (phi3, qwen2, llama, mistral), their shapes and GGUF
names; for Phi-3.5 it equals the 128 candidates exactly, and `candidate_paths_for_model`,
`filter_candidates` and patch files (optional `part_shapes`/`n_layers`) now work for the others.
`python -m src.product.fast_repair --model <HF id> --mode fast|exact` runs compress → score
original/3-bit on a seeded BBQ sample → find damage → Fast Scan (Person 2's `fast_scan`, with the
hedging penalty; fast mode sized to ~18,800 questions whatever the number of parts, see
`scan_config`/`plan_questions`) → check on held-out questions → report.json / patch.json /
events.jsonl. Search questions are a seeded random 300 flipped + 200 stated-answer probe (never the
first N: the lists are grouped by category). The model work sits behind a backend interface:
`tests/product/test_fast_repair.py` runs it with a rule-based stand-in, and
`test_fast_repair_tiny_model.py` with real HQQ, SwapRegistry and Qwen's tokenizer on a tiny model.

**The main repair (since Phase 2) is the hedging-aware one**, named in
`results/reference/MAIN_REPAIR.json` and read through `src/experiments/main_repair.py` by the final
tables, FairGate, Studio, groups, sensitivity, figures and the deck. It restores 5 attention-input
modules (layers 1, 14, 8, 13, 16; `results/patches/hedging_penalty.json`, 3.91% of weights, +81 MB):
86% [84, 87] of the lost "unknown" back, false "unknown" on stated-answer questions 10.9% (original
7.3%, +3.6 points [+3.1, +4.2]), stated-answer accuracy 83.7% (3-bit 80.8%), combined recovery 75%.
The first repair (search_top5, 92%, but 20.3% false "unknown", combined 64%) stays as `previous` and
as the patch the same-size baselines and prefix runs were built for; it beats every baseline (best:
largest-error, 71%). With the main repair, 9 of 23 targeted groups end more stereotyped than the
original. Numbers below about "the 92%" refer to the first repair.

Findings already computed from committed runs, to keep in mind when writing anything up: the 92%
recovery is stable across bootstrap settings and benchmark subsets; the repair moves 84% of the
search sample's flipped items back to "unknown" but 77% of those it never saw; and per targeted
group, 8 of 23 groups end significantly more stereotyped than the original model after repair.
Also: compression mostly makes the model stop saying "unknown" and guess (it picks a person 7,159
times vs the original's 1,840), and those guesses lean stereotyped (58%, against the original's
73% on its rare guesses) with a mild word-order lean (56% pick the second-mentioned person). So
write "compression makes the model guess, and the guesses lean stereotyped", not "it stereotypes
every time". Studio's demo examples are stereotyped in both word orders (BBQ's twin questions).
The 3-bit model also avoids answer letter A: it says "unknown" 14% of the time when "unknown" is
choice A, 53% when it is C (the original: 82% / 86%); the repair leaves 62% / 94%. So typed questions
in Studio are asked in all 6 choice orders, and part of the "fairness" damage is a letter effect.
Phase 2's hedging check (Oct 2026): on stated-answer questions the 5-module repair says
"unknown" 20.3% of the time vs the original's 7.3% (+13.0 points, CI +12.3 to +13.7) and its accuracy
is 73.9% vs 91.5% (3-bit: 80.8%) -- decision point 1's "hedging" outcome, so part of the 92% is hedging.
The fix is the penalised search (`--penalty-sample results/search_samples/hedging_probe_n200_seed0.json`,
built by `safe_set --kind hedging-probe`); `run_search --resume/--mirror` makes its ~7 h survivable.
Before any new GPU run, `python -m src.eval.check_reproduction` must PASS: a live Studio run gave a
different repaired answer from the saved run on one question, which points at library-version drift.

## Conventions

- Branch per person/feature (`person-a/...`, `person-b/...`, `person-c/...`), PR into `main`.
- Every result-writing script logs structured output (JSON/CSV) under `results/` via
  `src/eval` or `src/experiments` — never hand-copy numbers into the deck.
- `results/` is gitignored with six exceptions: `results/reference/` (the shared baseline
  and hqq_3bit runs), `results/flipped/` (the flipped set built from them),
  `results/search_samples/` (the fixed sample Stage 4 scores every candidate on — shared so
  every candidate, and every teammate, sees the identical questions), `results/patches/` and
  `results/quant_error/` (which modules each repair restores, so every teammate's baseline runs
  restore identical module sets), and `results/final/` (the final tables and the freeze
  manifest). The reference runs
  matter most: everything downstream is defined relative to them (the safe set is "what the baseline
  got right", the flipped set is "what changed against the baseline"), all three of us must use
  the identical file, and having it in the repo means the flipped-set, statistics and plotting
  work needs no GPU at all. It was lost three times before this rule existed — an interrupt, a
  dead Colab session, a dead Kaggle session — at ~8 GPU-minutes and a working GPU each time.
  Promote a run with `make promote-reference RUN_ID=<id> NAME=<name>`, which validates
  completeness first: a truncated or duplicated run promoted as "the baseline" would not crash
  anything, it would quietly redefine what every later stage compares against.
- Long runs checkpoint per category and resume with `--resume <RUN_ID>`; a category is only
  written once fully scored, so anything in the file is complete.
- `notebooks/` is for exploration only; final numbers come from `src/` scripts so they're
  reproducible.
- Write pytest tests alongside new functions in `src/data`, `src/eval`, `src/stats`.
