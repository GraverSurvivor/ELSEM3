# ELSEM3 — Symptom-Conditioned Fairness Repair for Quantized LLMs

Live, symptom-conditioned fairness repair for quantized LLMs: detect when HQQ 3-bit compression
flips a Phi-3.5-mini-Instruct answer on BBQ from "unknown" to stereotyped, localize which
attention/MLP modules are responsible using the failing examples as evidence, and selectively
restore only those modules to higher precision.

See [CLAUDE.md](CLAUDE.md) for the full project context, pipeline stages, and team ownership.

## Setup

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt  # for tests, notebooks, experiment tracking
```

## Layout

```
src/data/          BBQ loader, safe-set / flipped-set construction
src/eval/          deterministic option-scoring evaluator, USR/SRS metrics
src/stats/         chi-squared, bootstrap CI, effect size
src/quant/         HQQ wrappers, module swap / mixed-precision plumbing
src/attribution/   greedy attribution search
src/experiments/   headline experiment, sweeps
src/plotting/      plots and results tables
results/           gitignored run outputs
notebooks/         exploration only
tests/             pytest, mirrors src/
```

## Reproducing a run

```bash
make baseline   # fp16 eval on BBQ ambiguous, all 5 categories
make quantize   # HQQ 3-bit compress + flipped-set detection
make figures    # all charts + results table from the committed reference runs (CPU)
make headline   # full detect -> localize -> repair loop, 3 failure windows (GPU, resumable)
make headline-report HEADLINE_ID=headline_<ts>   # headline tables, overlap verdict, figures (CPU)
make disambig   # the control runs on BBQ's disambiguated questions (GPU)
make repair PATCH=results/patches/positional.json   # verify any patch on the full benchmark (GPU)
make controls   # hedging check + search vs matched-budget baselines (CPU)
make fairgate-report   # the FairGate HTML audit report (CPU)
```

The full sequence of GPU runs for the revised plan, with timings and what to commit, is in
[RUNBOOK.md](RUNBOOK.md).

## FairGate

A command-line tool over the same pipeline: does compressing a model make it more biased, and what
fixes it?

```bash
python -m src.product.fairgate check      # PASS/FAIL: original vs compressed
python -m src.product.fairgate export --patch results/patches/hedging_penalty.json --format llamacpp
python -m src.product.fairgate report     # results/fairgate/report.html
```

## FairGate Studio (the demo app)

The presentation app: a short guide, then four screens. **Ask** (three models answer the same
question side by side, and you can swap the two people or type your own question), **Diagnose** (a
replay of the search), **Repair** (restore parts yourself and watch the stereotyped answers drop,
live) and **Ship** (the verdict on the full benchmark, and the patch to download).

```bash
python -m src.product.studio --share     # live, on a Colab T4: notebooks/fairgate_studio.ipynb
python -m src.product.studio --offline   # preview on saved results, any laptop (needs gradio)
python -m src.product.studio --byom --share   # tab 5: diagnose any Hugging Face model (GPU)
python -m src.product.studio --byom --byom-pipeline fake   # tab 5 with a stand-in pipeline, any laptop
```

## Deck

`deck/ELSEM3_deck.pptx` is generated, not hand-edited: every number on it is read from the committed
results. After `make figures`, rebuild with `cd deck && npm install && node build_deck.js`.
