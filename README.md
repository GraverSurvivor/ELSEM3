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
make baseline   # BF16 eval on BBQ ambiguous, all 5 categories
make quantize   # HQQ 3-bit compress + flipped-set detection
make figures    # all charts + results table from the committed reference runs (CPU)
make headline   # full detect -> localize -> repair loop, 3 failure windows (GPU, resumable)
make headline-report HEADLINE_ID=headline_<ts>   # headline tables, overlap verdict, figures (CPU)
```

## Deck

`deck/ELSEM3_deck.pptx` is generated, not hand-edited: every number on it is read from the committed
results. After `make figures`, rebuild with `cd deck && npm install && node build_deck.js`.
