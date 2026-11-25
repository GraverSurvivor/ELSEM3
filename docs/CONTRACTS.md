# Sprint contracts: FairGate Fast Repair (16-20 Oct 2026)

The four sprint workstreams build against these, so they can run in parallel. Changes go through
Person 1. Status as of this file: all five implemented and wired together (C1, C2 merged in
PRs #45-47; C3-C5 merged with the pipeline fitted to C2's `ScanScore`).

## C1. Per-item logging (Person 2)
`run_search --log-items` appends one JSON line per trial to `results/<run_id>/items.jsonl`:

```json
{"round": 2, "path": "model.layers.14.self_attn.qkv_proj",
 "sample": "0110...", "probe": "0010..."}
```
* `path` is `null` for the model before any trial.
* `sample`: one character per sample question, in the sample's `question_ids` order; `1` = answered
  the question's label.
* `probe`: one character per probe question; `1` = said "unknown" although the answer is stated;
  `""` when there is no probe.
* Copied to `--mirror` like `trials.csv`; `headline.py` passes `--log-items` to every search.

## C2. Fast Scan (Person 2): `src/attribution/fast_scan.py`
```python
fast_scan(candidates, registry, scorer, sample_ids, probe_ids, config,
          on_progress=None, resume=None) -> FastScanResult    # a SearchResult with survivors, pool
scorer(sample_ids, probe_ids) -> ScanScore(objective, sample_bits, probe_bits)
on_progress(result)                            # after every trial and every keep, like greedy_search
FastScanConfig(mode="fast"|"exact", screen_sample=60, screen_probe=40, survivors=40, pool=16,
               max_rounds=5, threshold=0.01, nbits=8, penalty_weight=1.0, seed=0)
```
* objective = correct rate − penalty_weight × hedging. In `run_search` hedging is the false-"unknown"
  rate; `fast_repair` passes `excess_hedging`: only "unknown" the compressed model did NOT already give
  counts (Qwen2.5-3B: compression added hedging, and penalising all of it made the scan keep parts that
  remove hedging at fairness's expense).
* With `survivors = pool = len(candidates)` and full subsets it must equal `greedy_search` step for
  step (same tie-break: the earliest candidate wins).
* "fast" scores everything on the screening subset; "exact" re-checks round 1's best `survivors` on
  every question and runs rounds 2+ on all of them. Rounds 2+ try only the `pool` best of round 1.
* `fast_repair` (C4) returns a `ScanScore` with each question's result (`item_log.sample_bits` /
  `probe_bits`), counts the questions it asks for the progress bar, and sets the sizes per run:
  Fast Scan's defaults for a 128-part model like Phi-3.5 (18,800 questions in fast mode); for a model
  with more parts, fewer screening questions (at least 60) so the scan asks about as many.
  `plan_questions()` is that count, tested equal to what Fast Scan asks.

## C3. Model adapter (Person 1): `src/models/adapter.py` (done)
```python
adapter_for(config) -> ModelAdapter     # also adapter_for_model(model), adapter_for_id(model_id)
ModelAdapter.arch, .n_layers, .part_types, .part_shapes, .candidates(), .params(path),
             .total_params(), .gguf_name(path), .split(path)
```
* Tables for `phi3` (fused `qkv_proj` / `gate_up_proj`), `qwen2`, `llama`, `mistral`
  (`q/k/v/o_proj`, `gate/up/down_proj`); any other text-generation model is discovered from its
  structure (`ModelAdapter.layers_path`, `.transposed` for GPT-2 `Conv1D`; `gguf_name` may be None).
  Reports add `layers_path` and `gguf_names`.
* Candidates: grouped by part type, attention types first, each across every layer. For Phi-3.5 they
  equal `src/quant/candidates.py`'s 128 exactly (tested).
* GGUF names: `attn_qkv`, `attn_q`, `attn_k`, `attn_v`, `attn_output`, `ffn_up` (fused gate_up or
  up), `ffn_gate`, `ffn_down`, as `blk.<layer>.<name>`.
* Patches for non-Phi models carry `part_shapes` and `n_layers` (`src/attribution/patch.py`);
  Phi-3.5 patches are unchanged (all committed patch files re-write byte-identically).

## C4. One-command pipeline (Person 1): `python -m src.product.fast_repair` (done)
```
python -m src.product.fast_repair --model <HF id> --mode fast|exact --questions 1000 --out <dir>
    [--seed 0] [--batch-size 16] [--penalty-weight 1.0]
```
`<dir>/events.jsonl`, one complete line per update:
```json
{"time": "2026-10-19T10:02:11.120+00:00", "stage": "scan", "status": "progress",
 "done": 41, "total": 312, "message": "round 2 of 5"}
```
* `stage`: `download`, `quantize`, `score_original`, `score_3bit`, `find_damage`, `scan`,
  `verify`, `report`, always in this order.
* `status`: `started`, `progress`, `done`, `failed` (with a plain-language reason in `message`).

`<dir>/report.json`, written once at the end:
```json
{"model_id", "arch", "mode", "questions", "gpu", "minutes", "safe", "flipped",
 "original": {"usr", "srs", "false_unknown", "accuracy"}, "compressed": {...}, "repaired": {...},
 "recovery": {"point", "lower", "upper"}, "kept": ["model.layers.13.self_attn.q_proj", ...],
 "memory_mb", "gate": "PASS"|"FAIL", "gate_failures": [...], "patch": "patch.json" | null,
 "n_layers": 36, "part_types": ["self_attn.q_proj", ...],
 "checked_on": {"ambiguous": 503, "stated_answer": 497},
 "scan": {"engine": "fast_scan"|"none", "candidates", "screen_sample", "screen_probe",
          "survivors", "pool", "search_sample", "search_probe", "penalty_weight", "seed",
          "pool_parts", "trials", "stop_reason"}}     # the last three only when the scan ran
```
* The three cards and the recovery are measured on held-out questions the scan never used.
* `recovery` values are `null` when compression changed nothing to recover.
* `kept` is `[]` and `patch` is `null` when compression flipped too few answers to repair.
* `outcome`: `"repaired"`, `"rejected"` (the held-out check found the repair made fairness worse: the whole
  recovery interval below 0; `patch` is null, nothing is offered to ship), `"no_gain"` (damage found, no part
  worth keeping), `"no_damage"` (fewer than 10
  flipped search questions), or `"not_applicable"` (the original says "can't tell" on under 10% of
  ambiguous questions, or fewer than 50: no fair behaviour to break). `outcome_reason` says it in plain
  words. `not_applicable` runs no scan and has `gate: "N/A"`.
* `recovery` is computed only when compression cut "can't tell" answers by at least 2 points.
* `other_damage`: plain sentences for accuracy lost or "can't tell" added on stated-answer questions
  (5 points or more), for the compressed model and, if repaired, the repaired one. The fairness verdict
  doesn't cover these, so the Studio shows them next to it.
* `n_layers` and `part_types` (display order) were added for the Studio grid (Person 3's request).
* The scan stage's `done`/`total` count questions asked against the plan, so the bar moves with time;
  its `started` message carries the estimate ("252 parts · about 25 min") and every progress message
  the step and time left ("Step 1 of 2 · Quick look at every part: 37 of 252 · about 21 min left").

`<dir>/patch.json`: the project's patch format, with `part_shapes` and `n_layers` for non-Phi models.

Added for v2 (Nov 2026):
* `<dir>/answers.json`: `{"ambig": {question id: {"split": "search"|"final", "original": 2, "compressed": 0,
  "repaired": 2}}, "stated": {...}}`. Every question's pick from each model; `repaired` is null where the
  repaired model wasn't asked.
* `report.json` -> `fingerprint` (only for a repair worth shipping): 50 of the final check's questions with the
  repaired model's picks, `[{"question_id", "set", "pick"}]`. A rebuilt copy that answers them the same is the
  model that was measured.
* The real models moved to `src/product/torch_backend.py` (re-exported from `fast_repair`); `TorchBackend.probs`
  returns every choice's probability and `.revision` is the model's commit on Hugging Face.

`--pipeline v2` (off by default until it has won on the test models; decisions in `src/product/decisions.py`):
* Three splits by hash: `search` 40%, `validation` 20%, `final` 40%. Version 1 has `search` and `final` halves.
  v2's `final` lies inside v1's when both ran with the same `--questions` and seed (the sample depends on both, and
  the defaults differ): compare the versions only that way. `answers.json` names the split. `validation` is reserved
  for choosing between finished repairs (PR A2); nothing reads it yet.
* `report.json` adds `pipeline` (both versions), and for v2 `objective`, `splits` (questions of each set in each
  split) and `letter_balanced` (the cards' numbers on up to 400 + 400 final questions, each answered from the
  average over every rotation of its choices; `questions`, `original`, `compressed`, `repaired`).
* `<dir>/scan.jsonl`: one line per score of the scan, `{"path": part being tried or null, "kept": [...], "sample",
  "stated", "soft", "kl", "count"}`. `--objective` says which decides (default `soft`).
* `--study N` (at least 5) also writes `<dir>/study.jsonl` (N random parts, question by question; the first line
  records the scan's own question counts, `screen`, and the penalty `weight`);
  `python -m src.product.compare_objectives <dir>` turns it into `objectives.json`.
* 2,000 questions of each set by default (v1: 1,000).

## C7. Patch, version 2: `src/attribution/patch.py`
Three optional keys; a patch without them reads and re-writes byte-identically (tested on every committed patch):
```json
{"module_bits": {"model.layers.1.self_attn.q_proj": 4}, "revision": "<commit>", "prompt_style": "examples"}
```
* `Patch.bits(path)`: what a module is restored to (`module_bits` if listed, else `nbits`). `run_repair --patch`,
  `model_variants` and `check_reproduction` apply it. On a plain dict:
  `{p: patch.get("module_bits", {}).get(p, patch["nbits"]) for p in patch["modules"]}`.
* The kit, publishing and llama.cpp lines still assume one bit-width: they learn `module_bits` before any
  pipeline writes it.

## C5. Studio (Person 3): `src/product/byom.py` (on branch sprint/p3-studio-byom)
* Reads only `events.jsonl`, `report.json` and `patch.json`; starts the pipeline as a subprocess,
  one job at a time; Cancel; reload-safe.
* `python -m src.product.studio --byom` starts Studio without the Phi-3.5 models (the GPU is for the
  customer's model); `--byom-pipeline fake` and `--byom-replay <run dir>` need no GPU.
