# Sprint contracts: FairGate Fast Repair (16-20 Oct 2026)

The four sprint workstreams build against these, so they can run in parallel. Changes go through
Person 1. Status as of this file: C3, C4 and C5 implemented; C1 and C2 are Person 2's.

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
          on_progress=None, resume=None) -> SearchResult      # greedy_search.SearchResult
scorer(sample_ids, probe_ids) -> objective                   # see note
FastScanConfig(mode="fast"|"exact", screen_sample=60, screen_probe=40, survivors=40, pool=16,
               max_rounds=5, threshold=0.01, nbits=8, penalty_weight=1.0, seed=0)
```
* objective = correct rate − penalty_weight × false-"unknown" rate (as `run_search`).
* With `survivors = pool = len(candidates)` and full subsets it must equal `greedy_search` step for
  step (same tie-break: the earliest candidate wins).
* Note: `fast_repair` passes a scorer that returns the objective as a float. The draft contract
  said `ScanScore(objective, sample_bits, probe_bits)`; if Fast Scan needs the bits, return that
  object and `fast_repair.run_scan` will be updated in the same PR.
* Until this file exists, `fast_repair` uses a built-in scan with the same idea: screen every part
  on a subset, then `greedy_search` on the best `pool` parts with the full search questions.

## C3. Model adapter (Person 1): `src/models/adapter.py` (done)
```python
adapter_for(config) -> ModelAdapter     # also adapter_for_model(model), adapter_for_id(model_id)
ModelAdapter.arch, .n_layers, .part_types, .part_shapes, .candidates(), .params(path),
             .total_params(), .gguf_name(path), .split(path)
```
* Supports `phi3` (fused `qkv_proj` / `gate_up_proj`), `qwen2`, `llama`, `mistral`
  (`q/k/v/o_proj`, `gate/up/down_proj`).
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
 "scan": {"engine", "candidates", "screen_sample", "screen_probe", "pool", "search_probe",
          "penalty_weight", "seed"}}
```
* The three cards and the recovery are measured on held-out questions the scan never used.
* `recovery` values are `null` when compression changed nothing to recover.
* `kept` is `[]` and `patch` is `null` when compression flipped too few answers to repair.
* `n_layers` and `part_types` (display order) were added for the Studio grid (Person 3's request).

`<dir>/patch.json`: the project's patch format, with `part_shapes` and `n_layers` for non-Phi models.

## C5. Studio (Person 3): `src/product/byom.py` (on branch sprint/p3-studio-byom)
* Reads only `events.jsonl`, `report.json` and `patch.json`; starts the pipeline as a subprocess,
  one job at a time; Cancel; reload-safe.
* `python -m src.product.studio --byom` starts Studio without the Phi-3.5 models (the GPU is for the
  customer's model); `--byom-pipeline fake` and `--byom-replay <run dir>` need no GPU.
