# Shared contracts: FairGate Fast Repair sprint (Fri 16 – Tue 20 Oct 2026)

Everyone builds against these, so four people can work in parallel without breaking each other's
code. **Agreed before coding; changes only through Person 1.** If a contract needs to change, tell
Person 1 before changing code.

| Contract | Owner | Where |
|---|---|---|
| C1 Per-item logging | Person 2 | `src/attribution/run_search.py --log-items`, `src/attribution/item_log.py` |
| C2 Fast Scan | Person 2 | `src/attribution/fast_scan.py` |
| C3 Model adapter | Person 1 | `src/models/adapter.py` |
| C4 One-command pipeline | Person 1 | `src/product/fast_repair.py` |
| C5 Studio | Person 3 | `src/product/studio.py --byom` |

```
SHARED CONTRACTS (agreed before coding; changes only through Person 1)

C1. Per-item logging (Person 2, merged on Day 1 by 13:00)
  run_search --log-items appends one JSON line per trial to results/<run_id>/items.jsonl:
    {"round": int, "path": str or null (null = the model before any trial),
     "sample": "0110...", one char per sample question, in the sample's question_ids order,
                          1 = answered the question's label
     "probe": "0010..."}  one char per probe question, 1 = said "unknown" although the answer
                          is stated; "" when there is no probe
  Copied to --mirror like trials.csv. headline.py passes --log-items through to every search.

C2. Fast Scan (Person 2): src/attribution/fast_scan.py
  fast_scan(candidates, registry, scorer, sample_ids, probe_ids, config,
            on_progress=None, resume=None) -> SearchResult (the class in greedy_search.py)
  scorer(sample_ids, probe_ids) -> ScanScore(objective: float, sample_bits: str, probe_bits: str)
    objective = correct rate - penalty_weight x false-"unknown" rate (same as run_search today)
  FastScanConfig(mode="fast"|"exact", screen_sample=60, screen_probe=40, survivors=40, pool=16,
                 max_rounds=5, threshold=0.01, nbits=8, penalty_weight=1.0, seed=0)
    step 1 screen every candidate on a seeded subset; step 2 score the survivors
    ("exact": on the full sample + probe); step 3 greedy rounds 2..max_rounds only within the
    `pool` best of round 1. With survivors = pool = len(candidates) and full subsets it must
    equal greedy_search step for step (same tie-break: earliest candidate wins).
  run_search gains --fast fast|exact (default: off = today's search).

C3. Model adapter (Person 1): src/models/adapter.py
  adapter_for(config) -> ModelAdapter with .arch, .n_layers, .part_types (ordered),
    .candidates() (attention parts first, then MLP; same order rule as src/quant/candidates.py),
    .params(path), .gguf_name(path)
  Supports phi3 (fused qkv_proj / gate_up_proj), qwen2 and llama (q/k/v/o, gate/up/down;
  mistral = llama). For Phi-3.5 the candidates must equal today's 128 exactly.

C4. One-command pipeline (Person 1): python -m src.product.fast_repair
      --model <HF id> --mode fast|exact --questions 1000 --out <dir>
  <dir>/events.jsonl, one line per update:
    {"time": ISO, "stage": "download"|"quantize"|"score_original"|"score_3bit"|"find_damage"
      |"scan"|"verify"|"report", "status": "started"|"progress"|"done"|"failed",
     "done": int, "total": int, "message": str}
  <dir>/report.json:
    {"model_id", "arch", "mode", "questions", "gpu", "minutes", "safe": int, "flipped": int,
     "original": {...}, "compressed": {...}, "repaired": {...}   each {"usr", "srs",
                                                                 "false_unknown", "accuracy"},
     "recovery": {"point", "lower", "upper"}, "kept": [module paths], "memory_mb",
     "gate": "PASS"|"FAIL", "gate_failures": [str], "patch": "patch.json"}
  <dir>/patch.json in the existing patch format (src/attribution/patch.py).
  Search questions and verification questions never overlap.

C5. Studio (Person 3) reads only events.jsonl, report.json and patch.json, and starts the
  pipeline as a subprocess (one job at a time). python -m src.product.studio --byom starts the
  app WITHOUT loading the Phi-3.5 models, because the GPU is needed for the customer's model;
  the existing screens then use saved results.
```

## Team rules

- One branch per person, PR into `main`; Person 1 reviews and merges. Never push to `main` directly.
- Run `pytest tests -q` before every PR. All tests must pass.
- Don't edit files outside your scope unless your task list says so; ask the owner first.
- Tokens (GitHub, Hugging Face) only in Colab secrets. Never paste them into a cell, a chat or a commit.
- Daily 10-minute check-in (suggested 21:00): what's done, what's blocked, what's next.
- If a contract needs to change, tell Person 1 before changing code.
