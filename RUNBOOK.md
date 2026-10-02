# ELSEM3 runbook: the GPU runs for the revised plan

All the code for the revised plan's Weeks 1–5 is on `main`. What's left is running it: the steps
below need a GPU (Colab or Kaggle T4), and the analyses after them run on any laptop.

**Rule that has saved this project three times:** every finished run gets promoted into
`results/reference/` and committed, so nothing lives only on a Colab/Kaggle machine.

---

## 0. Before anything: the missing search folder

The 5-module repair came from search `search_20260926T155230Z`, whose folder was never committed.
Whoever ran it (Gadiputi) should copy it into `results/reference/search_fairness/` and commit it.
It holds `trials.csv`, which the concentration comparison (step 7) needs.

If it can't be found: the Round 1 search you ran yourself on the same sample
(`search_20260927T090520Z`) covers what step 7 needs. Copy its folder (the `trials.csv` you
downloaded, plus its `summary.json` from Drive) into `results/reference/search_fairness_round1/`.

## 1. Session setup (every new Colab/Kaggle session)

```
!nvidia-smi                                   # must show a Tesla T4
```
Clone with your token, install, restart, re-enter the folder: the same cells as before. On Kaggle,
first run `import os; os.environ["CUDA_VISIBLE_DEVICES"] = "0"`.

Mount Google Drive and point `results/` at it, so a disconnect can't erase a run (Colab):
```python
from google.colab import drive; drive.mount('/content/drive')
import os, shutil
DRIVE = "/content/drive/MyDrive/ELSEM3_results"; os.makedirs(DRIVE, exist_ok=True)
shutil.copytree("results", DRIVE, dirs_exist_ok=True); shutil.rmtree("results"); os.symlink(DRIVE, "results")
```

After each run below: `!python -m src.experiments.promote_reference <RUN_ID> --name <NAME>`, then
commit `results/reference/<NAME>/` through a PR.

---

## Phase 2: fairness controls (Improvement #1), about 5.5 GPU-hours, split across 3 accounts

### Account 1: control runs, about 1.5 h
| # | Command | Time | Promote as |
|---|---|---|---|
| 1 | `!python -m src.eval.run_baseline --context disambig --batch-size 16` | ~10 min | `baseline_fp16_disambig` |
| 2 | `!python -m src.eval.run_quantized --context disambig --batch-size 16` | ~30 min | `hqq_3bit_disambig` |
| 3 | `!python -m src.attribution.run_repair --patch results/patches/search_top5.json --context disambig --batch-size 16` | ~35 min | `hqq_repaired_disambig` |
| 4 | `!python -m src.eval.run_baseline --batch-size 16` (re-run, now records confidence) | ~8 min | `baseline_fp16_confidence` |
| 5 | `!python -m src.quant.quant_error` | ~10 min | commit `results/quant_error/` |

After step 5, on any machine: `python -m src.attribution.baselines --quant-error results/quant_error/hqq_3bit_g64.json`
(adds `results/patches/largest_error.json`) and commit it.

### Account 2: baselines, about 3 h (25 min each)
```
!python -m src.attribution.run_repair --patch results/patches/random_s0.json --batch-size 16
```
Repeat for `random_s1` … `random_s4`, `positional`, and (after Account 1 step 5) `largest_error`.
Promote each as `repair_<patch name>`, e.g. `repair_random_s0`.

### Account 3: prefix runs, about 1.7 h (25 min each)
```
!python -m src.attribution.run_repair --patch results/patches/search_top5_top1.json --batch-size 16
```
Repeat for `_top2`, `_top3`, `_top4`. Promote as `repair_search_top5_top1` … `_top4`.

### Then on a laptop (CPU)
```
python -m src.stats.controls            # hedging + search vs baselines + the verdict
python -m src.stats.flip_predictability --fp16 results/reference/baseline_fp16_confidence
python -m src.stats.groups
python -m src.stats.sensitivity
python -m src.plotting.make_figures
python -m src.product.fairgate report   # results/fairgate/report.html
```

### Decision point 1: everyone reads `controls.json`'s verdict
- `claim_holds`: continue.
- `baseline_matches`: reframe the claim (see the plan PDF); continue.
- `hedging`: re-run the search with the penalty (about 4 h): draw a disambiguated probe sample with
  `python -m src.attribution.search_sample --flipped results/flipped/capability_hqq_3bit.json --n 200 --out results/search_samples/hedging_probe.json`
  (after step 6), then `run_search --sample ... --penalty-sample results/search_samples/hedging_probe.json`.

---

## Phase 3: concentrated vs spread out, about 6 GPU-hours

### 6. General-ability flipped set and sample (CPU)
```
python -m src.data.safe_set --kind capability --baseline results/reference/baseline_fp16_disambig/question_results.jsonl --quantized results/reference/hqq_3bit_disambig/question_results.jsonl --out results/flipped/capability_hqq_3bit.json
python -m src.attribution.search_sample --flipped results/flipped/capability_hqq_3bit.json --out results/search_samples/capability_n300_seed0.json
```
Commit both files.

### 7. Searches (GPU)
| Command | Time | Copy the search folder to |
|---|---|---|
| `!python -m src.attribution.run_search --sample results/search_samples/capability_n300_seed0.json --batch-size 16` | ~4 h | `results/reference/search_capability/` |
| `!python -m src.attribution.run_search --sample results/search_samples/hqq_3bit_n300_seed0.json --candidates mlp --batch-size 16` | ~2 h | `results/reference/search_mlp_only/` |

Then verify the general-ability patch on both question sets:
```
!python -m src.attribution.run_repair --patch results/reference/search_capability/patch.json --batch-size 16
!python -m src.attribution.run_repair --patch results/reference/search_capability/patch.json --context disambig --batch-size 16
```
Promote as `repair_capability` and `repair_capability_disambig`.

### 8. The comparison (CPU)
```
python -m src.stats.dissociation --fairness-trials results/reference/search_fairness/trials.csv --capability-trials results/reference/search_capability/trials.csv --run fairness_ambig=results/reference/hqq_repaired fairness_disambig=results/reference/hqq_repaired_disambig capability_ambig=results/reference/repair_capability capability_disambig=results/reference/repair_capability_disambig
python -m src.plotting.make_figures --fairness-trials results/reference/search_fairness/trials.csv --capability-trials results/reference/search_capability/trials.csv
```

### Decision point 2
Is fairness damage more concentrated than general-ability damage? That sets the deck's headline sentence.

---

## Phase 4: headline experiment + beyond one test, about 14.5 GPU-hours

### Headline (Person C): one window per account, in parallel, about 4 h each
```
!python -m src.experiments.headline --windows age_heavy --batch-size 16
```
`ses_heavy` and `balanced` on the other two accounts. A dead session continues with
`--resume headline_<id>`. Then `python -m src.experiments.headline_report headline_<id>` on CPU.

### Beyond one test
| Command | Time | Promote / commit as |
|---|---|---|
| `!python -m src.eval.run_quantized --nbits 4 --batch-size 16` | ~30 min | `hqq_4bit` |
| `!python -m src.attribution.run_repair --patch results/patches/search_top5.json --base-nbits 4 --batch-size 16` | ~30 min | `repair_4bit` |
| `!python -m src.eval.run_crows_pairs --model fp16` (then `hqq_3bit`, then `patch --patch results/patches/search_top5.json`) | ~10 min each | copy to `results/reference/crows_<condition>/` |
| `!python -m src.eval.run_open_ended run --model fp16` (then `hqq_3bit`, then `patch --patch ...`) | ~20 min each | copy to `results/reference/open_ended_<condition>/` |

Fill in the `human_label` column of each `handcheck.csv` (stereotype / anti_stereotype / unknown /
unclear), then `python -m src.eval.run_open_ended report results/reference/open_ended_*`.

---

## GPU budget

| Phase | GPU hours |
|---|---|
| 2 | ~5.5 |
| 3 | ~6 |
| 4 | ~14.5 |
| **Total** | **~26** (Kaggle gives 30 h per account per week; there are three accounts) |
