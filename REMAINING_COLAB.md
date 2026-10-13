# What's left on the GPU: exact Colab cells

Everything Phase 2 needed is done and merged. These are the remaining GPU runs, in order of
importance. Each block is one Colab session on its own Google account, so they can all run at the
same time.

| Block | What it answers | Time | Account (`ACCOUNT =`) |
|---|---|---|---|
| **A. Headline**, one window each | Do different kinds of failure need the same repaired parts? (the project's original headline question) | ~7 h each | 5, 6, 7 |
| **B. Beyond one test** | Does the repair help at 4-bit, on a second bias test, and in free-text answers? | ~3 h | 8 |
| **C. Concentrated vs spread out** | Is fairness damage in a few parts while general damage is everywhere? | ~6 h | 9 |
| **D. llama.cpp transfer** (stretch) | Does the fix carry over to llama.cpp's own 3-bit format? | ~1.5 h | 10 (steps in RUNBOOK.md, "Week 6") |

**Every session starts the same way:** Cells 1, 2, 2b and 3 from `PHASE2_COLAB.md`, with that
block's `ACCOUNT` number in Cell 2. **Cell 2b must say PASS.** After a block's runs are saved, open a
PR for its `phase2/account<N>` branch and merge it.

---

## A. Headline experiment: one window per account (~7 h each)

Account 5 runs `age_heavy`, account 6 `ses_heavy`, account 7 `balanced`. Each search uses the hedging
penalty (like the main repair) and copies everything to Google Drive as it goes.

### A1. Drive and the window (after Cells 1, 2, 2b, 3)
```python
from google.colab import drive
drive.mount("/content/drive", force_remount=True)
MIRROR = "/content/drive/MyDrive/ELSEM3_headline"
os.makedirs(MIRROR, exist_ok=True)
WINDOW = "age_heavy"          # account 5: age_heavy   account 6: ses_heavy   account 7: balanced
```

### A2. Start
```python
!python -m src.experiments.headline --windows {WINDOW} --batch-size 16 --mirror {MIRROR}
```

### If the session dies: Cells 1, 2, 2b, 3 and A1 again, then this instead of A2
```python
HID = sorted(d for d in os.listdir(MIRROR) if d.startswith("headline_"))[-1]
print("Resuming", HID)
!python -m src.experiments.headline --resume {HID} --windows {WINDOW} --mirror {MIRROR}
```
It brings back everything the dead session lost and continues the search where it stopped.

### A3. When it prints "Done.": save to GitHub
```python
import json, shutil
HID = sorted(d for d in os.listdir(MIRROR) if d.startswith("headline_"))[-1]
w = json.load(open(f"results/{HID}/manifest.json"))["windows"][WINDOW]
dest = "results/reference/headline_runs"
for name in (HID, w["search_run_id"], w["repair_run_id"]):
    if name:
        shutil.copytree(f"results/{name}", f"{dest}/{name}", dirs_exist_ok=True)
sh(f"git add {dest} && git commit -q -m 'Headline: {WINDOW} window'")
sh(f"git push -q {REPO} HEAD:{BRANCH}")
print(HID, w)
```
Send the printed line to whoever writes the report. Once all three are merged, on a laptop:
```
python -m src.experiments.headline_report <HID of age_heavy> <HID of ses_heavy> <HID of balanced> --results-dir results/reference/headline_runs
```

---

## B. Beyond one test (~3 h, one account)

After Cells 1, 2 (`ACCOUNT = 8`), 2b, 3, add this helper cell once:
```python
def save_dir(prefix, name):
    """Copy the newest results/<prefix>_<timestamp> run to results/reference/<name> and push it."""
    import shutil
    runs = sorted(d for d in os.listdir("results") if d.startswith(prefix + "_") and d[len(prefix) + 1:][:1].isdigit())
    if not runs:
        raise SystemExit(f"No {prefix} run found -- did the run above finish?")
    shutil.copytree(f"results/{runs[-1]}", f"results/reference/{name}", dirs_exist_ok=True)
    sh(f"git add results/reference/{name} && git commit -q -m 'Beyond BBQ: add {name}'")
    sh(f"git push -q {REPO} HEAD:{BRANCH}")
    print("Saved to GitHub:", name)
```

**B1. 4-bit: does the same repair help a less-compressed model?** (~1 h)
```python
step("hqq_4bit", "python -m src.eval.run_quantized --nbits 4 --batch-size 16", "hqq_4bit")
```
```python
step("repair_4bit",
     "python -m src.attribution.run_repair --patch results/patches/hedging_penalty.json --base-nbits 4 --batch-size 16",
     "hqq_4bit_repaired")
```

**B2. CrowS-Pairs, a second bias test** (~10 min each)
```python
get_ipython().system("python -m src.eval.run_crows_pairs --model fp16"); save_dir("crows_fp16", "crows_fp16")
```
```python
get_ipython().system("python -m src.eval.run_crows_pairs --model hqq_3bit"); save_dir("crows_hqq_3bit", "crows_hqq_3bit")
```
```python
get_ipython().system("python -m src.eval.run_crows_pairs --model patch --patch results/patches/hedging_penalty.json")
save_dir("crows_hqq_repaired", "crows_hqq_repaired")
```

**B3. Free-text answers** (~20 min each)
```python
get_ipython().system("python -m src.eval.run_open_ended run --model fp16"); save_dir("open_ended_fp16", "open_ended_fp16")
```
```python
get_ipython().system("python -m src.eval.run_open_ended run --model hqq_3bit"); save_dir("open_ended_hqq_3bit", "open_ended_hqq_3bit")
```
```python
get_ipython().system("python -m src.eval.run_open_ended run --model patch --patch results/patches/hedging_penalty.json")
save_dir("open_ended_hqq_repaired", "open_ended_hqq_repaired")
```
After merging: open each `results/reference/open_ended_*/handcheck.csv`, fill in the `human_label`
column for the 50 rows (stereotype / anti_stereotype / unknown / unclear), and commit it. That
measures how often the automatic labels are right.

---

## C. Concentrated vs spread out (~6 h, one account)

The question set is already built and committed (`results/search_samples/capability_n300_seed0.json`:
300 of the 1,673 stated-answer questions compression got wrong).

### C1. Drive (after Cells 1, 2 with `ACCOUNT = 9`, 2b, 3)
```python
from google.colab import drive
drive.mount("/content/drive", force_remount=True)
MIRROR = "/content/drive/MyDrive/ELSEM3_capability"
os.makedirs(MIRROR, exist_ok=True)
```

### C2. The same search, aimed at general mistakes (~4 h)
No hedging penalty here: on these questions the right answer is never "unknown".
```python
!python -m src.attribution.run_search --sample results/search_samples/capability_n300_seed0.json --batch-size 16 --mirror {MIRROR}
```
If the session dies: Cells 1, 2, 2b, 3 and C1 again, then the same command with
`--resume {sorted(d for d in os.listdir(MIRROR) if d.startswith("search_"))[-1]}` added at the end.

### C3. Save it and verify its patch on both question sets (~35 min)
```python
import shutil
RUN_ID = sorted(d for d in os.listdir(MIRROR) if d.startswith("search_"))[-1]
shutil.copytree(f"{MIRROR}/{RUN_ID}", "results/reference/search_capability", dirs_exist_ok=True)
sh("git add results/reference/search_capability && git commit -q -m 'Phase 3: general-ability search'")
sh(f"git push -q {REPO} HEAD:{BRANCH}")
step("repair_capability",
     "python -m src.attribution.run_repair --patch results/reference/search_capability/patch.json --batch-size 16",
     "hqq_repaired")
step("repair_capability_disambig",
     "python -m src.attribution.run_repair --patch results/reference/search_capability/patch.json --context disambig --batch-size 16",
     "hqq_repaired_disambig")
```

### C4. Optional: MLP-only search (~2 h), the attention-vs-MLP ablation
```python
!python -m src.attribution.run_search --sample results/search_samples/hqq_3bit_n300_seed0.json --candidates mlp --penalty-sample results/search_samples/hedging_probe_n200_seed0.json --batch-size 16 --mirror {MIRROR}
```
Then copy its folder to `results/reference/search_mlp_only` the same way as in C3.

---

## After everything is merged (laptop, CPU)
```
python -m src.stats.dissociation --fairness-trials results/reference/search_hedging_penalty/trials.csv --capability-trials results/reference/search_capability/trials.csv --run fairness_ambig=results/reference/repair_hedging_penalty fairness_disambig=results/reference/repair_hedging_penalty_disambig capability_ambig=results/reference/repair_capability capability_disambig=results/reference/repair_capability_disambig
python -m src.stats.final_tables
python -m src.product.fairgate report
python -m src.experiments.freeze
```
`freeze` passes once every required run is in; then tag v1.0 as `RUNBOOK.md` describes.
