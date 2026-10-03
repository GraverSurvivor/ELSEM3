# Phase 2 on Colab: proving the repair is really about fairness

Three accounts can work in parallel; each pushes to its own branch, so nothing collides.

| Account | Runs | Time |
|---|---|---|
| 1 | control runs (disambiguated questions), confidence re-run, quantization error, largest-error baseline | ~2 h |
| 2 | 5 random baselines + positional baseline | ~2.5 h |
| 3 | the search's first 1, 2, 3, 4 modules | ~1.7 h |

Every run is promoted and **pushed to GitHub the moment it finishes**, so a disconnect loses at most
the run in progress. If the session dies, start again from Cell 1: finished runs are skipped.

---

## Before you start (once per account)
1. Colab: **Runtime → Change runtime type → T4 GPU**.
2. Click the **key icon** on the left, add a secret named `GITHUB_TOKEN` with a GitHub token that has
   access to `GraverSurvivor/ELSEM3`, and switch on "Notebook access". The token is never printed.

## Cell 1: check the GPU
```python
!nvidia-smi --query-gpu=name,memory.total --format=csv
```
Must say **Tesla T4**.

## Cell 2: get the code and install
Set `ACCOUNT` to 1, 2 or 3.
```python
ACCOUNT = 1

import os, subprocess
from google.colab import userdata
token = userdata.get("GITHUB_TOKEN")
REPO = f"https://{token}@github.com/GraverSurvivor/ELSEM3.git"
BRANCH = f"phase2/account{ACCOUNT}"

def sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print((r.stdout + r.stderr).replace(token, "***").strip())
    return r.returncode

if not os.path.exists("/content/ELSEM3"):
    sh(f"git clone -q {REPO} /content/ELSEM3")
os.chdir("/content/ELSEM3")
# Continue this account's branch if it already exists on GitHub, otherwise start it from main.
if sh(f"git fetch -q {REPO} {BRANCH}") == 0:
    sh(f"git checkout -q -B {BRANCH} FETCH_HEAD")
else:
    sh(f"git checkout -q -B {BRANCH} origin/main")
sh('git config user.name "ELSEM3 Colab" && git config user.email "GraverSurvivor@users.noreply.github.com"')
!pip install -q -r requirements.txt "transformers==5.0.0" "hqq==0.2.8.post1" "accelerate==1.13.0" "tokenizers==0.22.2" "safetensors==0.7.0"
```
The pinned versions are the ones the saved reference runs were made with (`requirements.lock.txt`).
If Colab asks to **restart the session**, restart, then run Cells 2 and 3 again (it skips the clone).

## Cell 2b: check this machine reproduces the saved runs (about 10 min, every new session)
```python
!python -m src.eval.check_reproduction
```
**Only continue if it ends with PASS.** If it says FAIL and `torch` differs, run
`!pip install -q torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128`, restart, and check again.
Runs made on a machine that fails this check can't be compared with the saved runs.

## Cell 3: helpers (run once per session)
```python
import os, subprocess
os.chdir("/content/ELSEM3")

def finish(condition, name):
    """Promote the newest run of `condition` as results/reference/<name>, then commit and push it."""
    prefix = condition + "_"
    runs = sorted(d for d in os.listdir("results") if d.startswith(prefix) and d[len(prefix):][:1].isdigit())
    if not runs:
        raise SystemExit(f"No {condition} run found -- did the run above finish?")
    get_ipython().system(f"python -m src.experiments.promote_reference {runs[-1]} --name {name}")
    if not os.path.exists(f"results/reference/{name}/question_results.jsonl"):
        raise SystemExit(f"Promotion of {runs[-1]} failed -- read the error above. Nothing was pushed.")
    sh(f"git add results/reference/{name} && git commit -q -m 'Phase 2: add {name}'")
    sh(f"git push -q {REPO} HEAD:{BRANCH}")
    print(f"Saved to GitHub: {name}")

def step(name, command, condition):
    """Run one benchmark run unless it is already promoted, then save it."""
    if os.path.exists(f"results/reference/{name}/question_results.jsonl"):
        print(f"Already done: {name} -- skipping.")
        return
    get_ipython().system(command)
    finish(condition, name)
```

---

## Account 1 (~2 h)
Run these cells one after another.
```python
step("baseline_fp16_disambig", "python -m src.eval.run_baseline --context disambig --batch-size 16", "fp16_disambig")
```
```python
step("hqq_3bit_disambig", "python -m src.eval.run_quantized --context disambig --batch-size 16", "hqq_3bit_disambig")
```
```python
step("hqq_repaired_disambig",
     "python -m src.attribution.run_repair --patch results/patches/search_top5.json --context disambig --batch-size 16",
     "hqq_repaired_disambig")
```
```python
step("baseline_fp16_confidence", "python -m src.eval.run_baseline --batch-size 16", "fp16")
```
Quantization error, then the largest-error baseline patch (this one is pushed as files, not a run):
```python
if not os.path.exists("results/patches/largest_error.json"):
    get_ipython().system("python -m src.quant.quant_error")
    get_ipython().system("python -m src.attribution.baselines --quant-error results/quant_error/hqq_3bit_g64.json")
    sh("git add results/quant_error results/patches && git commit -q -m 'Phase 2: quantization error and largest-error patch'")
    sh(f"git push -q {REPO} HEAD:{BRANCH}")
```
```python
step("repair_largest_error",
     "python -m src.attribution.run_repair --patch results/patches/largest_error.json --batch-size 16",
     "hqq_repaired")
```

## Account 2 (~2.5 h)
One cell runs all six; re-running it after a disconnect skips the finished ones.
```python
for patch in ["random_s0", "random_s1", "random_s2", "random_s3", "random_s4", "positional"]:
    step(f"repair_{patch}",
         f"python -m src.attribution.run_repair --patch results/patches/{patch}.json --batch-size 16",
         "hqq_repaired")
```

## Account 3 (~1.7 h)
```python
for k in [1, 2, 3, 4]:
    step(f"repair_search_top5_top{k}",
         f"python -m src.attribution.run_repair --patch results/patches/search_top5_top{k}.json --batch-size 16",
         "hqq_repaired")
```

---

## When all three are done
1. On GitHub, open a pull request for each branch (`phase2/account1`, `phase2/account2`,
   `phase2/account3`) into `main` and merge them. They touch different folders, so they merge cleanly.
2. On a laptop, after `git pull`:
```bash
python -m src.stats.controls
python -m src.stats.flip_predictability --fp16 results/reference/baseline_fp16_confidence
python -m src.stats.final_tables
```
3. Read the verdict at the end of `controls`: **claim_holds**, **baseline_matches** or **hedging**.
   `RUNBOOK.md` (Decision point 1) says what to do for each.
