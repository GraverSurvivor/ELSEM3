# Fast Scan on Phi-3.5: validation cells (for the person running the GPU)

Run on a free Colab T4. Set up exactly as in `PHASE2_COLAB.md`: **Cell 1** (check the GPU), **Cell 2** (clone,
install, use branch `sprint/p2-fast-scan` or `main` once merged) and **Cell 2b** (`check_reproduction`
must say PASS: never start a run on a FAIL). Then:

## Cell F1: fast scoring gives the same answers (about 10 min)
Scores 200 questions with the plain 3-bit model and again with `qkv_proj` restored to 8-bit in each of the
32 layers, both ways (all positions' logits vs the last position's only).
```python
!python -m src.attribution.check_fast_scoring --sample results/search_samples/hqq_3bit_n300_seed0.json
```
Needs **PASS: 33/33 agree** and a speed-up above 1x. Post the speed-up line. On FAIL, stop and send the
saved `check_fast_scoring.json`: the speed-ups are not to be used until this passes.

## Cell F2: Fast Scan, exact mode (about 45 min)
Same modules as the full 7-hour search, from the same sample and hedging probe.
```python
!python -m src.attribution.run_search --sample results/search_samples/hqq_3bit_n300_seed0.json \
    --penalty-sample results/search_samples/hedging_probe_n200_seed0.json --penalty-weight 1.0 \
    --fast exact --mirror /content/drive/MyDrive/ELSEM3_fastscan
```
Expected kept modules, in this order: `model.layers.{1, 14, 8, 13, 16}.self_attn.qkv_proj`. Mount Drive first
(`from google.colab import drive; drive.mount('/content/drive')`). If the session dies, rerun the cell with
`--resume <run_id>` added.

## Cell F3: Fast Scan, fast mode (about 15-20 min)
```python
!python -m src.attribution.run_search --sample results/search_samples/hqq_3bit_n300_seed0.json \
    --penalty-sample results/search_samples/hedging_probe_n200_seed0.json --penalty-weight 1.0 \
    --fast fast --mirror /content/drive/MyDrive/ELSEM3_fastscan
```

## Cell F4: what to post to the team
For each of F2 and F3, from `results/<run_id>/summary.json`:
```python
import json, glob
for path in sorted(glob.glob("results/search_*/summary.json"))[-2:]:
    s = json.load(open(path))
    print(path, s["kept"], f"{s['wall_clock_seconds'] / 60:.1f} min", s["n_trials"], "trials", s["stop_reason"])
```
Post: the kept modules, minutes, and whether each matches layers 1, 14, 8, 13, 16. A fast-mode result that
differs from the exact one is a valid finding, not a bug: Person 2 measures how often with `replay`.
Commit the two run folders (`summary.json`, `trials.csv`, `patch.json`) through a PR to `sprint/p4-results`.

## Optional: the full-search logs for replay
The headline searches run with `--log-items`, so each has an `items.jsonl`. Person 2 runs
`python -m src.attribution.replay items results/<search run>/` on them, no GPU needed.
