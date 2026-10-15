"""GPU check: fast scoring gives the same answers as normal scoring, for a module in every layer.

"Fast" scoring asks the model for the last position's logits only (`logits_to_keep=1`, see
`option_scorer.last_position_logits`); "normal" computes the logits at every position and slices the
last. They should agree to rounding. This script proves it on the real model before Fast Scan relies on
it: for the plain 3-bit model and again with one module (qkv_proj) restored to 8-bit in each layer, it
scores the same questions both ways and requires identical answers and confidences within 0.001.

    python -m src.attribution.check_fast_scoring --sample results/search_samples/hqq_3bit_n300_seed0.json
    python -m src.attribution.check_fast_scoring --sample ... --questions 200 --layers 0 15 31

Exit status 1 if any layer disagrees. Writes results/<run_id>/check_fast_scoring.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

TOLERANCE = 1e-3
QUESTIONS = 200


def compare_scores(normal: tuple[list[int], list[float]], fast: tuple[list[int], list[float]],
                   tolerance: float = TOLERANCE) -> dict:
    """Same answers and confidences within `tolerance`? Returns the figures behind the verdict."""
    (normal_pred, normal_conf), (fast_pred, fast_conf) = normal, fast
    if len(normal_pred) != len(fast_pred):
        raise ValueError(f"scored {len(normal_pred)} and {len(fast_pred)} questions")
    different = [i for i, (a, b) in enumerate(zip(normal_pred, fast_pred)) if a != b]
    worst = max((abs(a - b) for a, b in zip(normal_conf, fast_conf)), default=0.0)
    return {"n": len(normal_pred), "different_answers": len(different), "first_different": different[:5],
            "max_confidence_difference": worst, "ok": not different and worst <= tolerance}


def check_layer(score_both, label: str) -> dict:
    """`score_both()` returns (normal, fast, seconds_normal, seconds_fast) for the model as it stands."""
    normal, fast, t_normal, t_fast = score_both()
    return {"label": label, **compare_scores(normal, fast),
            "seconds_normal": round(t_normal, 2), "seconds_fast": round(t_fast, 2)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample", type=Path, required=True, help="search sample JSON (its first questions are used)")
    parser.add_argument("--questions", type=int, default=QUESTIONS)
    parser.add_argument("--layers", type=int, nargs="*", help="only these layers (default: all)")
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("No GPU found. This check needs CUDA.")
    device = "cuda"

    # Imported here so --help and the tests work without hqq/GPU.
    from src.attribution.run_search import COMPUTE_DTYPE, _load_sample_items
    from src.eval.model_loader import MODEL_ID
    from src.eval.option_scorer import DEFAULT_BATCH_SIZE, score_items_with_confidence
    from src.experiments.logger import RESULTS_DIR, _write_atomically, new_run_id
    from src.quant.hqq_quantizer import load_hqq_3bit_model
    from src.quant.module_swap import SwapRegistry
    from src.quant.weight_source import CheckpointWeightSource

    sample = json.loads(args.sample.read_text(encoding="utf-8"))
    sample = {**sample, "question_ids": sample["question_ids"][: args.questions]}
    items = _load_sample_items(sample)
    batch_size = args.batch_size or DEFAULT_BATCH_SIZE

    print(f"Loading Phi-3.5-mini-Instruct, HQQ 3-bit, {COMPUTE_DTYPE} compute...")
    model, tokenizer = load_hqq_3bit_model(device=device, dtype=COMPUTE_DTYPE)
    registry = SwapRegistry(model, CheckpointWeightSource(MODEL_ID), device=device)

    def timed(fast: bool):
        model._elsem_last_logits_only = fast  # False makes last_position_logits take the full-logits path
        torch.cuda.synchronize()
        start = time.time()
        result = score_items_with_confidence(model, tokenizer, items, device, batch_size=batch_size)
        torch.cuda.synchronize()
        return result, time.time() - start

    def score_both():
        normal, t_normal = timed(False)
        fast, t_fast = timed(True)
        return normal, fast, t_normal, t_fast

    layers = args.layers if args.layers else list(range(model.config.num_hidden_layers))
    rows = [check_layer(score_both, "3-bit, nothing restored")]
    print(f"{rows[0]['label']:34} {'OK' if rows[0]['ok'] else 'DIFFERENT'}")
    for layer in layers:
        path = f"model.layers.{layer}.self_attn.qkv_proj"
        with registry.trial(path, 8):
            rows.append(check_layer(score_both, path))
        row = rows[-1]
        print(f"{row['label']:34} {'OK' if row['ok'] else 'DIFFERENT'}  "
              f"max conf diff {row['max_confidence_difference']:.2e}  {row['seconds_normal']}s -> {row['seconds_fast']}s")

    seconds = sum(r["seconds_normal"] for r in rows), sum(r["seconds_fast"] for r in rows)
    failed = [r["label"] for r in rows if not r["ok"]]
    summary = {"questions": len(items), "tolerance": TOLERANCE, "checked": len(rows), "failed": failed,
               "seconds_normal": round(seconds[0], 1), "seconds_fast": round(seconds[1], 1),
               "speedup": round(seconds[0] / seconds[1], 2) if seconds[1] else None, "rows": rows}
    out_dir = RESULTS_DIR / new_run_id("fastscoring")
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_atomically(out_dir / "check_fast_scoring.json", lambda handle: json.dump(summary, handle, indent=2))
    print(f"\n{'PASS' if not failed else 'FAIL'}: {len(rows) - len(failed)}/{len(rows)} agree; "
          f"scoring {summary['speedup']}x faster ({seconds[0]:.0f}s -> {seconds[1]:.0f}s). Saved {out_dir.resolve()}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
