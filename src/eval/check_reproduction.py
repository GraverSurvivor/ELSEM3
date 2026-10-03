"""Before any new GPU run: does this machine reproduce the committed reference runs?

Every comparison in the project is against the committed runs in results/reference/. A new run is
only comparable with them if this machine builds the same 3-bit model, and HQQ's quantization and
transformers' attention code both change between library versions. A drifted environment doesn't
crash -- it quietly produces a slightly different model, and every "baseline vs search" difference
then mixes repair effects with version effects.

This scores a seeded sample of ambiguous questions with the 3-bit model and with the search's patch
applied (and the original model with --with-fp16), compares each answer with the committed run,
and prints the installed versions next to requirements.lock.txt. It fails if agreement is below the
threshold. A few disagreements are normal: batching pads prompts differently than the full run did,
and near-ties can tip. Systematic disagreement means the environment differs.

    python -m src.eval.check_reproduction                 # about 10 minutes on a T4
    python -m src.eval.check_reproduction --with-fp16 --n 600
"""

from __future__ import annotations

import argparse
import random
from importlib import metadata
from pathlib import Path

REFERENCE_DIR = Path("results/reference")
LOCK = Path("requirements.lock.txt")
PACKAGES = ("torch", "transformers", "hqq", "accelerate", "tokenizers", "safetensors")
DEFAULT_THRESHOLD = 0.97


def installed_vs_locked(lock_path: Path = LOCK, packages=PACKAGES) -> list[tuple[str, str, str]]:
    locked = {}
    if Path(lock_path).exists():
        for line in Path(lock_path).read_text(encoding="utf-8").splitlines():
            name, sep, version = line.strip().partition("==")
            if sep:
                locked[name.lower().replace("_", "-")] = version
    rows = []
    for name in packages:
        try:
            have = metadata.version(name)
        except metadata.PackageNotFoundError:
            have = "not installed"
        rows.append((name, have, locked.get(name, "not in lock")))
    return rows


def sample_ids(rows: list, n: int, seed: int = 0) -> list[str]:
    """`n` question ids, stratified by category in proportion, seeded."""
    by_category: dict[str, list[str]] = {}
    for r in rows:
        by_category.setdefault(r.category, []).append(r.question_id)
    total = sum(len(v) for v in by_category.values())
    rng = random.Random(seed)
    chosen = []
    for category in sorted(by_category):
        k = max(1, round(n * len(by_category[category]) / total))
        chosen.extend(rng.sample(sorted(by_category[category]), min(k, len(by_category[category]))))
    return chosen


def agreement(saved: dict, predictions: dict) -> dict:
    """saved / predictions: question id -> predicted option index."""
    ids = [q for q in predictions if q in saved]
    same = [q for q in ids if saved[q] == predictions[q]]
    return {"n": len(ids), "agree": len(same), "rate": len(same) / len(ids) if ids else 0.0,
            "disagree": [q for q in ids if saved[q] != predictions[q]]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--with-fp16", action="store_true", help="also check the original 16-bit model")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--patch", type=Path, default=Path("results/patches/search_top5.json"))
    args = parser.parse_args()

    print("Library versions (installed / used for the reference runs):")
    drift = False
    for name, have, locked in installed_vs_locked():
        mark = "" if have == locked or locked == "not in lock" else "   <-- differs"
        drift |= bool(mark)
        print(f"  {name:<14}{have:<22}{locked}{mark}")

    import torch

    from src.data.bbq_loader import load_all_categories
    from src.data.safe_set import read_results_file
    from src.eval.option_scorer import score_items

    if not torch.cuda.is_available():
        raise SystemExit("No GPU found -- this check needs the same CUDA setup as the runs.")
    saved = {name: {r.question_id: r.predicted_index
                    for r in read_results_file(REFERENCE_DIR / name / "question_results.jsonl")}
             for name in ("baseline_fp16", "hqq_3bit", "hqq_repaired")}
    rows = read_results_file(REFERENCE_DIR / "hqq_3bit" / "question_results.jsonl")
    ids = sample_ids(rows, args.n, args.seed)
    items_by_id = {it.question_id: it for its in load_all_categories("ambig").values() for it in its}
    items = [items_by_id[q] for q in ids]
    print(f"\nScoring {len(items)} questions per model...")

    def score(model, tokenizer, label):
        picks = score_items(model, tokenizer, items, "cuda", batch_size=args.batch_size, desc=label)
        return dict(zip(ids, picks))

    results = {}
    from src.eval.model_variants import load_variant

    model, tokenizer, _ = load_variant("hqq_3bit", "cuda")
    results["hqq_3bit"] = agreement(saved["hqq_3bit"], score(model, tokenizer, "3-bit"))
    from src.attribution.patch import load_patch
    from src.eval.model_loader import MODEL_ID
    from src.quant.module_swap import SwapRegistry
    from src.quant.weight_source import CheckpointWeightSource

    patch = load_patch(args.patch)
    registry = SwapRegistry(model, CheckpointWeightSource(MODEL_ID), device="cuda")
    for path in patch.modules:
        registry.commit(path, patch.nbits)
    results["hqq_repaired"] = agreement(saved["hqq_repaired"], score(model, tokenizer, "3-bit + patch"))
    del model, registry
    torch.cuda.empty_cache()
    if args.with_fp16:
        model, tokenizer, _ = load_variant("fp16", "cuda")
        results["baseline_fp16"] = agreement(saved["baseline_fp16"], score(model, tokenizer, "16-bit"))

    print(f"\nAgreement with the committed runs (pass at {args.threshold:.0%}):")
    failed = []
    for name, a in results.items():
        ok = a["rate"] >= args.threshold
        failed += [] if ok else [name]
        print(f"  {name:<15}{a['agree']:>4} / {a['n']}  {a['rate']:6.1%}  {'PASS' if ok else 'FAIL'}"
              + (f"   e.g. {', '.join(a['disagree'][:4])}" if a["disagree"] else ""))
    if failed:
        hint = ("Library versions differ from the lock file: install those versions and re-run this check."
                if drift else "Versions match, so look for a code or data difference before running anything.")
        raise SystemExit(f"\nFAIL: this machine does not reproduce {', '.join(failed)}. {hint}")
    print("\nPASS: this machine reproduces the committed runs. New runs are comparable with them.")


if __name__ == "__main__":
    main()
