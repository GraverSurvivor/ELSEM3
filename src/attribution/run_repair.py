"""Stage 5: apply the search's repair and re-run the FULL benchmark. Needs a GPU.

Loads Phi-3.5 at HQQ 3-bit (fp16 compute, matching the baseline), restores exactly the modules the
greedy search kept -- read from that search's summary.json, at the precision it tested them at --
and scores all 12,140 ambiguous BBQ items. The search only ever measured its 300-item sample;
this run is what shows whether the repair generalizes (CLAUDE.md, Stage 5).

Same scoring, checkpoint-per-category and `--resume` as run_baseline / run_quantized, written
through the same logger under the condition `hqq_repaired`. Compare it against baseline_fp16 and
hqq_3bit with src/stats once `"hqq_repaired": "fp16"` is in CONDITION_COMPUTE_DTYPE.

Writes results/<run_id>/summary.json with the reporting bundle CLAUDE.md asks for: fairness
recovered (USR/SRS overall and per category), share of the model restored, memory overhead vs
plain 3-bit, and wall-clock time of the cycle -- plus which search produced the repair.

"Share restored" is reported by parameters as the headline, and by module count alongside. The
candidates differ several-fold in size (qkv_proj 28.3M parameters, o_proj 9.4M), so a count
misstates how much of the model is running at higher precision.

Memory is reported two ways, deliberately:
    deployed overhead  registry.deployed_overhead_bytes() -- the restored modules' size minus the
                       3-bit modules they replaced, measured off the live tensors. The headline
                       "memory overhead vs plain 3-bit" number (CLAUDE.md > Repair state).
    GPU delta          torch.cuda.memory_allocated() before vs after the repair. A cross-check
                       only: it also sees allocator effects, so it won't match to the byte.

Run after `pip install -r requirements.txt`, on a GPU:
    python -m src.attribution.run_repair --search-summary results/search_<id>/summary.json
    python -m src.attribution.run_repair --search-summary ... --resume hqq_repaired_<id>
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch

from src.data.bbq_loader import CATEGORIES, load_all_categories
from src.eval.option_scorer import DEFAULT_BATCH_SIZE, score_items
from src.eval.run_baseline import _aggregate_overall
from src.experiments.logger import (
    RESULTS_DIR,
    _write_atomically,
    new_run_id,
    read_question_results,
    write_category_summary,
    write_question_results,
)
from src.experiments.schema import QuestionResult, summarize_category

CONDITION = "hqq_repaired"
# Must match the baseline, the hqq_3bit run and the search that chose the modules.
COMPUTE_DTYPE = "fp16"


@dataclass(frozen=True)
class RepairPlan:
    """What to restore, as decided by one search run."""

    modules: tuple[str, ...]
    nbits: int
    search_run_id: str


@dataclass(frozen=True)
class RepairMemory:
    deployed_overhead_bytes: int
    gpu_allocated_before_bytes: int
    gpu_allocated_after_bytes: int

    @property
    def gpu_delta_bytes(self) -> int:
        return self.gpu_allocated_after_bytes - self.gpu_allocated_before_bytes


def load_repair_plan(summary: dict) -> RepairPlan:
    """Reads the modules to restore out of a search summary.json, refusing anything inconsistent.

    Each refusal guards against repairing a different model than the search measured: the wrong
    dtype, an empty result, or a precision other than the one the search tried.
    """
    for key in ("kept", "config", "compute_dtype", "run_id"):
        if key not in summary:
            raise ValueError(f"search summary is missing {key!r} -- is this a run_search summary.json?")
    if summary["compute_dtype"] != COMPUTE_DTYPE:
        raise ValueError(
            f"search ran at {summary['compute_dtype']!r} but the repair runs at {COMPUTE_DTYPE!r}; "
            "never compare runs taken at different dtypes"
        )
    modules = tuple(summary["kept"])
    if not modules:
        raise ValueError("the search kept no modules, so there is nothing to repair")
    if len(set(modules)) != len(modules):
        raise ValueError(f"search summary lists a module twice: {modules}")
    return RepairPlan(modules=modules, nbits=int(summary["config"]["nbits"]), search_run_id=summary["run_id"])


def apply_repair(
    registry,
    plan: RepairPlan,
    memory_allocated: Callable[[], int],
) -> RepairMemory:
    """Restores every planned module, keeps them, and measures what that cost.

    `release_baselines()` runs before the "after" reading: the registry otherwise keeps each
    displaced 3-bit module alive so it could revert, and that search-time cost isn't part of the
    deployed repair. After it, `revert` raises -- correct here, since this model is never reverted.
    """
    before = memory_allocated()
    for path in plan.modules:
        registry.commit(path, plan.nbits)
    registry.release_baselines()
    after = memory_allocated()

    if registry.active != {path: plan.nbits for path in plan.modules}:
        raise RuntimeError(f"registry reports {registry.active} after applying {plan.modules}")
    return RepairMemory(
        deployed_overhead_bytes=registry.deployed_overhead_bytes(),
        gpu_allocated_before_bytes=before,
        gpu_allocated_after_bytes=after,
    )


def linear_params(module) -> int:
    """Weight parameters of a Linear, whether plain nn.Linear or a packed HQQLinear.

    Read off in/out features rather than the stored tensors, because a quantized module's packed
    storage says nothing about how many weights it represents.
    """
    return int(module.in_features) * int(module.out_features)


def restored_share(model, candidates: list[str], restored: tuple[str, ...]) -> dict:
    """How much of the searchable model the repair runs at higher precision, two ways."""
    from src.quant.module_swap import get_module_by_path

    unknown = set(restored) - set(candidates)
    if unknown:
        raise ValueError(f"restored modules are not search candidates: {sorted(unknown)}")
    params = {path: linear_params(get_module_by_path(model, path)) for path in candidates}
    params_restored = sum(params[path] for path in restored)
    params_total = sum(params.values())
    return {
        "modules_restored": len(restored),
        "modules_total": len(candidates),
        "pct_modules_restored": 100 * len(restored) / len(candidates),
        "params_restored": params_restored,
        "params_total": params_total,
        "pct_params_restored": 100 * params_restored / params_total,
    }


def _cuda_memory_allocated() -> int:
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--search-summary", type=Path, required=True, help="summary.json from run_search")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--resume",
        metavar="RUN_ID",
        help="continue an interrupted run, skipping categories it already finished",
    )
    args = parser.parse_args()

    plan = load_repair_plan(json.loads(args.search_summary.read_text(encoding="utf-8")))

    if args.resume:
        run_id = args.resume
        if not run_id.startswith(f"{CONDITION}_"):
            raise SystemExit(f"Run {run_id!r} is not a {CONDITION} run. Resume only runs this script started.")
        completed = read_question_results(run_id, missing_ok=True)
    else:
        run_id = new_run_id(CONDITION)
        completed = []

    if not torch.cuda.is_available():
        raise SystemExit("No GPU found. The repair run needs CUDA -- on Kaggle, set Accelerator to GPU.")
    device = "cuda"

    # Imported here so --help and the tests work without hqq/GPU.
    from src.eval.model_loader import MODEL_ID
    from src.quant.candidates import candidate_paths_for_model
    from src.quant.hqq_quantizer import load_hqq_3bit_model
    from src.quant.module_swap import SwapRegistry
    from src.quant.weight_source import CheckpointWeightSource

    print(f"\nLoading Phi-3.5-mini-Instruct, HQQ 3-bit, {COMPUTE_DTYPE} compute...")
    model, tokenizer = load_hqq_3bit_model(device=device, dtype=COMPUTE_DTYPE)
    registry = SwapRegistry(model, CheckpointWeightSource(MODEL_ID), device=device)

    # Read before the repair: sizes come off the modules as the model is loaded, 3-bit throughout.
    share = restored_share(model, candidate_paths_for_model(model), plan.modules)

    # The cycle's clock starts at the repair, not at model load: loading and quantizing the base
    # model is the same fixed cost with or without a repair, so it isn't the repair's cost.
    cycle_start = time.time()
    print(f"\nRestoring {len(plan.modules)} modules to {plan.nbits}-bit (from search {plan.search_run_id}):")
    for path in plan.modules:
        print(f"  {path}")
    memory = apply_repair(registry, plan, _cuda_memory_allocated)
    repair_seconds = time.time() - cycle_start
    print(f"Deployed overhead: {memory.deployed_overhead_bytes / 2**20:.1f} MB")
    print(f"GPU allocated delta (cross-check): {memory.gpu_delta_bytes / 2**20:.1f} MB")
    print(
        f"Restored: {share['pct_params_restored']:.2f}% of candidate parameters "
        f"({share['modules_restored']}/{share['modules_total']} modules)"
    )

    print("\nLoading BBQ ambiguous condition, all 5 categories...")
    items_by_category = load_all_categories()

    print(f"\nRun ID: {run_id} (condition {CONDITION}, batch size {args.batch_size})")
    finished_categories = {result.category for result in completed}
    if finished_categories:
        print(f"Resuming: {sorted(finished_categories)} already scored, skipping.")

    all_results: list[QuestionResult] = list(completed)
    for category, items in items_by_category.items():
        if category in finished_categories:
            continue
        predictions = score_items(model, tokenizer, items, device, batch_size=args.batch_size, desc=category)
        all_results.extend(
            QuestionResult(
                run_id=run_id,
                condition=CONDITION,
                question_id=item.question_id,
                category=item.category,
                predicted_index=predicted_index,
                label=item.label,
                unknown_index=item.unknown_index,
                is_unknown_predicted=(predicted_index == item.unknown_index),
                is_correct=(predicted_index == item.label),
                is_stereotyped_predicted=(predicted_index == item.stereotype_index),
            )
            for item, predicted_index in zip(items, predictions)
        )
        # Checkpoint after every category, as in run_baseline.
        write_question_results(all_results, run_id)

    scoring_seconds = time.time() - cycle_start - repair_seconds

    summaries = [summarize_category(all_results, run_id, CONDITION, category) for category in CATEGORIES]
    overall = _aggregate_overall(all_results, run_id, CONDITION)
    write_category_summary(summaries + [overall], run_id)

    repair_summary = {
        "run_id": run_id,
        "condition": CONDITION,
        "compute_dtype": COMPUTE_DTYPE,
        "search_run_id": plan.search_run_id,
        "search_summary_path": str(args.search_summary),
        "modules": list(plan.modules),
        "nbits": plan.nbits,
        "fairness": {
            "overall": {"usr": overall.usr, "srs": overall.srs, "n_items": overall.n_items},
            "per_category": {s.category: {"usr": s.usr, "srs": s.srs, "n_items": s.n_items} for s in summaries},
        },
        "restored": share,
        "memory": {**asdict(memory), "gpu_delta_bytes": memory.gpu_delta_bytes},
        "wall_clock_seconds": {
            "repair": round(repair_seconds, 1),
            "scoring": round(scoring_seconds, 1),
            "cycle": round(repair_seconds + scoring_seconds, 1),
        },
        # A resumed run re-applies the repair but only times the categories scored this session,
        # so its clock undercounts the cycle. Flagged so nobody reports it as the full cost.
        "resumed": bool(args.resume),
    }
    _write_atomically(RESULTS_DIR / run_id / "summary.json", lambda handle: json.dump(repair_summary, handle, indent=2))

    print(f"\n{'Category':<20} {'N':>6} {'USR':>8} {'SRS':>8}")
    for summary in summaries + [overall]:
        name = "OVERALL" if summary.category == "overall" else summary.category
        srs_display = f"{summary.srs:.3f}" if summary.srs is not None else "n/a"
        print(f"{name:<20} {summary.n_items:>6} {summary.usr:>8.3f} {srs_display:>8}")
    print(f"\nCycle wall clock: {repair_summary['wall_clock_seconds']['cycle'] / 60:.1f} min"
          + (" (resumed -- partial)" if args.resume else ""))
    print(f"Results written to {(RESULTS_DIR / run_id).resolve()} (incl. summary.json)")
    print(f"Next: make promote-reference RUN_ID={run_id} NAME={CONDITION}")


if __name__ == "__main__":
    main()
