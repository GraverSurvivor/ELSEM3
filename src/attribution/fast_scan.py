"""Fast Scan (contract C2): the greedy attribution search without the wasted work.

`greedy_search` re-scores all remaining candidates on the full sample in every round. Replaying the
real search showed the winners of every round sit in round 1's top 10 of 128, so Fast Scan

  1. screens every candidate once on a small seeded subset of the sample and probe;
  2. keeps the `survivors` best and scores them on the full sample + probe (exact mode only);
  3. runs rounds 2..max_rounds greedily, but only over the `pool` best of round 1.

Modes. "exact" scores step 2 and the later rounds on the full sample + probe. "fast" does everything
on the screening subset (steps 1-3 reuse the same questions), which is what makes it ~3x cheaper
again; `survivors` has no effect there because nothing is re-scored. How often fast mode picks the
same modules is measured offline by `replay.py`, not assumed.

With survivors = pool = len(candidates) and subsets covering the whole sample/probe, nothing is
pruned or re-scored and the result equals `greedy_search` step for step (same trials, same
tie-break: on an exact tie the earliest candidate wins). The property test pins that.

Round-1 trials: one per candidate. A survivor's trial carries its full-sample score; a pruned
candidate's carries its screening-subset score and gain against the screening baseline, so
`usr` of pruned and surviving rows are not on the same question set unless the subset is the sample.

Resume: a checkpoint from a finished round 1 onward continues exactly (kept modules are committed
again; in the unfinished round only untried candidates are scored; the pool comes from the
checkpoint). A checkpoint from before round 1 finished is restarted from the screen -- the pool is
not known yet.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

from src.attribution.greedy_search import (
    STOP_EXHAUSTED,
    STOP_MAX_ROUNDS,
    STOP_RECOVERED,
    STOP_THRESHOLD,
    Registry,
    SearchResult,
    Trial,
)

MODES = ("fast", "exact")


@dataclass(frozen=True)
class ScanScore:
    """What one scoring pass returns: the objective plus per-question results (see item_log.py)."""

    objective: float  # correct rate - penalty_weight x false-"unknown" rate
    sample_bits: str = ""
    probe_bits: str = ""


Scorer = Callable[[Sequence[str], Sequence[str]], ScanScore]


@dataclass(frozen=True)
class FastScanConfig:
    mode: str = "fast"
    screen_sample: int = 60
    screen_probe: int = 40
    survivors: int = 40
    pool: int = 16
    max_rounds: int = 5
    threshold: float = 0.01
    nbits: int = 8
    penalty_weight: float = 1.0  # used by the scorer the driver builds; recorded here for the run's provenance
    seed: int = 0

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        for name in ("screen_sample", "survivors", "pool", "max_rounds"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1, got {getattr(self, name)}")
        if self.screen_probe < 0:
            raise ValueError(f"screen_probe must be >= 0, got {self.screen_probe}")
        if not self.threshold > 0:
            raise ValueError(f"threshold must be > 0, got {self.threshold}")
        if self.nbits not in (4, 8):
            raise ValueError(f"nbits must be 4 or 8, got {self.nbits}")


@dataclass
class FastScanResult(SearchResult):
    """A SearchResult that also remembers round 1's screening outcome (needed to resume)."""

    survivors: list[str] = field(default_factory=list)
    pool: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {**super().to_dict(), "survivors": list(self.survivors), "pool": list(self.pool)}


def seeded_subset(ids: Sequence[str], size: int, rng: random.Random) -> list[str]:
    """`size` of the ids, chosen by `rng`, in their original order; all of them if size >= len(ids)."""
    ids = list(ids)
    if size >= len(ids):
        return ids
    keep = sorted(rng.sample(range(len(ids)), size))
    return [ids[i] for i in keep]


def _top(paths: Sequence[str], objective: dict[str, float], n: int, order: Sequence[str]) -> list[str]:
    """The n paths with the highest objective, ties to the earlier candidate, returned in candidate order."""
    position = {p: i for i, p in enumerate(order)}
    ranked = sorted(paths, key=lambda p: (-objective[p], position[p]))[:n]
    return sorted(ranked, key=position.__getitem__)


def fast_scan(
    candidates: Iterable[str],
    registry: Registry,
    scorer: Scorer,
    sample_ids: Sequence[str],
    probe_ids: Sequence[str],
    config: FastScanConfig = FastScanConfig(),
    on_progress: Callable[[SearchResult], None] | None = None,
    resume: SearchResult | None = None,
) -> FastScanResult:
    candidates = list(candidates)
    if len(set(candidates)) != len(candidates):
        raise ValueError("candidate list contains duplicates")
    sample_ids, probe_ids = list(sample_ids), list(probe_ids)

    rng = random.Random(config.seed)
    screen_s = seeded_subset(sample_ids, config.screen_sample, rng)
    screen_p = seeded_subset(probe_ids, config.screen_probe, rng)
    screen_is_full = len(screen_s) == len(sample_ids) and len(screen_p) == len(probe_ids)
    exact = config.mode == "exact"
    # The question set later rounds are scored on, and the one `start_usr`/`usr` are measured on.
    work_s, work_p = (sample_ids, probe_ids) if exact else (screen_s, screen_p)

    def work_score() -> float:
        return scorer(work_s, work_p).objective

    keep_resume = resume is not None and bool(resume.kept)
    if keep_resume:
        pool = list(getattr(resume, "pool", []))
        if not pool:
            raise ValueError("cannot resume a Fast Scan checkpoint that has kept modules but no pool")
        result = FastScanResult(start_usr=resume.start_usr, kept=list(resume.kept),
                                usr_after_round=list(resume.usr_after_round), trials=list(resume.trials),
                                survivors=list(getattr(resume, "survivors", [])), pool=pool)
        for count, path in enumerate(result.kept, start=1):
            registry.commit(path, config.nbits)
            if len(result.usr_after_round) < count:
                result.usr_after_round.append(work_score())
        current_usr = result.final_usr
    else:
        result = FastScanResult(start_usr=work_score())
        current_usr = result.start_usr

    for round_number in range(len(result.kept) + 1, config.max_rounds + 1):
        if current_usr >= 1.0:
            result.stop_reason = STOP_RECOVERED
            return result
        scope = candidates if round_number == 1 else result.pool
        remaining = [p for p in scope if p not in registry.active]
        if not remaining:
            result.stop_reason = STOP_EXHAUSTED
            return result

        if round_number == 1:
            best_index = _round_one(result, remaining, registry, scorer, config, exact, screen_is_full,
                                    (sample_ids, probe_ids), (screen_s, screen_p), candidates, on_progress)
        else:
            already = {t.path: i for i, t in enumerate(result.trials) if t.round == round_number}
            best_index = None
            for path in remaining:
                if path in already:
                    index = already[path]
                else:
                    with registry.trial(path, config.nbits):
                        usr = work_score()
                    result.trials.append(Trial(round=round_number, path=path, usr=usr,
                                               gain=usr - current_usr, kept=False))
                    index = len(result.trials) - 1
                if best_index is None or result.trials[index].gain > result.trials[best_index].gain:
                    best_index = index
                if on_progress is not None and path not in already:
                    on_progress(result)

        best = result.trials[best_index]
        if best.gain < config.threshold:
            result.stop_reason = STOP_THRESHOLD
            if on_progress is not None:
                on_progress(result)
            return result

        registry.commit(best.path, config.nbits)
        result.trials[best_index] = Trial(**{**best.to_dict(), "kept": True})
        result.kept.append(best.path)
        current_usr = work_score()
        result.usr_after_round.append(current_usr)
        if on_progress is not None:
            on_progress(result)

    result.stop_reason = STOP_MAX_ROUNDS
    return result


def _round_one(result, remaining, registry, scorer, config, exact, screen_is_full, full, screen, candidates,
               on_progress) -> int:
    """Steps 1-2: screen every candidate, keep the survivors, score them. Returns the winner's trial index."""
    (sample_ids, probe_ids), (screen_s, screen_p) = full, screen
    # Step 1: screen. In fast mode, or when the subset is the whole sample, this baseline is start_usr itself.
    screen_start = result.start_usr if (not exact or screen_is_full) else scorer(screen_s, screen_p).objective
    screen_obj: dict[str, float] = {}
    for path in remaining:
        with registry.trial(path, config.nbits):
            score = scorer(screen_s, screen_p).objective
        screen_obj[path] = score
        result.trials.append(Trial(round=1, path=path, usr=score, gain=score - screen_start, kept=False))
        if on_progress is not None:
            on_progress(result)

    # Step 2: survivors. Fast mode and full-subset screens have nothing to re-score.
    n_survivors = min(config.survivors, len(remaining)) if exact else len(remaining)
    survivors = _top(remaining, screen_obj, n_survivors, candidates)
    final_obj = dict(screen_obj)
    if exact and not screen_is_full:
        index_of = {t.path: i for i, t in enumerate(result.trials) if t.round == 1}
        for path in survivors:
            with registry.trial(path, config.nbits):
                score = scorer(sample_ids, probe_ids).objective
            final_obj[path] = score
            result.trials[index_of[path]] = Trial(round=1, path=path, usr=score,
                                                  gain=score - result.start_usr, kept=False)
            if on_progress is not None:
                on_progress(result)

    result.survivors = survivors
    result.pool = _top(survivors, final_obj, min(config.pool, len(survivors)), candidates)
    best_index = None
    for i, t in enumerate(result.trials):
        if t.round == 1 and t.path in survivors and (best_index is None or t.gain > result.trials[best_index].gain):
            best_index = i
    return best_index
