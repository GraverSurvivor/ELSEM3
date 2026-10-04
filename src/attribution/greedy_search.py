"""Stage 4: greedy symptom-conditioned attribution search.

Each round tries promoting every candidate not yet kept, one at a time, and measures USR on the
flipped-set sample with that one module restored. The single best candidate is kept if its gain
clears the threshold; otherwise the search stops. The kept modules, in the order they were kept,
are the attribution result -- the components the evidence says the compression damaged.

The loop is written against two small interfaces rather than the real model, so the decision
logic is testable on CPU:

- `registry` -- anything with `trial(path, nbits)` (a context manager that promotes and then
  puts back exactly what was there), `commit(path, nbits)`, and `active`. `SwapRegistry` is the
  real one; its revert-in-a-finally is what makes trying 128 candidates per round safe.
- `score_fn` -- a zero-argument callable returning USR on the search sample for the model as it
  currently stands. The GPU driver (`run_search.py`) builds it from the option scorer.

Why re-measure after each commit instead of reusing the winning trial's score: `commit` installs
a fresh 8-bit module rather than keeping the trial's object, and CLAUDE.md is explicit that HQQ's
quantizer is not assumed bit-deterministic. One extra scoring pass per round makes the next
round's gains relative to the model that actually exists.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Callable, ContextManager, Iterable, Protocol

# Stop reasons, recorded in the result so a run's log says why it ended.
STOP_THRESHOLD = "no candidate cleared the threshold"
STOP_MAX_ROUNDS = "reached max_rounds"
STOP_RECOVERED = "fully recovered (USR 1.0 on the sample)"
STOP_EXHAUSTED = "every candidate already kept"


class Registry(Protocol):
    @property
    def active(self) -> dict[str, int]: ...

    def trial(self, path: str, nbits: int) -> ContextManager[object]: ...

    def commit(self, path: str, nbits: int) -> None: ...


@dataclass(frozen=True)
class SearchConfig:
    """The search's knobs. The threshold is a setting, not a constant, per the plan."""

    # Minimum USR gain on the sample for a candidate to be kept. Must be > 0: at 0, a module
    # that changes nothing would still be "kept", inflating the repair's memory cost for free.
    threshold: float = 0.01
    # Precision a candidate is restored to while trying it and after keeping it.
    nbits: int = 8
    # Upper bound on modules kept, and so on rounds. Also bounds GPU time: each round costs one
    # scoring pass per remaining candidate.
    max_rounds: int = 5

    def __post_init__(self) -> None:
        if not self.threshold > 0:
            raise ValueError(f"threshold must be > 0, got {self.threshold}")
        if self.nbits not in (4, 8):
            raise ValueError(f"nbits must be 4 or 8 (HQQ's promotion targets here), got {self.nbits}")
        if self.max_rounds < 1:
            raise ValueError(f"max_rounds must be >= 1, got {self.max_rounds}")


@dataclass(frozen=True)
class Trial:
    """One candidate tried in one round. The search log is a list of these."""

    round: int
    path: str
    usr: float
    gain: float
    kept: bool

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SearchResult:
    start_usr: float
    kept: list[str] = field(default_factory=list)
    # USR measured after each keep; usr_after_round[i] is the model with kept[: i + 1] promoted.
    usr_after_round: list[float] = field(default_factory=list)
    trials: list[Trial] = field(default_factory=list)
    stop_reason: str = ""

    @property
    def final_usr(self) -> float:
        return self.usr_after_round[-1] if self.usr_after_round else self.start_usr

    def to_dict(self) -> dict:
        return {
            "start_usr": self.start_usr,
            "final_usr": self.final_usr,
            "kept": list(self.kept),
            "usr_after_round": list(self.usr_after_round),
            "stop_reason": self.stop_reason,
            "n_trials": len(self.trials),
        }


def greedy_search(
    candidates: Iterable[str],
    registry: Registry,
    score_fn: Callable[[], float],
    config: SearchConfig = SearchConfig(),
    on_progress: Callable[["SearchResult"], None] | None = None,
    resume: "SearchResult | None" = None,
) -> SearchResult:
    """Runs the search and returns which modules to keep, in the order they were kept.

    `candidates` should come from `candidate_paths_for_model()`, which orders attention before
    MLP. That order is also the tie-break: on an exact tie the earlier candidate wins, so equal
    recovery is bought with the cheaper (attention) module.

    `on_progress` is called with the result-so-far after every trial and after every keep -- the
    driver uses it to checkpoint the log, so an interrupted search still leaves every measurement
    it paid for on disk.

    `resume` continues an interrupted search from that checkpoint: its kept modules are committed
    again, finished rounds are not repeated, and in the unfinished round only the candidates not yet
    tried are scored. The result is the same search as an uninterrupted one, because scoring is
    deterministic and the tie-break walks the candidates in the same order.
    """
    candidates = list(candidates)
    if len(set(candidates)) != len(candidates):
        raise ValueError("candidate list contains duplicates")

    if resume is None:
        result = SearchResult(start_usr=score_fn())
        current_usr = result.start_usr
    else:
        result = SearchResult(start_usr=resume.start_usr, kept=list(resume.kept),
                              usr_after_round=list(resume.usr_after_round), trials=list(resume.trials))
        for path in result.kept:
            registry.commit(path, config.nbits)
        if len(result.usr_after_round) < len(result.kept):  # interrupted right after a keep
            result.usr_after_round.append(score_fn())
        current_usr = result.final_usr

    for round_number in range(len(result.kept) + 1, config.max_rounds + 1):
        if current_usr >= 1.0:
            result.stop_reason = STOP_RECOVERED
            return result

        remaining = [path for path in candidates if path not in registry.active]
        if not remaining:
            result.stop_reason = STOP_EXHAUSTED
            return result

        already = {t.path: i for i, t in enumerate(result.trials) if t.round == round_number}
        best_index: int | None = None
        for path in remaining:
            if path in already:
                index = already[path]
            else:
                with registry.trial(path, config.nbits):
                    usr = score_fn()
                result.trials.append(
                    Trial(round=round_number, path=path, usr=usr, gain=usr - current_usr, kept=False)
                )
                index = len(result.trials) - 1
            # Strict ">" so the earliest candidate wins a tie.
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
        current_usr = score_fn()
        result.usr_after_round.append(current_usr)
        if on_progress is not None:
            on_progress(result)

    result.stop_reason = STOP_MAX_ROUNDS
    return result
