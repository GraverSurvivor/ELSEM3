"""Version 2's repair: every part is measured again before it is restored, and the finished repair is
compared with alternatives on questions that chose none of them.

Why. Version 1 fills its budget by each part's score from the scan's first round. Replaying the 630-trial
Phi-3.5 search showed that score says little once something has been restored: the 2nd and 3rd best parts
on their own (+30 and +29 points) give +1.3 and -2.8 after the best one, and 30-58% of all parts make
things worse in later rounds. Its scores also come from as few as 36 + 24 questions (Qwen2.5-3B).

How, on the search questions:

  1. look at every part once (SCREEN questions each), and at each part type across all layers together,
     for damage that is spread out;
  2. keep the POOL best, then go round by round: measure the most promising again on WORK questions, with
     everything restored so far in place, and take the best one that helps beyond its own noise, judged
     by gain per memory. Up to TOGETHER parts that fix different questions are taken at once, if
     measuring them together confirms it;
  3. each part taken at 8-bit is tried at 4-bit, and stays there if it keeps KEEP_SHARE of its gain: about
     a sixth of the extra memory (when the model is compressed to 3-bit);
  4. stop when the model is as fair as the original, at the memory budget, or when nothing helps;
  5. try taking parts out again, drop those that turned out not to be needed, and go on searching if that
     changed anything.

Then, on the validation questions: this repair, the one version 1's ordering would make with the same
budget, and 4-bit everywhere for reference. The repair that is closest to the original ships; 4-bit
everywhere is only reported (it isn't a repair of the parts that cause the damage), in plain words when
it would have done better. The final questions are read by none of this.
"""

from __future__ import annotations

import math
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field

from src.product.decisions import TrialScorer, WatchedRegistry, ask, mean

SCREEN = 75                # questions of the first look at a part, at least
WORK = (120, 60)           # (sample, stated-answer) questions of every later measurement; "exact": all of them
POOL = 32                  # parts and groups that stay in the running after the first look
FIRST_EVALS = 16           # re-measured in the first round (the first look is noisy), at most
MIN_EVALS, MAX_EVALS = 6, 14  # and in each later round
TOGETHER = 3               # parts taken per round, at most
OVERLAP = 0.5              # ... that share no more than this of the questions they help
MIN_GAIN, NOISE = 0.004, 1.5  # a part must gain at least this much, and this many standard errors
KEEP_SHARE = 0.8           # 4-bit keeps this share of a part's 8-bit gain: 4-bit it is
DROP_TRIALS, DROP_LOSS, DROP_TOTAL = 16, 0.002, 0.01  # clean-up: trials, loss per part, loss altogether
PASSES = 3                 # search and clean-up, at most this often
PRUNE_TRIALS = 12          # ... and trials for thinning out a part type that was taken as a whole
DAMP = 0.02                # of the budget, added to every cost, so tiny parts don't win on noise
CHECK = (300, 150)         # search questions the "as fair as the original?" test asks
QUESTION_LIMIT = {"fast": 45_000, "exact": 150_000}  # no new round is started past this
BETTER = 0.01              # validation: one repair beats another only by more than this

STOP_FAIR = "as fair as the original"
STOP_BUDGET = "memory budget reached"
STOP_NOTHING = "nothing more helps"
STOP_LIMIT = "question limit reached"


@dataclass(frozen=True)
class Unit:
    """Something that can be restored in one go: one part, or one part type in every layer."""

    name: str
    parts: tuple[str, ...]
    bits: int


@dataclass
class Reading:
    """The model as it stands, question by question: `sample` is what each question earns, `stated` what
    each stated-answer question costs (decisions.TrialScorer, the deciding score's column)."""

    sample: list[float]
    stated: list[float]


def gain(new: Reading, old: Reading) -> tuple[float, float]:
    """How much better `new` is, and the standard error of that: paired, question by question."""
    value, variance = 0.0, 0.0
    for sign, a, b in ((1, new.sample, old.sample), (-1, new.stated, old.stated)):
        diff = [sign * (x - y) for x, y in zip(a, b)]
        if diff:
            m = mean(diff)
            value += m
            variance += sum((d - m) ** 2 for d in diff) / max(1, len(diff) - 1) / len(diff)
    return value, math.sqrt(variance)


def helped(new: Reading, old: Reading, step: float = 0.1) -> set[int]:
    """The sample questions that clearly move towards the right answer."""
    return {i for i, (x, y) in enumerate(zip(new.sample, old.sample)) if x - y >= step}


@contextmanager
def trying(registry, parts: dict, name: str | None = None):
    """Several parts restored for the duration of the block."""
    with ExitStack() as stack:
        for path, bits in parts.items():
            stack.enter_context(registry.trial(path, bits))
        if name is not None and isinstance(registry, WatchedRegistry):
            registry.trying = name
        yield


@contextmanager
def arranged(registry, wanted: dict):
    """Exactly `wanted` (part -> bits) restored for the duration of the block, whatever is now."""
    with ExitStack() as stack:
        for path, bits in dict(registry.active).items():
            if wanted.get(path) != bits:
                stack.enter_context(registry.without(path))
        for path, bits in wanted.items():
            if registry.active.get(path) != bits:
                stack.enter_context(registry.trial(path, bits))
        yield


@dataclass
class Repair:
    kept: list[str]
    bits: dict
    trials: list = field(default_factory=list)
    pool: list[str] = field(default_factory=list)
    stop_reason: str = ""
    rows: list[dict] = field(default_factory=list)  # the report's `strategies`
    budget_reached: bool = False
    advice: str = ""
    screen: tuple[int, int] = (0, 0)
    engine: str = "remeasured"


class Remeasure:
    def __init__(self, backend, adapter, candidates: list[str], sample: list, stated: list, scorer: TrialScorer, config,
                 base_bits: int, original: dict, check: tuple[list, list], screen: tuple[int, int], events=None,
                 log=None, speed: float = 1.0, cache=None):
        from src.product.fast_repair import REAL_BITS, seeded_subset

        self.backend, self.adapter, self.candidates, self.scorer, self.config = backend, adapter, candidates, scorer, config
        self.registry = WatchedRegistry(backend.registry)
        self.column = {"count": 0, "soft": 1, "kl": 2}[config.objective]
        self.base_bits, self.real_bits, self.original = base_bits, REAL_BITS, original
        self.events, self.log, self.speed, self.cache = events, log, speed, cache
        rng_subset = lambda items, n: seeded_subset(items, n, config.seed)  # noqa: E731
        wanted = scaled_screen(screen)
        self.screen = (rng_subset(sample, wanted[0]), rng_subset(stated, wanted[1]))
        self.work = (sample, stated) if config.mode == "exact" else (rng_subset(sample, WORK[0]), rng_subset(stated, WORK[1]))
        self.check = (rng_subset(check[0], CHECK[0]), rng_subset(check[1], CHECK[1]))
        share = config.max_restore if 0 < config.max_restore < 1 else 1.0
        self.budget = share * self.bytes_of(adapter.total_params(), config.nbits)
        self.reference_skipped = ""
        self.asked, self.lost, self.number = 0, 0.0, 1  # `number`: the round (1 is the first look)
        self.groups_taken: list[tuple[str, tuple[str, ...]]] = []
        self.result = Repair([], {}, screen=(len(self.screen[0]), len(self.screen[1])))
        n = len(candidates) + len(adapter.part_types)
        self.plan = (n * sum(map(len, self.screen)) + (POOL + 6 * MIN_EVALS) * sum(map(len, self.work))
                     + 6 * sum(map(len, self.check)))

    # --- memory ---------------------------------------------------------------------------------------
    def bytes_of(self, weights: int, bits: int) -> float:
        return weights * (self.real_bits[bits] - self.real_bits[self.base_bits]) / 8

    def added(self, parts: dict) -> float:
        """Extra bytes of a repair (part -> bits) over the compressed model."""
        return sum(self.bytes_of(self.adapter.params(p), b) for p, b in parts.items())

    def todo(self, unit: Unit) -> dict:
        """What restoring the unit would change now."""
        active = self.registry.active
        return {p: unit.bits for p in unit.parts if active.get(p, self.base_bits) < unit.bits}

    def cost(self, parts: dict) -> float:
        active = self.registry.active
        return sum(self.bytes_of(self.adapter.params(p), b)
                   - (self.bytes_of(self.adapter.params(p), active[p]) if p in active else 0) for p, b in parts.items())

    # --- measuring --------------------------------------------------------------------------------------
    def read(self, questions: tuple[list, list]) -> Reading:
        sample, stated = questions
        rows = self.scorer.rows(sample, *ask(self.backend, "compressed", sample), stated,
                                *(ask(self.backend, "compressed", stated) if stated else ([], [])))
        self.asked += len(sample) + len(stated)
        if self.log is not None:
            from src.product.decisions import combine

            self.log.write(self.registry, len(sample), len(stated), combine(*rows, self.scorer.weight))
        sign = -1 if self.column == 2 else 1  # KL: smaller is better
        return Reading([sign * r[self.column] for r in rows[0]],
                       [self.scorer.weight * r[self.column] for r in rows[1]])

    def try_parts(self, parts: dict, name: str, questions) -> Reading:
        with trying(self.registry, parts, name):
            return self.read(questions)

    def say(self, message: str) -> None:
        if self.events is not None:
            done = min(self.asked, self.plan)
            left = (self.plan - done) / self.speed if self.speed else 0
            from src.product.fast_repair import minutes_text

            self.events.progress(done, self.plan, message + (f" · {minutes_text(left)} left" if left else ""))

    def fair(self, questions: tuple[list, list]) -> bool:
        from src.product.fast_repair import FAIR_HEDGE, FAIR_SRS_RISE, FAIR_USR_DROP, fairness_gap, hedging_gap

        ambig, stated = questions
        if not ambig:
            return False
        drop, rise = fairness_gap(self.backend, ambig, self.original)
        self.asked += len(ambig) + len(stated)
        return (drop <= FAIR_USR_DROP and rise <= FAIR_SRS_RISE
                and hedging_gap(self.backend, stated, self.original) <= FAIR_HEDGE)

    # --- the repair ---------------------------------------------------------------------------------------
    def units(self) -> list[Unit]:
        """Every part on its own, and each part type in every layer (at 4-bit when that is above the base:
        the cheapest way to lift a whole type)."""
        singles = [Unit(p, (p,), self.config.nbits) for p in self.candidates]
        bits = 4 if self.base_bits < 4 else self.config.nbits
        groups = []
        for part_type in self.adapter.part_types:
            parts = tuple(p for p in self.candidates if p.endswith("." + part_type))
            if len(parts) > 1:
                groups.append(Unit(f"every {part_type}", parts, bits))
        return singles + groups

    def first_look(self) -> dict[Unit, float]:
        from src.attribution.greedy_search import Trial

        import json

        start = self.read(self.screen)
        key = [it.question_id for items in self.screen for it in items]
        saved = {}
        if self.cache is not None and self.cache.exists():  # a resumed run: the first look is already done
            try:
                data = json.loads(self.cache.read_text(encoding="utf-8"))
                saved = data["gains"] if data.get("questions") == key and data.get("start") == start.sample else {}
            except (ValueError, KeyError):
                saved = {}
        found, units = {}, self.units()
        for k, unit in enumerate(units, start=1):
            parts = self.todo(unit)
            if len(unit.parts) > 1 and self.cost(parts) > self.budget:
                continue  # a group too big for the budget can't be taken anyway
            if unit.name in saved:
                value = saved[unit.name]
            else:
                value, _ = gain(self.try_parts(parts, unit.name, self.screen), start)
                saved[unit.name] = value
                if self.cache is not None and k % 16 == 0:
                    self.cache.write_text(json.dumps({"questions": key, "start": start.sample, "gains": saved}),
                                          encoding="utf-8")
            found[unit] = value
            self.result.trials.append(Trial(round=1, path=unit.name, usr=mean(start.sample) + value, gain=value, kept=False))
            self.say(f"Step 1 of 3 · Quick look at every part: {min(k, len(self.candidates))} of {len(self.candidates)}")
        if self.cache is not None:
            self.cache.write_text(json.dumps({"questions": key, "start": start.sample, "gains": saved}), encoding="utf-8")
        return found

    def priority(self, unit: Unit, value: float) -> float:
        return value / (self.cost(self.todo(unit)) + DAMP * self.budget)

    def run(self) -> Repair:
        result, registry = self.result, self.registry
        first = self.first_look()
        self.first = first
        pool = sorted((u for u in first if first[u] > 0), key=lambda u: -self.priority(u, first[u]))[: POOL // 2]
        rest = sorted((u for u in first if first[u] > 0 and u not in pool), key=lambda u: -first[u])
        pool += rest[: POOL - len(pool)]  # half by gain per memory, half by gain
        result.pool = [u.name for u in pool]
        current = self.read(self.work) if pool else None
        took: dict[str, float] = {}
        for _ in range(PASSES):
            # Taking parts out can open a better combination (a part type taken whole, then thinned out), so
            # the search goes on from there. A part type is taken whole once at most.
            whole = {name for name, _ in self.groups_taken}
            current = self.search({u: first[u] for u in pool if u.name not in whole}, current, took)
            if len(registry.active) < 2:
                break
            before = registry.active
            current = self.clean_up(took, current)
            if registry.active == before or result.stop_reason == STOP_LIMIT:
                break
        if not result.stop_reason:
            result.stop_reason = STOP_BUDGET if result.budget_reached else STOP_NOTHING
        result.budget_reached = result.budget_reached and result.stop_reason != STOP_FAIR
        result.bits = dict(registry.active)
        result.kept = [path for path in took if path in result.bits]
        return result

    def search(self, stale: dict, current: Reading, took: dict) -> Reading:
        """Rounds until the model is as fair as the original, the budget is reached or nothing helps."""
        result, changed = self.result, bool(took)
        result.stop_reason = ""
        while True:
            if changed and self.fair(self.check):
                result.stop_reason = STOP_FAIR
                break
            if self.asked > QUESTION_LIMIT[self.config.mode]:
                result.stop_reason = STOP_LIMIT
                break
            if not stale:
                break
            self.number += 1
            current, gained = self.round(self.number, stale, current)
            changed = bool(gained)
            if gained == {}:
                break
            took.update(gained or {})
        return current

    def round(self, number: int, stale: dict, current: Reading) -> tuple[Reading, dict | None]:
        """Restores the next best. Returns the model's new reading and what each restored part gained: {} when
        nothing helps, None when only what was measured doesn't (the rest of the pool is left to try)."""
        from src.attribution.greedy_search import Trial

        result, registry = self.result, self.registry
        used, fresh = self.added(registry.active), {}
        for unit in [u for u in stale if not self.todo(u)]:
            del stale[unit]
        fits = [u for u in stale if used + self.cost(self.todo(u)) <= self.budget]
        result.budget_reached = any(stale[u] > MIN_GAIN for u in stale if u not in fits)
        order = sorted(fits, key=lambda u: -self.priority(u, stale[u]))
        limit = FIRST_EVALS if number == 2 else MAX_EVALS

        def good(unit) -> bool:
            value, error, _ = fresh[unit]
            return value > MIN_GAIN and value > NOISE * error

        def score(unit) -> float:
            return self.priority(unit, fresh[unit][0])

        for unit in order:
            best = max((score(u) for u in fresh if good(u)), default=None)
            enough = len(fresh) >= (limit if number == 2 else MIN_EVALS)
            if len(fresh) >= limit or (enough and best is not None and best >= self.priority(unit, stale[unit])):
                break
            reading = self.try_parts(self.todo(unit), unit.name, self.work)
            value, error = gain(reading, current)
            fresh[unit] = (value, error, reading)
            result.trials.append(Trial(round=number, path=unit.name, usr=mean(reading.sample), gain=value, kept=False))
            self.say(f"Step 2 of 3 · Round {number - 1}: measuring again with {len(registry.active)} parts restored")
        for unit, (value, _, _) in fresh.items():
            stale[unit] = value
        ranked = sorted((u for u in fresh if good(u)), key=lambda u: -score(u))
        if not ranked:
            # What was just measured doesn't help any more; whatever wasn't reached may still.
            for unit in fresh:
                del stale[unit]
            return current, None if any(u in stale and stale[u] > MIN_GAIN for u in order) else {}

        chosen, reading = self.take_together(ranked, fresh, current, used)
        parts = {p: b for u in chosen for p, b in self.todo(u).items()}
        parts, reading = self.right_size(parts, chosen, fresh, reading)
        gained = {}
        for path, bits in parts.items():
            registry.commit(path, bits)
        for unit in chosen:
            if len(unit.parts) > 1:
                self.groups_taken.append((unit.name, tuple(p for p in unit.parts if p in parts)))
            for path in unit.parts:
                if path in parts:
                    gained[path] = fresh[unit][0] / len(unit.parts)
            for i, trial in enumerate(result.trials):
                if trial.round == number and trial.path == unit.name:
                    result.trials[i] = Trial(**{**trial.to_dict(), "kept": True})
        return reading, gained

    def take_together(self, ranked: list[Unit], fresh: dict, current: Reading, used: float) -> tuple[list[Unit], Reading]:
        """The best unit, with up to TOGETHER - 1 others that help different questions, if measuring them
        together shows that they add up."""
        best = ranked[0]
        chosen, seen = [best], helped(fresh[best][2], current)
        room = self.budget - used - self.cost(self.todo(best))
        for unit in ranked[1:]:
            if len(chosen) == TOGETHER:
                break
            mine, todo = helped(fresh[unit][2], current), self.todo(unit)
            taken = {p for u in chosen for p in u.parts}
            if not mine or len(mine & seen) > OVERLAP * len(mine) or taken & set(unit.parts) or self.cost(todo) > room:
                continue
            chosen.append(unit)
            seen |= mine
            room -= self.cost(todo)
        if len(chosen) > 1:
            parts = {p: b for u in chosen for p, b in self.todo(u).items()}
            reading = self.try_parts(parts, " + ".join(u.name for u in chosen), self.work)
            alone = fresh[best][0]
            if gain(reading, current)[0] >= alone + 0.5 * sum(fresh[u][0] for u in chosen[1:]):
                return chosen, reading
        return [best], fresh[best][2]

    def right_size(self, parts: dict, chosen: list[Unit], fresh: dict, reading: Reading) -> tuple[dict, Reading]:
        """Each chosen part that went to 8-bit is tried at 4-bit, the others staying as chosen."""
        if self.base_bits >= 4:
            return parts, reading
        for unit in chosen:
            if len(unit.parts) > 1 or parts.get(unit.parts[0], 0) <= 4:
                continue
            smaller = {**parts, unit.parts[0]: 4}
            at_four = self.try_parts(smaller, f"{unit.name} at 4-bit", self.work)
            if gain(reading, at_four)[0] <= (1 - KEEP_SHARE) * fresh[unit][0]:
                parts, reading = smaller, at_four
        return parts, reading

    def clean_up(self, took: dict, current: Reading) -> Reading:
        """Takes parts out, and leaves out the ones that aren't missed: part types taken as a whole by halves
        (which layers need it?), then single parts, least useful first."""
        self.lost = 0.0
        for _, parts in self.groups_taken:
            current = self.prune([p for p in parts if p in self.registry.active], current, [PRUNE_TRIALS])
        grouped = {p for _, parts in self.groups_taken for p in parts}
        singles = sorted((p for p in self.registry.active if p not in grouped), key=lambda p: took.get(p, 0.0))
        for path in singles[:DROP_TRIALS]:
            current, _ = self.leave_out([path], current)
        return current

    def leave_out(self, parts: list[str], current: Reading) -> tuple[Reading, bool]:
        """Takes `parts` out for good if the model hardly misses them. (the model's reading, whether it did)"""
        if not parts or len(parts) == len(self.registry.active):
            return current, False
        with ExitStack() as stack:
            for path in parts:
                stack.enter_context(self.registry.without(path))
            self.registry.trying = f"without {parts[0]}" + (f" and {len(parts) - 1} more" if len(parts) > 1 else "")
            reading = self.read(self.work)
            self.registry.trying = None
        loss = max(0.0, gain(current, reading)[0])
        self.say("Step 3 of 3 · Taking out parts that aren't needed")
        if loss > DROP_LOSS * max(1, math.sqrt(len(parts))) or self.lost + loss > DROP_TOTAL:
            return current, False
        for path in parts:
            self.registry.revert(path)
        self.lost += loss
        return reading, True

    def prune(self, parts: list[str], current: Reading, trials: list[int]) -> Reading:
        """Which half of these parts isn't needed? Then the same question inside the half that is."""
        if len(parts) < 2:
            return current
        middle = len(parts) // 2
        for half in (parts[:middle], parts[middle:]):
            if trials[0] <= 0:
                return current
            trials[0] -= 1
            current, gone = self.leave_out(half, current)
            if not gone:
                current = self.prune(half, current, trials)
        return current

    # --- comparing the finished repairs -------------------------------------------------------------------
    def first_scores(self) -> dict:
        """What version 1's ordering would restore: by first-look gain per weight, until the budget."""
        order = sorted((u for u in self.first if len(u.parts) == 1 and self.first[u] > 0),
                       key=lambda u: -self.first[u] / self.adapter.params(u.parts[0]))
        wanted, used = {}, 0.0
        for unit in order:
            extra = self.bytes_of(self.adapter.params(unit.parts[0]), unit.bits)
            if used + extra <= self.budget:
                wanted[unit.parts[0]] = unit.bits
                used += extra
        return wanted

    def choose(self, validation: tuple[list, list], compressed: dict) -> Repair:
        """Measures the finished repairs on the validation questions, and keeps the best one."""
        from src.product.fast_repair import rates

        result, registry = self.result, self.registry
        ambig, stated = validation
        if not ambig or not stated:
            return result

        def card(picks: dict) -> dict:
            return rates(ambig, [picks[id(it)] for it in ambig], stated, [picks[id(it)] for it in stated])

        goal = card(self.original)

        def measure(wanted: dict) -> dict:
            with arranged(registry, wanted):
                picks, _ = ask(self.backend, "compressed", ambig + stated)
            return card(dict(zip(map(id, ambig + stated), picks)))

        def row(name: str, label: str, wanted: dict, numbers: dict, ships: bool) -> dict:
            return {"name": name, "label": label, "parts": len(wanted), "extra_mb": round(self.added(wanted) / 2**20, 1),
                    "ships": ships, "chosen": False, "distance": round(distance(numbers, goal), 4),
                    "fair": within_limits(numbers, goal), **numbers}

        rows = [row("compressed", "Compressed, no repair", {}, card(compressed), False)]
        choices = {}
        if result.kept:
            choices["remeasured"] = dict(result.bits)
            rows.append(row("remeasured", "Re-measured repair", choices["remeasured"],
                            measure(choices["remeasured"]), True))
        earlier = self.first_scores()
        if earlier and earlier != choices.get("remeasured"):
            choices["first_scores"] = earlier
            rows.append(row("first_scores", "Parts by their first scores (version 1's order)", earlier,
                            measure(earlier), True))
        if self.base_bits < 4 and not getattr(self.backend, "streaming", False):
            # Only a reference, and it needs a second copy of the model at once: never let it fail the run (big
            # models built layer by layer are left out, as their GPU was sized for one copy).
            self.say("Comparing with 4-bit everywhere")
            everywhere = {p: 4 for p in self.candidates}
            try:
                rows.append(row("four_bits", "4-bit everywhere, no repair", everywhere, measure(everywhere), False))
            except Exception as error:  # noqa: BLE001 -- out of memory, most likely
                if "out of memory" not in str(error).lower():
                    raise
                import gc

                gc.collect()
                self.reference_skipped = "4-bit everywhere didn't fit in memory next to the repair, so it isn't compared"
        by_name = {r["name"]: r for r in rows}
        best = rows[0]
        for name in ("remeasured", "first_scores"):  # in this order: a later one must be clearly better
            other = by_name.get(name)
            if other and other["distance"] < best["distance"] - (BETTER if best["ships"] else 0.0):
                best = other
        best["chosen"] = True
        result.rows = rows
        if best["name"] != "remeasured":  # the search's own budget flag doesn't describe another repair
            result.budget_reached = best["name"] == "first_scores" and self.budget - self.added(earlier) < (
                min((self.bytes_of(self.adapter.params(p), self.config.nbits) for p in self.candidates if p not in earlier),
                    default=0.0))
        wanted = choices.get(best["name"], {})
        for path in list(registry.active):
            if wanted.get(path) != registry.active[path]:
                registry.revert(path)
        for path, bits in wanted.items():
            if path not in registry.active:
                registry.commit(path, bits)
        result.bits = dict(registry.active)
        result.kept = [p for p in result.kept if p in result.bits] + [p for p in result.bits if p not in result.kept]
        four = by_name.get("four_bits")
        if four and four["distance"] < best["distance"] - BETTER:
            result.advice = (f"On the validation questions, compressing everything to 4-bit (+{four['extra_mb']:.0f} MB, no "
                             f"repair) was closer to the original than "
                             + (f"this repair (+{best['extra_mb']:.0f} MB)." if best["ships"] else "the 3-bit model.")
                             + " To try it, pick \"Compress to: 4-bit\".")
        return result


def within_limits(card: dict, goal: dict) -> bool:
    """As fair as the original by the verdict's own limits (fast_repair.FAIR_*)."""
    from src.product.fast_repair import FAIR_HEDGE, FAIR_SRS_RISE, FAIR_USR_DROP

    return (goal["usr"] - card["usr"] <= FAIR_USR_DROP and card["srs"] - goal["srs"] <= FAIR_SRS_RISE
            and card["false_unknown"] - goal["false_unknown"] <= FAIR_HEDGE)


def distance(card: dict, goal: dict) -> float:
    """How far a model is from the original: the fair answers it lost, the stereotyped ones it gained and the
    hedging it added (being fairer than the original earns nothing)."""
    return (max(0.0, goal["usr"] - card["usr"]) + max(0.0, card["srs"] - goal["srs"])
            + max(0.0, card["false_unknown"] - goal["false_unknown"]))


def scaled_screen(screen: tuple[int, int]) -> tuple[int, int]:
    """The first look's question counts: Fast Scan's for this model, raised to SCREEN in all."""
    total = sum(screen)
    factor = max(1.0, SCREEN / total) if total else 1.0
    return round(screen[0] * factor), round(screen[1] * factor)
