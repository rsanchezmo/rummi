"""`frugal`, with the opening turn replaced -- the one decision that does not recur.

Every ranking in this repo maximises how many tiles leave the rack this turn, and
`tools/oracle_regret.py` priced that choice in hindsight: over 39,903 alternative
turns rolled out against the real deck, shedding fewer tiles is worth -3.1% overall
and -8.1% in the endgame, and the **one** cell that points the other way is the
opening -- +9.3% +-5.7% for an alternative shedding fewer tiles than the maximum.

The mechanism that would explain it is specific to the opening, which is why it is
worth a direct arm. Before melding the table is untouchable, so the sets an agent
opens with are handed to the opponent as **rigid** sets it may lay off onto, while a
tile kept back is played on a later turn with full rearrangement rights. Every other
decision in a turn recurs -- `MacroAgent` decides again after every set -- so a set
kept out of the rack is usually played in the same turn anyway, which is what
`tools/rack_potential_ab.py` measured and found nothing in. The opening is the one
boundary that forecloses.

Each arm here differs from `frugal` **only** while the acting seat has not melded;
every other decision is delegated to `by_value` untouched, so a difference in score
is attributable to the opening alone (`test_opening.py` holds the post-meld
decisions to byte identity).

Pre-meld the macro space collapses to something small enough to plan exactly: lay-offs
and steals are illegal under `strict_initial_meld`, `REPARTITION` is offered only after
melding, so the only legal macros are new sets, `END_TURN` and `DRAW`. An opening is
therefore a sequence of templates drawn from the rack alone, and an arm is a rule for
choosing that sequence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rummi.agents.base import Observation, has_melded, table
from rummi.agents.macro import (
    MacroAgent,
    by_value,
    extend_offset,
    laid_tiles,
    playable,
    set_templates,
    shortfall,
    template_points,
)
from rummi.env.numpy.sets import evaluate_slots
from rummi.rules.config import RummiConfig
from rummi.rules.encoding import EMPTY


@dataclass(frozen=True)
class Opening:
    """How one arm builds its opening turn. Defaults are `by_value`'s own behaviour."""

    order: str = "dear"
    """Which playable template goes down first: `dear` (`by_value`'s ranking, highest
    points), `cheap` (its inverse), `runs` or `groups` (that shape first, then dearest)."""
    stop: bool = False
    """End the turn the moment the meld threshold is met, instead of playing on."""
    drop_last: bool = False
    """Lay `by_value`'s whole opening except its final set, where the rest still clears
    the threshold -- the smallest possible deviation from what `frugal` actually opens
    with, which is what the oracle's winning cell was measured against."""
    hold_joker: bool = False
    """Refuse a template the hand can only cover by substituting its joker."""
    solver: str = ""
    """`min_tiles` or `max_tiles`: CP-SAT decides the whole opening instead."""


ARMS: dict[str, Opening] = {
    "full": Opening(),
    "min_sets": Opening(stop=True),
    "minus_one": Opening(drop_last=True),
    "min_tiles": Opening(solver="min_tiles"),
    "max_tiles": Opening(solver="max_tiles"),
    "cheap": Opening(order="cheap", stop=True),
    "runs_first": Opening(order="runs", stop=True),
    "groups_first": Opening(order="groups", stop=True),
    "no_joker": Opening(hold_joker=True),
}
"""`full` is the correctness check and not an arm: it rebuilds `by_value`'s own opening
through this file's planner, so it must score *exactly* even against plain `frugal`.
`max_tiles` is the negative control -- what `optimal` opens with, which the oracle
priced at -1.1 -- and the three `stop` arms differ from each other only in ordering, so
they read against `min_sets` rather than against `frugal`."""


@dataclass
class OpeningStats:
    """Proof the arm acts: a flat score means nothing without it.

    Counted over pre-meld decisions only, which is the whole of what these arms touch.
    """

    decisions: int = 0
    turns: int = 0
    """Pre-meld turns a plan was built for -- most of them cannot reach the threshold
    at all and fall back."""
    fallbacks: int = 0
    """Plans the arm's own construction could not open with, played as `frugal` instead:
    an arm that gave up the opening would be measuring the delay, not the shape."""
    moved: int = 0
    """Decisions where the arm's macro differs from `by_value`'s pick."""
    stale: int = 0
    """Plans abandoned because the mask refused their next set."""

    def merge(self, other: OpeningStats) -> None:
        for name in self.__dataclass_fields__:
            setattr(self, name, getattr(self, name) + getattr(other, name))

    def report(self) -> str:
        decisions = max(self.decisions, 1)
        turns = max(self.turns, 1)
        return (
            f"pre-meld decisions {self.decisions:>7,}  moved {self.moved / decisions:>6.2%}  "
            f"plans {self.turns:>7,}  fell back {self.fallbacks / turns:>6.2%}  "
            f"stale {self.stale:>4}"
        )


def template_is_run(cfg: RummiConfig) -> np.ndarray:
    """`(T,)` -- is this template a run rather than a group?

    Read off the kinds a template holds rather than off the order `set_templates`
    happens to build them in, which is not part of its contract.
    """
    templates = set_templates(cfg)
    colours = np.arange(cfg.n_kinds) // cfg.n_numbers
    out = np.zeros(len(templates), dtype=bool)
    for t, row in enumerate(templates):
        out[t] = len(set(colours[np.flatnonzero(row)].tolist())) == 1
    return out


def set_value(cfg: RummiConfig, counts: np.ndarray) -> int:
    """What the env credits a set of these tiles towards the opening meld.

    Asked of `evaluate_slots` rather than summed from `template_points`, because a
    joker's value is positional and the env resolves it by the *best* window the slot
    admits -- so a template's own points are the wrong number for any set the hand
    could only cover with a joker.
    """
    kinds = np.repeat(np.arange(cfg.n_kinds), counts)
    row = np.full((1, 1, cfg.max_set_len), EMPTY, dtype=np.int16)
    row[0, 0, : kinds.size] = kinds
    return int(evaluate_slots(cfg, row).value[0, 0])


class OpeningChoice:
    """`by_value`, with the pre-meld turn built by `ARMS[arm]`.

    A whole opening is planned at the turn's first decision and replayed across the
    decisions that follow, because the arms that need CP-SAT cannot be asked again
    mid-turn: `solve_turn` credits the meld threshold against the sets it creates and
    knows nothing of the ones this turn already laid. The plan is a deterministic
    function of the observation at the turn boundary -- rebuilt whenever nothing has
    been played this turn, and abandoned to `by_value` the moment the mask refuses it.
    """

    def __init__(self, cfg: RummiConfig, arm: str, stats: OpeningStats | None = None) -> None:
        assert arm in ARMS, arm
        assert cfg.strict_initial_meld, "an open table pre-meld is a different game"
        self.cfg = cfg
        self.rule = ARMS[arm]
        self.base = by_value(cfg)
        self.templates = set_templates(cfg).astype(np.int64)
        self.sizes = self.templates.sum(-1)
        self.points = template_points(cfg).astype(np.float64)
        self.is_run = template_is_run(cfg)
        self.n_templates = extend_offset(cfg)
        self.stats = stats if stats is not None else OpeningStats()
        self._plans: dict[int, list[int] | None] = {}

    def reset(self) -> None:
        self._plans = {}

    def _order_key(self, hand: np.ndarray) -> np.ndarray:
        """`(T,)` ranking the arm lays sets by, highest first.

        `dear` is `by_value`'s own `template_points`, so `argmax`'s first-maximum tie
        goes to the same template it would -- the lowest index, which is the cheapest,
        lowest-numbered set.
        """
        order = self.rule.order
        if order == "dear":
            return self.points
        if order == "cheap":
            return -self.points
        prefer = self.is_run if order == "runs" else ~self.is_run
        # The shape dominates and points break the tie inside it, so the two shape arms
        # differ from `min_sets` in ordering and in nothing else.
        return self.points + prefer * (self.points.max() + 1.0)

    def _greedy(self, hand: np.ndarray, free: int) -> tuple[list[int], list[int]]:
        """Templates in the arm's own order, and what each is worth, until it runs out.

        Stops early once the threshold is met when the arm says to. `free` is the empty
        slots, which is what bounds an opening -- the same test `legal_macros` makes.
        """
        cfg = self.cfg
        rule = self.rule
        key = self._order_key(hand)
        hand = hand.copy()
        seq: list[int] = []
        values: list[int] = []
        while free > 0:
            ok = playable(cfg, hand[None])[0]
            if rule.hold_joker:
                ok = ok & (shortfall(cfg, hand[None])[0] == 0)
            if not ok.any():
                break
            pick = int(np.argmax(np.where(ok, key, -np.inf)))
            laid = laid_tiles(cfg, self.templates[pick], hand)
            seq.append(pick)
            values.append(set_value(cfg, laid))
            hand = hand - laid
            free -= 1
            if rule.stop and sum(values) >= cfg.initial_meld:
                break
        return seq, values

    def _template_for(self, counts: np.ndarray, hand: np.ndarray) -> int | None:
        """The template that lays exactly `counts` out of `hand`, or None.

        A solved set holding a joker does not name the tile the joker stands for, so
        the template is recovered by asking `laid_tiles` -- the same substitution
        `MacroAgent.expand` will make -- which one candidate reproduces and the rest
        do not.
        """
        real = counts.copy()
        real[self.cfg.joker_kind] = 0
        fits = (self.templates >= real).all(-1) & (self.sizes == int(counts.sum()))
        for t in np.flatnonzero(fits).tolist():
            if (laid_tiles(self.cfg, self.templates[t], hand) == counts).all():
                return t
        return None

    def _solved(self, hand: np.ndarray, board: np.ndarray) -> list[int] | None:
        """CP-SAT's opening as a sequence of templates, or None if it is not expressible.

        `min_tiles` scans `tiles_cap` upwards from the shortest set that could exist:
        pre-meld the model already demands `initial_meld`, so the first feasible cap is
        the fewest tiles any legal opening sheds. The maximum is solved first because it
        bounds the scan -- and where the two coincide there is nothing to choose.
        """
        from rummi.solver.ilp import solve_turn

        cfg = self.cfg
        best = solve_turn(cfg, hand, board, False)
        if not best.plays_anything:
            return None
        sets = best.sets
        if self.rule.solver == "min_tiles":
            for cap in range(cfg.min_set, best.tiles_played):
                trial = solve_turn(cfg, hand, board, False, tiles_min=cap, tiles_cap=cap)
                if trial.plays_anything:
                    sets = trial.sets
                    break

        out: list[int] = []
        left = hand.copy()
        # Sorted so the order the sets go down in is fixed by their contents rather than
        # by the solver's internal enumeration, which its time limit is free to reorder.
        for kinds in sorted(sets):
            counts = np.bincount(np.asarray(kinds, dtype=np.int64), minlength=cfg.n_kinds)
            t = self._template_for(counts, left)
            if t is None:
                return None
            out.append(t)
            left = left - laid_tiles(cfg, self.templates[t], left)
        return out

    def _build(self, obs: Observation, env: int, hand: np.ndarray) -> list[int] | None:
        """The arm's whole opening, or None to play the turn as `frugal` would.

        None is not a failure mode, it is the arm's own boundary: an arm that gave up an
        opening it could not build its way would be measured on the *delay*, which
        swamps the shape by an order of magnitude.
        """
        cfg = self.cfg
        free = int((np.asarray(table(obs)[env]).max(-1) < 0).sum())
        if self.rule.solver:
            return self._solved(hand, np.asarray(table(obs)[env]))

        seq, values = self._greedy(hand, free)
        if not seq or sum(values) < cfg.initial_meld:
            # `frugal` does not open here either, and the two play the identical turn.
            return None
        if self.rule.drop_last:
            return seq[:-1] if sum(values[:-1]) >= cfg.initial_meld else seq
        return seq

    def __call__(self, obs: Observation, env: int, legal: np.ndarray) -> int:
        if bool(has_melded(obs)[env]):
            return self.base(obs, env, legal)

        stats = self.stats
        stats.decisions += 1
        base = self.base(obs, env, legal)
        hand = np.asarray(obs["rack"][env]).astype(np.int64) + np.asarray(
            obs["workbench"][env]
        ).astype(np.int64)
        if not np.asarray(obs["placed_this_turn"][env]).any():
            stats.turns += 1
            self._plans[env] = self._build(obs, env, hand)
            stats.fallbacks += int(self._plans[env] is None)

        plan = self._plans.get(env)
        if plan is None:
            return base
        if not plan:
            end_macro = len(legal) - 2
            # The plan is finished, so the threshold is met and ending is legal. Where
            # it is not -- `minus_one` handing back an opening that never reached it --
            # `by_value` is left to draw, exactly as it would.
            chosen = end_macro if legal[end_macro] else base
        else:
            chosen = plan[0]
            if not legal[chosen]:
                # A stale plan is abandoned rather than forced: the rest of the turn is
                # `frugal`'s, which is the same rule `MacroAgent` applies to expansions.
                stats.stale += 1
                self._plans[env] = None
                return base
            plan.pop(0)
        stats.moved += int(chosen != base)
        return int(chosen)


class OpeningAgent(MacroAgent):
    """`frugal` with one of :data:`ARMS` in place of its opening turn.

    A `MacroAgent` over the same macro space with the same stuck-state solve, so the
    only difference from `frugal` is the `choose` above -- and `base` is `frugal`
    itself, built here so the mirror and the arms come off one code path.
    """

    def __init__(self, cfg: RummiConfig, arm: str, stats: OpeningStats | None = None) -> None:
        choice = None if arm == "base" else OpeningChoice(cfg, arm, stats)
        super().__init__(
            cfg, choose=by_value(cfg) if choice is None else choice, repartition=True
        )
        self.name = f"open-{arm}"
        self.choice = choice

    def reset(self, n_envs: int) -> None:
        super().reset(n_envs)
        if self.choice is not None:
            self.choice.reset()
