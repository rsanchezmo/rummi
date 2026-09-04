"""Optimal single-turn play via OR-Tools CP-SAT.

The Den Hertog & Hulshof formulation: choose how many instances of each candidate
set end up on the table, subject to every tile currently on the table still being
used and no more tiles being drawn on than the rack holds. Unrestricted
rearrangement is what makes this legal -- the solver is free to repartition the
whole table, so it finds plays the greedy baseline cannot see.

Jokers are handled in aggregate rather than per set. A joker substitutes for any
tile, so only the *total* number of substitutions per kind matters, and any
distribution of those substitutions across the sets that demand that kind is
realisable. That collapses what would be a ``candidates x kinds`` matrix of
variables into one integer per kind.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from functools import cache

import numpy as np

from rummi.rules.config import RummiConfig
from rummi.rules.encoding import EMPTY, tables
from rummi.env.numpy.sets import evaluate_slots
from rummi.solver.candidates import candidates

DEFAULT_TIME_LIMIT = 2.0
"""Wall-clock backstop, so a pathological model cannot hang a run."""

DETERMINISTIC_LIMIT = 1.0
"""The budget that actually makes a score reproducible.

Wall clock is not a contract: a solve that runs out of it returns ``FEASIBLE`` and
a different table, so on a slower or loaded machine the published
``standard-optimal`` numbers would drift under an unchanged ``PROTOCOL_VERSION``.
Deterministic time counts work rather than seconds, so it is the same bound on
every machine.

Calibrated over 2,409 solves across the three standard seat counts and all four
shapes a caller asks for -- openings, mid-game, frozen and k-best: the worst
consumed 0.0888 units (mid-game p95 0.035, k-best worst 0.0888), so 1.0 leaves
11x headroom on the worst case seen.
"""


class SolverUnavailable(RuntimeError):
    pass


def _cp_model():
    try:
        from ortools.sat.python import cp_model
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise SolverUnavailable(
            "the CP-SAT solver needs the optional extra: pip install 'rummi[solver]'"
        ) from exc
    return cp_model


class Objective(str, Enum):
    """What a solve maximises. An enum rather than a string namespace so a typo is
    refused instead of falling through to the tile-maximising branch."""

    MAX_TILES = "max_tiles"
    """Shed as many tiles as possible -- what actually wins games."""
    MAX_VALUE = "max_value"
    """Shed as much rack penalty as possible."""


@dataclass(frozen=True, slots=True)
class Solution:
    feasible: bool
    sets: tuple[tuple[int, ...], ...] = ()
    """The target table: one tuple of kind ids per set, jokers materialised."""
    played: np.ndarray | None = None
    """``(K,)`` tiles moved from the rack to the table."""
    tiles_played: int = 0
    value_played: int = 0
    meld_value: int = 0
    """Total value of newly created sets, for the opening-meld check."""
    set_counts: np.ndarray | None = None
    """``(n_candidates,)`` how many instances of each candidate the target holds.
    The identity of a solved *table*, which is what a no-good cut excludes."""
    status: str = "unknown"

    @property
    def plays_anything(self) -> bool:
        return self.feasible and self.tiles_played > 0


def _slot_contents(table: np.ndarray) -> list[tuple[int, ...]]:
    return [tuple(int(k) for k in row if k >= 0) for row in table if (row >= 0).any()]


def _current_instances(cfg: RummiConfig, table: np.ndarray) -> dict[tuple[int, ...], int]:
    """How many times each exact slot content already sits on the table.

    Used only as a tie-break to keep the table still, so joker-bearing slots not
    matching a joker-free candidate is harmless.
    """
    out: dict[tuple[int, ...], int] = {}
    for content in _slot_contents(table):
        key = tuple(sorted(content))
        out[key] = out.get(key, 0) + 1
    return out


def _materialise(
    cfg: RummiConfig,
    counts: np.ndarray,
    x: np.ndarray,
    subs: np.ndarray,
    frozen: Sequence[tuple[int, ...]] = (),
) -> list[tuple[int, ...]]:
    """Turn set counts plus per-kind joker substitutions into concrete tile lists.

    ``frozen`` is the standing sets that must survive intact. Their real tiles are
    pinned into whichever instance each one claims, so a substitution may not land
    on one: a joker standing in for a tile that is *present* frees that tile for
    another set, which leaves the multiset of sets untouched and takes both sets
    apart. The model bounds how many substitutions each claimed instance can
    absorb; this is where that bound is honoured tile by tile.
    """
    cand = candidates(cfg)
    instances: list[list[int]] = []
    for c in np.flatnonzero(x):
        tiles = [int(k) for k in cand.kinds[c] if k >= 0]
        instances.extend([list(tiles) for _ in range(int(x[c]))])

    # Claim an instance per standing set, exact matches first so a set already
    # standing where the target wants it is never spent on a longer one.
    pinned: list[Counter[int]] = [Counter() for _ in instances]
    unclaimed = set(range(len(instances)))
    order = sorted(range(len(frozen)), key=lambda i: -len(frozen[i]))
    for i in order:
        real = Counter(k for k in frozen[i] if k != cfg.joker_kind)
        exact = [j for j in unclaimed if Counter(instances[j]) == Counter(frozen[i])]
        fits = exact or [j for j in unclaimed if real <= Counter(instances[j])]
        if not fits:
            continue
        claimed = fits[0]
        unclaimed.discard(claimed)
        pinned[claimed] = real

    remaining = subs.copy()
    for kind in np.flatnonzero(remaining):
        # Least pinned first: an instance with room to spare cannot be the one the
        # placement gets stuck on, and the model guarantees room exists somewhere.
        for j in sorted(range(len(instances)), key=lambda j: pinned[j][int(kind)]):
            inst = instances[j]
            if remaining[kind] == 0:
                break
            free = inst.count(int(kind)) - pinned[j][int(kind)]
            while remaining[kind] and free > 0:
                inst[inst.index(int(kind))] = cfg.joker_kind
                remaining[kind] -= 1
                free -= 1
    if remaining.any():  # pragma: no cover - guarded by the model's constraints
        raise AssertionError("could not place every joker substitution")
    return [tuple(sorted(inst)) for inst in instances]


@dataclass(frozen=True, slots=True)
class _Constants:
    """What the model needs that depends only on the config.

    Rebuilding these per solve was most of the time a solve took: the C++ search is
    a few milliseconds and constructing the model around it was twice that, nearly
    all of it Python loops over the ~330 candidates that never change.
    """

    involved: tuple[tuple[int, ...], ...]
    """Candidates using each numbered kind."""
    by_content: dict[tuple[int, ...], int]
    """Candidate index for a set's sorted contents."""
    longest_first: tuple[int, ...]
    length: tuple[int, ...]
    value: tuple[int, ...]


@cache
def _constants(cfg: RummiConfig) -> _Constants:
    cand = candidates(cfg)
    n_cand = len(cand)
    return _Constants(
        involved=tuple(
            tuple(int(c) for c in np.flatnonzero(cand.counts[:, k]))
            for k in range(cfg.n_numbered_kinds)
        ),
        by_content={
            tuple(sorted(int(k) for k in cand.kinds[c] if k >= 0)): c for c in range(n_cand)
        },
        longest_first=tuple(int(c) for c in np.argsort(-cand.length)),
        length=tuple(int(v) for v in cand.length),
        value=tuple(int(v) for v in cand.value),
    )


def solve_turn(
    cfg: RummiConfig,
    rack: np.ndarray,
    table: np.ndarray,
    has_melded: bool,
    objective: Objective | str = Objective.MAX_TILES,
    time_limit: float = DEFAULT_TIME_LIMIT,
    keep_weight: int = 1,
    tiles_min: int = 1,
    tiles_cap: int | None = None,
    exclude: Sequence[np.ndarray] = (),
    freeze_table: bool = False,
) -> Solution:
    """Best legal turn for the acting player, or an infeasible solution.

    Before melding, the official rule forbids using table tiles or rearranging,
    so the problem shrinks to "best set of sets built from the rack alone, worth
    at least ``initial_meld``" and the existing table is left untouched.

    The four restrictions below exist so a caller can ask for turns *other* than
    the best one, which is what pricing the choice between them needs. They are
    additive constraints: at their defaults the model is the one above.

    ``tiles_min`` and ``tiles_cap`` bracket how many tiles leave the rack, so
    fixing both to the optimum turns the model into an enumerator over the tables
    that shed the same number. ``exclude`` forbids target tables already found --
    a no-good cut per :attr:`Solution.set_counts`, which is what makes that
    enumeration k-best rather than the same answer repeated. ``freeze_table``
    forbids rearrangement: every set now on the table must survive into the
    target, extended but never taken apart. Both halves of a slot are pinned. Its
    own jokers keep their positions, so the target must have room for them on top
    of its real tiles; and a joker may stand in only for a kind the slot does
    *not* already supply, which is what stops a substitution taking a real tile's
    place and freeing it for another set -- a steal that leaves the multiset of
    sets intact while physically taking two sets apart. Buying a joker back falls
    to the same rule: it is rearrangement, and reaching it means dissolving the
    set.
    """
    try:
        objective = Objective(objective)
    except ValueError:
        raise ValueError(
            f"unknown objective {objective!r}; expected one of "
            f"{', '.join(o.value for o in Objective)}"
        ) from None

    cp_model = _cp_model()
    cand = candidates(cfg)
    t = tables(cfg)
    n_cand = len(cand)

    rack = np.asarray(rack).astype(np.int64)
    table_counts = np.zeros(cfg.n_kinds, dtype=np.int64)
    for content in _slot_contents(table):
        for kind in content:
            table_counts[kind] += 1
    occupied_slots = len(_slot_contents(table))

    opening = not has_melded
    # Two different things, and only `strict_initial_meld` makes them coincide:
    # whether the meld is still owed, and whether the table may be touched. The
    # engine gates the table on `may_touch_table`, which is unconditional when the
    # rule is relaxed -- see SPEC.md section 5.
    locked = opening and cfg.strict_initial_meld
    # While it is locked the model only sees the rack, so the solved sets are
    # additions rather than a repartition.
    visible_table = np.zeros_like(table_counts) if locked else table_counts
    available = visible_table + rack
    slot_budget = cfg.max_sets - (occupied_slots if locked else 0)
    if slot_budget <= 0:
        return Solution(feasible=False, status="no_free_slots")

    model = cp_model.CpModel()
    joker_available = int(available[cfg.joker_kind])
    # Per-candidate bound from what is actually reachable. Without it every one of
    # the ~330 candidates is nominally placeable several times over and CP-SAT
    # spends its whole budget proving otherwise.
    scarcest = np.where(cand.counts > 0, available[None, :], np.iinfo(np.int64).max).min(-1)
    upper = np.minimum(cfg.n_copies + cfg.n_jokers, scarcest + joker_available)
    x = [model.NewIntVar(0, int(upper[c]), f"x{c}") for c in range(n_cand)]

    const = _constants(cfg)
    numbered = np.arange(cfg.n_numbered_kinds)
    demand = {
        int(k): (sum(x[c] for c in const.involved[k]) if const.involved[k] else 0)
        for k in numbered
    }

    use = {}
    subs = {}
    for k in (int(k) for k in numbered):
        kind_upper = int(available[k])
        use[k] = model.NewIntVar(int(visible_table[k]), kind_upper, f"use{k}")
        subs[k] = model.NewIntVar(0, cfg.n_jokers, f"sub{k}")
        # Every slot a set demands is filled either by a real tile or by a joker.
        model.Add(demand[k] == use[k] + subs[k])

    jokers_used = model.NewIntVar(int(visible_table[cfg.joker_kind]), joker_available, "jokers")
    model.Add(jokers_used == sum(subs.values()))

    model.Add(sum(x) <= slot_budget)
    # Redundant but decisive: total tiles consumed cannot exceed what exists. It
    # follows from the per-kind constraints, yet stating it as one linear bound is
    # what lets the solver prune whole regions instead of enumerating them.
    model.Add(
        sum(const.length[c] * x[c] for c in range(n_cand)) <= int(available.sum())
    )

    played = {k: v - int(visible_table[k]) for k, v in use.items()}
    played_jokers = jokers_used - int(visible_table[cfg.joker_kind])
    tiles_played = sum(played.values()) + played_jokers
    value_played = (
        sum(int(t.value[k]) * v for k, v in played.items()) + cfg.joker_penalty * played_jokers
    )
    meld_value = sum(const.value[c] * x[c] for c in range(n_cand))

    model.Add(tiles_played >= tiles_min)
    if opening:
        # SPEC.md section 7 credits the opening two ways. Strict, the table is
        # untouchable so every target set is new and the joker is worth the
        # position it fills. Relaxed, only the face value of what left the rack
        # counts, and an unclaimed joker is worth nothing.
        credited = (
            meld_value
            if cfg.strict_initial_meld
            else sum(int(t.value[k]) * v for k, v in played.items())
        )
        model.Add(credited >= cfg.initial_meld)
    if tiles_cap is not None:
        model.Add(tiles_played <= tiles_cap)

    if freeze_table and not locked:
        # Each existing set claims one instance of a target set containing it, and
        # two sets cannot claim the same instance -- so a set may grow by a lay-off
        # and can never be split, which is exactly "no rearrangement".
        claims: dict[int, list] = {}
        # Per kind, the joker substitutions that land inside a claimed instance and
        # the demand those instances account for. Both are needed because a
        # substitution is only forbidden *where a frozen slot already supplies a
        # real tile*: bounding it per kind over the whole table cannot say that, and
        # that is how the steal got through.
        landed: dict[int, list] = {}
        accounted: dict[int, list] = {}
        # A standing joker is itself a substitution of whatever kind it stands for,
        # so it has to draw on that kind's budget. Without this the model can leave
        # the budget at zero and move the joker to another set entirely.
        own_claims: dict[int, list] = {}
        for slot, content in enumerate(_slot_contents(table)):
            need = np.zeros(cfg.n_kinds, dtype=np.int64)
            for kind in content:
                need[kind] += 1
            held_jokers = int(need[cfg.joker_kind])
            need[cfg.joker_kind] = 0
            # A frozen slot's own jokers keep their positions, so its target must
            # have room for them on top of its real tiles.
            supersets = np.flatnonzero(
                (cand.counts >= need[None, :]).all(-1) & (cand.length >= len(content))
            )
            if supersets.size == 0:
                return Solution(feasible=False, status="frozen_unmatched_slot")
            picks = [model.NewBoolVar(f"frz{slot}_{sup}") for sup in supersets]
            model.Add(sum(picks) == 1)
            for sup, pick in zip(supersets, picks, strict=True):
                claims.setdefault(int(sup), []).append(pick)

            # The positions of the claimed instance the slot does not already fill
            # with a real tile. Everything else in it is pinned: a joker there would
            # be standing in for a tile that is present, which frees that tile for
            # another set -- the multiset of sets intact and both sets taken apart.
            slack = np.maximum(cand.counts[supersets].astype(np.int64) - need[None, :], 0)
            counts = cand.counts[supersets].astype(np.int64)
            room: dict[int, object] = {}
            for gap in (int(g) for g in np.flatnonzero(slack.any(0))):
                room[gap] = sum(
                    int(slack[row, gap]) * pick
                    for row, pick in enumerate(picks)
                    if slack[row, gap]
                )
            for kind in (int(k) for k in np.flatnonzero(counts.any(0))):
                accounted.setdefault(kind, []).append(
                    sum(
                        int(counts[row, kind]) * pick
                        for row, pick in enumerate(picks)
                        if counts[row, kind]
                    )
                )
                # No slack for this kind means no joker may stand for it here.
                cap = int(slack[:, kind].max()) if kind in room else 0
                here = model.NewIntVar(0, cap, f"frzs{slot}_{kind}")
                if cap:
                    model.Add(here <= room[kind])
                landed.setdefault(kind, []).append(here)

            if not held_jokers:
                continue
            # Which kinds those jokers stand for is not determined by the slot --
            # a joker beside two 7s could be either missing colour -- so the model
            # chooses, from the same room a laid-off joker would use.
            own = []
            for gap in room:
                mine = model.NewIntVar(0, held_jokers, f"frzj{slot}_{gap}")
                # Its own jokers are part of what landed here, not extra to it.
                model.Add(mine <= landed[gap][-1])
                own.append(mine)
                own_claims.setdefault(gap, []).append(mine)
            model.Add(sum(own) == held_jokers)

        for kept_index, group in claims.items():
            model.Add(sum(group) <= x[kept_index])
        # `subs` is keyed by the numbered-kind array, so it is already in kind
        # order; a list of it indexes by the plain ints collected above.
        per_kind: list = list(subs.values())
        for kind, group in landed.items():
            # Every substitution of this kind is either inside a claimed instance,
            # where the room above bounds it, or in one of the fresh sets, which
            # accounts for the rest of the demand.
            outside = demand[kind] - sum(accounted.get(kind, []))
            model.Add(per_kind[kind] <= sum(group) + outside)
        for kind, group in own_claims.items():
            model.Add(sum(group) <= per_kind[kind])

    total_sets = sum(x) if exclude else 0
    for cut, previous in enumerate(exclude):
        # A target table differs from an excluded one iff some candidate it used
        # appears a different number of times, or some candidate it did not use
        # appears at all.
        counts = np.asarray(previous).astype(np.int64)
        support = [int(used) for used in np.flatnonzero(counts)]
        literals = []
        for used in support:
            below = model.NewBoolVar(f"cut{cut}_lo{used}")
            model.Add(x[used] <= int(counts[used]) - 1).OnlyEnforceIf(below)
            above = model.NewBoolVar(f"cut{cut}_hi{used}")
            model.Add(x[used] >= int(counts[used]) + 1).OnlyEnforceIf(above)
            literals += [below, above]
        outside = model.NewBoolVar(f"cut{cut}_outside")
        model.Add(total_sets - sum(x[used] for used in support) >= 1).OnlyEnforceIf(outside)
        literals.append(outside)
        model.AddBoolOr(literals)

    # Leaving existing sets alone is worth a tie-break: it shortens the resulting
    # micro-action sequence, which is the only cost the score does not capture.
    kept_terms = []
    if keep_weight and not locked:
        current = _current_instances(cfg, table)
        for content, count in current.items():
            c = const.by_content.get(content)
            if c is None:
                continue
            kept = model.NewIntVar(0, count, f"keep{c}")
            model.Add(kept <= x[c])
            kept_terms.append(kept)

    # Weights make this a strict lexicographic order: tiles, then value, then stillness.
    span_value = max(1, cfg.n_tiles * max(cfg.n_numbers, cfg.joker_penalty))
    span_keep = max(1, cfg.max_sets + 1)
    if Objective(objective) is Objective.MAX_VALUE:
        primary, secondary = value_played, tiles_played
        primary_scale = span_keep * (cfg.n_tiles + 1)
    else:
        primary, secondary = tiles_played, value_played
        primary_scale = span_keep * (span_value + 1)
    model.Maximize(
        primary * primary_scale
        + secondary * span_keep
        + (sum(kept_terms) if kept_terms else 0)
    )

    solver = cp_model.CpSolver()
    # Deterministic time is the reproducible bound; the wall clock is only a
    # backstop, and with the headroom above it should never be what stops a solve.
    solver.parameters.max_deterministic_time = DETERMINISTIC_LIMIT
    solver.parameters.max_time_in_seconds = time_limit
    # Single worker keeps the result reproducible, which matters for a baseline.
    solver.parameters.num_workers = 1
    solver.parameters.random_seed = 0
    # Place the biggest sets first: it finds a strong incumbent immediately, which
    # is what makes the bound bite.
    model.AddDecisionStrategy(
        [x[c] for c in const.longest_first], cp_model.CHOOSE_FIRST, cp_model.SELECT_MAX_VALUE
    )
    status = solver.Solve(model)
    status_name = solver.StatusName(status)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return Solution(feasible=False, status=status_name)

    x_val = np.array([solver.Value(v) for v in x], dtype=np.int64)
    sub_val = np.zeros(cfg.n_kinds, dtype=np.int64)
    for k, v in subs.items():
        sub_val[k] = solver.Value(v)

    sets = _materialise(
        cfg, cand.counts, x_val, sub_val,
        _slot_contents(table) if freeze_table and not locked else (),
    )
    played_counts = np.zeros(cfg.n_kinds, dtype=np.int16)
    for k, v in use.items():
        played_counts[k] = solver.Value(v) - int(visible_table[k])
    played_counts[cfg.joker_kind] = solver.Value(jokers_used) - int(visible_table[cfg.joker_kind])

    if sets:
        padded = np.full((len(sets), cfg.max_set_len), EMPTY, dtype=np.int16)
        for i, content in enumerate(sets):
            padded[i, : len(content)] = content
        if not evaluate_slots(cfg, padded).is_valid.all():  # pragma: no cover
            raise AssertionError("solver returned a set the rules kernel rejects")

    return Solution(
        feasible=True,
        sets=tuple(sets),
        played=played_counts,
        tiles_played=int(played_counts.sum()),
        value_played=int(
            played_counts[numbered].astype(np.int64) @ t.value[numbered].astype(np.int64)
            + played_counts[cfg.joker_kind] * cfg.joker_penalty
        ),
        meld_value=int(sum(int(cand.value[c]) * int(x_val[c]) for c in range(n_cand))),
        set_counts=x_val,
        status=status_name,
    )
