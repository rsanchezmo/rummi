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

from dataclasses import dataclass, field

import numpy as np

from rummi.rules.config import RummiConfig
from rummi.rules.encoding import EMPTY, tables
from rummi.env.numpy.sets import evaluate_slots
from rummi.solver.candidates import candidates

DEFAULT_TIME_LIMIT = 2.0


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


class Objective:
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
    kept_sets: int = 0
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
    cfg: RummiConfig, counts: np.ndarray, x: np.ndarray, subs: np.ndarray
) -> list[tuple[int, ...]]:
    """Turn set counts plus per-kind joker substitutions into concrete tile lists."""
    cand = candidates(cfg)
    instances: list[list[int]] = []
    for c in np.flatnonzero(x):
        tiles = [int(k) for k in cand.kinds[c] if k >= 0]
        instances.extend([list(tiles) for _ in range(int(x[c]))])

    remaining = subs.copy()
    for kind in np.flatnonzero(remaining):
        for inst in instances:
            if remaining[kind] == 0:
                break
            if kind in inst:
                inst[inst.index(int(kind))] = cfg.joker_kind
                remaining[kind] -= 1
    if remaining.any():  # pragma: no cover - guarded by the model's constraints
        raise AssertionError("could not place every joker substitution")
    return [tuple(sorted(inst)) for inst in instances]


def solve_turn(
    cfg: RummiConfig,
    rack: np.ndarray,
    table: np.ndarray,
    has_melded: bool,
    objective: str = Objective.MAX_TILES,
    time_limit: float = DEFAULT_TIME_LIMIT,
    keep_weight: int = 1,
) -> Solution:
    """Best legal turn for the acting player, or an infeasible solution.

    Before melding, the official rule forbids using table tiles or rearranging,
    so the problem shrinks to "best set of sets built from the rack alone, worth
    at least ``initial_meld``" and the existing table is left untouched.
    """
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
    # Pre-meld the table is off limits, so the model only sees the rack and the
    # solved sets are additions rather than a repartition.
    visible_table = np.zeros_like(table_counts) if opening else table_counts
    available = visible_table + rack
    slot_budget = cfg.max_sets - (occupied_slots if opening else 0)
    if slot_budget <= 0:
        return Solution(feasible=False, status="no_free_slots")

    model = cp_model.CpModel()
    joker_available = int(available[cfg.joker_kind])
    # Per-candidate bound from what is actually reachable. Without it every one of
    # the ~330 candidates is nominally placeable several times over and CP-SAT
    # spends its whole budget proving otherwise.
    scarcest = np.array(
        [available[np.flatnonzero(cand.counts[c])].min() for c in range(n_cand)],
        dtype=np.int64,
    )
    upper = np.minimum(cfg.n_copies + cfg.n_jokers, scarcest + joker_available)
    x = [model.NewIntVar(0, int(upper[c]), f"x{c}") for c in range(n_cand)]

    numbered = np.arange(cfg.n_numbered_kinds)
    demand = {}
    for k in numbered:
        involved = np.flatnonzero(cand.counts[:, k])
        demand[k] = sum(x[int(c)] for c in involved) if involved.size else 0

    use = {}
    subs = {}
    for k in numbered:
        upper = int(available[k])
        use[k] = model.NewIntVar(int(visible_table[k]), upper, f"use{k}")
        subs[k] = model.NewIntVar(0, cfg.n_jokers, f"sub{k}")
        # Every slot a set demands is filled either by a real tile or by a joker.
        model.Add(demand[k] == use[k] + subs[k])

    jokers_used = model.NewIntVar(int(visible_table[cfg.joker_kind]), joker_available, "jokers")
    model.Add(jokers_used == sum(subs[k] for k in numbered))

    model.Add(sum(x) <= slot_budget)
    # Redundant but decisive: total tiles consumed cannot exceed what exists. It
    # follows from the per-kind constraints, yet stating it as one linear bound is
    # what lets the solver prune whole regions instead of enumerating them.
    model.Add(
        sum(int(cand.length[c]) * x[c] for c in range(n_cand)) <= int(available.sum())
    )

    played = {k: use[k] - int(visible_table[k]) for k in numbered}
    played_jokers = jokers_used - int(visible_table[cfg.joker_kind])
    tiles_played = sum(played.values()) + played_jokers
    value_played = (
        sum(int(t.value[k]) * played[k] for k in numbered) + cfg.joker_penalty * played_jokers
    )
    meld_value = sum(int(cand.value[c]) * x[c] for c in range(n_cand))

    if opening:
        model.Add(meld_value >= cfg.initial_meld)
        model.Add(tiles_played >= 1)
    else:
        model.Add(tiles_played >= 1)

    # Leaving existing sets alone is worth a tie-break: it shortens the resulting
    # micro-action sequence, which is the only cost the score does not capture.
    kept_terms = []
    if keep_weight and not opening:
        current = _current_instances(cfg, table)
        by_content = {tuple(sorted(int(k) for k in cand.kinds[c] if k >= 0)): c for c in range(n_cand)}
        for content, count in current.items():
            c = by_content.get(content)
            if c is None:
                continue
            kept = model.NewIntVar(0, count, f"keep{c}")
            model.Add(kept <= x[c])
            kept_terms.append(kept)

    # Weights make this a strict lexicographic order: tiles, then value, then stillness.
    span_value = max(1, cfg.n_tiles * max(cfg.n_numbers, cfg.joker_penalty))
    span_keep = max(1, cfg.max_sets + 1)
    if objective == Objective.MAX_VALUE:
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
    solver.parameters.max_time_in_seconds = time_limit
    # Single worker keeps the result reproducible, which matters for a baseline.
    solver.parameters.num_workers = 1
    solver.parameters.random_seed = 0
    # Place the biggest sets first: it finds a strong incumbent immediately, which
    # is what makes the bound bite.
    model.AddDecisionStrategy(
        [x[c] for c in np.argsort(-cand.length)], cp_model.CHOOSE_FIRST, cp_model.SELECT_MAX_VALUE
    )
    status = solver.Solve(model)
    status_name = solver.StatusName(status)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return Solution(feasible=False, status=status_name)

    x_val = np.array([solver.Value(v) for v in x], dtype=np.int64)
    sub_val = np.zeros(cfg.n_kinds, dtype=np.int64)
    for k in numbered:
        sub_val[k] = solver.Value(subs[k])

    sets = _materialise(cfg, cand.counts, x_val, sub_val)
    played_counts = np.zeros(cfg.n_kinds, dtype=np.int16)
    for k in numbered:
        played_counts[k] = solver.Value(use[k]) - int(visible_table[k])
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
        kept_sets=int(sum(solver.Value(k) for k in kept_terms)) if kept_terms else 0,
        status=status_name,
    )
