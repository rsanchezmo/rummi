"""Greedy one-turn-lookahead baseline.

Plans a whole turn when the turn starts, then emits the micro-actions that
realise it. Pre-resolving slots is exact: nobody else touches the table while one
player is mid-turn, so the plan can predict which empty slot each new set lands
in.

Deliberately does *not* rearrange the table -- it only appends to existing sets
and lays new ones from its own rack. That limit is the point: the gap to
:class:`~rummi.agents.optimal_agent.OptimalAgent` is the value of rearrangement,
and :class:`~rummi.agents.rearrange_agent.RearrangeAgent` sits between them by
allowing exactly one stolen tile.
"""

from __future__ import annotations

import numpy as np

from rummi.agents.base import Observation, PlanningAgent, has_melded, table
from rummi.rules.actions import encode_assign, encode_place
from rummi.rules.config import RummiConfig
from rummi.rules.encoding import tables
from rummi.env.numpy.sets import evaluate_slots
from rummi.solver.candidates import candidates


def _offload_values(cfg: RummiConfig) -> np.ndarray:
    """How much rack penalty each kind sheds when played."""
    v = tables(cfg).value.astype(np.int32).copy()
    v[cfg.joker_kind] = cfg.joker_penalty
    return v


def appendable(cfg: RummiConfig, table: np.ndarray, rack: np.ndarray) -> np.ndarray:
    """``(S, K)`` may kind ``k`` be appended to slot ``s`` leaving it valid?

    The verdict comes from :func:`evaluate_slots` over the *grown* row rather than
    from colour/number arithmetic over the tiles already there, and that is what makes
    a joker answerable at all: the run window accepts whichever reading of it keeps
    the set legal, so a set holding one still takes tiles and the rack's own joker
    lays off onto anything with room. Arithmetic can express neither. `macro.EXTEND`
    shares this for exactly that reason -- one feasibility test, so the space's
    legality and its expansion cannot disagree.
    """
    s, ell = table.shape
    k = cfg.n_kinds
    base = evaluate_slots(cfg, table)
    has_room = np.asarray((table < 0).any(-1))
    pos = np.argmax(table < 0, axis=-1)

    grown = np.repeat(table[:, None, :], k, axis=1)
    grown[np.arange(s)[:, None], np.arange(k)[None, :], pos[:, None]] = np.arange(
        k, dtype=np.int16
    )
    grown_valid = evaluate_slots(cfg, grown).is_valid

    return grown_valid & base.is_valid[:, None] & has_room[:, None] & np.asarray(rack > 0)[None, :]


def _appendable_row(cfg: RummiConfig, table: np.ndarray, rack: np.ndarray, slot: int) -> np.ndarray:
    """`(K,)` -- :func:`appendable` for one slot.

    Appending a tile changes that slot's row and nothing else, so the planning loop
    refreshes one row instead of all `S`. On the standard config that is 53 grown
    variants to evaluate per iteration rather than 1855, and `slot_stats` over the
    grown table was 75% of the agent's remaining runtime.
    """
    k = cfg.n_kinds
    row = table[slot : slot + 1]
    base = evaluate_slots(cfg, row)
    has_room = bool(np.asarray((row < 0).any(-1))[0])
    if not has_room or not bool(np.asarray(base.is_valid)[0]):
        return np.zeros(k, dtype=bool)

    pos = int(np.argmax(row[0] < 0))
    grown = np.repeat(row[:, None, :], k, axis=1)
    grown[0, np.arange(k), pos] = np.arange(k, dtype=np.int16)
    grown_valid = np.asarray(evaluate_slots(cfg, grown).is_valid)[0]
    return grown_valid & np.asarray(rack > 0)


def _realise(cfg: RummiConfig, counts: np.ndarray, rack: np.ndarray) -> list[int] | None:
    """Tile list for a candidate, substituting jokers for whatever is missing."""
    missing = np.maximum(0, counts - rack)
    needed = int(missing.sum())
    if needed > int(rack[cfg.joker_kind]):
        return None
    tiles: list[int] = []
    for kind in np.flatnonzero(counts):
        have = min(int(counts[kind]), int(rack[kind]))
        tiles.extend([int(kind)] * have)
        tiles.extend([cfg.joker_kind] * (int(counts[kind]) - have))
    return tiles


def _best_new_set(
    cfg: RummiConfig, rack: np.ndarray, by_value: bool
) -> tuple[list[int], int] | None:
    """Highest-scoring set formable from ``rack`` alone, or ``None``.

    Ranked across every candidate at once, then realised only for the winner.
    The loop this replaces called :func:`_realise` per candidate -- 329 of them on
    the standard config -- and profiling put 643k of those calls, each a handful of
    tiny NumPy reductions, at 60% of the whole agent's runtime.

    Two facts make the vectorised form exact rather than approximate. A realised
    candidate always has `counts.sum()` tiles (a missing tile becomes a joker, so
    the count is unchanged), which is precomputed as `cand.length`; and feasibility
    is just "the shortfall fits in the jokers held", which is one matrix op.
    """
    cand = candidates(cfg)
    counts = cand.counts.astype(np.int64)

    shortfall = np.maximum(0, counts - rack[None, :].astype(np.int64)).sum(-1)
    feasible = shortfall <= int(rack[cfg.joker_kind])
    if not feasible.any():
        return None

    length = cand.length.astype(np.int64)
    value = cand.value.astype(np.int64)
    # Melding needs points; afterwards, shedding tiles is what wins. Packed into
    # one integer so `argmax` does the lexicographic comparison -- and `argmax`
    # returns the *first* maximum, which is what the strict `>` in the previous
    # loop over ascending indices did.
    if by_value:
        key = value * (int(length.max()) + 1) + length
    else:
        key = length * (int(value.max()) + 1) + value

    best = int(np.argmax(np.where(feasible, key, -1)))
    tiles = _realise(cfg, cand.counts[best], rack)
    assert tiles is not None, "the feasibility test and _realise disagree"
    return tiles, int(cand.value[best])


def plan_turn(
    cfg: RummiConfig, rack: np.ndarray, table: np.ndarray, has_melded: bool
) -> list[int]:
    """Micro-action ids realising one greedy turn; empty when nothing is worth doing."""
    rack = rack.copy()
    table = table.copy()
    lengths = (table >= 0).sum(-1)
    placements: list[tuple[int, int]] = []

    if has_melded:
        offload = _offload_values(cfg)
        # Computed in full once, then refreshed a row at a time: an append touches
        # the slot it landed in, and the kind's column only if the rack ran out.
        allowed = appendable(cfg, table, rack)
        while True:
            if not allowed.any():
                break
            # Shed the most expensive tile available.
            scores = np.where(allowed, offload[None, :], -1)
            flat_slot, flat_kind = np.unravel_index(int(np.argmax(scores)), scores.shape)
            slot, kind = int(flat_slot), int(flat_kind)
            table[slot, int(np.argmax(table[slot] < 0))] = kind
            table[slot] = np.sort(np.where(table[slot] >= 0, table[slot], cfg.n_kinds))
            table[slot] = np.where(table[slot] == cfg.n_kinds, -1, table[slot])
            rack[kind] -= 1
            lengths[slot] += 1
            placements.append((kind, slot))

            allowed[slot] = _appendable_row(cfg, table, rack, slot)
            if rack[kind] <= 0:
                allowed[:, kind] = False

    empty = [i for i in range(cfg.max_sets) if lengths[i] == 0]
    meld_total = 0
    while empty:
        best = _best_new_set(cfg, rack, by_value=not has_melded)
        if best is None:
            break
        tiles, value = best
        slot = empty.pop(0)
        for kind in tiles:
            rack[kind] -= 1
            placements.append((kind, slot))
        meld_total += value
        if not has_melded and meld_total >= cfg.initial_meld:
            break

    if not placements:
        return []
    if not has_melded and meld_total < cfg.initial_meld:
        return []

    # PLACE everything first, then ASSIGN in slot order so each new set's first
    # tile arrives while its slot is still the lowest empty one.
    out = [encode_place(cfg, kind) for kind, _ in placements]
    out += [encode_assign(cfg, kind, slot) for kind, slot in placements]
    out.append(cfg.end_turn_action)
    return out


class GreedyAgent(PlanningAgent):
    name = "greedy"

    def plan(self, obs: Observation, env: int) -> list[int]:
        return plan_turn(
            self.cfg, obs["rack"][env], table(obs)[env], bool(has_melded(obs)[env])
        )
