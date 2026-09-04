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
from rummi.rules.actions import encode_assign_batch, encode_place_batch
from rummi.rules.config import RummiConfig
from rummi.rules.encoding import tables
from rummi.env.numpy.sets import evaluate_slots
from rummi.solver.candidates import candidates


def _offload_values(cfg: RummiConfig) -> np.ndarray:
    """How much rack penalty each kind sheds when played."""
    v = tables(cfg).value.astype(np.int32)
    v[cfg.joker_kind] = cfg.joker_penalty
    return v


def _appendable_rows(cfg: RummiConfig, rows: np.ndarray, rack: np.ndarray) -> np.ndarray:
    """``(m, K)`` may kind ``k`` be appended to each of ``m`` slot rows, one rack each?

    The verdict comes from :func:`evaluate_slots` over the *grown* row rather than
    from colour/number arithmetic over the tiles already there, and that is what makes
    a joker answerable at all: the run window accepts whichever reading of it keeps
    the set legal, so a set holding one still takes tiles and the rack's own joker
    lays off onto anything with room. Arithmetic can express neither. `macro.EXTEND`
    shares this for exactly that reason -- one feasibility test, so the space's
    legality and its expansion cannot disagree.
    """
    k = cfg.n_kinds
    out = np.zeros((rows.shape[0], k), dtype=bool)
    base = evaluate_slots(cfg, rows)
    # Only a valid set with room can take a tile, so only those rows are worth
    # growing. On a real table most slots are empty, and growing one costs K rows.
    worth = np.flatnonzero(np.asarray(base.is_valid) & (rows < 0).any(-1))
    if worth.size:
        keep = rows[worth]
        grown = np.repeat(keep[:, None, :], k, axis=1)
        at = np.broadcast_to(np.argmax(keep < 0, axis=-1)[:, None, None], (worth.size, k, 1))
        np.put_along_axis(grown, at, np.arange(k, dtype=np.int16)[:, None], axis=-1)
        out[worth] = evaluate_slots(cfg, grown).is_valid
    return out & (rack > 0)


def appendable(cfg: RummiConfig, table: np.ndarray, rack: np.ndarray) -> np.ndarray:
    """``(..., S, K)`` may kind ``k`` be appended to slot ``s`` leaving it valid?

    :func:`_appendable_rows` decides it; this only spreads each rack across its own
    slots, so a whole table -- or a whole batch of them -- is one call.
    """
    table = np.asarray(table)
    rack = np.asarray(rack)
    s, ell = table.shape[-2:]
    racks = np.repeat(rack.reshape(-1, cfg.n_kinds), s, axis=0)
    grown = _appendable_rows(cfg, table.reshape(-1, ell), racks)
    return grown.reshape(*table.shape[:-1], cfg.n_kinds)


def _sorted_rows(cfg: RummiConfig, rows: np.ndarray) -> np.ndarray:
    """Canonical order within a slot: kinds ascending, ``EMPTY`` pushed to the end."""
    key = np.sort(np.where(rows >= 0, rows, np.int16(cfg.n_kinds)), axis=-1)
    return np.where(key == cfg.n_kinds, np.int16(-1), key).astype(np.int16)


def _best_new_sets(
    cfg: RummiConfig, rack: np.ndarray, by_value: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per rack, the highest-scoring set formable from it: its tiles, value, and
    whether one exists at all.

    Ranked across every candidate at once, then realised only for the winner.
    Realising each candidate to score it -- 329 of them on the standard config, a
    handful of tiny NumPy reductions apiece -- was 643k calls and 60% of the whole
    agent's runtime.

    Two facts make the vectorised form exact rather than approximate. A candidate
    holds each kind at most once, so its shortfall is just how many of its kinds the
    rack is out of, and a missing tile becomes a joker rather than shortening the
    set -- so `cand.length` is the realised length whatever the rack holds.
    """
    cand = candidates(cfg)
    n = rack.shape[0]
    kinds = cand.kinds.astype(np.int64)                       # (C, L), EMPTY padded
    held = np.concatenate([rack > 0, np.ones((n, 1), bool)], axis=1)
    at = np.where(kinds >= 0, kinds, cfg.n_kinds)             # padding lands on the True
    have = held[:, at]                                        # (n, C, L)

    shortfall = (~have).sum(-1)
    feasible = shortfall <= rack[:, cfg.joker_kind, None]

    length = cand.length.astype(np.int64)
    value = cand.value.astype(np.int64)
    # Melding needs points; afterwards, shedding tiles is what wins. Packed into one
    # integer so `argmax` does the lexicographic comparison -- and `argmax` returns
    # the *first* maximum, so ties go to the lowest candidate index and the pick is
    # a function of the rack alone.
    by_points = value * (int(length.max()) + 1) + length
    by_size = length * (int(value.max()) + 1) + value
    key = np.where(by_value[:, None], by_points[None, :], by_size[None, :])

    best = np.argmax(np.where(feasible, key, -1), axis=-1)
    chosen = kinds[best]                                      # (n, L)
    rows = np.arange(n)
    # A kind the rack is out of is played as a joker; the padding stays EMPTY.
    tiles = np.where(
        chosen < 0,
        np.int64(-1),
        np.where(rack[rows[:, None], np.maximum(chosen, 0)] > 0, chosen, cfg.joker_kind),
    )
    return tiles, cand.value[best].astype(np.int64), feasible.any(-1)


def plan_turn(
    cfg: RummiConfig, rack: np.ndarray, table: np.ndarray, has_melded: bool
) -> list[int]:
    """Micro-action ids realising one greedy turn; empty when nothing is worth doing."""
    return plan_turns(
        cfg, np.asarray(rack)[None], np.asarray(table)[None], np.array([has_melded])
    )[0]


def plan_turns(
    cfg: RummiConfig, racks: np.ndarray, tables: np.ndarray, melded: np.ndarray
) -> list[list[int]]:
    """One greedy turn per env, planned for the whole batch at once.

    The policy is exactly the one-env policy -- every choice below is the same
    argmax over the same scores, and the tests hold the two to identical plans. What
    changes is that the arrays carry the batch, and that is the whole cost of this
    agent: the work per turn is tiny and there is a great deal of it. Planned one at
    a time, 512 turns spent 93% of a rollout inside `slot_stats`, called 28,709
    times on rows of thirteen tiles.
    """
    n = racks.shape[0]
    if n == 0:
        return []
    rack = np.asarray(racks, dtype=np.int16).copy()
    table = np.asarray(tables, dtype=np.int16).copy()
    melded = np.asarray(melded, dtype=bool)
    rows = np.arange(n)
    # One entry per placed tile: which kind, into which slot, for which envs.
    steps: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    offload = _offload_values(cfg)
    allowed = appendable(cfg, table, rack) & melded[:, None, None]
    # Appends per turn, bounded by the *dealt* rack size. A rack drawn into holds
    # more than that, so the bound caps what greedy plays -- and greedy is the suite
    # opponent, so raising it moves every published score.
    for _ in range(cfg.rack_size):
        live = allowed.any((1, 2))
        if not live.any():
            break
        # Shed the most expensive tile available. `argmax` takes the first maximum,
        # so ties go to the lowest slot and then the lowest kind.
        flat = np.argmax(np.where(allowed, offload, -1).reshape(n, -1), axis=-1)
        slot, kind = flat // cfg.n_kinds, flat % cfg.n_kinds

        row = table[rows, slot]
        row[rows, np.argmax(row < 0, axis=-1)] = kind.astype(np.int16)
        row = _sorted_rows(cfg, row)
        acting = rows[live]
        table[acting, slot[live]] = row[live]
        rack[acting, kind[live]] -= 1
        steps.append((kind, slot, live))

        # An append changes the slot it landed in, and the kind's column only if
        # the rack ran out of it.
        allowed[acting, slot[live]] = _appendable_rows(cfg, row[live], rack[live])
        allowed &= (rack > 0)[:, None, :]

    # Appends only lengthen occupied slots, so the empty ones are still these.
    is_empty = (table >= 0).sum(-1) == 0
    in_order = np.argsort(~is_empty, axis=-1, kind="stable")
    n_empty = is_empty.sum(-1)
    meld_total = np.zeros(n, dtype=np.int64)
    done = np.zeros(n, dtype=bool)

    for j in range(int(n_empty.max(initial=0))):
        live = ~done & (j < n_empty)
        if not live.any():
            break
        tiles, value, feasible = _best_new_sets(cfg, rack, by_value=~melded)
        live &= feasible
        if not live.any():
            break
        slot = in_order[:, j]
        for t in range(tiles.shape[1]):
            kind = tiles[:, t]
            placing = live & (kind >= 0)
            rack[rows[placing], kind[placing]] -= 1
            steps.append((kind, slot, placing))
        meld_total += np.where(live, value, 0)
        # Stop an env that could not place, and one that has met the opening meld.
        done |= ~live | (~melded & (meld_total >= cfg.initial_meld))

    return _to_plans(cfg, steps, melded, meld_total)


def _to_plans(
    cfg: RummiConfig,
    steps: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    melded: np.ndarray,
    meld_total: np.ndarray,
) -> list[list[int]]:
    """The action ids each env's placements come to.

    PLACE everything first, then ASSIGN in placement order, so each new set's first
    tile arrives while its slot is still the lowest empty one.
    """
    n = melded.shape[0]
    if not steps:
        return [[] for _ in range(n)]

    kinds = np.stack([k for k, _, _ in steps])          # (T, n)
    slots = np.stack([s for _, s, _ in steps])
    placed = np.stack([p for _, _, p in steps])
    # Env-major, so each env's placements come out in the order they were made.
    env, step = np.nonzero(placed.T)
    kind, slot = kinds[step, env], slots[step, env]
    place = encode_place_batch(cfg, kind)
    assign = encode_assign_batch(cfg, kind, slot)

    bounds = np.cumsum(placed.sum(0))[:-1]
    # A turn that placed nothing, or that failed to reach the opening meld, is not
    # worth taking: DRAW is what the caller falls back to.
    keep = (placed.any(0)) & (melded | (meld_total >= cfg.initial_meld))
    return [
        [*p.tolist(), *a.tolist(), cfg.end_turn_action] if keep[i] else []
        for i, (p, a) in enumerate(zip(np.split(place, bounds), np.split(assign, bounds), strict=True))
    ]


class GreedyAgent(PlanningAgent):
    name = "greedy"

    def plan(self, obs: Observation, env: int) -> list[int]:
        return plan_turn(
            self.cfg, obs["rack"][env], table(obs)[env], bool(has_melded(obs)[env])
        )

    def plan_batch(self, obs: Observation, envs: np.ndarray) -> list[list[int]]:
        return plan_turns(
            self.cfg, obs["rack"][envs], table(obs)[envs], has_melded(obs)[envs]
        )
