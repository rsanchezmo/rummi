"""Greedy, plus a single stolen tile.

Greedy's weakness is specific and measurable: it stalls out in ~98% of games,
because once it can neither append to a set nor lay one from its rack it simply
draws. The reason is that it never takes an existing set apart.

This agent adds the cheapest possible rearrangement -- steal **one** tile from a
set that stays legal without it, and use it to complete a new set. That is the
canonical Rummikub manoeuvre in its smallest form: extend a run, then lift a tile
out of it to finish a group.

It exists because the ceiling was unreachable. CP-SAT beats greedy 100-0, which
tells a newcomer nothing about whether their agent is any good. This sits in
between, and the distance from here to CP-SAT is the value of rearranging *more
than one tile at a time*.
"""

from __future__ import annotations

import numpy as np

from rummi.agents.base import Observation, PlanningAgent, has_melded, table
from rummi.agents.greedy_agent import plan_turn
from rummi.env.numpy.sets import evaluate_slots, pad_slot
from rummi.rules.actions import encode_assign, encode_pick, encode_place
from rummi.rules.config import RummiConfig
from rummi.solver.candidates import candidates


def _stealable(cfg: RummiConfig, board: np.ndarray) -> list[tuple[int, int, int]]:
    """``(slot, position, kind)`` for every tile whose set survives losing it."""
    out = []
    verdict = evaluate_slots(cfg, board)
    for slot in range(cfg.max_sets):
        if not verdict.is_valid[slot]:
            continue
        tiles = [int(k) for k in board[slot] if k >= 0]
        for pos, kind in enumerate(tiles):
            remainder = tiles[:pos] + tiles[pos + 1 :]
            if len(remainder) < cfg.min_set:
                continue
            if bool(evaluate_slots(cfg, pad_slot(cfg, remainder)[None]).is_valid[0]):
                out.append((slot, pos, kind))
    return out


def _set_using(cfg: RummiConfig, rack: np.ndarray, borrowed: int):
    """Best set formable from the rack *plus* one borrowed tile, or ``None``.

    Scored by how many rack tiles it sheds -- the borrowed one was already on the
    table, so moving it around is worth nothing by itself.
    """
    cand = candidates(cfg)
    jokers = int(rack[cfg.joker_kind])
    best = None
    best_key = (0, 0)

    for i in range(len(cand)):
        counts = cand.counts[i]
        if not counts[borrowed]:
            continue
        need = counts.copy()
        need[borrowed] -= 1                       # this one comes off the table
        missing = np.maximum(0, need - rack)
        if int(missing.sum()) > jokers:
            continue

        tiles, from_rack = [borrowed], 0
        for kind in np.flatnonzero(need):
            have = min(int(need[kind]), int(rack[kind]))
            tiles.extend([int(kind)] * have)
            tiles.extend([cfg.joker_kind] * (int(need[kind]) - have))
            from_rack += int(need[kind])
        key = (from_rack, int(cand.value[i]))
        if from_rack and key > best_key:
            best_key, best = key, tiles
    return best


def steal_one(cfg: RummiConfig, rack: np.ndarray, board: np.ndarray) -> list[int]:
    """Micro-actions for a one-tile steal, or ``[]`` if none pays."""
    empty = [s for s in range(cfg.max_sets) if not (board[s] >= 0).any()]
    if not empty:
        return []

    best, best_key = None, (0, 0)
    for slot, pos, kind in _stealable(cfg, board):
        tiles = _set_using(cfg, rack, kind)
        if tiles is None:
            continue
        played = sum(1 for t in tiles if t != kind)
        key = (played, len(tiles))
        if key > best_key:
            best_key, best = key, (slot, pos, kind, tiles)
    if best is None:
        return []

    slot, pos, kind, tiles = best
    target = empty[0]
    actions = [encode_pick(cfg, slot, pos)]
    actions += [encode_place(cfg, t) for t in tiles if t != kind]
    actions += [encode_assign(cfg, t, target) for t in tiles]
    actions.append(cfg.end_turn_action)
    return actions


class RearrangeAgent(PlanningAgent):
    name = "rearrange"

    def plan(self, obs: Observation, env: int) -> list[int]:
        cfg = self.cfg
        rack, board = obs["rack"][env], table(obs)[env]
        melded = bool(has_melded(obs)[env])

        greedy = plan_turn(cfg, rack, board, melded)
        if greedy or not melded:
            # Before melding the table may not be touched at all, so there is no
            # steal to attempt -- and when greedy already has a play, take it.
            return greedy
        return steal_one(cfg, rack, board)
