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
from rummi.agents.greedy_agent import plan_turn, plan_turns
from rummi.env.numpy.sets import evaluate_slots
from rummi.rules.actions import encode_assign, encode_pick, encode_place
from rummi.rules.config import RummiConfig
from rummi.rules.encoding import EMPTY
from rummi.solver.candidates import candidates


def _survivors(cfg: RummiConfig, boards: np.ndarray) -> np.ndarray:
    """``(n, S, L)`` true where the tile at that position can be lifted out and the
    set it leaves behind is still a set.

    Every removal at once: dropping a position and closing the gap keeps a sorted
    row sorted, so the whole ``(S, L)`` grid of remainders is one gather and one
    :func:`evaluate_slots`.
    """
    ell = cfg.max_set_len
    # Row `p` of this holds every position but `p`, in order, with `p` pushed last.
    order = np.argsort(np.eye(ell, dtype=np.int8), axis=-1, kind="stable")
    remainder = np.take(boards, order, axis=-1)
    remainder[..., -1] = EMPTY
    return (
        np.asarray(evaluate_slots(cfg, remainder).is_valid)
        & np.asarray(evaluate_slots(cfg, boards).is_valid)[..., None]
        & (boards >= 0)
    )


def _sets_using(
    cfg: RummiConfig, rack: np.ndarray, borrowed: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per ``(rack, borrowed tile)`` pair, the best set formable from the rack *plus*
    that tile: the winning candidate, and ``-1`` where there is none.

    Scored by how many rack tiles it sheds -- the borrowed one was already on the
    table, so moving it around is worth nothing by itself. That count is the
    candidate's length minus the borrowed tile whatever the rack holds, since a tile
    the rack is out of is played as a joker rather than dropped, so the ranking is
    the same lexicographic ``(length, value)`` the enumeration is already sorted for.
    """
    cand = candidates(cfg)
    kinds = cand.kinds.astype(np.int64)
    length = cand.length.astype(np.int64)
    value = cand.value.astype(np.int64)

    # The padding column indexes a slot that always reads "held", so it never counts.
    at = np.where(kinds >= 0, kinds, cfg.n_kinds)
    held = np.concatenate([rack > 0, np.ones((rack.shape[0], 1), bool)], axis=1)
    have = held[:, at]                                              # (P, C, L)
    others = (kinds >= 0)[None] & (kinds[None] != borrowed[:, None, None])

    shortfall = (others & ~have).sum(-1)
    feasible = (
        (cand.counts > 0).T[borrowed]
        & (shortfall <= rack[:, cfg.joker_kind, None])
    )
    key = length * (int(value.max()) + 1) + value
    best = np.argmax(np.where(feasible, key[None, :], -1), axis=-1)
    return np.where(feasible[np.arange(best.shape[0]), best], best, -1), have


def _tiles_of(
    cfg: RummiConfig, chosen: np.ndarray, have: np.ndarray, borrowed: int
) -> list[int]:
    """The candidate as tiles: the borrowed one first, then the rack's own in kind
    order, each replaced by a joker where the rack is out of it."""
    others = (chosen >= 0) & (chosen != borrowed)
    played = np.where(have[others], chosen[others], cfg.joker_kind)
    return [int(borrowed), *played.tolist()]


def steal_one(cfg: RummiConfig, rack: np.ndarray, board: np.ndarray) -> list[int]:
    """Micro-actions for a one-tile steal, or ``[]`` if none pays."""
    return steal_one_batch(cfg, np.asarray(rack)[None], np.asarray(board)[None])[0]


def steal_one_batch(
    cfg: RummiConfig, racks: np.ndarray, boards: np.ndarray
) -> list[list[int]]:
    """One steal per env, decided for the whole batch at once.

    The same choice as one env at a time -- longest set the borrowed tile completes,
    ties to the lowest slot then the lowest position -- but the candidate scan runs
    over every (tile, candidate) pair in one pass instead of a Python loop per tile.
    """
    n = racks.shape[0]
    empty = ~(boards >= 0).any(-1)
    target = np.argmax(empty, axis=-1)
    stealable = _survivors(cfg, boards) & np.asarray(empty.any(-1))[:, None, None]

    env, slot, pos = np.nonzero(stealable)
    if env.size == 0:
        return [[] for _ in range(n)]

    borrowed = boards[env, slot, pos].astype(np.int64)
    best, have = _sets_using(cfg, racks[env], borrowed)

    # The longest set wins; `np.nonzero` walked slots then positions, so the earliest
    # of an equal-scoring group is the one the tie-break wants.
    length = candidates(cfg).length.astype(np.int64)
    score = np.where(best >= 0, length[best], -1)
    order = np.lexsort((np.arange(env.size), -score, env))
    first = np.searchsorted(env[order], np.arange(n))
    plans: list[list[int]] = [[] for _ in range(n)]
    for e in np.flatnonzero(np.bincount(env, minlength=n)):
        i = order[first[e]]
        if score[i] < 0:
            continue
        tiles = _tiles_of(cfg, candidates(cfg).kinds[best[i]], have[i, best[i]], borrowed[i])
        actions = [encode_pick(cfg, int(slot[i]), int(pos[i]))]
        actions += [encode_place(cfg, t) for t in tiles if t != borrowed[i]]
        actions += [encode_assign(cfg, t, int(target[e])) for t in tiles]
        actions.append(cfg.end_turn_action)
        plans[e] = actions
    return plans


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

    def plan_batch(self, obs: Observation, envs: np.ndarray) -> list[list[int]]:
        """The greedy half for the whole batch at once; the steal only where that
        half came back empty, which is the only case it is reached in."""
        cfg = self.cfg
        racks, boards = obs["rack"][envs], table(obs)[envs]
        melded = has_melded(obs)[envs]
        plans = plan_turns(cfg, racks, boards, melded)
        stealing = np.flatnonzero(
            melded & np.array([not plan for plan in plans], dtype=bool)
        )
        if stealing.size:
            steals = steal_one_batch(cfg, racks[stealing], boards[stealing])
            for i, steal in zip(stealing.tolist(), steals, strict=True):
                plans[i] = steal
        return plans
