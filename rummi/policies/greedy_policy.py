"""Greedy one-turn-lookahead baseline.

Plans a whole turn when the turn starts, then emits the micro-actions that
realise it. Planning per turn rather than per step keeps the Python cost off the
per-step path, and pre-resolving slots is exact: nobody else touches the table
while one player is mid-turn, so the policy can predict which empty slot each new
set will land in.

Deliberately does *not* rearrange the table -- it only appends to existing sets
and lays new ones from its own rack. Rearrangement is what the CP-SAT optimal
policy is for, so keeping greedy simple leaves a real gap between the two.

Reads only what the acting player may see: their own rack and the table.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from rummi.rules.actions import encode_assign, encode_place
from rummi.rules.config import RummiConfig
from rummi.rules.encoding import tables
from rummi.env.numpy.sets import evaluate_slots
from rummi.env.numpy.state import BatchState
from rummi.solver.candidates import candidates


def _offload_values(cfg: RummiConfig) -> np.ndarray:
    """How much rack penalty each kind sheds when played."""
    v = tables(cfg).value.astype(np.int32).copy()
    v[cfg.joker_kind] = cfg.joker_penalty
    return v


def _appendable(cfg: RummiConfig, table: np.ndarray, rack: np.ndarray) -> np.ndarray:
    """``(S, K)`` may kind ``k`` be appended to slot ``s`` leaving it valid?"""
    s, ell = table.shape
    k = cfg.n_kinds
    base = evaluate_slots(cfg, table)
    has_room = (table < 0).any(-1)
    pos = np.argmax(table < 0, axis=-1)

    grown = np.repeat(table[:, None, :], k, axis=1)
    grown[np.arange(s)[:, None], np.arange(k)[None, :], pos[:, None]] = np.arange(
        k, dtype=np.int16
    )
    grown_valid = evaluate_slots(cfg, grown).is_valid

    return grown_valid & base.is_valid[:, None] & has_room[:, None] & (rack > 0)[None, :]


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
    """Highest-scoring set formable from ``rack`` alone, or ``None``."""
    cand = candidates(cfg)
    best: tuple[list[int], int] | None = None
    best_key = (-1, -1)
    for i in range(len(cand)):
        tiles = _realise(cfg, cand.counts[i], rack)
        if tiles is None:
            continue
        value = int(cand.value[i])
        # Melding needs points; afterwards, shedding tiles is what wins.
        key = (value, len(tiles)) if by_value else (len(tiles), value)
        if key > best_key:
            best_key, best = key, (tiles, value)
    return best


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
        while True:
            allowed = _appendable(cfg, table, rack)
            if not allowed.any():
                break
            # Shed the most expensive tile available.
            scores = np.where(allowed, offload[None, :], -1)
            slot, kind = np.unravel_index(int(np.argmax(scores)), scores.shape)
            table[slot, int(np.argmax(table[slot] < 0))] = kind
            table[slot] = np.sort(np.where(table[slot] >= 0, table[slot], cfg.n_kinds))
            table[slot] = np.where(table[slot] == cfg.n_kinds, -1, table[slot])
            rack[kind] -= 1
            lengths[slot] += 1
            placements.append((int(kind), int(slot)))

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


class GreedyPolicy:
    def __init__(self, cfg: RummiConfig, strict: bool = True) -> None:
        self.cfg = cfg
        self.strict = strict
        self._plans: dict[int, deque[int]] = {}

    def act(
        self, state: BatchState, mask: np.ndarray, envs: np.ndarray | None = None
    ) -> np.ndarray:
        """``envs`` restricts which envs this policy owns.

        It must be honoured rather than ignored: plans are cached per env and
        consumed one action at a time, so touching an env another policy controls
        would desynchronise that env's plan.
        """
        cfg = self.cfg
        out = np.full(state.batch_size, cfg.draw_action, dtype=np.int64)

        for env in range(state.batch_size):
            if state.done[env] or (envs is not None and not envs[env]):
                continue
            if state.micro_count[env] == 0:
                player = int(state.current[env])
                self._plans[env] = deque(
                    plan_turn(
                        cfg,
                        state.racks[env, player],
                        state.table_sets[env],
                        bool(state.melded[env, player]),
                    )
                )
            plan = self._plans.get(env)
            if not plan:
                continue
            action = plan.popleft()
            if not mask[env, action]:
                if self.strict:
                    from rummi.rules.actions import action_name

                    raise AssertionError(
                        f"greedy planned an illegal {action_name(cfg, action)} in env {env}"
                    )
                plan.clear()
                continue
            out[env] = action
        return out
