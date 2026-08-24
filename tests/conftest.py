"""Shared helpers for building hand-made states."""

from __future__ import annotations

import numpy as np

from rummi.rules.config import RummiConfig
from rummi.env.numpy.deal import reset
from rummi.rules.encoding import tables
from rummi.env.numpy.masks import legal_actions
from rummi.env.numpy.engine import step
from rummi.env.numpy.state import BatchState


def rebalance_pool(state: BatchState) -> None:
    """Push every tile that is not in a rack or on the table back into the pool.

    Hand-made states are built by overwriting racks and slots directly, which
    breaks conservation; this restores it so ``check_invariants`` stays meaningful.
    """
    cfg = state.cfg
    state.pool[:] = (
        tables(cfg).copies[None, :]
        - state.racks.sum(1)
        - state.table_counts()
        - state.workbench
    )
    state.draw_ptr[:] = cfg.n_tiles - state.pool.sum(-1)
    state.check_invariants()


def state_with(
    cfg: RummiConfig,
    rack: list[int] | None = None,
    table: list[list[int]] | None = None,
    batch_size: int = 1,
    seed: int = 0,
    melded: bool = False,
) -> BatchState:
    """A dealt state overwritten so the acting player holds ``rack`` and the table
    holds ``table`` (one inner list per slot)."""
    s = reset(cfg, batch_size, seed=seed)
    if rack is not None:
        s.racks[:, 0] = 0
        for k in rack:
            s.racks[:, 0, k] += 1
    for slot, kinds in enumerate(table or []):
        s.table_sets[:, slot, : len(kinds)] = kinds
    s.table_snapshot[:] = s.table_sets
    if melded:
        s.melded[:] = True
    rebalance_pool(s)
    return s


def play(state: BatchState, actions) -> np.ndarray:
    """Apply a sequence of actions to every env, asserting each is legal."""
    for a in actions:
        mask = legal_actions(state)
        assert mask[:, a].all(), f"action {a} illegal"
        step(state, np.full(state.batch_size, a), mask)
    return legal_actions(state)


def drain_pool(state: BatchState, to_player: int = -1) -> None:
    """Empty the pool by dealing what is left into one player's rack.

    Zeroing the pool directly would delete tiles and trip conservation, so the
    honest way to reach an exhausted pool is to give the remainder to somebody.
    """
    cfg = state.cfg
    player = to_player % cfg.n_players
    state.racks[:, player] += state.pool
    state.pool[:] = 0
    state.draw_ptr[:] = cfg.n_tiles
    state.check_invariants()
