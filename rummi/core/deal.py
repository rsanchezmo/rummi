"""Reset and dealing.

The whole deck is permuted once per env at reset and stored in ``deck_order``;
every later draw just advances ``draw_ptr``. So the step function contains no
randomness at all, per-env streams are independent by construction, and a
batched rollout is bit-identical to the same envs run one at a time.
"""

from __future__ import annotations

import numpy as np

from rummi.core.config import RummiConfig
from rummi.core.encoding import EMPTY, tables
from rummi.core.state import NO_WINNER, BatchState, allocate


def deck(cfg: RummiConfig) -> np.ndarray:
    """``(n_tiles,)`` every physical tile as a kind id."""
    return np.repeat(np.arange(cfg.n_kinds, dtype=np.int16), tables(cfg).copies)


def env_seeds(seed: int, batch_size: int) -> list[np.random.SeedSequence]:
    """Per-env seeds that depend only on ``(seed, env_index)``."""
    return np.random.SeedSequence(seed).spawn(batch_size)


def derived_seeds(base: int, step: int, envs) -> list[np.random.SeedSequence]:
    """Seeds for re-dealing ``envs`` at ``step``, derived from position alone.

    Because each seed depends only on ``(base, step, env)`` and not on how many
    re-deals came before, a recorded trajectory can be replayed without carrying
    any counter state -- which is what keeps the golden fixtures self-contained.
    """
    return [np.random.SeedSequence([base, step, int(env)]) for env in envs]


def reset(cfg: RummiConfig, batch_size: int, seed: int = 0) -> BatchState:
    state = allocate(cfg, batch_size)
    reset_envs(state, np.arange(batch_size), env_seeds(seed, batch_size))
    return state


def reset_envs(
    state: BatchState,
    which: np.ndarray,
    seeds: list[np.random.SeedSequence],
) -> None:
    """Re-deal the given envs in place. ``seeds`` is parallel to ``which``."""
    cfg = state.cfg
    base = deck(cfg)
    if len(seeds) != len(which):
        raise ValueError("one seed per env being reset")

    for env, seed in zip(np.atleast_1d(which), seeds):
        order = np.random.default_rng(seed).permutation(base)
        state.deck_order[env] = order

        state.racks[env] = 0
        for player in range(cfg.n_players):
            dealt = order[player * cfg.rack_size : (player + 1) * cfg.rack_size]
            np.add.at(state.racks[env, player], dealt, 1)

        state.draw_ptr[env] = cfg.n_players * cfg.rack_size
        state.pool[env] = tables(cfg).copies - state.racks[env].sum(0)

    state.table_sets[which] = EMPTY
    state.table_snapshot[which] = EMPTY
    state.workbench[which] = 0
    state.placed_rack[which] = 0
    state.slot_new[which] = False
    state.melded[which] = False
    state.current[which] = 0
    state.micro_count[which] = 0
    state.turn_count[which] = 0
    state.consecutive_draws[which] = 0
    state.last_action[which] = -1
    state.action_history[which] = -1
    state.winner[which] = NO_WINNER
    state.done[which] = False
    state.truncated[which] = False
