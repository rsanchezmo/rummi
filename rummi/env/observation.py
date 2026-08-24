"""Observation encoding for the acting seat.

Two properties matter here.

*Seat-relative.* Per-player fields are rotated so index 0 is always the acting
seat. That is what lets a single shared policy play every seat in self-play
without learning P separate conventions.

*Information-correct.* The observation exposes only what the acting player may
see: their own rack, the table, and ``unseen`` -- the tiles whose location they
cannot know, which is the pool and the opponents' racks combined. Individual
opponent racks are never revealed, only their sizes.
"""

from __future__ import annotations

import numpy as np

from rummi.rules.config import RummiConfig
from rummi.rules.encoding import tables
from rummi.env.numpy.masks import current_rack, meld_value
from rummi.env.numpy.sets import evaluate, slot_stats
from rummi.env.numpy.state import BatchState

SLOT_FEATURES = 10
"""[len, run_valid, group_valid, extendable, colour, lo, hi, n_jokers, value, is_new]"""

# Index into obs["scalars"]. Named so an agent never has to count positions.
POOL_SIZE = 0
MELD_PROGRESS = 1
MELD_REMAINING = 2
MICRO_COUNT = 3
N_SCALARS = 4

# Index into obs["slot_features"].
F_LEN = 0
F_RUN_VALID = 1
F_GROUP_VALID = 2
F_EXTENDABLE = 3
F_COLOR = 4
F_LO = 5
F_HI = 6
F_JOKERS = 7
F_VALUE = 8
F_IS_NEW = 9


def observation_space(cfg: RummiConfig):
    from gymnasium import spaces

    k, s, ell, p = cfg.n_kinds, cfg.max_sets, cfg.max_set_len, cfg.n_players
    max_copies = max(cfg.n_copies, cfg.n_jokers)
    return spaces.Dict(
        {
            "rack": spaces.Box(0, max_copies, (k,), dtype=np.int16),
            "table_sets": spaces.Box(-1, k - 1, (s, ell), dtype=np.int16),
            "slot_features": spaces.Box(
                -2, max(k, cfg.n_numbers * ell, cfg.n_numbers * cfg.n_colors), (s, SLOT_FEATURES), dtype=np.int32
            ),
            "workbench": spaces.Box(0, max_copies, (k,), dtype=np.int16),
            "placed_this_turn": spaces.Box(0, max_copies, (k,), dtype=np.int16),
            "unseen": spaces.Box(0, max_copies, (k,), dtype=np.int16),
            "rack_sizes": spaces.Box(0, cfg.n_tiles, (p,), dtype=np.int16),
            "melded": spaces.Box(0, 1, (p,), dtype=np.int8),
            # Per-element bounds: these four quantities have genuinely different
            # ranges, and a single shared bound is overflowed by micro_count.
            "scalars": spaces.Box(
                low=np.zeros(N_SCALARS, dtype=np.int32),
                high=np.array(
                    [cfg.n_tiles, _max_meld_value(cfg), cfg.initial_meld, cfg.max_micro_per_turn],
                    dtype=np.int32,
                ),
                dtype=np.int32,
            ),
        }
    )
    # scalars = [pool_size, meld_progress, initial_meld_remaining, micro_count]


def _max_meld_value(cfg: RummiConfig) -> int:
    """Loosest bound on a turn's declared meld value: every slot a top-value set."""
    top_run = sum(range(cfg.n_numbers - cfg.max_set_len + 1, cfg.n_numbers + 1))
    top_group = cfg.n_colors * cfg.n_numbers
    return cfg.max_sets * max(top_run, top_group)


def _seat_rotation(state: BatchState) -> np.ndarray:
    """``(B, P)`` seat indices ordered from the acting seat outwards."""
    p = state.cfg.n_players
    return (state.current[:, None].astype(np.int64) + np.arange(p)[None, :]) % p


def encode(state: BatchState) -> dict[str, np.ndarray]:
    cfg = state.cfg
    stats = slot_stats(cfg, state.table_sets)
    ev = evaluate(cfg, stats)

    rack = current_rack(state)
    table_counts = state.table_counts()
    unseen = (
        tables(cfg).copies[None, :].astype(np.int16) - rack - table_counts - state.workbench
    )

    slot_features = np.stack(
        [
            stats.n.astype(np.int32),
            ev.run_valid.astype(np.int32),
            ev.group_valid.astype(np.int32),
            ev.is_extendable.astype(np.int32),
            stats.color.astype(np.int32),
            stats.lo.astype(np.int32),
            stats.hi.astype(np.int32),
            stats.n_jokers.astype(np.int32),
            ev.value.astype(np.int32),
            state.slot_new.astype(np.int32),
        ],
        axis=-1,
    )

    rotation = _seat_rotation(state)
    progress = meld_value(state, ev.value).astype(np.int32)
    has_melded = state.melded[np.arange(state.batch_size), state.current]
    remaining = np.where(has_melded, 0, np.maximum(0, cfg.initial_meld - progress))

    return {
        "rack": rack.astype(np.int16),
        "table_sets": state.table_sets.astype(np.int16),
        "slot_features": slot_features,
        "workbench": state.workbench.astype(np.int16),
        "placed_this_turn": state.placed_rack.astype(np.int16),
        "unseen": unseen.astype(np.int16),
        "rack_sizes": np.take_along_axis(state.racks.sum(-1), rotation, axis=1).astype(np.int16),
        "melded": np.take_along_axis(state.melded, rotation, axis=1).astype(np.int8),
        "scalars": np.stack(
            [
                state.pool_size.astype(np.int32),
                progress,
                remaining.astype(np.int32),
                state.micro_count.astype(np.int32),
            ],
            axis=-1,
        ),
    }
