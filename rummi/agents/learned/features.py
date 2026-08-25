"""What the network sees, defined once so two frameworks cannot drift on it.

The observation is a dict of nine fields with different shapes and wildly
different magnitudes -- counts, tile ids, a turn counter, a meld value. This
module fixes two things about turning it into a flat vector: **the order** the
fields are concatenated in, and **the divisor** each position is scaled by. Both
implementations read them from here, so a mismatch is impossible rather than
merely unlikely.

*Why `table_sets` is left out.* It holds raw kind ids, and a kind id is not an
ordinal quantity -- kind 5 and kind 6 may be different colours. Feeding it to an
MLP as a scaled scalar would be worse than not feeding it, and one-hotting it
costs `S*L*K` inputs. It is also close to redundant: `slot_features` carries
`len`, `colour`, `lo`, `hi`, `n_jokers` and `value` per slot, which pins a run's
contents exactly and a group's up to which colours are present. A set-aware
encoder -- embed each tile, attend over slots -- is the obvious way to use the
real thing, and is exactly what this reference model is a baseline for.
"""

from __future__ import annotations

import numpy as np

from rummi.rules.config import RummiConfig
from rummi.rules.observation import (
    F_COLOR,
    F_EXTENDABLE,
    F_GROUP_VALID,
    F_HI,
    F_IS_NEW,
    F_JOKERS,
    F_LEN,
    F_LO,
    F_RUN_VALID,
    F_VALUE,
    MELD_PROGRESS,
    MELD_REMAINING,
    MICRO_COUNT,
    POOL_SIZE,
    SLOT_FEATURES,
    max_meld_value,
)

FEATURE_FIELDS: tuple[str, ...] = (
    "rack",
    "workbench",
    "placed_this_turn",
    "unseen",
    "slot_features",
    "rack_sizes",
    "melded",
    "scalars",
)
"""Concatenation order. Flattened per field, batch dimension kept."""


def feature_dim(cfg: RummiConfig) -> int:
    k, s, p = cfg.n_kinds, cfg.max_sets, cfg.n_players
    return 4 * k + s * SLOT_FEATURES + 2 * p + 4


def _slot_column_scales(cfg: RummiConfig) -> np.ndarray:
    """One divisor per `slot_features` column, in the order SPEC.md section 8 lists.

    Booleans divide by 1; the rest by their own ceiling, so no column arrives
    orders of magnitude larger than its neighbours.
    """
    scales = np.ones(SLOT_FEATURES, dtype=np.float32)
    scales[F_LEN] = cfg.max_set_len
    scales[F_RUN_VALID] = 1.0
    scales[F_GROUP_VALID] = 1.0
    scales[F_EXTENDABLE] = 1.0
    scales[F_COLOR] = max(1, cfg.n_colors)
    scales[F_LO] = max(1, cfg.n_numbers)
    scales[F_HI] = max(1, cfg.n_numbers)
    scales[F_JOKERS] = max(1, cfg.n_jokers)
    scales[F_VALUE] = max(1, cfg.n_numbers * cfg.max_set_len)
    scales[F_IS_NEW] = 1.0
    return scales


def feature_scale(cfg: RummiConfig) -> np.ndarray:
    """`(feature_dim,)` divisor per position, so every input lands near [0, 1].

    Returned as NumPy on purpose. Both networks convert it once at construction;
    a JAX lookup table must never be a traced array, and a torch one wants to be
    a registered buffer rather than rebuilt per forward pass.
    """
    k, s, p = cfg.n_kinds, cfg.max_sets, cfg.n_players
    copies = float(max(cfg.n_copies, cfg.n_jokers, 1))

    parts = [
        np.full(k, copies, dtype=np.float32),          # rack
        np.full(k, copies, dtype=np.float32),          # workbench
        np.full(k, copies, dtype=np.float32),          # placed_this_turn
        np.full(k, copies, dtype=np.float32),          # unseen
        np.tile(_slot_column_scales(cfg), s),          # slot_features, row-major
        np.full(p, float(cfg.n_tiles), dtype=np.float32),   # rack_sizes
        np.ones(p, dtype=np.float32),                  # melded
        np.zeros(4, dtype=np.float32),                 # scalars, filled below
    ]
    scalars = parts[-1]
    scalars[POOL_SIZE] = max(1, cfg.n_tiles)
    scalars[MELD_PROGRESS] = max(1, max_meld_value(cfg))
    scalars[MELD_REMAINING] = max(1, cfg.initial_meld)
    scalars[MICRO_COUNT] = max(1, cfg.max_micro_per_turn)

    scale = np.concatenate(parts).astype(np.float32)
    assert scale.shape == (feature_dim(cfg),), (scale.shape, feature_dim(cfg))
    assert (scale > 0).all(), "a zero divisor would produce inf features"
    return scale
