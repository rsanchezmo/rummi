"""What the network sees, defined once so two frameworks cannot drift on it.

The observation is a dict of nine fields with different shapes and wildly
different magnitudes -- counts, tile ids, a turn counter, a meld value. This
module fixes two things about turning it into a flat vector: **the order** the
fields are concatenated in, and **the divisor** each position is scaled by. Both
implementations read them from here, so a mismatch is impossible rather than
merely unlikely.

*Why `table_sets` is not here, tested rather than argued.* Every bundled agent
reads it, so a network without it cannot represent their policy exactly, and that
looked like the obvious cause of cloning plateauing at 75% agreement. It is not.
Adding it as per-slot kind counts `(S, K)` -- exact, order-free, the same
representation `rack` and `unseen` use -- at matched budget on `standard` gave
**74.5% agreement and -360.6**, against **75.6% and -348.3** without. Slightly
worse, not better.

Two readings, and the data does not separate them: either `slot_features` already
carries what matters (`run_valid`, `group_valid`, `is_extendable` and `value` are
precisely the relational facts a policy needs, and `lo`/`hi`/`colour` pin a run
exactly), or 1855 extra inputs into a 256-unit layer are diluted and 40 epochs
under-trains them -- the with-table run ended on a *higher* NLL, which is what
under-fitting looks like.

Either way it is not the cheap win it appeared to be, so the baseline stays lean.
:func:`slot_counts_numpy` and the per-network `slot_counts` methods are kept and
tested, because a factored or bilinear action head needs per-slot representations
and will want them.
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
"""Observation fields, in concatenation order. Flattened, batch dimension kept.

`slot_counts` is deliberately *not* among them -- see the module docstring for the
measurement that settled it.
"""


def slot_counts_numpy(cfg: RummiConfig, table_sets: np.ndarray) -> np.ndarray:
    """`(B, S, K)` count of each kind in each slot. The reference for the ports.

    `EMPTY` padding contributes nothing, which is the only subtlety: it is `-1`, so
    it has to be masked out rather than clamped into kind 0.
    """
    b, s, _ = table_sets.shape
    counts = np.zeros((b, s, cfg.n_kinds), dtype=np.float32)
    valid = table_sets >= 0
    idx = np.where(valid, table_sets, 0).astype(np.int64)
    np.add.at(
        counts,
        (
            np.arange(b)[:, None, None].repeat(s, 1).repeat(table_sets.shape[2], 2),
            np.arange(s)[None, :, None].repeat(b, 0).repeat(table_sets.shape[2], 2),
            idx,
        ),
        valid.astype(np.float32),
    )
    return counts


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
