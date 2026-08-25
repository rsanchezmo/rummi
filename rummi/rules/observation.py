"""The observation's layout: field names, column order, scalar order.

Backend-free and definitional, which is why it lives here rather than beside any
one encoder. Three implementations of :func:`encode` read these, and a column
order they disagreed on would be a silent, shape-clean divergence -- the worst
kind. SPEC.md section 8 is the prose version of this file.
"""

from __future__ import annotations

from rummi.rules.config import RummiConfig

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

FIELDS: tuple[str, ...] = (
    "rack",
    "table_sets",
    "slot_features",
    "workbench",
    "placed_this_turn",
    "unseen",
    "rack_sizes",
    "melded",
    "scalars",
)
"""Every key an encoder must produce, and nothing else."""


def max_meld_value(cfg: RummiConfig) -> int:
    """Loosest bound on a turn's declared meld value: every slot a top-value set."""
    top_run = sum(range(cfg.n_numbers - cfg.max_set_len + 1, cfg.n_numbers + 1))
    top_group = cfg.n_colors * cfg.n_numbers
    return cfg.max_sets * max(top_run, top_group)
