"""Tile-kind encoding and the lookup tables derived from a config.

A *kind* is a tile identity, not an instance: ``n_copies`` tiles share a kind.
Numbered kinds are laid out colour-major so that a run occupies consecutive
kind ids within one colour::

    kind = color * n_numbers + (number - 1)      0 <= kind < n_colors * n_numbers
    kind = n_colors * n_numbers                  the joker

``-1`` marks an empty slot position throughout the simulator.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from rummi.rules.config import RummiConfig

EMPTY = -1
"""Sentinel stored in ``table_sets`` for an unoccupied position."""


def kind_of(cfg: RummiConfig, color: int, number: int) -> int:
    """Kind id of a numbered tile. ``number`` is 1-based."""
    if not 0 <= color < cfg.n_colors:
        raise ValueError(f"color {color} outside [0, {cfg.n_colors})")
    if not 1 <= number <= cfg.n_numbers:
        raise ValueError(f"number {number} outside [1, {cfg.n_numbers}]")
    return color * cfg.n_numbers + (number - 1)


@dataclass(frozen=True, slots=True)
class Tables:
    """Per-config lookup tables, indexed by kind id.

    Every array has length ``K`` and the joker occupies the final entry, so the
    tables can be indexed directly by anything the engine stores. They are plain
    NumPy int arrays so a torch/JAX backend can adopt them verbatim.
    """

    cfg: RummiConfig
    color: np.ndarray
    """Colour of each kind; ``-1`` for the joker."""
    number: np.ndarray
    """Number of each kind; ``-1`` for the joker."""
    value: np.ndarray
    """Face value used for scoring; ``0`` for the joker, whose value is
    positional and must be resolved from the set it sits in."""
    is_joker: np.ndarray
    copies: np.ndarray
    """How many physical tiles exist of each kind."""

    @property
    def total_copies(self) -> int:
        return int(self.copies.sum())


@lru_cache(maxsize=None)
def tables(cfg: RummiConfig) -> Tables:
    """Build (and memoise) the lookup tables for ``cfg``."""
    k = cfg.n_kinds
    color = np.full(k, -1, dtype=np.int16)
    number = np.full(k, -1, dtype=np.int16)

    numbered = np.arange(cfg.n_numbered_kinds, dtype=np.int16)
    color[: cfg.n_numbered_kinds] = numbered // cfg.n_numbers
    number[: cfg.n_numbered_kinds] = numbered % cfg.n_numbers + 1

    value = np.where(number > 0, number, 0).astype(np.int16)

    is_joker = np.zeros(k, dtype=bool)
    is_joker[cfg.joker_kind] = True

    copies = np.full(k, cfg.n_copies, dtype=np.int16)
    copies[cfg.joker_kind] = cfg.n_jokers

    return Tables(
        cfg=cfg,
        color=color,
        number=number,
        value=value,
        is_joker=is_joker,
        copies=copies,
    )


_COLOR_LETTERS = "RBYKGMCW"


def color_letter(color: int) -> str:
    return _COLOR_LETTERS[color % len(_COLOR_LETTERS)]


def kind_name(cfg: RummiConfig, kind: int) -> str:
    """Short human label, e.g. ``R7`` or ``*`` for the joker, ``.`` for empty."""
    if kind == EMPTY:
        return "."
    if kind == cfg.joker_kind:
        return "*"
    t = tables(cfg)
    return f"{color_letter(int(t.color[kind]))}{int(t.number[kind])}"


def counts_to_kinds(counts: np.ndarray) -> np.ndarray:
    """Expand a ``(K,)`` count vector into a flat array of kind ids."""
    return np.repeat(np.arange(counts.shape[-1], dtype=np.int16), counts)


def kinds_to_counts(cfg: RummiConfig, kinds) -> np.ndarray:
    """Collapse an iterable of kind ids into a ``(K,)`` count vector."""
    counts = np.zeros(cfg.n_kinds, dtype=np.int16)
    for kind in kinds:
        if kind != EMPTY:
            counts[kind] += 1
    return counts
