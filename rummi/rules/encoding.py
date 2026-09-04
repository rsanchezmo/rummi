"""Tile-kind encoding and the lookup tables derived from a config.

A *kind* is a tile identity, not an instance: ``n_copies`` tiles share a kind.
Numbered kinds are laid out colour-major so that a run occupies consecutive
kind ids within one colour::

    kind = color * n_numbers + (number - 1)      0 <= kind < n_colors * n_numbers
    kind = n_colors * n_numbers                  the joker

``-1`` marks an empty slot position throughout the simulator.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from functools import cache

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


@cache
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


@dataclass(frozen=True, slots=True)
class SlotCode:
    """One integer per tile kind, packed so a slot's whole summary falls out of
    two reductions over it: a bitwise OR and a sum.

    Fields from the low bit up: the colours present, the numbers present, and the
    count of real tiles. Each is padded with enough spare bits that summing
    ``max_set_len`` entries can never carry out of its own field. That is what makes
    ``sum == or`` exactly "no colour (or number) appears twice", and lets the count
    be read straight off the sum -- so neither costs a pass of its own over the
    positions. The joker's entry is empty but for the position it occupies: its
    colour and number stay undetermined until the set around it resolves them.

    Indexed by ``kind + 1`` so ``EMPTY`` lands on a zero entry and needs no mask.

    The three backends share this packing, which is why it is kept under 31 bits:
    JAX is 32-bit unless x64 is enabled globally.
    """

    code: np.ndarray
    """``(K + 1,)`` packed constant per kind, indexed by ``kind + 1``."""
    number_shift: int
    real_shift: int
    color_bits: int
    """The colour bits alone, without the field's padding."""
    number_bits: int
    color_field: int
    """The colour field including its padding, which is what reading the sum needs."""
    number_field: int


@cache
def slot_code(cfg: RummiConfig) -> SlotCode:
    """Build (and memoise) the packed per-kind constants for ``cfg``."""
    t = tables(cfg)
    pad = cfg.max_set_len.bit_length()  # 2**pad > max_set_len, so a field cannot carry out
    number_shift = cfg.n_colors + pad
    real_shift = number_shift + cfg.n_numbers + pad
    width = real_shift + pad
    if width > 31:
        raise ValueError(
            f"the slot packing needs {width} bits for this config, and it has to fit in "
            "the 32-bit integers the JAX backend runs on"
        )
    if cfg.n_numbers > 24:
        # The kernels read a bit index off a float32 exponent, exact only below 2**24.
        raise ValueError(f"n_numbers={cfg.n_numbers} exceeds the 24 the bit tricks assume")

    numbered = np.arange(cfg.n_numbered_kinds)
    code = np.zeros(cfg.n_kinds + 1, dtype=np.int32)
    code[numbered + 1] = (
        (1 << t.color[numbered].astype(np.int32))
        | (1 << (number_shift + t.number[numbered].astype(np.int32) - 1))
        | (1 << real_shift)
    )
    return SlotCode(
        code=code,
        number_shift=number_shift,
        real_shift=real_shift,
        color_bits=(1 << cfg.n_colors) - 1,
        number_bits=(1 << cfg.n_numbers) - 1,
        color_field=(1 << number_shift) - 1,
        number_field=(1 << (cfg.n_numbers + pad)) - 1,
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


def kinds_to_counts(cfg: RummiConfig, kinds: Iterable[int]) -> np.ndarray:
    """Collapse an iterable of kind ids into a ``(K,)`` count vector."""
    counts = np.zeros(cfg.n_kinds, dtype=np.int16)
    for kind in kinds:
        if kind != EMPTY:
            counts[kind] += 1
    return counts
