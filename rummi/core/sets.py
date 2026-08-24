"""Batched validity kernel for table slots.

A *slot* is one candidate set, stored as ``(..., L)`` kind ids with ``EMPTY``
padding. Everything the engine needs to know about a slot is derived here in two
stages:

``slot_stats``
    Reduce each slot to a handful of summary scalars.
``evaluate`` / ``assign_open``
    Answer every question in closed form from those scalars, so the
    "what if I added kind ``k``?" mask over ``(S, K)`` costs no rescanning.

Both stages are pure array arithmetic over a leading batch of arbitrary rank,
with no data-dependent control flow, so a torch/JAX port is mechanical.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rummi.core.config import RummiConfig
from rummi.core.encoding import EMPTY, tables

_NO_COLOR = np.int16(-2)
"""Sentinel that compares unequal to every real colour, so a mixed-colour slot
never satisfies ``color == c``."""

_NO_NUMBER = np.int16(-2)


@dataclass(frozen=True, slots=True)
class SlotStats:
    """Per-slot summary. Every field has the slots' batch shape ``(...)``."""

    n: np.ndarray
    """Occupied positions."""
    n_jokers: np.ndarray
    n_real: np.ndarray
    color: np.ndarray
    """Shared colour of the real tiles, or ``_NO_COLOR`` if there is none."""
    number: np.ndarray
    """Shared number of the real tiles, or ``_NO_NUMBER`` if there is none."""
    lo: np.ndarray
    """Lowest real number; ``n_numbers`` when there are no real tiles, which is
    the loosest value for the run-window test."""
    hi: np.ndarray
    """Highest real number; ``1`` when there are no real tiles."""
    color_mask: np.ndarray
    number_mask: np.ndarray
    same_color: np.ndarray
    same_number: np.ndarray
    distinct_real: np.ndarray
    """No real kind appears twice. Combined with ``same_color`` this is exactly
    "distinct numbers"; with ``same_number`` it is exactly "distinct colours"."""


@dataclass(frozen=True, slots=True)
class SlotEval:
    """Per-slot verdicts. Every field has the slots' batch shape ``(...)``."""

    is_empty: np.ndarray
    run_valid: np.ndarray
    group_valid: np.ndarray
    is_valid: np.ndarray
    run_open: np.ndarray
    group_open: np.ndarray
    is_extendable: np.ndarray
    """Could still become a valid set by adding tiles. Empty slots qualify."""
    value: np.ndarray
    """Resolved face-value total of a valid set, jokers included; ``0`` if the
    slot is not valid. Where a joker's identity is ambiguous the best-case
    reading is taken, matching the rule that the player declares it."""


def slot_stats(cfg: RummiConfig, slots: np.ndarray) -> SlotStats:
    """Reduce ``(..., L)`` kind ids to per-slot summary scalars."""
    t = tables(cfg)
    slots = np.asarray(slots)

    occupied = slots >= 0
    is_joker = occupied & (slots == cfg.joker_kind)
    real = occupied & ~is_joker

    safe = np.where(real, slots, 0)
    color = t.color[safe].astype(np.int16)
    number = t.number[safe].astype(np.int16)

    n = occupied.sum(-1).astype(np.int16)
    n_jokers = is_joker.sum(-1).astype(np.int16)
    n_real = real.sum(-1).astype(np.int16)
    no_real = n_real == 0

    c_min = np.where(real, color, np.int16(cfg.n_colors)).min(-1)
    c_max = np.where(real, color, np.int16(-1)).max(-1)
    lo = np.where(real, number, np.int16(cfg.n_numbers)).min(-1)
    hi = np.where(real, number, np.int16(1)).max(-1)

    same_color = no_real | (c_min == c_max)
    same_number = no_real | (lo == hi)

    shared_color = np.where(same_color & ~no_real, c_max, _NO_COLOR).astype(np.int16)
    shared_number = np.where(same_number & ~no_real, hi, _NO_NUMBER).astype(np.int16)

    one = np.int64(1)
    color_mask = np.bitwise_or.reduce(
        np.where(real, one << color.astype(np.int64), np.int64(0)), axis=-1
    )
    number_mask = np.bitwise_or.reduce(
        np.where(real, one << (number.astype(np.int64) - 1), np.int64(0)), axis=-1
    )

    # Duplicate real kinds are detected by sorting reals to the front; jokers and
    # empties are pushed past them because they may legitimately repeat.
    key = np.where(real, slots, np.int16(cfg.n_kinds)).astype(np.int16)
    key = np.sort(key, axis=-1)
    adjacent_dup = (key[..., 1:] == key[..., :-1]) & (key[..., :-1] < cfg.n_kinds)
    distinct_real = ~adjacent_dup.any(-1)

    return SlotStats(
        n=n,
        n_jokers=n_jokers,
        n_real=n_real,
        color=shared_color,
        number=shared_number,
        lo=lo.astype(np.int16),
        hi=hi.astype(np.int16),
        color_mask=color_mask,
        number_mask=number_mask,
        same_color=same_color,
        same_number=same_number,
        distinct_real=distinct_real,
    )


def evaluate(cfg: RummiConfig, s: SlotStats) -> SlotEval:
    """Decide validity, extendability and value from summary scalars."""
    n = s.n.astype(np.int32)
    min_set = np.int32(cfg.min_set)
    n_numbers = np.int32(cfg.n_numbers)
    n_colors = np.int32(cfg.n_colors)

    is_empty = s.n == 0
    long_enough = n >= min_set

    # A run of length n needs a window [start, start + n - 1] inside [1, n_numbers]
    # that covers every real number; jokers fill whatever positions are left.
    win_lo = np.maximum(np.int32(1), s.hi.astype(np.int32) - n + 1)
    win_hi = np.minimum(s.lo.astype(np.int32), n_numbers - n + 1)
    window_fits = win_lo <= win_hi

    run_shape = s.same_color & s.distinct_real
    run_valid = run_shape & long_enough & (n <= n_numbers) & window_fits

    group_shape = s.same_number & s.distinct_real
    group_valid = group_shape & long_enough & (n <= n_colors)

    is_valid = run_valid | group_valid

    # Openness needs no window arithmetic: a partial run can always be completed
    # inside [1, n_numbers] as long as its numbers are distinct and it still fits.
    run_open = run_shape & (n <= n_numbers)
    group_open = group_shape & (n <= n_colors) & bool(cfg.group_possible)

    # Best-case window start maximises the run's total.
    run_total = n * win_hi + n * (n - 1) // 2
    group_number = np.where(s.n_real > 0, s.number.astype(np.int32), n_numbers)
    group_total = n * group_number

    value = np.maximum(
        np.where(run_valid, run_total, np.int32(0)),
        np.where(group_valid, group_total, np.int32(0)),
    )

    return SlotEval(
        is_empty=is_empty,
        run_valid=run_valid,
        group_valid=group_valid,
        is_valid=is_valid,
        run_open=run_open,
        group_open=group_open,
        is_extendable=run_open | group_open,
        value=value.astype(np.int32),
    )


def evaluate_slots(cfg: RummiConfig, slots: np.ndarray) -> SlotEval:
    return evaluate(cfg, slot_stats(cfg, slots))


def assign_open(cfg: RummiConfig, s: SlotStats) -> np.ndarray:
    """``(..., K)`` bool: would adding one tile of each kind leave the slot extendable?

    This is the ASSIGN legality test. Availability of the tile is the caller's
    concern; this answers only whether the resulting set could still be completed.
    """
    t = tables(cfg)
    n_next = s.n.astype(np.int32)[..., None] + 1
    one = np.int64(1)

    kind_color = t.color.astype(np.int16)
    kind_number = t.number.astype(np.int16)
    kind_is_joker = t.is_joker

    fits_run = n_next <= np.int32(cfg.n_numbers)
    fits_group = n_next <= np.int32(cfg.n_colors)
    distinct = s.distinct_real[..., None]

    # Numbered tile: it must not duplicate a colour (group) or a number (run)
    # already present, and must agree with whatever the slot has settled on.
    color_free = (s.color_mask[..., None] & (one << kind_color.astype(np.int64))) == 0
    number_free = (
        s.number_mask[..., None] & (one << (kind_number.astype(np.int64) - 1))
    ) == 0
    color_agrees = (s.n_real == 0)[..., None] | (s.color[..., None] == kind_color)
    number_agrees = (s.n_real == 0)[..., None] | (s.number[..., None] == kind_number)

    numbered_run = fits_run & distinct & color_agrees & number_free
    numbered_group = fits_group & distinct & number_agrees & color_free

    # A joker constrains nothing beyond the shape the slot already has.
    joker_run = fits_run & distinct & s.same_color[..., None]
    joker_group = fits_group & distinct & s.same_number[..., None]

    run_ok = np.where(kind_is_joker, joker_run, numbered_run)
    group_ok = np.where(kind_is_joker, joker_group, numbered_group)
    return run_ok | (group_ok & bool(cfg.group_possible))


def pad_slot(cfg: RummiConfig, kinds) -> np.ndarray:
    """Build one ``(L,)`` slot row from a list of kind ids, padded with ``EMPTY``."""
    kinds = list(kinds)
    if len(kinds) > cfg.max_set_len:
        raise ValueError(f"{len(kinds)} tiles exceeds max_set_len={cfg.max_set_len}")
    row = np.full(cfg.max_set_len, EMPTY, dtype=np.int16)
    row[: len(kinds)] = kinds
    return row
