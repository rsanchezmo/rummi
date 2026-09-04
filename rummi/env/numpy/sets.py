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

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from rummi.rules.config import RummiConfig
from rummi.rules.encoding import EMPTY, slot_code, tables

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
    distinct_colors: np.ndarray
    """No colour appears twice among the real tiles."""
    distinct_numbers: np.ndarray
    """No number appears twice among the real tiles.

    A run is shaped by ``same_color & distinct_numbers`` and a group by
    ``same_number & distinct_colors``; under either conjunction the flag says
    exactly "no kind appears twice", which is the rule being applied. Neither is
    read outside such a pairing, so the general duplicate test -- which costs a
    sort -- is never needed.
    """


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


def _bit_length(mask: np.ndarray) -> np.ndarray:
    """Index of the highest set bit, plus one; ``0`` for an empty mask.

    A float's exponent *is* that value, so ``frexp`` reads it off directly -- and
    unlike a lookup table it does not grow with the number range.
    """
    return np.frexp(mask.astype(np.float32))[1]


def slot_stats(cfg: RummiConfig, slots: np.ndarray) -> SlotStats:
    """Reduce ``(..., L)`` kind ids to per-slot summary scalars.

    One gather and two reductions over the positions; everything after them is
    per-slot bit arithmetic. See :class:`~rummi.rules.encoding.SlotCode` for the
    packing that makes an OR and a sum enough.
    """
    code = slot_code(cfg)
    slots = np.asarray(slots)

    packed = code.code[slots + 1]
    present = np.bitwise_or.reduce(packed, axis=-1)
    total = packed.sum(-1)

    color_mask = (present & code.color_bits).astype(np.int32)
    number_mask = ((present >> code.number_shift) & code.number_bits).astype(np.int32)

    n = (slots >= 0).sum(-1).astype(np.int16)
    n_real = (total >> code.real_shift).astype(np.int16)
    n_jokers = (n - n_real).astype(np.int16)
    no_real = n_real == 0

    # A field's sum matches its OR exactly when no bit in it was set twice.
    distinct_colors = (total & code.color_field) == color_mask
    distinct_numbers = ((total >> code.number_shift) & code.number_field) == number_mask

    # `x & (x - 1)` clears the lowest set bit, so this is "one bit at most".
    same_color = (color_mask & (color_mask - 1)) == 0
    same_number = (number_mask & (number_mask - 1)) == 0

    # With no real tiles the masks are empty, which lands these on the loosest
    # values for the run-window test.
    hi = np.maximum(_bit_length(number_mask), 1).astype(np.int16)
    lo = np.where(
        no_real, np.int16(cfg.n_numbers), _bit_length(number_mask & -number_mask)
    ).astype(np.int16)
    c_max = (_bit_length(color_mask) - 1).astype(np.int16)

    return SlotStats(
        n=n,
        n_jokers=n_jokers,
        n_real=n_real,
        color=np.where(same_color & ~no_real, c_max, _NO_COLOR).astype(np.int16),
        number=np.where(same_number & ~no_real, hi, _NO_NUMBER).astype(np.int16),
        lo=lo,
        hi=hi,
        color_mask=color_mask,
        number_mask=number_mask,
        same_color=same_color,
        same_number=same_number,
        distinct_colors=distinct_colors,
        distinct_numbers=distinct_numbers,
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

    run_shape = s.same_color & s.distinct_numbers
    run_valid = run_shape & long_enough & (n <= n_numbers) & window_fits

    group_shape = s.same_number & s.distinct_colors
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


@dataclass(frozen=True, slots=True)
class SlotSummary:
    """Both stages for one table, so the two readers of it can share the work.

    The action mask and the observation are computed from the same state and both
    need every field here; computing it twice cost about a fifth of a step.
    """

    stats: SlotStats
    ev: SlotEval


def summarize(cfg: RummiConfig, slots: np.ndarray) -> SlotSummary:
    stats = slot_stats(cfg, slots)
    return SlotSummary(stats=stats, ev=evaluate(cfg, stats))


@dataclass(frozen=True, slots=True)
class AssignCode:
    """ASSIGN legality, factored into a colour half and a number half.

    A numbered kind *is* a ``(colour, number)`` pair -- that is how kind ids are
    laid out -- and every term of the predicate constrains only one of the two. So
    the ``(..., K)`` answer is a product of an ``(..., C)`` and an ``(..., N)``
    table, and the caller materialises it in whichever memory order it wants: the
    action mask is kind-major, :func:`assign_open` is kind-minor. Bit 0 of each
    code carries the run branch and bit 1 the group branch, so a single bitwise
    AND resolves both.
    """

    color: np.ndarray
    """``(..., C)`` uint8."""
    number: np.ndarray
    """``(..., N)`` uint8."""
    joker: np.ndarray
    """``(...)`` bool: the joker is one kind, so it is not part of the grid."""


def assign_codes(cfg: RummiConfig, s: SlotStats) -> AssignCode:
    """The two halves of the ASSIGN legality test. See :class:`AssignCode`."""
    colors = np.arange(cfg.n_colors, dtype=np.int32)
    numbers = np.arange(1, cfg.n_numbers + 1, dtype=np.int32)
    n_next = s.n.astype(np.int32) + 1

    run_slot = (n_next <= cfg.n_numbers) & s.distinct_numbers
    group_slot = (n_next <= cfg.n_colors) & s.distinct_colors & bool(cfg.group_possible)
    no_real = (s.n_real == 0)[..., None]

    # Numbered tile: it must not duplicate a colour (group) or a number (run)
    # already present, and must agree with whatever the slot has settled on.
    run_color = (no_real | (s.color[..., None] == colors)) & run_slot[..., None]
    group_color = ((s.color_mask[..., None] & (1 << colors)) == 0) & group_slot[..., None]
    run_number = (s.number_mask[..., None] & (1 << (numbers - 1))) == 0
    group_number = no_real | (s.number[..., None] == numbers)

    return AssignCode(
        color=run_color.view(np.uint8) | (group_color.view(np.uint8) << 1),
        number=run_number.view(np.uint8) | (group_number.view(np.uint8) << 1),
        # A joker constrains nothing beyond the shape the slot already has.
        joker=(run_slot & s.same_color) | (group_slot & s.same_number),
    )


def assign_open(cfg: RummiConfig, s: SlotStats) -> np.ndarray:
    """``(..., K)`` bool: would adding one tile of each kind leave the slot extendable?

    This is the ASSIGN legality test. Availability of the tile is the caller's
    concern; this answers only whether the resulting set could still be completed.
    """
    codes = assign_codes(cfg, s)
    shape = s.n.shape
    out = np.empty((*shape, cfg.n_kinds), dtype=bool)
    grid = codes.color[..., :, None] & codes.number[..., None, :]
    out[..., : cfg.n_numbered_kinds] = (grid != 0).reshape(*shape, cfg.n_numbered_kinds)
    out[..., cfg.joker_kind] = codes.joker
    return out


def assign_open_at(
    cfg: RummiConfig, s: SlotStats, envs: np.ndarray, kinds: np.ndarray
) -> np.ndarray:
    """``(pairs, S)`` :func:`assign_open` for each ``(env, kind)`` pair, over every slot.

    The same predicate, evaluated only where the answer can be anything but false.
    A tile has to be in hand to be assigned and a workbench holds a handful of kinds
    -- 5.8 of 53 on the standard config -- so taking the (colour x number) product at
    the kinds held costs an order of magnitude less than taking it over the whole
    grid and masking afterwards. The two are held equal by
    `test_the_mask_matches_the_dense_predicate`, and by every backend comparison:
    torch and JAX build the grid, because a shape that depends on the data is what
    ``jit`` and ``compile`` cannot have.
    """
    t = tables(cfg)
    color = t.color[kinds].astype(np.int32)[:, None]
    number = t.number[kinds].astype(np.int32)[:, None]
    is_joker = t.is_joker[kinds][:, None]

    n_next = s.n[envs].astype(np.int32) + 1
    no_real = s.n_real[envs] == 0
    run_slot = (n_next <= cfg.n_numbers) & s.distinct_numbers[envs]
    group_slot = (
        (n_next <= cfg.n_colors) & s.distinct_colors[envs] & bool(cfg.group_possible)
    )

    # Clamped only to keep the shift legal: where the kind is the joker, the joker
    # branch of the `where` is the one taken.
    number_free = (s.number_mask[envs] & (1 << np.maximum(number - 1, 0))) == 0
    color_free = (s.color_mask[envs] & (1 << np.maximum(color, 0))) == 0
    color_agrees = no_real | (s.color[envs] == color)
    number_agrees = no_real | (s.number[envs] == number)

    # A joker constrains nothing beyond the shape the slot already has.
    run_ok = np.where(is_joker, s.same_color[envs], color_agrees & number_free)
    group_ok = np.where(is_joker, s.same_number[envs], number_agrees & color_free)
    return (run_slot & run_ok) | (group_slot & group_ok)


def pad_slot(cfg: RummiConfig, kinds: Iterable[int]) -> np.ndarray:
    """Build one ``(L,)`` slot row from a list of kind ids, padded with ``EMPTY``."""
    kinds = list(kinds)
    if len(kinds) > cfg.max_set_len:
        raise ValueError(f"{len(kinds)} tiles exceeds max_set_len={cfg.max_set_len}")
    row = np.full(cfg.max_set_len, EMPTY, dtype=np.int16)
    row[: len(kinds)] = kinds
    return row
