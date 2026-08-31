"""JAX port of the slot-validity kernel. See SPEC.md section 3.

Written against the spec, not by calling the NumPy code, so the benchmark
compares implementations rather than one implementation's plumbing.

The whole module is functional and shape-static, so it traces cleanly under
``jit``. ``cfg`` is a static argument everywhere: it is a frozen dataclass and
therefore hashable, so JAX can specialise on it and every derived shape becomes a
compile-time constant.
"""

from __future__ import annotations

from functools import cache
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from rummi.rules.config import RummiConfig
from rummi.rules.encoding import SlotCode, slot_code, tables

NO_COLOR = -2
NO_NUMBER = -2


class Lookup(NamedTuple):
    """Constant tables held as **NumPy** arrays, deliberately.

    They are cached, and a cache first populated from inside a ``jit`` trace would
    store tracers and leak them into every later call. NumPy arrays cannot be
    tracers; JAX folds them in as constants at each use site, which for tables
    this small costs nothing.
    """

    color: np.ndarray
    number: np.ndarray
    value: np.ndarray
    is_joker: np.ndarray
    copies: np.ndarray
    offload: np.ndarray
    """Face value with the joker at ``joker_penalty``: what a tile sheds when played."""
    packing: SlotCode
    """The :class:`~rummi.rules.encoding.SlotCode` table and its field layout."""


@cache
def lookup(cfg: RummiConfig) -> Lookup:
    t = tables(cfg)
    offload = t.value.astype(np.int32).copy()
    offload[cfg.joker_kind] = cfg.joker_penalty
    return Lookup(
        color=t.color.astype(np.int32),
        number=t.number.astype(np.int32),
        value=t.value.astype(np.int32),
        is_joker=t.is_joker.copy(),
        copies=t.copies.astype(np.int32),
        offload=offload,
        packing=slot_code(cfg),
    )


class SlotStats(NamedTuple):
    n: jax.Array
    n_jokers: jax.Array
    n_real: jax.Array
    color: jax.Array
    number: jax.Array
    lo: jax.Array
    hi: jax.Array
    color_mask: jax.Array
    number_mask: jax.Array
    same_color: jax.Array
    same_number: jax.Array
    distinct_colors: jax.Array
    distinct_numbers: jax.Array


class SlotEval(NamedTuple):
    is_empty: jax.Array
    run_valid: jax.Array
    group_valid: jax.Array
    is_valid: jax.Array
    run_open: jax.Array
    group_open: jax.Array
    is_extendable: jax.Array
    value: jax.Array


def _or_reduce(x: jax.Array) -> jax.Array:
    """Bitwise OR along the last dim.

    Unrolled: the axis is ``max_set_len``, static and small, and JAX has no
    ufunc-style reduce for bitwise ops.
    """
    out = x[..., 0]
    for i in range(1, x.shape[-1]):
        out = out | x[..., i]
    return out


def _bit_length(mask: jax.Array) -> jax.Array:
    """Index of the highest set bit, plus one; ``0`` for an empty mask.

    A float's exponent *is* that value, so ``frexp`` reads it off directly -- and
    unlike a lookup table it does not grow with the number range.
    """
    return jnp.frexp(mask.astype(jnp.float32))[1]


def slot_stats(cfg: RummiConfig, slots: jax.Array) -> SlotStats:
    """Reduce ``(..., L)`` kind ids to per-slot summary scalars.

    One gather and two reductions over the positions; everything after them is
    per-slot bit arithmetic. See :class:`~rummi.rules.encoding.SlotCode` for the
    packing that makes an OR and a sum enough.
    """
    code = lookup(cfg).packing
    slots = slots.astype(jnp.int32)

    packed = jnp.asarray(code.code)[slots + 1]
    present = _or_reduce(packed)
    total = packed.sum(-1)

    color_mask = present & code.color_bits
    number_mask = (present >> code.number_shift) & code.number_bits

    n = (slots >= 0).sum(-1).astype(jnp.int32)
    n_real = total >> code.real_shift
    n_jokers = n - n_real
    no_real = n_real == 0

    # A field's sum matches its OR exactly when no bit in it was set twice.
    distinct_colors = (total & code.color_field) == color_mask
    distinct_numbers = ((total >> code.number_shift) & code.number_field) == number_mask

    # `x & (x - 1)` clears the lowest set bit, so this is "one bit at most".
    same_color = (color_mask & (color_mask - 1)) == 0
    same_number = (number_mask & (number_mask - 1)) == 0

    # With no real tiles the masks are empty, which lands these on the loosest
    # values for the run-window test.
    hi = jnp.maximum(_bit_length(number_mask), 1)
    lo = jnp.where(no_real, cfg.n_numbers, _bit_length(number_mask & -number_mask))
    c_max = _bit_length(color_mask) - 1

    return SlotStats(
        n=n,
        n_jokers=n_jokers,
        n_real=n_real,
        color=jnp.where(same_color & ~no_real, c_max, NO_COLOR),
        number=jnp.where(same_number & ~no_real, hi, NO_NUMBER),
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
    n = s.n
    is_empty = n == 0
    long_enough = n >= cfg.min_set

    # A run of length n needs a window inside [1, n_numbers] covering every real
    # number; jokers fill whatever positions are left over.
    win_lo = jnp.maximum(1, s.hi - n + 1)
    win_hi = jnp.minimum(s.lo, cfg.n_numbers - n + 1)
    window_fits = win_lo <= win_hi

    run_shape = s.same_color & s.distinct_numbers
    run_valid = run_shape & long_enough & (n <= cfg.n_numbers) & window_fits

    group_shape = s.same_number & s.distinct_colors
    group_valid = group_shape & long_enough & (n <= cfg.n_colors)

    run_open = run_shape & (n <= cfg.n_numbers)
    group_open = group_shape & (n <= cfg.n_colors) & bool(cfg.group_possible)

    run_total = n * win_hi + n * (n - 1) // 2
    group_number = jnp.where(s.n_real > 0, s.number, cfg.n_numbers)
    value = jnp.maximum(
        jnp.where(run_valid, run_total, 0), jnp.where(group_valid, n * group_number, 0)
    )
    return SlotEval(
        is_empty=is_empty,
        run_valid=run_valid,
        group_valid=group_valid,
        is_valid=run_valid | group_valid,
        run_open=run_open,
        group_open=group_open,
        is_extendable=run_open | group_open,
        value=value,
    )


def evaluate_slots(cfg: RummiConfig, slots: jax.Array) -> SlotEval:
    return evaluate(cfg, slot_stats(cfg, slots))


class AssignCode(NamedTuple):
    """ASSIGN legality, factored into a colour half and a number half.

    A numbered kind *is* a ``(colour, number)`` pair -- that is how kind ids are
    laid out -- and every term of the predicate constrains only one of the two. So
    the ``(..., K)`` answer is a product of an ``(..., C)`` and an ``(..., N)``
    table, and the caller materialises it in whichever order it wants: the action
    mask is kind-major, :func:`assign_open` is kind-minor. Bit 0 of each code
    carries the run branch and bit 1 the group branch, so one AND resolves both.
    """

    color: jax.Array
    """``(..., C)`` uint8."""
    number: jax.Array
    """``(..., N)`` uint8."""
    joker: jax.Array
    """``(...)`` bool: the joker is one kind, so it is not part of the grid."""


def assign_codes(cfg: RummiConfig, s: SlotStats) -> AssignCode:
    """The two halves of the ASSIGN legality test. See :class:`AssignCode`."""
    colors = jnp.arange(cfg.n_colors, dtype=jnp.int32)
    numbers = jnp.arange(1, cfg.n_numbers + 1, dtype=jnp.int32)

    run_slot = (s.n + 1 <= cfg.n_numbers) & s.distinct_numbers
    group_slot = (s.n + 1 <= cfg.n_colors) & s.distinct_colors & bool(cfg.group_possible)
    no_real = (s.n_real == 0)[..., None]

    # Numbered tile: it must not duplicate a colour (group) or a number (run)
    # already present, and must agree with whatever the slot has settled on.
    run_color = (no_real | (s.color[..., None] == colors)) & run_slot[..., None]
    group_color = ((s.color_mask[..., None] & (1 << colors)) == 0) & group_slot[..., None]
    run_number = (s.number_mask[..., None] & (1 << (numbers - 1))) == 0
    group_number = no_real | (s.number[..., None] == numbers)

    return AssignCode(
        color=run_color.astype(jnp.uint8) | (group_color.astype(jnp.uint8) << 1),
        number=run_number.astype(jnp.uint8) | (group_number.astype(jnp.uint8) << 1),
        # A joker constrains nothing beyond the shape the slot already has.
        joker=(run_slot & s.same_color) | (group_slot & s.same_number),
    )


def assign_open(cfg: RummiConfig, s: SlotStats) -> jax.Array:
    """``(..., K)`` would adding one tile of each kind leave the slot extendable?"""
    codes = assign_codes(cfg, s)
    grid = (codes.color[..., :, None] & codes.number[..., None, :]) != 0
    return jnp.concatenate(
        [grid.reshape(*s.n.shape, cfg.n_numbered_kinds), codes.joker[..., None]], axis=-1
    )


class SlotSummary(NamedTuple):
    """Both stages for one table, so the two readers of it can share the work."""

    stats: SlotStats
    ev: SlotEval


def summarize(cfg: RummiConfig, slots: jax.Array) -> SlotSummary:
    stats = slot_stats(cfg, slots)
    return SlotSummary(stats=stats, ev=evaluate(cfg, stats))
