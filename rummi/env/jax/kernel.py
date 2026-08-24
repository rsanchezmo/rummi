"""JAX port of the slot-validity kernel. See SPEC.md section 3.

Written against the spec, not by calling the NumPy code, so the benchmark
compares implementations rather than one implementation's plumbing.

The whole module is functional and shape-static, so it traces cleanly under
``jit``. ``cfg`` is a static argument everywhere: it is a frozen dataclass and
therefore hashable, so JAX can specialise on it and every derived shape becomes a
compile-time constant.
"""

from __future__ import annotations

from functools import lru_cache
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from rummi.rules.config import RummiConfig
from rummi.rules.encoding import tables

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


@lru_cache(maxsize=None)
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
    distinct_real: jax.Array


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


def slot_stats(cfg: RummiConfig, slots: jax.Array) -> SlotStats:
    """Reduce ``(..., L)`` kind ids to per-slot summary scalars."""
    t = lookup(cfg)
    slots = slots.astype(jnp.int32)

    occupied = slots >= 0
    is_joker = occupied & (slots == cfg.joker_kind)
    real = occupied & ~is_joker

    safe = jnp.clip(slots, 0, None)
    color = jnp.asarray(t.color)[safe]
    number = jnp.asarray(t.number)[safe]

    n = occupied.sum(-1)
    n_jokers = is_joker.sum(-1)
    n_real = real.sum(-1)
    no_real = n_real == 0

    # Fill values are each reduction's identity, so a slot with no real tiles
    # comes out with the loosest possible bounds.
    c_min = jnp.where(real, color, cfg.n_colors).min(-1)
    c_max = jnp.where(real, color, -1).max(-1)
    lo = jnp.where(real, number, cfg.n_numbers).min(-1)
    hi = jnp.where(real, number, 1).max(-1)

    same_color = no_real | (c_min == c_max)
    same_number = no_real | (lo == hi)

    one = jnp.int32(1)
    color_mask = _or_reduce(jnp.where(real, one << color, 0))
    number_mask = _or_reduce(jnp.where(real, one << (number - 1), 0))

    # Duplicate real kinds: sort reals to the front, then compare neighbours.
    # Jokers may legitimately repeat, so they are pushed past the reals.
    key = jnp.sort(jnp.where(real, slots, cfg.n_kinds), axis=-1)
    adjacent_dup = (key[..., 1:] == key[..., :-1]) & (key[..., :-1] < cfg.n_kinds)
    distinct_real = ~adjacent_dup.any(-1)

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
        distinct_real=distinct_real,
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

    run_shape = s.same_color & s.distinct_real
    run_valid = run_shape & long_enough & (n <= cfg.n_numbers) & window_fits

    group_shape = s.same_number & s.distinct_real
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


def assign_open(cfg: RummiConfig, s: SlotStats) -> jax.Array:
    """``(..., K)`` would adding one tile of each kind leave the slot extendable?"""
    t = lookup(cfg)
    n_next = s.n[..., None] + 1
    one = jnp.int32(1)

    fits_run = n_next <= cfg.n_numbers
    fits_group = n_next <= cfg.n_colors
    distinct = s.distinct_real[..., None]
    no_real = (s.n_real == 0)[..., None]

    kind_color = jnp.asarray(t.color)
    kind_number = jnp.asarray(t.number)
    color_free = (s.color_mask[..., None] & (one << jnp.clip(kind_color, 0, None))) == 0
    number_free = (s.number_mask[..., None] & (one << (jnp.clip(kind_number, 1, None) - 1))) == 0
    color_agrees = no_real | (s.color[..., None] == kind_color)
    number_agrees = no_real | (s.number[..., None] == kind_number)

    numbered_run = fits_run & distinct & color_agrees & number_free
    numbered_group = fits_group & distinct & number_agrees & color_free

    # A joker constrains nothing beyond the shape the slot already has.
    joker_run = fits_run & distinct & s.same_color[..., None]
    joker_group = fits_group & distinct & s.same_number[..., None]

    is_joker = jnp.asarray(t.is_joker)
    run_ok = jnp.where(is_joker, joker_run, numbered_run)
    group_ok = jnp.where(is_joker, joker_group, numbered_group)
    return run_ok | (group_ok & bool(cfg.group_possible))
