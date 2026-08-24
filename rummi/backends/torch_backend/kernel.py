"""Torch port of the slot-validity kernel. See SPEC.md section 3.

Written against the spec rather than by calling the NumPy code, so the benchmark
compares two implementations instead of one implementation's plumbing. The shapes
and semantics are identical; the arithmetic is expressed in torch primitives.

Everything here is branch-free and shape-static, which is what lets the whole
kernel go through ``torch.compile`` and run without a host synchronisation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from rummi.core.config import RummiConfig
from rummi.core.encoding import tables

NO_COLOR = -2
NO_NUMBER = -2


@dataclass(frozen=True, slots=True)
class Lookup:
    """Per-config constant tables, resident on the target device."""

    cfg: RummiConfig
    color: torch.Tensor
    number: torch.Tensor
    value: torch.Tensor
    is_joker: torch.Tensor
    copies: torch.Tensor
    offload: torch.Tensor
    """Face value with the joker at ``joker_penalty``: what a tile sheds when played."""


_CACHE: dict[tuple[int, str, str], Lookup] = {}


def lookup(cfg: RummiConfig, device: torch.device) -> Lookup:
    key = (id(cfg), str(device), "v1")
    hit = _CACHE.get(key)
    if hit is not None:
        return hit
    t = tables(cfg)
    to = lambda a, dtype=torch.int64: torch.as_tensor(a.copy(), dtype=dtype, device=device)
    offload = t.value.astype("int64").copy()
    offload[cfg.joker_kind] = cfg.joker_penalty
    built = Lookup(
        cfg=cfg,
        color=to(t.color),
        number=to(t.number),
        value=to(t.value),
        is_joker=to(t.is_joker, torch.bool),
        copies=to(t.copies),
        offload=to(offload),
    )
    _CACHE[key] = built
    return built


@dataclass(frozen=True, slots=True)
class SlotStats:
    n: torch.Tensor
    n_jokers: torch.Tensor
    n_real: torch.Tensor
    color: torch.Tensor
    number: torch.Tensor
    lo: torch.Tensor
    hi: torch.Tensor
    color_mask: torch.Tensor
    number_mask: torch.Tensor
    same_color: torch.Tensor
    same_number: torch.Tensor
    distinct_real: torch.Tensor


@dataclass(frozen=True, slots=True)
class SlotEval:
    is_empty: torch.Tensor
    run_valid: torch.Tensor
    group_valid: torch.Tensor
    is_valid: torch.Tensor
    run_open: torch.Tensor
    group_open: torch.Tensor
    is_extendable: torch.Tensor
    value: torch.Tensor


def slot_stats(cfg: RummiConfig, slots: torch.Tensor) -> SlotStats:
    """Reduce ``(..., L)`` kind ids to per-slot summary scalars."""
    t = lookup(cfg, slots.device)
    slots = slots.to(torch.int64)

    occupied = slots >= 0
    is_joker = occupied & (slots == cfg.joker_kind)
    real = occupied & ~is_joker

    safe = slots.clamp(min=0)
    color = t.color[safe]
    number = t.number[safe]

    n = occupied.sum(-1)
    n_jokers = is_joker.sum(-1)
    n_real = real.sum(-1)
    no_real = n_real == 0

    # Fill values are the identity for each reduction, so a slot with no real
    # tiles yields the loosest possible bounds.
    c_min = torch.where(real, color, cfg.n_colors).amin(-1)
    c_max = torch.where(real, color, -1).amax(-1)
    lo = torch.where(real, number, cfg.n_numbers).amin(-1)
    hi = torch.where(real, number, torch.ones_like(number)).amax(-1)

    same_color = no_real | (c_min == c_max)
    same_number = no_real | (lo == hi)

    one = torch.ones((), dtype=torch.int64, device=slots.device)
    color_mask = _or_reduce(torch.where(real, one << color, torch.zeros_like(color)))
    number_mask = _or_reduce(torch.where(real, one << (number - 1), torch.zeros_like(number)))

    # Duplicate real kinds: sort reals to the front, then compare neighbours.
    # Jokers may legitimately repeat, so they are pushed past the reals.
    key = torch.where(real, slots, torch.full_like(slots, cfg.n_kinds)).sort(-1).values
    adjacent_dup = (key[..., 1:] == key[..., :-1]) & (key[..., :-1] < cfg.n_kinds)
    distinct_real = ~adjacent_dup.any(-1)

    return SlotStats(
        n=n,
        n_jokers=n_jokers,
        n_real=n_real,
        color=torch.where(same_color & ~no_real, c_max, torch.full_like(c_max, NO_COLOR)),
        number=torch.where(same_number & ~no_real, hi, torch.full_like(hi, NO_NUMBER)),
        lo=lo,
        hi=hi,
        color_mask=color_mask,
        number_mask=number_mask,
        same_color=same_color,
        same_number=same_number,
        distinct_real=distinct_real,
    )


def _or_reduce(x: torch.Tensor) -> torch.Tensor:
    """Bitwise OR along the last dim. Torch has no reduce for this, and the last
    dim is ``max_set_len`` -- small and fixed -- so unrolling it is fine."""
    out = x[..., 0]
    for i in range(1, x.shape[-1]):
        out = out | x[..., i]
    return out


def evaluate(cfg: RummiConfig, s: SlotStats) -> SlotEval:
    n = s.n
    is_empty = n == 0
    long_enough = n >= cfg.min_set

    # A run of length n needs a window inside [1, n_numbers] covering every real
    # number; jokers fill whatever positions are left over.
    win_lo = torch.clamp(s.hi - n + 1, min=1)
    win_hi = torch.clamp(s.lo, max=cfg.n_numbers - n + 1)
    window_fits = win_lo <= win_hi

    run_shape = s.same_color & s.distinct_real
    run_valid = run_shape & long_enough & (n <= cfg.n_numbers) & window_fits

    group_shape = s.same_number & s.distinct_real
    group_valid = group_shape & long_enough & (n <= cfg.n_colors)

    run_open = run_shape & (n <= cfg.n_numbers)
    group_open = group_shape & (n <= cfg.n_colors) & bool(cfg.group_possible)

    run_total = n * win_hi + n * (n - 1) // 2
    group_number = torch.where(s.n_real > 0, s.number, torch.full_like(s.number, cfg.n_numbers))
    group_total = n * group_number

    zero = torch.zeros_like(run_total)
    value = torch.maximum(
        torch.where(run_valid, run_total, zero), torch.where(group_valid, group_total, zero)
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


def evaluate_slots(cfg: RummiConfig, slots: torch.Tensor) -> SlotEval:
    return evaluate(cfg, slot_stats(cfg, slots))


def assign_open(cfg: RummiConfig, s: SlotStats) -> torch.Tensor:
    """``(..., K)`` would adding one tile of each kind leave the slot extendable?"""
    t = lookup(cfg, s.n.device)
    n_next = s.n.unsqueeze(-1) + 1
    one = torch.ones((), dtype=torch.int64, device=s.n.device)

    fits_run = n_next <= cfg.n_numbers
    fits_group = n_next <= cfg.n_colors
    distinct = s.distinct_real.unsqueeze(-1)
    no_real = (s.n_real == 0).unsqueeze(-1)

    color_free = (s.color_mask.unsqueeze(-1) & (one << t.color.clamp(min=0))) == 0
    number_free = (s.number_mask.unsqueeze(-1) & (one << (t.number.clamp(min=1) - 1))) == 0
    color_agrees = no_real | (s.color.unsqueeze(-1) == t.color)
    number_agrees = no_real | (s.number.unsqueeze(-1) == t.number)

    numbered_run = fits_run & distinct & color_agrees & number_free
    numbered_group = fits_group & distinct & number_agrees & color_free

    # A joker constrains nothing beyond the shape the slot already has.
    joker_run = fits_run & distinct & s.same_color.unsqueeze(-1)
    joker_group = fits_group & distinct & s.same_number.unsqueeze(-1)

    run_ok = torch.where(t.is_joker, joker_run, numbered_run)
    group_ok = torch.where(t.is_joker, joker_group, numbered_group)
    return run_ok | (group_ok & bool(cfg.group_possible))
