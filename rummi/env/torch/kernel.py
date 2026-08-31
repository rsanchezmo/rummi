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

from rummi.rules.config import RummiConfig
from rummi.rules.encoding import SlotCode, slot_code, tables

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
    slot_code: torch.Tensor
    """The :class:`~rummi.rules.encoding.SlotCode` table, indexed by ``kind + 1``."""
    packing: SlotCode
    """Its shifts and field masks. Held here rather than looked up in ``slot_stats``:
    ``slot_code`` is ``lru_cache``-wrapped and Dynamo refuses to trace through one."""


_CACHE: dict[tuple[int, str, str], Lookup] = {}


def lookup(cfg: RummiConfig, device: torch.device) -> Lookup:
    key = (id(cfg), str(device), "v1")
    hit = _CACHE.get(key)
    if hit is not None:
        return hit
    t = tables(cfg)
    def to(a, dtype=torch.int64):
        return torch.as_tensor(a.copy(), dtype=dtype, device=device)

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
        slot_code=to(slot_code(cfg).code),
        packing=slot_code(cfg),
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
    distinct_colors: torch.Tensor
    distinct_numbers: torch.Tensor


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


def _bit_length(mask: torch.Tensor, width: int) -> torch.Tensor:
    """Index of the highest set bit, plus one; ``0`` for an empty mask.

    Counted by thresholds rather than read off a float exponent: MPS has no
    ``frexp``, and ``width`` is static and small enough that this stays one pass.
    """
    powers = 1 << torch.arange(width, device=mask.device)
    return (mask.unsqueeze(-1) >= powers).sum(-1)


def slot_stats(cfg: RummiConfig, slots: torch.Tensor) -> SlotStats:
    """Reduce ``(..., L)`` kind ids to per-slot summary scalars.

    One gather and two reductions over the positions; everything after them is
    per-slot bit arithmetic. See :class:`~rummi.rules.encoding.SlotCode` for the
    packing that makes an OR and a sum enough.
    """
    t = lookup(cfg, slots.device)
    code = t.packing
    slots = slots.to(torch.int64)

    packed = t.slot_code[slots + 1]
    present = _or_reduce(packed)
    total = packed.sum(-1)

    color_mask = present & code.color_bits
    number_mask = (present >> code.number_shift) & code.number_bits

    n = (slots >= 0).sum(-1)
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
    hi = _bit_length(number_mask, cfg.n_numbers).clamp(min=1)
    lo = torch.where(
        no_real,
        torch.full_like(hi, cfg.n_numbers),
        _bit_length(number_mask & -number_mask, cfg.n_numbers),
    )
    c_max = _bit_length(color_mask, cfg.n_colors) - 1

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
        distinct_colors=distinct_colors,
        distinct_numbers=distinct_numbers,
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

    run_shape = s.same_color & s.distinct_numbers
    run_valid = run_shape & long_enough & (n <= cfg.n_numbers) & window_fits

    group_shape = s.same_number & s.distinct_colors
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


@dataclass(frozen=True, slots=True)
class AssignCode:
    """ASSIGN legality, factored into a colour half and a number half.

    A numbered kind *is* a ``(colour, number)`` pair -- that is how kind ids are
    laid out -- and every term of the predicate constrains only one of the two. So
    the ``(..., K)`` answer is a product of an ``(..., C)`` and an ``(..., N)``
    table, and the caller materialises it in whichever order it wants: the action
    mask is kind-major, :func:`assign_open` is kind-minor. Bit 0 of each code
    carries the run branch and bit 1 the group branch, so one AND resolves both.
    """

    color: torch.Tensor
    """``(..., C)`` uint8."""
    number: torch.Tensor
    """``(..., N)`` uint8."""
    joker: torch.Tensor
    """``(...)`` bool: the joker is one kind, so it is not part of the grid."""


def assign_codes(cfg: RummiConfig, s: SlotStats) -> AssignCode:
    """The two halves of the ASSIGN legality test. See :class:`AssignCode`."""
    dev = s.n.device
    colors = torch.arange(cfg.n_colors, device=dev)
    numbers = torch.arange(1, cfg.n_numbers + 1, device=dev)

    run_slot = (s.n + 1 <= cfg.n_numbers) & s.distinct_numbers
    group_slot = (s.n + 1 <= cfg.n_colors) & s.distinct_colors & bool(cfg.group_possible)
    no_real = (s.n_real == 0).unsqueeze(-1)

    # Numbered tile: it must not duplicate a colour (group) or a number (run)
    # already present, and must agree with whatever the slot has settled on.
    run_color = (no_real | (s.color.unsqueeze(-1) == colors)) & run_slot.unsqueeze(-1)
    group_color = ((s.color_mask.unsqueeze(-1) & (1 << colors)) == 0) & group_slot.unsqueeze(-1)
    run_number = (s.number_mask.unsqueeze(-1) & (1 << (numbers - 1))) == 0
    group_number = no_real | (s.number.unsqueeze(-1) == numbers)

    return AssignCode(
        color=run_color.to(torch.uint8) | (group_color.to(torch.uint8) << 1),
        number=run_number.to(torch.uint8) | (group_number.to(torch.uint8) << 1),
        # A joker constrains nothing beyond the shape the slot already has.
        joker=(run_slot & s.same_color) | (group_slot & s.same_number),
    )


def assign_open(cfg: RummiConfig, s: SlotStats) -> torch.Tensor:
    """``(..., K)`` would adding one tile of each kind leave the slot extendable?"""
    codes = assign_codes(cfg, s)
    grid = (codes.color.unsqueeze(-1) & codes.number.unsqueeze(-2)) != 0
    return torch.cat(
        [grid.flatten(-2, -1), codes.joker.unsqueeze(-1)], dim=-1
    )


@dataclass(frozen=True, slots=True)
class SlotSummary:
    """Both stages for one table, so the two readers of it can share the work."""

    stats: SlotStats
    ev: SlotEval


def summarize(cfg: RummiConfig, slots: torch.Tensor) -> SlotSummary:
    stats = slot_stats(cfg, slots)
    return SlotSummary(stats=stats, ev=evaluate(cfg, stats))
