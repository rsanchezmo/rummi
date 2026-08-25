"""Torch port of the observation encoder. See SPEC.md section 8.

Written against the spec rather than against
:mod:`rummi.env.observation`, like the rest of this backend. It returns tensors
on the state's device: converting here would throw away the reason for stepping
on a device at all, and Gymnasium's own ``wrappers.vector`` conversions handle the
boundary through ``from_dlpack`` when a caller wants something else.
"""

from __future__ import annotations

import torch

from rummi.rules.observation import N_SCALARS, SLOT_FEATURES
from rummi.env.torch.kernel import evaluate, lookup, slot_stats
from rummi.env.torch.sim import TorchState, current_rack, meld_value


def encode(state: TorchState) -> dict[str, torch.Tensor]:
    cfg = state.cfg
    dev = state.device
    b, p = state.batch_size, cfg.n_players

    stats = slot_stats(cfg, state.table_sets)
    ev = evaluate(cfg, stats)

    rack = current_rack(state)
    unseen = lookup(cfg, dev).copies.unsqueeze(0) - rack - state.table_counts() - state.workbench

    slot_features = torch.stack(
        [
            stats.n,
            ev.run_valid,
            ev.group_valid,
            ev.is_extendable,
            stats.color,
            stats.lo,
            stats.hi,
            stats.n_jokers,
            ev.value,
            state.slot_new,
        ],
        dim=-1,
    ).to(torch.int32)
    assert slot_features.shape[-1] == SLOT_FEATURES

    # Seat `i` is `(current + i) mod P`, so one policy plays every seat.
    rotation = (state.current.view(-1, 1) + torch.arange(p, device=dev).view(1, -1)) % p

    progress = meld_value(state, ev.value).to(torch.int32)
    has_melded = state.melded.gather(1, state.current.view(-1, 1)).squeeze(1)
    remaining = torch.where(
        has_melded, torch.zeros_like(progress), (cfg.initial_meld - progress).clamp(min=0)
    )

    scalars = torch.stack(
        [state.pool_size.to(torch.int32), progress, remaining, state.micro_count.to(torch.int32)],
        dim=-1,
    )
    assert scalars.shape == (b, N_SCALARS)

    return {
        "rack": rack.to(torch.int16),
        "table_sets": state.table_sets.to(torch.int16),
        "slot_features": slot_features,
        "workbench": state.workbench.to(torch.int16),
        "placed_this_turn": state.placed_rack.to(torch.int16),
        "unseen": unseen.to(torch.int16),
        "rack_sizes": state.racks.sum(-1).gather(1, rotation).to(torch.int16),
        "melded": state.melded.gather(1, rotation).to(torch.int8),
        "scalars": scalars,
    }
