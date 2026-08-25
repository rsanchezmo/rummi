"""JAX port of the observation encoder. See SPEC.md section 8.

Written against the spec rather than against
:mod:`rummi.env.observation`, like the rest of this backend. ``cfg`` is a static
argument for the same reason it is everywhere here -- the shapes are derived from
it, so ``jit`` has to specialise on it -- and the constant tables come from
``lookup`` as NumPy arrays, never cached tracers.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from rummi.rules.config import RummiConfig
from rummi.rules.observation import SLOT_FEATURES
from rummi.env.jax.kernel import evaluate, lookup, slot_stats
from rummi.env.jax.sim import JaxState, current_rack, meld_value, pool_size, table_counts


@partial(jax.jit, static_argnums=0)
def encode(cfg: RummiConfig, state: JaxState) -> dict[str, jax.Array]:
    p = cfg.n_players

    stats = slot_stats(cfg, state.table_sets)
    ev = evaluate(cfg, stats)

    rack = current_rack(state)
    copies = jnp.asarray(lookup(cfg).copies)
    unseen = copies[None, :] - rack - table_counts(cfg, state) - state.workbench

    slot_features = jnp.stack(
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
        axis=-1,
    ).astype(jnp.int32)
    assert slot_features.shape[-1] == SLOT_FEATURES

    # Seat `i` is `(current + i) mod P`, so one policy plays every seat.
    rotation = (state.current[:, None] + jnp.arange(p)[None, :]) % p

    progress = meld_value(cfg, state, ev.value).astype(jnp.int32)
    has_melded = jnp.take_along_axis(state.melded, state.current[:, None], axis=1)[:, 0]
    remaining = jnp.where(has_melded, 0, jnp.maximum(0, cfg.initial_meld - progress))

    return {
        "rack": rack.astype(jnp.int16),
        "table_sets": state.table_sets.astype(jnp.int16),
        "slot_features": slot_features,
        "workbench": state.workbench.astype(jnp.int16),
        "placed_this_turn": state.placed_rack.astype(jnp.int16),
        "unseen": unseen.astype(jnp.int16),
        "rack_sizes": jnp.take_along_axis(state.racks.sum(-1), rotation, axis=1).astype(jnp.int16),
        "melded": jnp.take_along_axis(state.melded, rotation, axis=1).astype(jnp.int8),
        "scalars": jnp.stack(
            [
                pool_size(cfg, state).astype(jnp.int32),
                progress,
                remaining.astype(jnp.int32),
                state.micro_count.astype(jnp.int32),
            ],
            axis=-1,
        ),
    }
