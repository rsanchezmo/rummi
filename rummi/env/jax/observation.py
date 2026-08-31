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
from rummi.env.jax.kernel import SlotSummary, summarize
from rummi.env.jax.sim import JaxState, current_rack, legal_actions, meld_value, pool_size


@partial(jax.jit, static_argnums=0)
def encode(
    cfg: RummiConfig, state: JaxState, summary: SlotSummary | None = None
) -> dict[str, jax.Array]:
    """``summary`` is this table's :func:`~rummi.env.jax.kernel.summarize`, passed in
    by a caller that is also building a mask from the same state."""
    p = cfg.n_players

    if summary is None:
        summary = summarize(cfg, state.table_sets)
    stats, ev = summary.stats, summary.ev

    rack = current_rack(state)
    # The pool plus every other rack. Tile conservation makes that identical to
    # `copies - rack - table - workbench`, and it is a (P, K) sum rather than a
    # scatter over every position on the table.
    unseen = state.racks.sum(1) - rack + state.pool

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


@partial(jax.jit, static_argnums=0)
def observe(cfg: RummiConfig, state: JaxState) -> tuple[jax.Array, dict[str, jax.Array]]:
    """The mask and the observation of one state, sharing the table's summary."""
    summary = summarize(cfg, state.table_sets)
    return legal_actions(cfg, state, summary), encode(cfg, state, summary)
