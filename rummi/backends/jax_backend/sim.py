"""JAX port of state, dealing, masks and the step function. See SPEC.md.

Fully functional: ``step`` returns a new state rather than mutating one, so the
whole thing is a single traceable graph and ``jit`` can fuse it end to end.

Three things follow from that and are worth knowing before use.

* ``cfg`` is a static argument. It is a frozen dataclass, so it is hashable and
  JAX specialises on it; every shape is then a compile-time constant.
* Action validation is *not* inside the jitted step. Checking the mask means
  reading a boolean off the device, which would force a synchronisation and break
  the trace, so it lives in :func:`check_actions` for callers who want it.
* Deck shuffling stays on NumPy's ``SeedSequence``. The permutation is part of
  the contract, not the implementation -- a JAX PRNG would deal different tiles
  and fail conformance against the golden trajectories.
"""

from __future__ import annotations

import hashlib
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from rummi.backends.jax_backend.kernel import assign_open, evaluate, lookup, slot_stats
from rummi.core.config import RewardMode, RummiConfig
from rummi.core.encoding import EMPTY, tables

NO_WINNER = -1
HISTORY_LEN = 8

DIGEST_FIELDS = (
    "racks", "table_sets", "workbench", "placed_rack", "slot_new",
    "table_snapshot", "pool", "deck_order", "draw_ptr", "melded",
    "current", "micro_count", "turn_count", "consecutive_draws",
    "winner", "done", "truncated",
)


class JaxState(NamedTuple):
    """A pytree of arrays. ``cfg`` is deliberately absent -- it travels as a
    static argument so it never becomes a traced leaf."""

    racks: jax.Array
    table_sets: jax.Array
    workbench: jax.Array
    placed_rack: jax.Array
    slot_new: jax.Array
    table_snapshot: jax.Array
    pool: jax.Array
    deck_order: jax.Array
    draw_ptr: jax.Array
    melded: jax.Array
    current: jax.Array
    micro_count: jax.Array
    turn_count: jax.Array
    consecutive_draws: jax.Array
    last_action: jax.Array
    action_history: jax.Array
    winner: jax.Array
    done: jax.Array
    truncated: jax.Array


class StepResult(NamedTuple):
    rewards: jax.Array
    terminated: jax.Array
    truncated: jax.Array


def digest(state: JaxState) -> str:
    """Must match ``rummi.core.state.BatchState.digest`` bit for bit."""
    h = hashlib.sha256()
    for name in DIGEST_FIELDS:
        arr = np.asarray(getattr(state, name))
        h.update(name.encode())
        h.update(str(arr.shape).encode())
        h.update(np.ascontiguousarray(arr).astype(np.int64).tobytes())
    return h.hexdigest()


def batch_size(state: JaxState) -> int:
    return int(state.racks.shape[0])


def pool_size(cfg: RummiConfig, state: JaxState) -> jax.Array:
    return cfg.n_tiles - state.draw_ptr


def counts_of(cfg: RummiConfig, kinds: jax.Array) -> jax.Array:
    """``(B, K)`` counts from a ``(B, ...)`` array of kind ids with EMPTY padding."""
    b = kinds.shape[0]
    flat = kinds.reshape(b, -1).astype(jnp.int32)
    rows = jnp.arange(b)[:, None]
    out = jnp.zeros((b, cfg.n_kinds), jnp.int32)
    return out.at[rows, jnp.clip(flat, 0, None)].add((flat >= 0).astype(jnp.int32))


def table_counts(cfg: RummiConfig, state: JaxState) -> jax.Array:
    return counts_of(cfg, state.table_sets)


def rack_values(cfg: RummiConfig, state: JaxState) -> jax.Array:
    return state.racks @ jnp.asarray(lookup(cfg).offload)


def check_invariants(cfg: RummiConfig, state: JaxState) -> None:
    total = state.racks.sum(1) + table_counts(cfg, state) + state.workbench + state.pool
    expected = lookup(cfg).copies
    bad = np.argwhere(np.asarray(total) != expected[None, :])
    if bad.size:
        b, k = bad[0]
        raise AssertionError(f"tile conservation violated in env {b} for kind {k}")


# --- dealing ------------------------------------------------------------------
def _decks(cfg: RummiConfig, seeds) -> np.ndarray:
    base = np.repeat(np.arange(cfg.n_kinds), tables(cfg).copies)
    return np.stack([np.random.default_rng(s).permutation(base) for s in seeds])


def deck_orders(cfg: RummiConfig, seed: int, count: int) -> jax.Array:
    return jnp.asarray(_decks(cfg, np.random.SeedSequence(seed).spawn(count)), jnp.int32)


def derived_deck_orders(cfg: RummiConfig, base: int, step_index: int, envs) -> jax.Array:
    seeds = [np.random.SeedSequence([base, step_index, int(e)]) for e in envs]
    return jnp.asarray(_decks(cfg, seeds), jnp.int32)


def allocate(cfg: RummiConfig, b: int) -> JaxState:
    p, k, s, ell = cfg.n_players, cfg.n_kinds, cfg.max_sets, cfg.max_set_len
    z = lambda shape: jnp.zeros(shape, jnp.int32)
    return JaxState(
        racks=z((b, p, k)),
        table_sets=jnp.full((b, s, ell), EMPTY, jnp.int32),
        workbench=z((b, k)),
        placed_rack=z((b, k)),
        slot_new=jnp.zeros((b, s), bool),
        table_snapshot=jnp.full((b, s, ell), EMPTY, jnp.int32),
        pool=z((b, k)),
        deck_order=z((b, cfg.n_tiles)),
        draw_ptr=z(b),
        melded=jnp.zeros((b, p), bool),
        current=z(b),
        micro_count=z(b),
        turn_count=z(b),
        consecutive_draws=z(b),
        last_action=jnp.full((b,), -1, jnp.int32),
        action_history=jnp.full((b, HISTORY_LEN), -1, jnp.int32),
        winner=jnp.full((b,), NO_WINNER, jnp.int32),
        done=jnp.zeros(b, bool),
        truncated=jnp.zeros(b, bool),
    )


def reset(cfg: RummiConfig, b: int, seed: int = 0) -> JaxState:
    return reset_envs(cfg, allocate(cfg, b), jnp.arange(b), deck_orders(cfg, seed, b))


@partial(jax.jit, static_argnums=0)
def reset_envs(
    cfg: RummiConfig, state: JaxState, which: jax.Array, orders: jax.Array
) -> JaxState:
    """Re-deal the given envs; ``orders`` is parallel to ``which``."""
    n = orders.shape[0]
    dealt = orders[:, : cfg.n_players * cfg.rack_size].reshape(
        n, cfg.n_players, cfg.rack_size
    )
    racks = (
        jnp.zeros((n, cfg.n_players, cfg.n_kinds), jnp.int32)
        .at[
            jnp.arange(n)[:, None, None],
            jnp.arange(cfg.n_players)[None, :, None],
            dealt,
        ]
        .add(1)
    )
    empty_table = jnp.full((n, cfg.max_sets, cfg.max_set_len), EMPTY, jnp.int32)
    return state._replace(
        deck_order=state.deck_order.at[which].set(orders),
        racks=state.racks.at[which].set(racks),
        draw_ptr=state.draw_ptr.at[which].set(cfg.n_players * cfg.rack_size),
        pool=state.pool.at[which].set(jnp.asarray(lookup(cfg).copies) - racks.sum(1)),
        table_sets=state.table_sets.at[which].set(empty_table),
        table_snapshot=state.table_snapshot.at[which].set(empty_table),
        workbench=state.workbench.at[which].set(0),
        placed_rack=state.placed_rack.at[which].set(0),
        slot_new=state.slot_new.at[which].set(False),
        melded=state.melded.at[which].set(False),
        current=state.current.at[which].set(0),
        micro_count=state.micro_count.at[which].set(0),
        turn_count=state.turn_count.at[which].set(0),
        consecutive_draws=state.consecutive_draws.at[which].set(0),
        last_action=state.last_action.at[which].set(-1),
        action_history=state.action_history.at[which].set(-1),
        winner=state.winner.at[which].set(NO_WINNER),
        done=state.done.at[which].set(False),
        truncated=state.truncated.at[which].set(False),
    )


# --- masks --------------------------------------------------------------------
def current_rack(state: JaxState) -> jax.Array:
    return jnp.take_along_axis(state.racks, state.current[:, None, None], axis=1)[:, 0]


def meld_value(cfg: RummiConfig, state: JaxState, slot_value: jax.Array) -> jax.Array:
    if cfg.strict_initial_meld:
        return (slot_value * state.slot_new).sum(-1)
    return state.placed_rack @ jnp.asarray(lookup(cfg).value)


@partial(jax.jit, static_argnums=0)
def legal_actions(cfg: RummiConfig, state: JaxState) -> jax.Array:
    b = state.racks.shape[0]
    stats = slot_stats(cfg, state.table_sets)
    ev = evaluate(cfg, stats)

    rack = current_rack(state)
    has_melded = jnp.take_along_axis(state.melded, state.current[:, None], axis=1)[:, 0]
    lengths = stats.n
    occupied_slot = lengths > 0

    may_touch = has_melded if cfg.strict_initial_meld else jnp.ones_like(has_melded)
    mine = may_touch[:, None] | state.slot_new

    empty_slot = ~occupied_slot
    # Only the lowest empty slot is offered, collapsing the S! equivalent choices.
    first_empty = jnp.argmax(empty_slot, axis=-1)
    is_first_empty = (jnp.arange(cfg.max_sets)[None, :] == first_empty[:, None]) & empty_slot.any(
        -1, keepdims=True
    )

    touchable = occupied_slot & mine
    place = rack > 0
    pick = (state.table_sets >= 0) & touchable[..., None]
    dissolve = touchable
    assign = (
        assign_open(cfg, stats)
        & (state.workbench > 0)[:, None, :]
        & (lengths < cfg.max_set_len)[..., None]
        & (touchable | is_first_empty)[..., None]
    )

    table_whole = (ev.is_valid | ev.is_empty).all(-1)
    played_something = state.placed_rack.sum(-1) > 0
    meld_ok = has_melded | (meld_value(cfg, state, ev.value) >= cfg.initial_meld)
    end_turn = (state.workbench.sum(-1) == 0) & table_whole & played_something & meld_ok
    playable = (state.micro_count < cfg.max_micro_per_turn) & ~state.done
    gate = playable[:, None]

    return jnp.concatenate(
        [
            place & gate,
            pick.reshape(b, -1) & gate,
            dissolve & gate,
            # ASSIGN ids are kind-major, so transpose the (S, K) predicate.
            assign.transpose(0, 2, 1).reshape(b, -1) & gate,
            (end_turn & playable)[:, None],
            jnp.ones((b, 1), bool),
        ],
        axis=-1,
    )


def check_actions(mask: jax.Array, actions: jax.Array, live: jax.Array) -> None:
    """Host-side legality check, kept out of the jitted step on purpose."""
    chosen = np.asarray(mask)[np.arange(mask.shape[0]), np.asarray(actions)]
    bad = ~chosen & np.asarray(live)
    if bad.any():
        env = int(np.argmax(bad))
        raise ValueError(f"illegal action {int(np.asarray(actions)[env])} in env {env}")


# --- engine -------------------------------------------------------------------
def _sort_slots(cfg: RummiConfig, table: jax.Array) -> jax.Array:
    """Canonical order *within* each slot: kinds ascending, EMPTY last."""
    key = jnp.sort(jnp.where(table >= 0, table, cfg.n_kinds), axis=-1)
    return jnp.where(key == cfg.n_kinds, EMPTY, key)


def _sort_slot_order(cfg: RummiConfig, table: jax.Array) -> jax.Array:
    """Canonical order *of* the slots, so one table has one representation.

    JAX has ``lexsort``, so this can stay the direct expression of the rule --
    the torch port had to pack digits into integer keys instead.

    Applied only when a turn commits: slot indices must be stable while a player
    is mid-turn, or a multi-step plan would aim at moving targets.
    """
    key = jnp.where(table >= 0, table, cfg.n_kinds)
    # lexsort takes the last key as primary, so hand it the columns in reverse.
    order = jnp.lexsort([key[..., i] for i in range(cfg.max_set_len - 1, -1, -1)], axis=-1)
    return jnp.take_along_axis(table, order[..., None], axis=1)


@partial(jax.jit, static_argnums=0)
def step(
    cfg: RummiConfig,
    state: JaxState,
    actions: jax.Array,
    active: jax.Array | None = None,
) -> tuple[JaxState, StepResult]:
    """Apply one action per env, returning the new state.

    Legality is *not* checked here -- see :func:`check_actions`. Every branch
    assumes its precondition already holds, exactly as in the NumPy reference.
    """
    b = state.racks.shape[0]
    actions = actions.astype(jnp.int32)
    rows = jnp.arange(b)

    live = ~state.done
    if active is not None:
        live = live & active

    is_place = actions < cfg.pick_offset
    is_pick = (actions >= cfg.pick_offset) & (actions < cfg.dissolve_offset)
    is_dissolve = (actions >= cfg.dissolve_offset) & (actions < cfg.assign_offset)
    is_assign = (actions >= cfg.assign_offset) & (actions < cfg.end_turn_action)
    is_end = actions == cfg.end_turn_action
    is_draw = actions == cfg.draw_action

    pick_rel = actions - cfg.pick_offset
    assign_rel = actions - cfg.assign_offset
    kind = jnp.where(is_place, actions, jnp.where(is_assign, assign_rel // cfg.max_sets, 0))
    slot = jnp.where(
        is_pick,
        pick_rel // cfg.max_set_len,
        jnp.where(
            is_dissolve,
            actions - cfg.dissolve_offset,
            jnp.where(is_assign, assign_rel % cfg.max_sets, 0),
        ),
    )
    pos = jnp.where(is_pick, pick_rel % cfg.max_set_len, 0)

    kind_sel = jnp.arange(cfg.n_kinds)[None, :] == kind[:, None]
    slot_sel = jnp.arange(cfg.max_sets)[None, :] == slot[:, None]

    rewards = jnp.zeros((b, cfg.n_players), jnp.float32)
    last_action = jnp.where(live, actions, state.last_action)
    action_history = jnp.where(
        live[:, None],
        jnp.concatenate([state.action_history[:, 1:], actions[:, None]], axis=1),
        state.action_history,
    )

    racks, workbench, placed = state.racks, state.workbench, state.placed_rack
    table, slot_new = state.table_sets, state.slot_new

    # --- PLACE: rack -> workbench
    hit = is_place & live
    moved = (kind_sel & hit[:, None]).astype(jnp.int32)
    racks = racks.at[rows, state.current].add(-moved)
    workbench = workbench + moved
    placed = placed + moved

    # --- PICK: one tile of a set -> workbench
    hit = is_pick & live
    flat = table.reshape(b, -1)
    at = slot * cfg.max_set_len + pos
    taken = flat[rows, at]
    workbench = workbench.at[rows, jnp.clip(taken, 0, None)].add(
        (hit & (taken >= 0)).astype(jnp.int32)
    )
    table = flat.at[rows, at].set(jnp.where(hit, EMPTY, taken)).reshape(table.shape)

    # --- DISSOLVE: whole set -> workbench
    hit = is_dissolve & live
    row = table[rows, slot]
    workbench = workbench.at[rows[:, None], jnp.clip(row, 0, None)].add(
        ((row >= 0) & hit[:, None]).astype(jnp.int32)
    )
    table = table.at[rows, slot].set(jnp.where(hit[:, None], EMPTY, row))

    # --- ASSIGN: workbench -> set slot
    hit = is_assign & live
    row = table[rows, slot]
    was_empty = (row < 0).all(-1)
    free = jnp.argmax(row < 0, axis=-1)
    filled = row.at[rows, free].set(kind)
    table = table.at[rows, slot].set(jnp.where(hit[:, None], filled, row))
    workbench = workbench - (kind_sel & hit[:, None]).astype(jnp.int32)
    slot_new = slot_new | (slot_sel & (hit & was_empty)[:, None])

    # Sorting is idempotent on an already-sorted slot, so doing it unconditionally
    # is both correct and cheaper than branching on which slot changed.
    table = _sort_slots(cfg, table)

    mid_turn = (is_place | is_pick | is_dissolve | is_assign) & live
    micro_count = state.micro_count + mid_turn
    if cfg.micro_step_cost:
        rewards = rewards.at[rows, state.current].add(-cfg.micro_step_cost * mid_turn)

    # --- END_TURN
    hit = is_end & live
    if cfg.tiles_placed_bonus:
        rewards = rewards.at[rows, state.current].add(
            cfg.tiles_placed_bonus * (placed.sum(-1) * hit).astype(jnp.float32)
        )
    if cfg.rack_value_delta:
        rewards = rewards.at[rows, state.current].add(
            cfg.rack_value_delta * ((placed @ jnp.asarray(lookup(cfg).value)) * hit).astype(jnp.float32)
        )
    melded = state.melded | (
        (jnp.arange(cfg.n_players)[None, :] == state.current[:, None]) & hit[:, None]
    )
    table = jnp.where(hit[:, None, None], _sort_slot_order(cfg, table), table)
    consecutive_draws = jnp.where(hit, 0, state.consecutive_draws)

    # --- DRAW: revert the turn, take a tile, pass
    hit = is_draw & live
    table = jnp.where(hit[:, None, None], state.table_snapshot, table)
    racks = racks.at[rows, state.current].add(placed * hit[:, None])

    can_draw = hit & (state.draw_ptr < cfg.n_tiles)
    drawn = state.deck_order[rows, jnp.clip(state.draw_ptr, 0, cfg.n_tiles - 1)]
    got = ((jnp.arange(cfg.n_kinds)[None, :] == drawn[:, None]) & can_draw[:, None]).astype(jnp.int32)
    racks = racks.at[rows, state.current].add(got)
    pool = state.pool - got
    draw_ptr = state.draw_ptr + can_draw
    consecutive_draws = consecutive_draws + hit

    # --- begin turn, for whichever envs committed
    committed = (is_end | is_draw) & live
    keep = ~committed
    state = state._replace(
        racks=racks,
        table_sets=table,
        workbench=workbench * keep[:, None],
        placed_rack=placed * keep[:, None],
        slot_new=slot_new & keep[:, None],
        pool=pool,
        draw_ptr=draw_ptr,
        melded=melded,
        micro_count=jnp.where(committed, 0, micro_count),
        turn_count=state.turn_count + committed,
        consecutive_draws=consecutive_draws,
        table_snapshot=jnp.where(committed[:, None, None], table, state.table_snapshot),
        current=jnp.where(committed, (state.current + 1) % cfg.n_players, state.current),
        last_action=last_action,
        action_history=action_history,
    )
    return _resolve_terminal(cfg, state, committed, rewards)


def _resolve_terminal(
    cfg: RummiConfig, state: JaxState, committed: jax.Array, rewards: jax.Array
) -> tuple[JaxState, StepResult]:
    actor = (state.current - 1) % cfg.n_players
    rack_totals = state.racks.sum(-1)
    emptied = committed & (jnp.take_along_axis(rack_totals, actor[:, None], 1)[:, 0] == 0)

    stalled = (
        committed & (pool_size(cfg, state) == 0) & (state.consecutive_draws >= cfg.n_players)
    )
    over_time = committed & (state.turn_count >= cfg.max_turns)

    values = rack_values(cfg, state)
    lowest = jnp.argmin(values, axis=-1).astype(jnp.int32)
    winner = jnp.where(emptied, actor, jnp.where(stalled, lowest, state.winner))

    finishing = emptied | stalled | over_time
    newly_done = finishing & ~state.done
    done = state.done | finishing
    truncated = state.truncated | (over_time & ~emptied & ~stalled)

    is_winner = jnp.arange(cfg.n_players)[None, :] == jnp.clip(winner, 0, None)[:, None]
    if cfg.reward_mode is RewardMode.WIN_LOSS:
        terminal = jnp.where(is_winner, 1.0, -1.0 / (cfg.n_players - 1)).astype(jnp.float32)
    else:
        paid = jnp.where(is_winner, 0, values).astype(jnp.float32)
        terminal = jnp.where(is_winner, paid.sum(-1, keepdims=True), -paid)
        if cfg.reward_mode is RewardMode.SCORE_NORMALIZED:
            terminal = terminal / max(1, cfg.rack_size * max(cfg.n_numbers, cfg.joker_penalty))

    paying = newly_done & ~truncated
    rewards = rewards + terminal * paying[:, None]
    state = state._replace(winner=winner, done=done, truncated=truncated)
    return state, StepResult(
        rewards=rewards, terminated=done & ~truncated, truncated=truncated
    )
