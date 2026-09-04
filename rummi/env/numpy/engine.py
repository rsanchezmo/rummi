"""The step function: apply one primitive action per env.

Legality is decided entirely in :mod:`rummi.env.numpy.masks`; this module only applies
effects, so every branch here assumes its precondition already holds. Effects are
applied family by family over the envs that chose that family, which keeps each
update a single vectorised scatter with no duplicate indices.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rummi.rules.actions import DecodedActions, decode_batch
from rummi.rules.config import RewardMode, RummiConfig
from rummi.rules.encoding import EMPTY, tables
from rummi.env.numpy.state import BatchState, counts_of


@dataclass(frozen=True, slots=True)
class StepResult:
    rewards: np.ndarray
    """``(B, P)`` reward credited to each seat this step."""
    terminated: np.ndarray
    truncated: np.ndarray


def _sort_slot_rows(cfg: RummiConfig, rows: np.ndarray) -> np.ndarray:
    """Canonical order within a slot: kinds ascending, ``EMPTY`` pushed to the end."""
    key = np.where(rows >= 0, rows, np.int16(cfg.n_kinds))
    key = np.sort(key, axis=-1)
    return np.where(key == cfg.n_kinds, np.int16(EMPTY), key).astype(np.int16)


def _sort_slot_order(cfg: RummiConfig, table: np.ndarray) -> np.ndarray:
    """Canonical order *of* the slots, so the same table always looks the same.

    Applied only when a turn commits: slot indices must stay stable while a player
    is mid-turn, or their multi-step plan would be aimed at moving targets.
    """
    key = np.where(table >= 0, table, np.int16(cfg.n_kinds))
    # np.lexsort takes the last key as primary, so hand it the columns in reverse.
    order = np.lexsort([key[..., i] for i in range(cfg.max_set_len - 1, -1, -1)], axis=-1)
    return np.take_along_axis(table, order[..., None], axis=1)


def step(
    state: BatchState,
    actions: np.ndarray,
    mask: np.ndarray | None = None,
    active: np.ndarray | None = None,
) -> StepResult:
    """Apply one action per env, mutating ``state`` in place.

    ``mask`` is the output of :func:`rummi.env.numpy.masks.legal_actions` for this
    state. Passing it is free for the caller -- the env computes it anyway -- and
    turns an illegal action into a loud failure rather than a corrupt state.

    ``active`` opts envs out of this step entirely. The vector env needs it for
    Gymnasium's next-step autoreset, where the action supplied alongside a just
    re-dealt env must be discarded rather than played.
    """
    cfg = state.cfg
    b = state.batch_size
    actions = np.asarray(actions, dtype=np.int64)
    if actions.shape != (b,):
        raise ValueError(f"expected {(b,)} actions, got {actions.shape}")
    checkable = ~state.done if active is None else (~state.done & active)
    if mask is not None and not mask[np.arange(b), actions][checkable].all():
        offender = int(np.argmax(~mask[np.arange(b), actions] & checkable))
        raise ValueError(f"illegal action {int(actions[offender])} in env {offender}")

    d = decode_batch(cfg, actions)
    live = ~state.done
    if active is not None:
        live = live & active
    rows = np.arange(b)
    rewards = np.zeros((b, cfg.n_players), dtype=np.float32)
    state.last_action = np.where(live, actions, state.last_action).astype(np.int32)
    acted = np.flatnonzero(live)
    if acted.size:
        state.action_history[acted] = np.concatenate(
            [state.action_history[acted, 1:], actions[acted, None].astype(np.int32)], axis=1
        )

    _apply_place(state, d, live)
    _apply_pick(state, d, live)
    _apply_dissolve(state, d, live)
    _apply_assign(state, d, live)

    mid_turn = (d.is_place | d.is_pick | d.is_dissolve | d.is_assign) & live
    state.micro_count[mid_turn] += 1
    if cfg.micro_step_cost:
        rewards[rows[mid_turn], state.current[mid_turn]] -= cfg.micro_step_cost

    _apply_end_turn(state, d, live, rewards)
    _apply_draw(state, d, live)

    _resolve_terminal(state, d, live, rewards)
    return StepResult(rewards=rewards, terminated=state.done & ~state.truncated,
                      truncated=state.truncated.copy())


def _apply_place(state: BatchState, d: DecodedActions, live: np.ndarray) -> None:
    sel = np.flatnonzero(d.is_place & live)
    if not sel.size:
        return
    kind = d.kind[sel]
    player = state.current[sel]
    state.racks[sel, player, kind] -= 1
    state.workbench[sel, kind] += 1
    state.placed_rack[sel, kind] += 1


def _apply_pick(state: BatchState, d: DecodedActions, live: np.ndarray) -> None:
    cfg = state.cfg
    sel = np.flatnonzero(d.is_pick & live)
    if not sel.size:
        return
    slot, pos = d.slot[sel], d.pos[sel]
    kind = state.table_sets[sel, slot, pos]
    state.table_sets[sel, slot, pos] = EMPTY
    state.workbench[sel, kind] += 1
    state.table_sets[sel, slot] = _sort_slot_rows(cfg, state.table_sets[sel, slot])


def _apply_dissolve(state: BatchState, d: DecodedActions, live: np.ndarray) -> None:
    cfg = state.cfg
    sel = np.flatnonzero(d.is_dissolve & live)
    if not sel.size:
        return
    slot = d.slot[sel]
    rows = state.table_sets[sel, slot]
    state.workbench[sel] += counts_of(cfg, rows[:, None, :])
    state.table_sets[sel, slot] = EMPTY


def _apply_assign(state: BatchState, d: DecodedActions, live: np.ndarray) -> None:
    cfg = state.cfg
    sel = np.flatnonzero(d.is_assign & live)
    if not sel.size:
        return
    kind, slot = d.kind[sel], d.slot[sel]
    rows = state.table_sets[sel, slot]
    was_empty = (rows < 0).all(-1)
    # The mask guarantees room, so the first empty position always exists.
    pos = (rows < 0).argmax(-1)
    rows[np.arange(sel.size), pos] = kind
    state.table_sets[sel, slot] = _sort_slot_rows(cfg, rows)
    state.workbench[sel, kind] -= 1
    state.slot_new[sel[was_empty], slot[was_empty]] = True


def _apply_end_turn(
    state: BatchState, d: DecodedActions, live: np.ndarray, rewards: np.ndarray
) -> None:
    cfg = state.cfg
    sel = np.flatnonzero(d.is_end_turn & live)
    if not sel.size:
        return
    player = state.current[sel]

    if cfg.tiles_placed_bonus:
        rewards[sel, player] += cfg.tiles_placed_bonus * state.placed_rack[sel].sum(-1)
    if cfg.rack_value_delta:
        # The joker's value is positional, so it cannot be credited from the rack.
        face_value = tables(cfg).value.astype(np.int32)
        rewards[sel, player] += (
            cfg.rack_value_delta * state.placed_rack[sel].astype(np.int32) @ face_value
        )

    state.melded[sel, player] = True
    state.table_sets[sel] = _sort_slot_order(cfg, state.table_sets[sel])
    state.consecutive_draws[sel] = 0
    _begin_turn(state, sel)


def _apply_draw(state: BatchState, d: DecodedActions, live: np.ndarray) -> None:
    cfg = state.cfg
    sel = np.flatnonzero(d.is_draw & live)
    if not sel.size:
        return
    player = state.current[sel]

    # Undo everything the turn attempted: the table goes back to its snapshot and
    # the tiles that left the rack return to it.
    state.table_sets[sel] = state.table_snapshot[sel]
    state.racks[sel, player] += state.placed_rack[sel]

    can_draw = state.draw_ptr[sel] < cfg.n_tiles
    drawers = sel[can_draw]
    if drawers.size:
        kind = state.deck_order[drawers, state.draw_ptr[drawers]]
        state.racks[drawers, state.current[drawers], kind] += 1
        state.pool[drawers, kind] -= 1
        state.draw_ptr[drawers] += 1

    state.consecutive_draws[sel] += 1
    _begin_turn(state, sel)


def _begin_turn(state: BatchState, sel: np.ndarray) -> None:
    """Clear per-turn bookkeeping and hand play to the next seat."""
    cfg = state.cfg
    state.workbench[sel] = 0
    state.placed_rack[sel] = 0
    state.slot_new[sel] = False
    state.micro_count[sel] = 0
    state.turn_count[sel] += 1
    state.table_snapshot[sel] = state.table_sets[sel]
    state.current[sel] = (state.current[sel] + 1) % cfg.n_players


def _resolve_terminal(
    state: BatchState, d: DecodedActions, live: np.ndarray, rewards: np.ndarray
) -> None:
    """Detect end of game and credit terminal reward."""
    cfg = state.cfg
    committed = (d.is_end_turn | d.is_draw) & live
    if not committed.any():
        return

    rack_totals = state.racks.sum(-1)
    # The seat that just played is the one before `current` after _begin_turn.
    actor = (state.current - 1) % cfg.n_players
    emptied = committed & (rack_totals[np.arange(state.batch_size), actor] == 0)

    pool_empty = state.pool_size == 0
    stalled = committed & pool_empty & (state.consecutive_draws >= cfg.n_players)
    over_time = committed & (state.turn_count >= cfg.max_turns)

    values = state.rack_values()
    lowest = np.argmin(values, axis=-1).astype(np.int16)

    state.winner = np.where(emptied, actor, np.where(stalled, lowest, state.winner)).astype(
        np.int16
    )
    newly_done = (emptied | stalled | over_time) & ~state.done
    state.done |= emptied | stalled | over_time
    state.truncated |= over_time & ~emptied & ~stalled

    finished = np.flatnonzero(newly_done & ~state.truncated)
    if finished.size:
        rewards[finished] += _terminal_rewards(cfg, values[finished], state.winner[finished])


def _terminal_rewards(
    cfg: RummiConfig, values: np.ndarray, winner: np.ndarray
) -> np.ndarray:
    """``(n, P)`` terminal reward for finished envs."""
    n = values.shape[0]
    is_winner = np.zeros((n, cfg.n_players), dtype=bool)
    is_winner[np.arange(n), winner] = True

    if cfg.reward_mode is RewardMode.WIN_LOSS:
        # Zero-sum so self-play cannot inflate its own return.
        return np.where(
            is_winner, 1.0, -1.0 / (cfg.n_players - 1)
        ).astype(np.float32)

    # Official scoring: each loser pays their rack, the winner collects the lot.
    paid = np.where(is_winner, 0, values).astype(np.float32)
    out = -paid
    out[np.arange(n), winner] = paid.sum(-1)
    if cfg.reward_mode is RewardMode.SCORE_NORMALIZED:
        out /= max(1, cfg.rack_size * max(cfg.n_numbers, cfg.joker_penalty))
    return out.astype(np.float32)
