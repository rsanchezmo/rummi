"""Legal-action masks.

Every rule that constrains *what may be done* lives here; ``engine.py`` only
applies effects. The split matters because the mask is the agent's only view of
legality, so anything the engine would reject must be masked out here rather
than penalised after the fact.

``DRAW`` is deliberately never masked. It reverts the turn and passes, so the MDP
cannot deadlock mid-turn and no row of the mask is ever all-zero -- including
once the per-turn micro-action budget is spent, where it is the only legal move,
and after an env is done, where an empty row would hand a policy a NaN.
"""

from __future__ import annotations

import numpy as np

from rummi.rules.config import RummiConfig
from rummi.rules.encoding import tables
from rummi.env.numpy.sets import SlotStats, SlotSummary, assign_open_at, summarize
from rummi.env.numpy.state import BatchState


def current_rack(state: BatchState) -> np.ndarray:
    """``(B, K)`` the acting player's rack."""
    return state.racks[np.arange(state.batch_size), state.current]


def _face_values(cfg: RummiConfig) -> np.ndarray:
    """Face value per kind with the joker at zero: its worth is positional and
    can only be resolved from the set it ends up in."""
    return tables(cfg).value.astype(np.int32)


def meld_value(state: BatchState, slot_value: np.ndarray) -> np.ndarray:
    """``(B,)`` value credited towards the acting player's initial meld.

    Under the official rule the opening meld is built solely from the player's
    own tiles, so the sets created this turn *are* the tiles they played and the
    joker's value is resolved by the set it sits in. When that restriction is
    relaxed the table may be rearranged, the two no longer coincide, and only the
    face value of the tiles that left the rack can be credited.
    """
    if state.cfg.strict_initial_meld:
        return (slot_value * state.slot_new).sum(-1)
    return state.placed_rack.astype(np.int32) @ _face_values(state.cfg)


def legal_actions(state: BatchState, summary: SlotSummary | None = None) -> np.ndarray:
    """``(B, A)`` boolean mask of currently legal actions.

    ``summary`` is the table's :func:`~rummi.env.numpy.sets.summarize`, which the
    caller passes when it is also encoding an observation from the same state.
    """
    cfg = state.cfg
    b = state.batch_size
    rows = np.arange(b)

    if summary is None:
        summary = summarize(cfg, state.table_sets)
    stats, ev = summary.stats, summary.ev

    rack = current_rack(state)
    has_melded = state.melded[rows, state.current]
    lengths = stats.n
    occupied_slot = lengths > 0

    # Before melding, the official rule forbids touching the existing table: the
    # player may only build fresh sets from their own tiles. Restricting them to
    # slots created this turn is exactly equivalent, and leaves them free to undo
    # their own work-in-progress.
    may_touch_table = has_melded if cfg.strict_initial_meld else np.ones(b, dtype=bool)
    mine = may_touch_table[:, None] | state.slot_new

    empty_slot = ~occupied_slot
    has_empty = empty_slot.any(-1)
    # Offering only the lowest empty slot collapses the S! ways of saying the same thing.
    is_first_empty = np.zeros_like(empty_slot)
    is_first_empty[rows[has_empty], empty_slot.argmax(-1)[has_empty]] = True

    touchable = occupied_slot & mine
    place = rack > 0
    pick = (state.table_sets >= 0) & touchable[..., None]
    dissolve = touchable

    table_whole = (ev.is_valid | ev.is_empty).all(-1)
    played_something = state.placed_rack.sum(-1) > 0
    meld_ok = has_melded | (meld_value(state, ev.value) >= cfg.initial_meld)
    end_turn = (state.workbench.sum(-1) == 0) & table_whole & played_something & meld_ok

    playable = (state.micro_count < cfg.max_micro_per_turn) & ~state.done
    # Everything ASSIGN asks of the slot rather than of the tile.
    slot_ok = (
        (lengths < cfg.max_set_len)             # slot has room
        & (touchable | is_first_empty)
        & playable[:, None]
    )

    mask = np.zeros((b, cfg.n_actions), dtype=bool)
    mask[:, cfg.place_offset : cfg.pick_offset] = place & playable[:, None]
    mask[:, cfg.pick_offset : cfg.dissolve_offset] = pick.reshape(b, -1) & playable[:, None]
    mask[:, cfg.dissolve_offset : cfg.assign_offset] = dissolve & playable[:, None]
    _fill_assign(cfg, state, stats, slot_ok, mask)
    mask[:, cfg.end_turn_action] = end_turn & playable
    # Unconditionally legal, finished envs included: the engine ignores actions
    # for done envs anyway, and an all-zero row would hand a policy a NaN.
    mask[:, cfg.draw_action] = True
    return mask


def _fill_assign(
    cfg: RummiConfig,
    state: BatchState,
    stats: SlotStats,
    slot_ok: np.ndarray,
    mask: np.ndarray,
) -> None:
    """Write the ASSIGN block of ``mask``, for the kinds that could be legal at all.

    A tile has to be in hand, and a workbench holds a handful of kinds. Every other
    column of the block is false -- which is what ``mask`` already holds -- so only
    the held ones are computed: see
    :func:`~rummi.env.numpy.sets.assign_open_at`. ASSIGN ids are kind-major, so
    each pair lands in one contiguous row.
    """
    env, kind = np.nonzero(state.workbench > 0)
    if env.size == 0:
        return
    block = mask[:, cfg.assign_offset : cfg.end_turn_action].reshape(
        state.batch_size, cfg.n_kinds, cfg.max_sets
    )
    block[env, kind] = assign_open_at(cfg, stats, env, kind) & slot_ok[env]
