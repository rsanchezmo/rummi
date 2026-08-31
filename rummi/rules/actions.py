"""The flat discrete action space.

One ``step`` applies one primitive table operation, so a player's whole turn is a
*sequence* of actions terminated by ``END_TURN`` or ``DRAW``. Action ids are laid
out as six contiguous blocks (see :class:`~rummi.rules.config.RummiConfig`)::

    PLACE(kind)          rack      -> workbench
    PICK(slot, pos)      one tile of a set -> workbench
    DISSOLVE(slot)       whole set -> workbench
    ASSIGN(kind, slot)   workbench -> set slot
    END_TURN             commit, only legal when the table is whole again
    DRAW                 revert the turn, take a tile, pass

Keeping tiles in a per-turn *workbench* is what makes every legality test local:
the table is only ever required to be fully valid at ``END_TURN``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from rummi.rules.config import RummiConfig
from rummi.rules.encoding import kind_name


class ActionKind(IntEnum):
    PLACE = 0
    PICK = 1
    DISSOLVE = 2
    ASSIGN = 3
    END_TURN = 4
    DRAW = 5


def encode_place(cfg: RummiConfig, kind: int) -> int:
    return cfg.place_offset + kind


def encode_place_batch(cfg: RummiConfig, kinds: np.ndarray) -> np.ndarray:
    """:func:`encode_place` over an array of kinds."""
    return (cfg.place_offset + np.asarray(kinds)).astype(np.int64)


def encode_assign_batch(cfg: RummiConfig, kinds: np.ndarray, slots: np.ndarray) -> np.ndarray:
    """:func:`encode_assign` over arrays of kinds and slots."""
    return (cfg.assign_offset + np.asarray(kinds) * cfg.max_sets + np.asarray(slots)).astype(
        np.int64
    )


def encode_pick(cfg: RummiConfig, slot: int, pos: int) -> int:
    return cfg.pick_offset + slot * cfg.max_set_len + pos


def encode_dissolve(cfg: RummiConfig, slot: int) -> int:
    return cfg.dissolve_offset + slot


def encode_assign(cfg: RummiConfig, kind: int, slot: int) -> int:
    return cfg.assign_offset + kind * cfg.max_sets + slot


@dataclass(frozen=True, slots=True)
class DecodedActions:
    """Vectorised decode of ``(B,)`` action ids into per-family selectors.

    ``kind``, ``slot`` and ``pos`` are only meaningful where the matching
    ``is_*`` selector is set; elsewhere they are clamped to ``0`` so they stay
    safe to use as gather indices under a ``where``.
    """

    is_place: np.ndarray
    is_pick: np.ndarray
    is_dissolve: np.ndarray
    is_assign: np.ndarray
    is_end_turn: np.ndarray
    is_draw: np.ndarray
    kind: np.ndarray
    slot: np.ndarray
    pos: np.ndarray


def decode_batch(cfg: RummiConfig, actions: np.ndarray) -> DecodedActions:
    a = np.asarray(actions, dtype=np.int64)
    if a.size and (a.min() < 0 or a.max() >= cfg.n_actions):
        raise ValueError(f"action out of range [0, {cfg.n_actions})")

    is_place = a < cfg.pick_offset
    is_pick = (a >= cfg.pick_offset) & (a < cfg.dissolve_offset)
    is_dissolve = (a >= cfg.dissolve_offset) & (a < cfg.assign_offset)
    is_assign = (a >= cfg.assign_offset) & (a < cfg.end_turn_action)
    is_end_turn = a == cfg.end_turn_action
    is_draw = a == cfg.draw_action

    pick_rel = a - cfg.pick_offset
    assign_rel = a - cfg.assign_offset

    kind = np.where(is_place, a, np.where(is_assign, assign_rel // cfg.max_sets, 0))
    slot = np.where(
        is_pick,
        pick_rel // cfg.max_set_len,
        np.where(is_dissolve, a - cfg.dissolve_offset, np.where(is_assign, assign_rel % cfg.max_sets, 0)),
    )
    pos = np.where(is_pick, pick_rel % cfg.max_set_len, 0)

    return DecodedActions(
        is_place=is_place,
        is_pick=is_pick,
        is_dissolve=is_dissolve,
        is_assign=is_assign,
        is_end_turn=is_end_turn,
        is_draw=is_draw,
        kind=kind,
        slot=slot,
        pos=pos,
    )


def decode(cfg: RummiConfig, action: int) -> tuple[ActionKind, int, int]:
    """Scalar decode into ``(kind_of_action, arg0, arg1)``, for logs and tests."""
    d = decode_batch(cfg, np.asarray([action]))
    if d.is_place[0]:
        return ActionKind.PLACE, int(d.kind[0]), -1
    if d.is_pick[0]:
        return ActionKind.PICK, int(d.slot[0]), int(d.pos[0])
    if d.is_dissolve[0]:
        return ActionKind.DISSOLVE, int(d.slot[0]), -1
    if d.is_assign[0]:
        return ActionKind.ASSIGN, int(d.kind[0]), int(d.slot[0])
    if d.is_end_turn[0]:
        return ActionKind.END_TURN, -1, -1
    return ActionKind.DRAW, -1, -1


def action_name(cfg: RummiConfig, action: int) -> str:
    what, a, b = decode(cfg, action)
    if what is ActionKind.PLACE:
        return f"PLACE({kind_name(cfg, a)})"
    if what is ActionKind.PICK:
        return f"PICK(slot={a}, pos={b})"
    if what is ActionKind.DISSOLVE:
        return f"DISSOLVE(slot={a})"
    if what is ActionKind.ASSIGN:
        return f"ASSIGN({kind_name(cfg, a)}, slot={b})"
    return what.name
