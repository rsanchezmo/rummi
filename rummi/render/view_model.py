"""One env, sliced out of the batch as a plain-Python snapshot.

Both renderers read this, so layout decisions -- what counts as a partial set,
which seat is acting, how far the opening meld has got -- are made once here
rather than twice in drawing code.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from rummi.rules.actions import ActionKind, action_name, decode
from rummi.rules.config import RummiConfig
from rummi.rules.encoding import counts_to_kinds, kind_name
from rummi.rules.encoding import tables
from rummi.env.numpy.sets import evaluate_slots
from rummi.env.numpy.state import BatchState


class SlotShape(str, Enum):
    EMPTY = "empty"
    RUN = "run"
    GROUP = "group"
    PARTIAL = "partial"
    """Not a legal set yet, but could still become one."""
    BROKEN = "broken"
    """Cannot become a legal set at all; must be taken apart before the turn ends."""


@dataclass(frozen=True, slots=True)
class SlotView:
    index: int
    tiles: tuple[int, ...]
    """As stored: kinds ascending, jokers last. Positions here are what ``PICK``
    indexes, so this order must not be rearranged for looks."""
    shown: tuple[int, ...]
    """As a person reads it: a joker sits in the gap it fills rather than at the
    end, so a run displays as ``10 11 * 13`` instead of ``10 11 13 *``."""
    shape: SlotShape
    value: int
    is_new: bool

    @property
    def is_valid(self) -> bool:
        return self.shape in (SlotShape.RUN, SlotShape.GROUP)

    @property
    def blocks_end_turn(self) -> bool:
        return self.shape in (SlotShape.PARTIAL, SlotShape.BROKEN)


@dataclass(frozen=True, slots=True)
class GameView:
    cfg: RummiConfig
    env_index: int
    slots: tuple[SlotView, ...]
    workbench: tuple[int, ...]
    rack: tuple[int, ...]
    """The acting seat's tiles, sorted."""
    placed_this_turn: tuple[int, ...]
    current_player: int
    rack_sizes: tuple[int, ...]
    melded: tuple[bool, ...]
    pool_size: int
    turn: int
    micro: int
    micro_budget: int
    last_action: str | None
    history: tuple[str, ...]
    """Recent actions, most recent first."""
    touched_slot: int | None
    """Slot the last action acted on, for highlighting; ``None`` if it acted
    elsewhere (PLACE, END_TURN, DRAW) or if the slots have since been reordered."""
    n_legal: int
    meld_progress: int
    done: bool
    truncated: bool
    winner: int

    @property
    def occupied_slots(self) -> tuple[SlotView, ...]:
        return tuple(s for s in self.slots if s.shape is not SlotShape.EMPTY)

    @property
    def table_whole(self) -> bool:
        return not any(s.blocks_end_turn for s in self.slots)

    @property
    def needs_meld(self) -> bool:
        return not self.melded[self.current_player]

    def label(self, kind: int) -> str:
        return kind_name(self.cfg, kind)


def reading_order(cfg: RummiConfig, tiles: tuple[int, ...], is_run: bool) -> tuple[int, ...]:
    """Lay a set out the way it reads, putting each joker in the gap it fills.

    Only runs carry positional meaning: a joker in ``R10 R11 R13 *`` stands for
    the 12, and showing it at the end makes the reader reconstruct that. In a
    group every tile shares a number, so there is no gap to sit in.

    Uses the same window the value calculation picks, so the layout and the
    printed score always tell the same story.
    """
    if not is_run or not tiles:
        return tiles

    t = tables(cfg)
    reals = sorted(int(t.number[k]) for k in tiles if k != cfg.joker_kind)
    jokers = len(tiles) - len(reals)
    if not jokers:
        return tiles
    if not reals:
        return tiles

    n = len(tiles)
    # The best-case start, matching SlotEval.value.
    start = min(reals[0], cfg.n_numbers - n + 1)
    by_number = {}
    for kind in tiles:
        if kind != cfg.joker_kind:
            by_number.setdefault(int(t.number[kind]), []).append(kind)

    out: list[int] = []
    spare = jokers
    for number in range(start, start + n):
        if by_number.get(number):
            out.append(by_number[number].pop())
        elif spare:
            out.append(cfg.joker_kind)
            spare -= 1
    # Anything unplaced (a window that did not line up) keeps its stored order,
    # so this can only ever reorder, never drop a tile.
    if len(out) != n:
        return tiles
    return tuple(out)


def _meld_progress(state: BatchState, b: int, slot_value: np.ndarray) -> int:
    """Value credited so far towards the acting seat's opening meld.

    Mirrors :func:`rummi.env.numpy.masks.meld_value` for a single env: under the
    official rule the sets created this turn *are* the tiles played, so the joker
    is resolved by its set; when that restriction is relaxed only face value can
    be credited.
    """
    cfg = state.cfg
    if cfg.strict_initial_meld:
        return int((slot_value * state.slot_new[b]).sum())
    return int(state.placed_rack[b].astype(np.int32) @ tables(cfg).value.astype(np.int32))


def view(state: BatchState, env_index: int = 0, mask: np.ndarray | None = None) -> GameView:
    """Snapshot one env. Only this env's slice is touched, which is what keeps
    rendering cheap under the torch/JAX backends."""
    cfg = state.cfg
    b = env_index
    ev = evaluate_slots(cfg, state.table_sets[b : b + 1])

    slots: list[SlotView] = []
    for i in range(cfg.max_sets):
        tiles = tuple(int(k) for k in state.table_sets[b, i] if k >= 0)
        if not tiles:
            shape = SlotShape.EMPTY
        elif bool(ev.run_valid[0, i]):
            shape = SlotShape.RUN
        elif bool(ev.group_valid[0, i]):
            shape = SlotShape.GROUP
        elif bool(ev.is_extendable[0, i]):
            shape = SlotShape.PARTIAL
        else:
            shape = SlotShape.BROKEN
        slots.append(
            SlotView(
                index=i,
                tiles=tiles,
                shown=reading_order(cfg, tiles, shape is SlotShape.RUN),
                shape=shape,
                value=int(ev.value[0, i]),
                is_new=bool(state.slot_new[b, i]),
            )
        )

    player = int(state.current[b])
    last = int(state.last_action[b])
    touched = None
    if last >= 0:
        what, arg0, arg1 = decode(cfg, last)
        if what in (ActionKind.PICK, ActionKind.DISSOLVE):
            touched = arg0
        elif what is ActionKind.ASSIGN:
            touched = arg1
    return GameView(
        cfg=cfg,
        env_index=b,
        slots=tuple(slots),
        workbench=tuple(int(k) for k in counts_to_kinds(state.workbench[b])),
        rack=tuple(int(k) for k in counts_to_kinds(state.racks[b, player])),
        placed_this_turn=tuple(int(k) for k in counts_to_kinds(state.placed_rack[b])),
        current_player=player,
        rack_sizes=tuple(int(x) for x in state.racks[b].sum(-1)),
        melded=tuple(bool(x) for x in state.melded[b]),
        pool_size=int(state.pool_size[b]),
        turn=int(state.turn_count[b]),
        micro=int(state.micro_count[b]),
        micro_budget=cfg.max_micro_per_turn,
        last_action=action_name(cfg, last) if last >= 0 else None,
        history=tuple(
            action_name(cfg, int(a)) for a in reversed(state.action_history[b]) if a >= 0
        ),
        touched_slot=touched,
        n_legal=int(mask[b].sum()) if mask is not None else -1,
        meld_progress=_meld_progress(state, b, ev.value[0]),
        done=bool(state.done[b]),
        truncated=bool(state.truncated[b]),
        winner=int(state.winner[b]),
    )
