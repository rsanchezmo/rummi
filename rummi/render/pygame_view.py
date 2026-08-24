"""Live pygame window, blit-only with dirty-rect updates.

Occupied slots are *packed* into a fixed grid of display rows rather than being
given one reserved row each. A worst-case table needs ``n_tiles // min_set``
slots, but real tables use far fewer and long runs, so reserving every slot
wasted most of the window. Packing keeps the window compact while the grid stays
fixed, which is what dirty-rect tracking needs: each row's hash includes the slot
index it is showing, so a shift in packing simply marks those rows dirty.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from rummi.core.config import RummiConfig
from rummi.render.atlas import (
    BACKGROUND,
    INVALID_EDGE,
    NEW_EDGE,
    TOUCH_EDGE,
    PANEL,
    TEXT,
    TEXT_DIM,
    Atlas,
    Variant,
    build,
)
from rummi.render.view_model import GameView, SlotShape

PAD = 8
LABEL_W = 24
TAG_W = 74
STATUS_H = 44
LOG_H = 26
STRIP_H = 46
STRIP_LABEL_W = 76
DEFAULT_CAPACITY = 20
"""Slots the window can show at once, matching the peak measured in real games.
Overflow is never hidden silently -- the status line reports it."""

SHAPE_TAG = {SlotShape.RUN: "run", SlotShape.GROUP: "grp", SlotShape.PARTIAL: "part", SlotShape.BROKEN: "bad"}


@dataclass(slots=True)
class Layout:
    cfg: RummiConfig
    atlas: Atlas
    columns: int
    rows: int
    slot_w: int
    slot_h: int
    width: int
    height: int
    table_top: int
    workbench_top: int
    rack_top: int
    log_top: int

    @property
    def capacity(self) -> int:
        return self.columns * self.rows

    def row_rect(self, display_index: int):
        """Row-major fill, so both columns are used from the first set onwards
        rather than one column filling completely before the other starts."""
        import pygame

        row, col = divmod(display_index, self.columns)
        return pygame.Rect(
            PAD + col * (self.slot_w + PAD),
            self.table_top + row * self.slot_h,
            self.slot_w,
            self.slot_h,
        )


def layout_for(
    cfg: RummiConfig, atlas: Atlas, columns: int = 2, capacity: int | None = None
) -> Layout:
    capacity = min(cfg.max_sets, capacity or DEFAULT_CAPACITY)
    rows = -(-capacity // columns)
    slot_w = LABEL_W + cfg.max_set_len * atlas.tile_w + TAG_W
    slot_h = atlas.tile_h + 4
    table_top = STATUS_H
    workbench_top = table_top + rows * slot_h + PAD
    rack_top = workbench_top + STRIP_H
    log_top = rack_top + STRIP_H + 2
    return Layout(
        cfg=cfg,
        atlas=atlas,
        columns=columns,
        rows=rows,
        slot_w=slot_w,
        slot_h=slot_h,
        width=PAD + columns * (slot_w + PAD),
        height=log_top + LOG_H + PAD,
        table_top=table_top,
        workbench_top=workbench_top,
        rack_top=rack_top,
        log_top=log_top,
    )


@dataclass(slots=True)
class PygameView:
    """Either a live window (``headless=False``) or an offscreen surface."""

    cfg: RummiConfig
    headless: bool = False
    tile_w: int = 26
    tile_h: int = 36
    columns: int = 2
    capacity: int | None = None
    caption: str = "rummi"
    _atlas: Atlas | None = field(default=None, init=False)
    _layout: Layout | None = field(default=None, init=False)
    _surface: object = field(default=None, init=False)
    _font: object = field(default=None, init=False)
    _small: object = field(default=None, init=False)
    _row_hashes: dict[int, int] = field(default_factory=dict, init=False)
    _strip_hashes: dict[str, int] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if self.headless:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        import pygame

        pygame.init()
        self._atlas = build(self.cfg, self.tile_w, self.tile_h)
        self._layout = layout_for(self.cfg, self._atlas, self.columns, self.capacity)
        size = (self._layout.width, self._layout.height)
        if self.headless:
            self._surface = pygame.Surface(size)
        else:
            self._surface = pygame.display.set_mode(size)
            pygame.display.set_caption(self.caption)
        self._font = pygame.font.Font(None, 21)
        self._small = pygame.font.Font(None, 18)
        self._surface.fill(BACKGROUND)
        if not self.headless:
            pygame.display.flip()

    @property
    def size(self) -> tuple[int, int]:
        return (self._layout.width, self._layout.height)

    # --- drawing -------------------------------------------------------------
    def _text(self, text: str, pos, color=TEXT, small: bool = False) -> None:
        font = self._small if small else self._font
        self._surface.blit(font.render(text, True, color), pos)

    def _draw_row(self, slot, rect, touched: bool = False) -> None:
        """Draw one table slot, or clear the row when there is nothing to show."""
        import pygame

        a, s = self._atlas, self._surface
        s.fill(BACKGROUND, rect)
        if slot is None or slot.shape is SlotShape.EMPTY:
            if slot is not None:
                # The one empty slot on offer, drawn as a landing zone.
                pygame.draw.rect(s, PANEL, rect, width=1, border_radius=4)
                self._text(f"{slot.index}", (rect.x + 4, rect.y + rect.h // 2 - 8), TEXT_DIM, True)
            return

        pygame.draw.rect(s, PANEL, rect, border_radius=4)
        # Precedence: a broken set is the most urgent thing to show, then what the
        # last action just touched, then what was created this turn.
        if slot.blocks_end_turn:
            pygame.draw.rect(s, INVALID_EDGE, rect, width=2, border_radius=4)
        elif touched:
            pygame.draw.rect(s, TOUCH_EDGE, rect, width=2, border_radius=4)
        elif slot.is_new:
            pygame.draw.rect(s, NEW_EDGE, rect, width=2, border_radius=4)

        self._text(f"{slot.index}", (rect.x + 4, rect.y + rect.h // 2 - 8), TEXT_DIM, True)
        variant = Variant.NEW if slot.is_new else Variant.NORMAL
        for i, kind in enumerate(slot.tiles):
            s.blit(a.surface, (rect.x + LABEL_W + i * a.tile_w, rect.y + 2), a.rect(kind, variant))

        tag = SHAPE_TAG[slot.shape]
        note = f"{tag} {slot.value}" if slot.is_valid else tag.upper()
        color = TEXT_DIM if slot.is_valid else INVALID_EDGE
        self._text(note, (rect.right - TAG_W + 6, rect.y + rect.h // 2 - 8), color, True)

    def _draw_strip(self, label: str, kinds, top: int, variant: Variant, note: str = "") -> None:
        import pygame

        a, s = self._atlas, self._surface
        rect = pygame.Rect(PAD, top, self._layout.width - 2 * PAD, STRIP_H - 4)
        s.fill(BACKGROUND, rect)
        pygame.draw.rect(s, PANEL, rect, border_radius=4)
        self._text(label, (rect.x + 8, rect.y + rect.h // 2 - 8), TEXT_DIM, True)
        for i, kind in enumerate(kinds):
            x = rect.x + STRIP_LABEL_W + i * a.tile_w
            if x + a.tile_w > rect.right - 40:
                self._text("...", (x, rect.y + rect.h // 2 - 8), TEXT_DIM, True)
                break
            s.blit(a.surface, (x, rect.y + 2), a.rect(kind, variant))
        if note:
            self._text(note, (rect.right - 46, rect.y + rect.h // 2 - 8), TEXT_DIM, True)

    def _visible(self, view: GameView) -> tuple[list, int]:
        """Occupied slots plus the one empty slot ASSIGN can target, and how many
        did not fit. Overflow is returned rather than dropped so the caller can
        say so: a silently truncated table would read as a complete one."""
        shown = list(view.occupied_slots)
        empty = next((s for s in view.slots if s.shape is SlotShape.EMPTY), None)
        if empty is not None:
            shown.append(empty)
        capacity = self._layout.capacity
        return shown[:capacity], max(0, len(shown) - capacity)

    def draw(self, view: GameView) -> list:
        """Redraw only what changed; returns the dirty rects."""
        import pygame

        lay = self._layout
        dirty: list = []

        visible, hidden = self._visible(view)
        head = (
            f"turn {view.turn}   seat {view.current_player}/{view.cfg.n_players}   "
            f"pool {view.pool_size}   micro {view.micro}/{view.micro_budget}   "
            f"legal {view.n_legal}   last {view.last_action or '-'}"
        )
        seats = "  ".join(
            f"{'>' if i == view.current_player else ' '}p{i}:{n}{'*' if view.melded[i] else ''}"
            for i, n in enumerate(view.rack_sizes)
        )
        sub = f"seats {seats}    sets {len(view.occupied_slots)}"
        if hidden:
            sub += f"  (+{hidden} not shown)"
        if view.needs_meld:
            sub += f"    meld {view.meld_progress}/{view.cfg.initial_meld}"
        if not view.table_whole:
            sub += "    TABLE BROKEN"
        if view.done:
            sub += "    " + ("TRUNCATED" if view.truncated else f"SEAT {view.winner} WINS")

        if self._strip_hashes.get("status") != hash((head, sub)):
            rect = pygame.Rect(0, 0, lay.width, STATUS_H)
            self._surface.fill(BACKGROUND, rect)
            self._text(head, (PAD, 6))
            self._text(sub, (PAD, 25), TEXT_DIM, True)
            self._strip_hashes["status"] = hash((head, sub))
            dirty.append(rect)

        for i in range(lay.capacity):
            slot = visible[i] if i < len(visible) else None
            touched = slot is not None and slot.index == view.touched_slot
            key = hash(
                (slot.index, slot.tiles, slot.shape, slot.is_new, touched) if slot else None
            )
            if self._row_hashes.get(i) == key:
                continue
            rect = lay.row_rect(i)
            self._draw_row(slot, rect, touched)
            self._row_hashes[i] = key
            dirty.append(rect)

        for name, kinds, top, variant, note in (
            ("workbench", view.workbench, lay.workbench_top, Variant.GHOST, ""),
            ("rack", view.rack, lay.rack_top, Variant.NORMAL, f"({len(view.rack)})"),
        ):
            key = hash((kinds, note))
            if self._strip_hashes.get(name) == key:
                continue
            self._draw_strip(name, kinds, top, variant, note)
            self._strip_hashes[name] = key
            dirty.append(pygame.Rect(PAD, top, lay.width - 2 * PAD, STRIP_H - 4))

        log = " <- ".join(view.history[:6]) or "-"
        if self._strip_hashes.get("log") != hash(log):
            rect = pygame.Rect(0, lay.log_top, lay.width, LOG_H)
            self._surface.fill(BACKGROUND, rect)
            self._text("log", (PAD + 8, rect.y + 5), TEXT_DIM, True)
            self._text(view.history[0] if view.history else "-", (PAD + 44, rect.y + 5), TOUCH_EDGE, True)
            if len(view.history) > 1:
                tail = " <- ".join(view.history[1:6])
                offset = PAD + 52 + self._small.size(view.history[0])[0]
                self._text(tail, (offset, rect.y + 5), TEXT_DIM, True)
            self._strip_hashes["log"] = hash(log)
            dirty.append(rect)

        return dirty

    def render(self, view: GameView) -> None:
        import pygame

        dirty = self.draw(view)
        if not self.headless:
            if dirty:
                pygame.display.update(dirty)
            # Draining events keeps the window responsive rather than "not responding".
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt("render window closed")

    def rgb_array(self, view: GameView):
        import numpy as np
        import pygame

        self.draw(view)
        return np.transpose(pygame.surfarray.array3d(self._surface), (1, 0, 2))

    def close(self) -> None:
        import pygame

        pygame.quit()
