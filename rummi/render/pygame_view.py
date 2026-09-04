"""The pygame renderer: one window, laid out from the shared board geometry.

Watching a rollout and playing a hand are the same renderer with a different
overlay, so the board a player clicks is the board the env draws and there is no
second layout to keep in step.

Every frame is painted whole. Tracking dirty rects instead would mean rectangles
that never move, and a layout that cannot move its rectangles is a fixed grid --
the cost the board is laid out to avoid. What keeps a fast rollout cheap is the
throttle in ``driver.py``, which decides how often it may draw at all.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from rummi.render.atlas import (
    BACKGROUND,
    CARD,
    CARD_EDGE,
    DISABLED,
    DROP_EDGE,
    FELT,
    FELT_EDGE,
    INVALID_EDGE,
    NEW_EDGE,
    PANEL,
    RACK,
    RACK_EDGE,
    RACK_LEDGE,
    TEXT,
    TEXT_DIM,
    TOUCH_EDGE,
    TRAY,
    Atlas,
    Variant,
    build,
)
from rummi.render.board import (
    CARD_PAD,
    CAPTION_H,
    PAD,
    RACK_LIP_H,
    STATUS_H,
    rack_ledges,
    Card,
    Metrics,
    Regions,
    metrics_for,
    regions_for,
)
from rummi.render.view_model import GameView, SlotShape
from rummi.rules.config import RummiConfig

if TYPE_CHECKING:  # pygame is imported in __post_init__, not at module scope
    import pygame

SHAPE_TAG = {
    SlotShape.RUN: "run",
    SlotShape.GROUP: "group",
    SlotShape.PARTIAL: "PARTIAL",
    SlotShape.BROKEN: "BROKEN",
}


@dataclass(frozen=True, slots=True)
class Overlay:
    """What the interactive window adds on top of a plain board.

    Empty by default, which is exactly the read-only view: rendering a rollout
    and playing a hand differ only in this.
    """

    held: int = -1
    """Tile kind the player has picked up, or ``-1``."""
    drop_slots: frozenset[int] = frozenset()
    """Slots the held tile may go to. Taken from the mask, never inferred."""
    drag_pos: tuple[int, int] | None = None
    """Cursor position while dragging, where the held tile is drawn."""
    you: int = -1
    """Which seat the player is sitting in, so the window can say "you" rather
    than a seat number."""
    rival: str = ""
    can_end_turn: bool = False
    """Read off the mask by the caller, so the button and the engine cannot
    disagree about whether the turn may be committed."""
    can_undo: bool = False
    hint: str = ""


@dataclass(slots=True)
class PygameView:
    """Either a live window (``headless=False``) or an offscreen surface."""

    cfg: RummiConfig
    headless: bool = False
    tile_w: int = 26
    tile_h: int = 36
    interactive: bool = False
    """The play window: buttons, a status of chips, and no action log. Off, this is
    the view an env renders -- telemetry and a log, to read a rollout by."""
    caption: str = "rummi"
    # Non-optional and defaultless: __post_init__ sets every one unconditionally,
    # and typing them `| None` made each of the ~90 uses below a type error.
    _atlas: Atlas = field(init=False)
    _metrics: Metrics = field(init=False)
    _surface: pygame.Surface = field(init=False)
    _font: pygame.font.Font = field(init=False)
    _small: pygame.font.Font = field(init=False)
    _big: pygame.font.Font = field(init=False)

    def __post_init__(self) -> None:
        if self.headless:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        import pygame

        pygame.init()
        self._atlas = build(self.cfg, self.tile_w, self.tile_h)
        self._metrics = metrics_for(self.cfg, self.tile_w, self.tile_h, self.interactive)
        size = (self._metrics.width, self._metrics.height)
        if self.headless:
            self._surface = pygame.Surface(size)
        else:
            self._surface = pygame.display.set_mode(size)
            pygame.display.set_caption(self.caption)
        self._font = pygame.font.Font(None, 21)
        self._small = pygame.font.Font(None, 17)
        self._big = pygame.font.Font(None, 58)
        self._surface.fill(BACKGROUND)
        if not self.headless:
            pygame.display.flip()

    @property
    def metrics(self) -> Metrics:
        return self._metrics

    @property
    def size(self) -> tuple[int, int]:
        return (self._metrics.width, self._metrics.height)

    def regions(self, view: GameView) -> Regions:
        return regions_for(self._metrics, view)

    # --- drawing -------------------------------------------------------------
    def _text(self, text: str, pos, color=TEXT, small: bool = False) -> None:
        font = self._small if small else self._font
        self._surface.blit(font.render(text, True, color), pos)

    def _caption(self, rect, note: str, color, index: int) -> None:
        """The strip under a card's tiles.

        The play window shows what the set is worth and nothing else: run or group
        is plain from the tiles, and the slot number exists to cross-reference the
        log and the terminal view, neither of which a player has. The read-only
        view keeps both, and drops the number rather than let it overlap.
        """
        top = rect.bottom - CAPTION_H
        glyph = self._small.render(note, True, color)
        self._surface.blit(glyph, (rect.x + CARD_PAD, top))
        if self.interactive:
            return
        tag = self._small.render(f"#{index}", True, TEXT_DIM)
        if rect.w - 2 * CARD_PAD - glyph.get_width() >= tag.get_width() + 6:
            self._surface.blit(tag, (rect.right - CARD_PAD - tag.get_width(), top))

    def _dashed_rect(self, color, rect, dash: int = 7, gap: int = 5) -> None:
        """A dashed outline, for the empty slot. Solid would read as a set that
        happens to be empty; dashed reads as somewhere to put one."""
        import pygame

        for x in range(rect.left, rect.right, dash + gap):
            end = min(x + dash, rect.right)
            pygame.draw.line(self._surface, color, (x, rect.top), (end, rect.top))
            pygame.draw.line(self._surface, color, (x, rect.bottom - 1), (end, rect.bottom - 1))
        for y in range(rect.top, rect.bottom, dash + gap):
            end = min(y + dash, rect.bottom)
            pygame.draw.line(self._surface, color, (rect.left, y), (rect.left, end))
            pygame.draw.line(self._surface, color, (rect.right - 1, y), (rect.right - 1, end))

    def _card_edge(self, card: Card, view: GameView, overlay: Overlay):
        """Which ring a card gets, and none is a valid answer.

        While a tile is held the ring answers "where can this go?", so a legal
        drop outranks everything -- including a broken set, which is often
        precisely the set the held tile repairs. With nothing in hand it answers
        "what is wrong, and what just changed?" instead.
        """
        lit = card.slot in overlay.drop_slots
        if lit and overlay.held >= 0:
            return DROP_EDGE, 3
        if card.shape in (SlotShape.PARTIAL, SlotShape.BROKEN):
            return INVALID_EDGE, 2
        if card.slot == view.touched_slot:
            return TOUCH_EDGE, 2
        if view.slots[card.slot].is_new:
            return NEW_EDGE, 2
        return CARD_EDGE, 1

    def _draw_card(self, card: Card, view: GameView, overlay: Overlay) -> None:
        import pygame

        a, s = self._atlas, self._surface
        lit = card.slot in overlay.drop_slots and overlay.held >= 0

        if card.is_landing:
            self._dashed_rect(DROP_EDGE if lit else PANEL, card.rect)
            label = "new set" if lit else "empty"
            glyph = self._small.render(label, True, DROP_EDGE if lit else TEXT_DIM)
            s.blit(glyph, glyph.get_rect(center=(card.rect.centerx, card.rect.centery - 4)))
            self._caption(card.rect, "", TEXT_DIM, card.slot)
            return

        pygame.draw.rect(s, CARD, card.rect, border_radius=5)
        color, width = self._card_edge(card, view, overlay)
        pygame.draw.rect(s, color, card.rect, width=width, border_radius=5)

        slot = view.slots[card.slot]
        variant = Variant.NEW if slot.is_new else Variant.NORMAL
        # Drawn from the spots, which carry `shown` order: a joker sits in the gap
        # it fills, while the position each spot reports is the stored one PICK
        # indexes. Keeping both on the spot is what lets a click lift the tile the
        # player actually pointed at.
        for spot in card.spots:
            s.blit(a.surface, spot.rect.topleft, a.rect(spot.kind, variant))

        tag = SHAPE_TAG[card.shape]
        if not slot.is_valid:
            note = tag
        else:
            note = str(slot.value) if self.interactive else f"{tag} {slot.value}"
        self._caption(
            card.rect, note, TEXT_DIM if slot.is_valid else INVALID_EDGE, card.slot
        )

    def _wrapped(self, text: str, width: int) -> list[str]:
        lines, line = [], ""
        for word in text.split():
            trial = f"{line} {word}".strip()
            if line and self._small.size(trial)[0] > width:
                lines.append(line)
                line = word
            else:
                line = trial
        return [*lines, line] if line else lines

    def _draw_hint(self, rect, text: str) -> None:
        """The margin opposite the buttons: what to do next, in one sentence, where
        there is room for it without a bar of its own."""
        lines = self._wrapped(text, rect.w)
        top = rect.centery - len(lines) * 9
        for i, line in enumerate(lines):
            self._text(line, (rect.x, top + i * 18), TEXT_DIM, True)

    def _draw_rack(self, rect, spots) -> None:
        """The rack, drawn as the object it is: tiles standing on tiers.

        Each ledge goes down after the tiles that stand in it, so it covers their
        feet -- that overlap is what makes a rack out of a row of tiles. The back
        tier is drawn empty when the hand fits on the front one, the way a real
        rack looks.
        """
        import pygame

        a, s = self._atlas, self._surface
        pygame.draw.rect(s, RACK, rect, border_radius=7)
        for spot in spots:
            s.blit(a.surface, spot.rect.topleft, a.rect(spot.kind, Variant.NORMAL))
        for ledge in rack_ledges(rect, self.tile_h):
            last = ledge.bottom >= rect.bottom - RACK_LIP_H
            pygame.draw.rect(
                s,
                RACK_LEDGE,
                ledge,
                border_bottom_left_radius=7 if last else 0,
                border_bottom_right_radius=7 if last else 0,
            )
            pygame.draw.line(s, RACK_EDGE, (ledge.x + 2, ledge.y), (ledge.right - 3, ledge.y))
        pygame.draw.rect(s, RACK_EDGE, rect, width=1, border_radius=7)

    def _draw_tray(self, rect, label: str, spots, ring: int = -1) -> None:
        import pygame

        a, s = self._atlas, self._surface
        pygame.draw.rect(s, TRAY, rect, border_radius=5)
        pygame.draw.rect(s, PANEL, rect, width=1, border_radius=5)
        glyph = self._small.render(label, True, DISABLED)
        s.blit(glyph, (rect.x + 10, rect.centery - glyph.get_height() // 2))
        for spot in spots:
            s.blit(a.surface, spot.rect.topleft, a.rect(spot.kind, Variant.GHOST))
        if ring >= 0:
            # Ring the tile in hand. Without it the hint line is the only clue
            # about what was picked up, which is unreadable once the workbench
            # holds several tiles.
            for spot in spots:
                if spot.kind == ring:
                    pygame.draw.rect(
                        s, DROP_EDGE, spot.rect.inflate(4, 4), width=3, border_radius=5
                    )
                    break

    def _draw_status(self, view: GameView) -> None:
        """The read-only view's status: everything about the step, in one line.

        This is the view an env renders, so it is read while a rollout runs and
        the telemetry is the point of it -- which is exactly why the play window
        does not use it.
        """
        # `action_mask=False` leaves no mask to count, which `view` reports as -1.
        legal = f"legal {view.n_legal}   " if view.n_legal >= 0 else ""
        head = (
            f"turn {view.turn}   seat {view.current_player}/{view.cfg.n_players}   "
            f"pool {view.pool_size}   micro {view.micro}/{view.micro_budget}   "
            f"{legal}last {view.last_action or '-'}"
        )
        seats = "  ".join(
            f"{'>' if i == view.current_player else ' '}p{i}:{n}{'*' if view.melded[i] else ''}"
            for i, n in enumerate(view.rack_sizes)
        )
        sub = f"seats {seats}    sets {len(view.occupied_slots)}"
        if view.needs_meld:
            sub += f"    meld {view.meld_progress}/{view.cfg.initial_meld}"
        if not view.table_whole:
            sub += "    TABLE BROKEN"
        if view.done:
            sub += "    " + ("TRUNCATED" if view.truncated else f"SEAT {view.winner} WINS")
        self._text(head, (PAD, 7))
        self._text(sub, (PAD, 27), TEXT_DIM, True)

    def _pill(self, rect, on: bool) -> None:
        import pygame

        pygame.draw.rect(self._surface, CARD if on else BACKGROUND, rect, border_radius=rect.h // 2)
        pygame.draw.rect(
            self._surface,
            DROP_EDGE if on else PANEL,
            rect,
            width=2 if on else 1,
            border_radius=rect.h // 2,
        )

    def _seat_pills(self, view: GameView, overlay: Overlay, top: int, h: int) -> None:
        """Who is on, and how much is left to play. A pill each, the acting one lit."""
        import pygame

        x = PAD
        for seat, size in enumerate(view.rack_sizes):
            name = "you" if seat == overlay.you else (overlay.rival or f"seat {seat}")
            on = seat == view.current_player
            label = self._small.render(name, True, TEXT if on else TEXT_DIM)
            count = self._font.render(str(size), True, TEXT if on else TEXT_DIM)
            rect = pygame.Rect(x, top, 34 + label.get_width() + count.get_width(), h)
            self._pill(rect, on)
            self._surface.blit(label, (rect.x + 16, rect.centery - label.get_height() // 2))
            self._surface.blit(
                count, (rect.right - 14 - count.get_width(), rect.centery - count.get_height() // 2)
            )
            if view.melded[seat]:
                # Opened. Until a seat has, nothing it does can touch the table.
                pygame.draw.circle(self._surface, NEW_EDGE, (rect.x + 9, rect.centery), 3)
            x = rect.right + 8

    def _pool_pill(self, view: GameView, right: int, top: int, h: int) -> int:
        """The pool drawn as the stack it is. Returns its left edge."""
        import pygame

        count = self._font.render(str(view.pool_size), True, TEXT_DIM)
        rect = pygame.Rect(right - (46 + count.get_width()), top, 46 + count.get_width(), h)
        self._pill(rect, False)
        for i in range(3):
            face = pygame.Rect(rect.x + 12 + i * 3, rect.centery - 7 + i, 10, 13)
            pygame.draw.rect(self._surface, DISABLED, face, border_radius=2)
            pygame.draw.rect(self._surface, PANEL, face, width=1, border_radius=2)
        self._surface.blit(
            count, (rect.right - 14 - count.get_width(), rect.centery - count.get_height() // 2)
        )
        return rect.x

    def _meld_bar(self, view: GameView, right: int, top: int, h: int) -> None:
        """How close the opening meld is, as a bar.

        The one rule a newcomer trips on, and the one number that changes what is
        legal, so it gets a shape rather than a fraction buried in a status line.
        """
        import pygame

        need = view.cfg.initial_meld
        got = min(view.meld_progress, need)
        bar = pygame.Rect(right - 128, top + h // 2 - 6, 128, 12)
        label = self._small.render("meld", True, TEXT_DIM)
        self._surface.blit(label, (bar.x - 8 - label.get_width(), bar.centery - label.get_height() // 2))
        pygame.draw.rect(self._surface, PANEL, bar, border_radius=6)
        if got:
            filled = pygame.Rect(bar.x, bar.y, max(6, bar.w * got // need), bar.h)
            pygame.draw.rect(
                self._surface, NEW_EDGE if got >= need else TOUCH_EDGE, filled, border_radius=6
            )
        text = self._small.render(f"{got}/{need}", True, TEXT)
        self._surface.blit(text, text.get_rect(center=bar.center))

    def _draw_hud(self, view: GameView, overlay: Overlay) -> None:
        """The play window's status: what a player acts on, and nothing else."""
        top, h = 9, STATUS_H - 20
        self._seat_pills(view, overlay, top, h)
        left = self._pool_pill(view, self._metrics.table.right, top, h)
        if view.needs_meld and not view.done:
            self._meld_bar(view, left - 14, top, h)

    def _draw_banner(self, view: GameView, overlay: Overlay) -> None:
        """The result, across the middle of the table.

        The board is dimmed behind it: a finished game should stop the eye, not be
        deduced from a rack that quietly stopped changing.
        """
        import pygame

        table = self._metrics.table
        scrim = pygame.Surface(table.size, pygame.SRCALPHA)
        scrim.fill((10, 14, 12, 190))
        self._surface.blit(scrim, table.topleft)

        if view.truncated:
            head, tail = "OUT OF TURNS", "nobody went out"
        elif view.winner == overlay.you:
            head, tail = "YOU WIN", "rack empty"
        else:
            name = overlay.rival or f"seat {view.winner}"
            head, tail = f"{name.upper()} WINS", f"you were left holding {len(view.rack)}"
        title = self._big.render(head, True, TEXT)
        note = self._font.render(tail, True, TEXT_DIM)

        panel = pygame.Rect(0, 0, max(title.get_width(), note.get_width()) + 110, 132)
        panel.center = table.center
        pygame.draw.rect(self._surface, CARD, panel, border_radius=12)
        pygame.draw.rect(self._surface, TOUCH_EDGE, panel, width=3, border_radius=12)
        self._surface.blit(title, title.get_rect(center=(panel.centerx, panel.centery - 18)))
        self._surface.blit(note, note.get_rect(center=(panel.centerx, panel.centery + 32)))

    def _draw_log(self, view: GameView) -> None:
        rect = self._metrics.log
        self._text("log", (rect.x + 2, rect.y + 4), TEXT_DIM, True)
        recent = view.history[0] if view.history else "-"
        self._text(recent, (rect.x + 36, rect.y + 4), TOUCH_EDGE, True)
        if len(view.history) > 1:
            offset = rect.x + 44 + self._small.size(recent)[0]
            self._text(" <- ".join(view.history[1:6]), (offset, rect.y + 4), TEXT_DIM, True)

    def _draw_controls(self, regions: Regions, view: GameView, overlay: Overlay) -> None:
        import pygame

        for rect, label, enabled in (
            (regions.end_turn, "END TURN", overlay.can_end_turn),
            (regions.undo, "UNDO", overlay.can_undo),
            (regions.draw, "DRAW", not view.done),
        ):
            pygame.draw.rect(self._surface, CARD if enabled else BACKGROUND, rect, border_radius=6)
            pygame.draw.rect(
                self._surface, DROP_EDGE if enabled else DISABLED, rect, width=2, border_radius=6
            )
            glyph = self._font.render(label, True, TEXT if enabled else DISABLED)
            self._surface.blit(glyph, glyph.get_rect(center=rect.center))

    def _draw_drag(self, overlay: Overlay) -> None:
        """The held tile under the cursor. Drawn last, over everything."""
        import pygame

        a, s = self._atlas, self._surface
        x = overlay.drag_pos[0] - self.tile_w // 2
        y = overlay.drag_pos[1] - self.tile_h // 2
        pygame.draw.rect(
            s, (12, 13, 16), (x + 3, y + 4, self.tile_w, self.tile_h), border_radius=4
        )
        s.blit(a.surface, (x, y), a.rect(overlay.held, Variant.NORMAL))

    def draw(self, view: GameView, overlay: Overlay | None = None) -> Regions:
        """Paint a whole frame and return what is clickable in it."""
        import pygame

        overlay = overlay or Overlay()
        m = self._metrics
        regions = regions_for(m, view)

        self._surface.fill(BACKGROUND)
        if self.interactive:
            self._draw_hud(view, overlay)
        else:
            self._draw_status(view)
        pygame.draw.rect(self._surface, FELT, m.table, border_radius=8)
        pygame.draw.rect(self._surface, FELT_EDGE, m.table, width=1, border_radius=8)
        for card in regions.cards:
            self._draw_card(card, view, overlay)
        if view.done and self.interactive:
            self._draw_banner(view, overlay)
        self._draw_tray(regions.workbench_tray, "workbench", regions.workbench, overlay.held)
        self._draw_rack(regions.rack_tray, regions.rack)
        if m.log is not None:
            self._draw_log(view)
        if m.controls is not None:
            self._draw_controls(regions, view, overlay)
        if m.hint is not None and overlay.hint:
            self._draw_hint(m.hint, overlay.hint)
        if overlay.drag_pos is not None and overlay.held >= 0:
            self._draw_drag(overlay)
        return regions

    def flip(self) -> None:
        """Present the drawn frame. Separate from ``draw`` so an interactive
        caller can own the event queue instead of having frames drain it."""
        import pygame

        if not self.headless:
            pygame.display.flip()

    def render(self, view: GameView, overlay: Overlay | None = None) -> Regions:
        """Draw and present, for a caller that is not reading events itself --
        a rollout, where an undrained queue shows up as "not responding"."""
        import pygame

        regions = self.draw(view, overlay)
        self.flip()
        if not self.headless and pygame.event.get(pygame.QUIT):
            raise KeyboardInterrupt("render window closed")
        return regions

    def rgb_array(self, view: GameView, overlay: Overlay | None = None):
        import numpy as np
        import pygame

        self.draw(view, overlay)
        return np.transpose(pygame.surfarray.array3d(self._surface), (1, 0, 2))

    def close(self) -> None:
        """Give back what this view owns, and nothing else.

        A live window owns the display it opened, so that goes; an offscreen view
        owns only surfaces, which are the garbage collector's business. Shutting
        pygame down instead would take the fonts and the display out from under
        every other view in the process -- a renderer closing its window is not a
        reason for the next one to find nothing to draw on.
        """
        import pygame

        if not self.headless:
            pygame.display.quit()
