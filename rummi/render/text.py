"""Terminal rendering: a static frame, and a live in-place view.

The static frame is what gets printed inside a fuzz or test failure, so it must
work with colour stripped and no cursor control at all. The live view is the same
frame redrawn in place, rewriting only the lines that changed -- which is what
keeps it usable while a rollout runs at thousands of steps a second.
"""

from __future__ import annotations

import os
import sys
from typing import TextIO

from rummi.core.encoding import color_letter, tables
from rummi.render.view_model import GameView, SlotShape

TILE_WIDTH = 4

# Truecolour faces, chosen to stay legible on both light and dark terminals.
COLOR_RGB: tuple[tuple[int, int, int], ...] = (
    (216, 62, 62),    # R
    (58, 122, 224),   # B
    (206, 158, 32),   # Y
    (74, 80, 96),     # K -- dark slate: true black vanishes, mid grey reads as
                      # disabled text against the dim UI chrome
    (72, 168, 108),
    (176, 96, 200),
    (64, 176, 190),
    (200, 120, 70),
)
JOKER_RGB = (196, 108, 220)
TOUCH_RGB = (226, 178, 72)
DIM = "\x1b[2m"
BOLD = "\x1b[1m"
RESET = "\x1b[0m"
WARN_RGB = (226, 96, 96)
NEW_RGB = (120, 200, 130)


def supports_color(stream: TextIO) -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return bool(getattr(stream, "isatty", lambda: False)())


def _fg(rgb: tuple[int, int, int]) -> str:
    r, g, b = rgb
    return f"\x1b[38;2;{r};{g};{b}m"


class Palette:
    """Colour helpers that collapse to plain text when colour is off."""

    def __init__(self, color: bool) -> None:
        self.color = color

    def paint(self, text: str, rgb: tuple[int, int, int], bold: bool = False) -> str:
        if not self.color:
            return text
        return f"{BOLD if bold else ''}{_fg(rgb)}{text}{RESET}"

    def dim(self, text: str) -> str:
        return f"{DIM}{text}{RESET}" if self.color else text


def tile(view: GameView, kind: int, palette: Palette) -> str:
    label = view.label(kind).rjust(TILE_WIDTH)
    if kind == view.cfg.joker_kind:
        return palette.paint(label, JOKER_RGB, bold=True)
    color = int(tables(view.cfg).color[kind])
    return palette.paint(label, COLOR_RGB[color % len(COLOR_RGB)])


def _tiles(view: GameView, kinds, palette: Palette) -> str:
    return "".join(tile(view, k, palette) for k in kinds)


_SHAPE_TAG = {
    SlotShape.RUN: "run",
    SlotShape.GROUP: "grp",
    SlotShape.PARTIAL: "PARTIAL",
    SlotShape.BROKEN: "BROKEN",
}


def frame(view: GameView, palette: Palette | None = None, width: int = 78) -> str:
    """Render one snapshot as a block of text."""
    p = palette or Palette(False)
    cfg = view.cfg
    rule = p.dim("-" * width)
    lines: list[str] = []

    status = (
        f"turn {view.turn}   seat {view.current_player}/{cfg.n_players}   "
        f"pool {view.pool_size}   micro {view.micro}/{view.micro_budget}"
    )
    if view.done:
        outcome = "TRUNCATED" if view.truncated else f"seat {view.winner} wins"
        status += "   " + p.paint(outcome, WARN_RGB, bold=True)
    lines.append(status)
    lines.append(rule)

    occupied = view.occupied_slots
    n_tiles = sum(len(s.tiles) for s in occupied)
    lines.append(p.dim(f" table  {len(occupied)} sets, {n_tiles} tiles"))
    if not occupied:
        lines.append(p.dim("   (empty)"))
    for slot in occupied:
        body = _tiles(view, slot.tiles, p).ljust(
            TILE_WIDTH * min(cfg.max_set_len, 7) + (0 if not p.color else 0)
        )
        tag = _SHAPE_TAG[slot.shape]
        note = f"{tag:<8}"
        if slot.is_valid:
            note += f"{slot.value:>3}"
        else:
            note = p.paint(f"{tag:<8}", WARN_RGB, bold=True) + "  <- blocks END_TURN"
        if slot.index == view.touched_slot:
            mark = p.paint(">", TOUCH_RGB, bold=True)
        elif slot.is_new:
            mark = p.paint("+", NEW_RGB, bold=True)
        else:
            mark = " "
        lines.append(f"  {slot.index:>2}{mark}{body}  {note}")

    lines.append(rule)
    if view.workbench:
        lines.append(" workbench" + _tiles(view, view.workbench, p))
    if view.needs_meld:
        lines.append(
            p.dim(f" meld     {view.meld_progress}/{cfg.initial_meld} needed to open")
        )
    lines.append(f" rack     {_tiles(view, view.rack, p)}  ({len(view.rack)})")

    seats = []
    for i, size in enumerate(view.rack_sizes):
        arrow = ">" if i == view.current_player else " "
        flag = "*" if view.melded[i] else "-"
        seats.append(f"{arrow}p{i}:{size}{flag}")
    lines.append(p.dim(" seats    " + "  ".join(seats) + "   (* = melded)"))

    lines.append(f" last     {p.paint(view.last_action or '-', TOUCH_RGB, bold=True)}")
    if view.history:
        log = p.dim(" <- ").join(view.history[1:6])
        lines.append(p.dim(" before   ") + log if log else p.dim(" before   -"))
    if view.n_legal >= 0:
        lines.append(p.dim(f" legal    {view.n_legal}/{cfg.n_actions} actions"))
    return "\n".join(lines)


class TerminalView:
    """Live in-place terminal rendering.

    Falls back to append-only output when the stream is not a TTY, so piping to a
    file or a CI log produces a readable sequence of frames instead of a mess of
    cursor escapes.
    """

    def __init__(
        self,
        stream: TextIO | None = None,
        color: bool | None = None,
        live: bool | None = None,
    ) -> None:
        self.stream = stream or sys.stdout
        self.color = supports_color(self.stream) if color is None else color
        self.live = self.color if live is None else live
        self.palette = Palette(self.color)
        self._previous: list[str] = []

    def frame(self, view: GameView) -> str:
        return frame(view, self.palette)

    def render(self, view: GameView) -> None:
        lines = self.frame(view).split("\n")
        if not self.live:
            self.stream.write("\n".join(lines) + "\n\n")
            self.stream.flush()
            return

        out: list[str] = []
        if self._previous:
            out.append(f"\x1b[{len(self._previous)}A")
        for i, line in enumerate(lines):
            # Only rewrite what changed; unchanged lines cost one cursor-down.
            if i < len(self._previous) and self._previous[i] == line:
                out.append("\x1b[1B")
            else:
                out.append("\r\x1b[2K" + line + "\n" if i == len(lines) - 1 else "\r\x1b[2K" + line + "\x1b[1B\r")
        # Clear any lines the previous, taller frame left behind.
        for _ in range(max(0, len(self._previous) - len(lines))):
            out.append("\r\x1b[2K\x1b[1B")
        self.stream.write("".join(out))
        self.stream.flush()
        self._previous = lines

    def close(self) -> None:
        if self.live and self._previous:
            self.stream.write("\n")
            self.stream.flush()
        self._previous = []
