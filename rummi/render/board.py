"""Where everything sits on screen, as plain rectangles.

Both pygame paths -- the read-only renderer and the interactive window -- lay out
from here, so the rectangle a set is drawn in is the rectangle a click resolves
against. Nothing in this module touches a surface or a display, which is what
lets a whole hand be played headlessly in the tests.

Sets are **cards sized to their contents**, flowed across the table: a card is
only as wide as the set inside it, and every occupied slot gets one. A grid of
fixed rows is the other way to lay this out and it costs twice -- each row has to
reserve the worst case, a 13-tile run, while real sets are four or five tiles, and
a fixed row count runs out on a busy table, leaving slots with no rectangle at
all. A slot with no rectangle cannot be clicked.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from rummi.render.view_model import GameView, SlotShape
from rummi.rules.config import RummiConfig

if TYPE_CHECKING:
    from pygame import Rect

PAD = 10
CARD_PAD = 6
CARD_GAP = 10
CAPTION_H = 15
STATUS_H = 46
LOG_H = 24
BUTTON_W, BUTTON_H, BUTTON_GAP = 128, 34, 10
MIN_STRIDE = 9
"""Narrowest a tray tile may be fanned to. Overlapping keeps every tile inside
the tray -- and so keeps it clickable -- however many tiles are held."""
RACK_PAD = 12
RACK_LIP_H = 10
"""The ledge the tiles stand in, drawn over their feet. It is what makes a row of
tiles read as a rack you are holding rather than a row of tiles on a strip."""
RACK_SLACK = 5
"""Tiles of headroom past a full rack, so playing one does not resize the rack
under the pointer."""
RACK_SEAT = 6
"""How deep a tile stands in the ledge. The ledge is drawn over its foot, and that
overlap is the whole difference between a rack and a row of tiles on a strip."""
RACK_TOP = 6
RACK_TIERS = 2
"""A real rack has two tiers whether or not the hand needs both, so the back one is
drawn empty rather than the rack changing height as tiles come and go. Two tiers of
tiles at full width hold more than a player can ever be dealt into."""
TARGET_ROWS = 5
"""Rows of set cards to aim for when choosing the window width. How many rows are
then actually reserved is not a guess -- see :func:`rows_needed`."""


class Zone(str, Enum):
    RACK = "rack"
    WORKBENCH = "workbench"
    SLOT = "slot"
    END_TURN = "end_turn"
    UNDO = "undo"
    DRAW = "draw"


@dataclass(frozen=True, slots=True)
class TileSpot:
    rect: Rect
    kind: int
    pos: int = -1
    """Index in the *stored* order for a table tile, which is what ``PICK`` takes."""


@dataclass(frozen=True, slots=True)
class Card:
    rect: Rect
    slot: int
    tiles: tuple[int, ...]
    shape: SlotShape
    spots: tuple[TileSpot, ...]
    """One per drawn tile, in reading order."""

    @property
    def is_landing(self) -> bool:
        """The single empty slot on offer -- where a new set starts."""
        return self.shape is SlotShape.EMPTY


@dataclass(frozen=True, slots=True)
class Hit:
    zone: Zone
    slot: int = -1
    kind: int = -1
    pos: int = -1
    """Stored position of the tile pointed at, or ``-1`` for a click that landed
    on a card but not on one of its tiles."""
    tiles: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class Regions:
    """Everything clickable in one frame, in the coordinates it was drawn at."""

    cards: tuple[Card, ...] = ()
    rack: tuple[TileSpot, ...] = ()
    workbench: tuple[TileSpot, ...] = ()
    rack_tray: Rect | None = None
    """The rack as drawn. Wider than the fixed one when a hand this big could not
    otherwise keep every tile inside it, so drawing and hit-testing agree."""
    workbench_tray: Rect | None = None
    end_turn: Rect | None = None
    undo: Rect | None = None
    draw: Rect | None = None


@dataclass(frozen=True, slots=True)
class Metrics:
    """Fixed geometry: the areas, and the window they add up to.

    Fixed because the window cannot resize mid-game, so every area has to be big
    enough for the worst state the config allows.
    """

    cfg: RummiConfig
    tile_w: int
    tile_h: int
    width: int
    height: int
    table: Rect
    workbench: Rect
    rack: Rect
    log: Rect | None
    """Dropped in the interactive window: what the opponent just did is told by the
    ring on the card it touched and by the pace it moves at, not by a line of
    action ids."""
    controls: Rect | None
    """Column beside the rack, not a bar under it. Centring the rack leaves a wide
    margin either side, and the buttons belong where the hand already is."""
    hint: Rect | None
    """The other margin, opposite the buttons."""

    @property
    def tier_h(self) -> int:
        return self.tile_h + RACK_LIP_H

    @property
    def card_h(self) -> int:
        return 2 * CARD_PAD + self.tile_h + CAPTION_H


def card_width(n_tiles: int, tile_w: int) -> int:
    return 2 * CARD_PAD + max(1, n_tiles) * tile_w


def rows_needed(cfg: RummiConfig, tile_w: int, usable: int) -> int:
    """Rows to reserve so that no legal table can overflow the area.

    Sized from the config rather than from what games do: a card pushed out of the
    table is a set with no rectangle, and a set with no rectangle cannot be clicked.

    The bound sweeps *uniform* tables -- every set the same size -- because uniform
    widths are the worst case for greedy wrapping. A narrower card mixed in only
    ever fills slack that the uniform table wastes at the end of a row.
    """
    worst = 1
    for size in range(1, cfg.max_set_len + 1):
        width = card_width(max(size, cfg.min_set), tile_w) + CARD_GAP
        per_row = max(1, (usable - CARD_GAP) // width)
        cards = min(cfg.max_sets, max(1, cfg.n_tiles // size)) + 1  # + the landing card
        worst = max(worst, -(-cards // per_row))
    return worst


def metrics_for(
    cfg: RummiConfig, tile_w: int = 26, tile_h: int = 36, interactive: bool = False
) -> Metrics:
    """``interactive`` is the play window: it reserves the button bar and drops the
    action log, where the read-only view keeps the log and the telemetry with it."""
    import pygame

    widest = card_width(cfg.max_set_len, tile_w)
    # Worst case the config permits: every slot occupied, every tile in the deck
    # on the table, plus the landing card. Greedy wrapping can waste up to one
    # card's width per row, so a row is only guaranteed to carry the rest.
    tiles = max(cfg.n_tiles, cfg.max_sets * cfg.min_set)
    content = cfg.max_sets * (2 * CARD_PAD + CARD_GAP) + tiles * tile_w + widest
    # The button bar sets a floor on width whether or not it is drawn: without it
    # a reduced config produces a window narrow enough that the table needs every
    # one of TARGET_ROWS for three sets.
    floor = 3 * BUTTON_W + 2 * BUTTON_GAP + 2 * PAD
    usable = max(floor, widest, -(-content // TARGET_ROWS) + widest)
    rows = rows_needed(cfg, tile_w, usable)

    card_h = 2 * CARD_PAD + tile_h + CAPTION_H
    # Centred and sized to a full rack rather than spanning the window: the rack is
    # the one object on screen that is yours, and a strip pinned to both edges
    # reads as another panel of the interface.
    tray_w = min(usable, 2 * RACK_PAD + (cfg.rack_size + RACK_SLACK) * tile_w)
    tray_x = PAD + (usable - tray_w) // 2
    table = pygame.Rect(PAD, STATUS_H, usable, rows * (card_h + CARD_GAP) + CARD_GAP)
    # Half the rack: the workbench holds what you have picked up, which is one or
    # two tiles almost always. It widens when it has to, in ``tray_rect``.
    bench_w = max(2 * RACK_PAD + 8 * tile_w, tray_w // 2)
    workbench = pygame.Rect(PAD + (usable - bench_w) // 2, table.bottom + PAD, bench_w, tile_h + 12)
    rack = pygame.Rect(
        tray_x, workbench.bottom + PAD, tray_w, RACK_TOP + RACK_TIERS * (tile_h + RACK_LIP_H) + 8
    )
    log = None if interactive else pygame.Rect(PAD, rack.bottom + PAD, usable, LOG_H)
    # The margins the centred trays leave, spanning both of them.
    top, height = workbench.y, rack.bottom - workbench.y
    margin = max(BUTTON_W, (usable - tray_w) // 2) - PAD
    bar = pygame.Rect(rack.right + PAD, top, margin, height) if interactive else None
    hint = pygame.Rect(PAD, top, margin, height) if interactive else None
    return Metrics(
        cfg=cfg,
        tile_w=tile_w,
        tile_h=tile_h,
        width=usable + 2 * PAD,
        height=(log or rack).bottom + PAD,
        table=table,
        workbench=workbench,
        rack=rack,
        log=log,
        controls=bar,
        hint=hint,
    )


def storage_order(tiles: tuple[int, ...], shown: tuple[int, ...]) -> tuple[int, ...]:
    """Stored position of each *displayed* tile.

    ``shown`` moves a joker into the gap it fills, while ``PICK`` indexes
    ``tiles``, so lifting the tile a player actually pointed at means translating
    back. Two tiles of the same kind are interchangeable, which is what makes
    first-unused exact rather than a guess.
    """
    used = [False] * len(tiles)
    out: list[int] = []
    for kind in shown:
        for i, stored in enumerate(tiles):
            if not used[i] and stored == kind:
                used[i] = True
                out.append(i)
                break
        else:
            out.append(-1)
    return tuple(out)


def _stride(n: int, room: int, tile_w: int) -> int:
    """How far apart tiles sit. They fan -- overlap -- rather than run off the end,
    because a tile with no rectangle cannot be picked up, and a rack that quietly
    stops drawing at its edge reads as a shorter rack."""
    if n < 2 or n * tile_w <= room:
        return tile_w
    return max(MIN_STRIDE, (room - tile_w) // (n - 1))


def tray_rect(m: Metrics, base: Rect, n: int) -> Rect:
    """The tray as drawn: the fixed one, widened only when even a full fan could
    not hold ``n`` tiles. Growing is the honest failure -- the alternative is tiles
    drawn past the edge, which is the same as tiles that cannot be picked up."""
    import pygame

    need = (n - 1) * MIN_STRIDE + m.tile_w if n else 0
    if need <= base.w - 2 * RACK_PAD:
        return base
    w = min(m.table.w, need + 2 * RACK_PAD)
    return pygame.Rect(m.table.x + (m.table.w - w) // 2, base.y, w, base.h)


def rack_ledges(rect: Rect, tile_h: int) -> tuple[Rect, ...]:
    """The ledge of each tier, drawn over the feet of the tiles standing in it."""
    import pygame

    pitch = tile_h + RACK_LIP_H
    return tuple(
        pygame.Rect(rect.x, rect.y + RACK_TOP + (t + 1) * pitch - RACK_LIP_H, rect.w, RACK_LIP_H)
        for t in range(RACK_TIERS)
    )


def _rack_spots(
    rect: Rect, kinds: tuple[int, ...], tile_w: int, tile_h: int
) -> tuple[TileSpot, ...]:
    """Seat a hand on the rack's tiers.

    One tier while the hand fits on one, and the front one, since that is the tier
    a hand of fourteen sits on. Past that it splits in half and reads the way it is
    written: left to right along the back tier, then along the front.
    """
    if not kinds:
        return ()
    ledges = rack_ledges(rect, tile_h)
    per_tier = max(1, (rect.w - 2 * RACK_PAD) // tile_w)
    if len(kinds) <= per_tier:
        tiers = [(ledges[-1], kinds, 0)]
    else:
        head = -(-len(kinds) // 2)
        tiers = [(ledges[0], kinds[:head], 0), (ledges[-1], kinds[head:], head)]

    out: list[TileSpot] = []
    for ledge, chunk, offset in tiers:
        seated = _tray_spots(rect, chunk, tile_w, tile_h, ledge.y + RACK_SEAT)
        out.extend(TileSpot(spot.rect, spot.kind, offset + i) for i, spot in enumerate(seated))
    return tuple(out)


def _tray_spots(
    rect: Rect, kinds: tuple[int, ...], tile_w: int, tile_h: int, baseline: int | None = None
) -> tuple[TileSpot, ...]:
    """Seat tiles in a tray, centred across it, their feet on ``baseline``."""
    import pygame

    if not kinds:
        return ()
    stride = _stride(len(kinds), rect.w - 2 * RACK_PAD, tile_w)
    span = (len(kinds) - 1) * stride + tile_w
    left = rect.x + (rect.w - span) // 2
    top = (baseline if baseline is not None else rect.centery + tile_h // 2) - tile_h
    return tuple(
        TileSpot(pygame.Rect(left + i * stride, top, tile_w, tile_h), kind, i)
        for i, kind in enumerate(kinds)
    )


def cards_for(m: Metrics, view: GameView) -> tuple[Card, ...]:
    """Flow the occupied sets across the table, plus one landing card.

    Slot order, so a card keeps its place for as long as the slots do: within a
    turn slot identity is stable, and a multi-step plan aimed at a card that
    moved under it would be aiming at the wrong set.
    """
    import pygame

    shown = list(view.occupied_slots)
    landing = next((s for s in view.slots if s.shape is SlotShape.EMPTY), None)
    if landing is not None:
        shown.append(landing)

    out: list[Card] = []
    x, y = m.table.x + CARD_GAP, m.table.y + CARD_GAP
    for slot in shown:
        # Never narrower than the smallest legal set, so a card mid-rearrangement
        # still has room for its caption -- and so the empty slot looks like
        # somewhere a set could go.
        w = card_width(max(len(slot.shown), m.cfg.min_set), m.tile_w)
        if x > m.table.x + CARD_GAP and x + w > m.table.right - CARD_GAP:
            x, y = m.table.x + CARD_GAP, y + m.card_h + CARD_GAP
        rect = pygame.Rect(x, y, w, m.card_h)
        spots = tuple(
            TileSpot(
                pygame.Rect(rect.x + CARD_PAD + i * m.tile_w, rect.y + CARD_PAD, m.tile_w, m.tile_h),
                kind,
                pos,
            )
            for i, (kind, pos) in enumerate(
                zip(slot.shown, storage_order(slot.tiles, slot.shown), strict=False)
            )
        )
        out.append(Card(rect=rect, slot=slot.index, tiles=slot.tiles, shape=slot.shape, spots=spots))
        x = rect.right + CARD_GAP
    return tuple(out)


def regions_for(m: Metrics, view: GameView) -> Regions:
    """Everything clickable, derived from the same geometry that draws it."""
    import pygame

    bar = m.controls
    buttons: list[Rect | None] = [None, None, None]
    if bar is not None:
        stride = BUTTON_H + BUTTON_GAP
        top = bar.y + (bar.h - 3 * stride + BUTTON_GAP) // 2
        buttons = [
            pygame.Rect(bar.x, top + i * stride, min(BUTTON_W, bar.w), BUTTON_H) for i in range(3)
        ]
    workbench = tray_rect(m, m.workbench, len(view.workbench))
    return Regions(
        cards=cards_for(m, view),
        rack=_rack_spots(m.rack, view.rack, m.tile_w, m.tile_h),
        workbench=_tray_spots(workbench, view.workbench, m.tile_w, m.tile_h),
        rack_tray=m.rack,
        workbench_tray=workbench,
        end_turn=buttons[0],
        undo=buttons[1],
        draw=buttons[2],
    )


def hit(regions: Regions, pos: tuple[int, int]) -> Hit | None:
    """Which region a click landed in. Pure: no pygame state, no display."""
    for rect, zone in (
        (regions.end_turn, Zone.END_TURN),
        (regions.undo, Zone.UNDO),
        (regions.draw, Zone.DRAW),
    ):
        if rect is not None and rect.collidepoint(pos):
            return Hit(zone)

    # Reversed: fanned tiles overlap, and the one drawn last is the one on top.
    for spots, zone in ((regions.rack, Zone.RACK), (regions.workbench, Zone.WORKBENCH)):
        for spot in reversed(spots):
            if spot.rect.collidepoint(pos):
                return Hit(zone, kind=spot.kind, pos=spot.pos)

    for card in regions.cards:
        if not card.rect.collidepoint(pos):
            continue
        for spot in card.spots:
            if spot.rect.collidepoint(pos):
                return Hit(Zone.SLOT, slot=card.slot, kind=spot.kind, pos=spot.pos, tiles=card.tiles)
        return Hit(Zone.SLOT, slot=card.slot, tiles=card.tiles)
    return None
