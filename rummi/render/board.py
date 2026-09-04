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
from functools import cache
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
"""Rows of set cards the table is allowed. The window is widened until the worst
table the config permits wraps onto no more than this, because height is the
expensive direction: a row is a whole card tall, while the width is already spent
on a full rack with a button column beside it."""


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
    def card_h(self) -> int:
        return 2 * CARD_PAD + self.tile_h + CAPTION_H


def card_width(n_tiles: int, tile_w: int) -> int:
    return 2 * CARD_PAD + max(1, n_tiles) * tile_w


@cache
def rows_needed(cfg: RummiConfig, tile_w: int, usable: int) -> int:
    """Rows to reserve so that no table the config allows can overflow the area.

    Sized from the config rather than from what games do: a card pushed out of the
    table is a set with no rectangle, and a set with no rectangle cannot be clicked.

    *Mixed* widths are the worst case for greedy wrapping. A slot holding fewer
    than ``min_set`` tiles is still drawn ``min_set`` wide, so a narrow card costs
    one tile and still eats a whole card's width, while the widest run the deck
    allows keeps forcing the wrap -- a uniform table of wide sets spends its tiles
    far faster for the same number of wraps, and a uniform table of narrow ones
    never forces one. What holds the worst case down is two budgets pulling against
    each other, ``max_sets`` cards and the deck's tiles to fill them with, so it is
    searched over every way of spending them rather than swept over one family.

    The search is what makes this a bound and not a sample: for each number of
    cards, tiles and rows it keeps the fullest a row can be left in, and a fuller
    row can only wrap sooner. Cards are placed exactly as ``cards_for`` places
    them, so the two cannot disagree about where one lands.
    """
    span = usable - 2 * CARD_GAP  # what a row has for its cards and the gaps between
    # One entry per drawn width, with what that width costs in tiles. The narrowest
    # card costs a single tile because anything under ``min_set`` draws the same.
    widths = [(card_width(cfg.min_set, tile_w), 1)] + [
        (card_width(size, tile_w), size) for size in range(cfg.min_set + 1, cfg.max_set_len + 1)
    ]
    # The landing card is drawn only while a slot is free, so ``max_sets`` bounds
    # the cards whether or not it is one of them; the spare tile pays for the fact
    # that it carries none.
    budget = cfg.n_tiles + 1
    # (cards, tiles, rows) -> the fullest the row in progress can be left.
    reach: dict[tuple[int, int, int], int] = {(0, 0, 1): 0}
    worst = 1
    for _ in range(cfg.max_sets):
        nxt: dict[tuple[int, int, int], int] = {}
        for (cards, tiles, rows), filled in reach.items():
            for width, cost in widths:
                if tiles + cost > budget:
                    continue
                if filled and filled + CARD_GAP + width > span:
                    key, room = (cards + 1, tiles + cost, rows + 1), width
                else:
                    room = filled + CARD_GAP + width if filled else width
                    key = (cards + 1, tiles + cost, rows)
                if nxt.get(key, -1) < room:
                    nxt[key] = room
                worst = max(worst, key[2])
        if not nxt:
            break
        reach = nxt
    return worst


@cache
def width_for_rows(cfg: RummiConfig, tile_w: int, floor: int, rows: int = TARGET_ROWS) -> int:
    """The narrowest table at least ``floor`` wide whose worst case fits ``rows``.

    Bisected rather than solved, because ``rows_needed`` is a search and not a
    formula. The invariant is that the upper end always fits -- a table wide enough
    for every card the config can draw needs one row -- so what comes back fits
    too, whatever the shape of the curve in between.
    """
    if rows_needed(cfg, tile_w, floor) <= rows:
        return floor
    lo, hi = floor, cfg.max_sets * (card_width(cfg.max_set_len, tile_w) + CARD_GAP) + CARD_GAP
    while lo < hi:
        mid = (lo + hi) // 2
        if rows_needed(cfg, tile_w, mid) <= rows:
            hi = mid
        else:
            lo = mid + 1
    return hi


def metrics_for(
    cfg: RummiConfig, tile_w: int = 26, tile_h: int = 36, interactive: bool = False
) -> Metrics:
    """``interactive`` is the play window: it reserves the button bar and drops the
    action log, where the read-only view keeps the log and the telemetry with it."""
    import pygame

    # The widest card the config allows, and the gaps a row keeps either side of
    # it: the first card of a row is placed whether or not it fits, so a table
    # narrower than this would draw one straight off the right-hand edge.
    widest = card_width(cfg.max_set_len, tile_w) + 2 * CARD_GAP
    # A floor on width whether or not the buttons are drawn: without it a reduced
    # config produces a window narrow enough that the table needs every one of
    # TARGET_ROWS for three sets.
    floor = 3 * BUTTON_W + 2 * BUTTON_GAP + 2 * PAD
    # Centred and sized to a full rack rather than spanning the window: the rack is
    # the one object on screen that is yours, and a strip pinned to both edges
    # reads as another panel of the interface.
    tray_target = 2 * RACK_PAD + (cfg.rack_size + RACK_SLACK) * tile_w
    if interactive:
        # The buttons stack *beside* the rack, one column each side, so the window
        # needs the tray plus a full button either way. The floor above is three
        # buttons side by side, which is a wider number on the standard config and
        # a narrower one on a reduced deck -- where the bar then left the window.
        floor = max(floor, tray_target + 2 * BUTTON_W)
    usable = width_for_rows(cfg, tile_w, max(floor, widest))
    rows = rows_needed(cfg, tile_w, usable)

    card_h = 2 * CARD_PAD + tile_h + CAPTION_H
    tray_w = min(usable, tray_target)
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
