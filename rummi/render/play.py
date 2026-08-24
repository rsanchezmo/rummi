"""Play a hand yourself, against any bundled agent.

The point of this module, beyond being fun, is that **the mask drives the UI**.
The same ``action_mask`` an RL agent consumes decides which tiles are clickable
and which slots light up, so an illegal move is not rejected -- it is
inexpressible. If you can click it, it is legal.

Interaction is two clicks: a rack tile goes to the workbench and stays selected,
then a slot takes it. Table tiles can be picked up the same way, which is how you
rearrange. Everything is a real micro-action, so playing a turn here produces
exactly the action sequence an agent would have to emit.

The hit-testing is deliberately pure -- :func:`hit` and :func:`action_for` take
rectangles and a click position and return an action id -- so the whole thing can
be tested headless without an event loop or a display.

    python -m rummi.render.play --opponent optimal
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from rummi.agents import build
from rummi.agents.base import act_on_state
from rummi.env.numpy.deal import reset
from rummi.env.numpy.engine import step
from rummi.env.numpy.masks import legal_actions
from rummi.render.pygame_view import LOG_H, PAD, STRIP_H, STRIP_LABEL_W, PygameView
from rummi.render.view_model import GameView, view
from rummi.rules.actions import encode_assign, encode_pick, encode_place
from rummi.rules.config import STANDARD, TINY_GROUPS, RummiConfig

CONFIGS = {"standard": STANDARD, "tiny_groups": TINY_GROUPS}
BUTTON_W, BUTTON_H, BUTTON_GAP = 130, 34, 10


class Zone(str, Enum):
    RACK = "rack"
    WORKBENCH = "workbench"
    SLOT = "slot"
    END_TURN = "end_turn"
    DRAW = "draw"


@dataclass(frozen=True, slots=True)
class Hit:
    zone: Zone
    index: int = -1
    """Slot id, for :data:`Zone.SLOT`."""
    kind: int = -1
    """Tile kind, for the rack and workbench strips."""
    tiles: tuple[int, ...] = ()
    """Contents of the clicked slot, so a pick can choose a position."""


@dataclass(slots=True)
class Selection:
    """What the player has picked up, if anything."""

    kind: int = -1

    @property
    def active(self) -> bool:
        return self.kind >= 0

    def clear(self) -> None:
        self.kind = -1


@dataclass(slots=True)
class Regions:
    """Clickable rectangles for one rendered frame."""

    rack: list[tuple[object, int]] = field(default_factory=list)
    workbench: list[tuple[object, int]] = field(default_factory=list)
    slots: list[tuple[object, int, tuple[int, ...]]] = field(default_factory=list)
    end_turn: object = None
    draw: object = None


def regions_for(window: PygameView, snapshot: GameView) -> Regions:
    """Where everything is on screen, derived from the same layout that drew it."""
    import pygame

    lay = window._layout
    atlas = window._atlas
    out = Regions()

    visible, _ = window.visible_slots(snapshot)
    for display_index, slot in enumerate(visible):
        rect = lay.row_rect(display_index)
        out.slots.append((rect, slot.index, slot.tiles))

    for name, kinds, top in (
        ("workbench", snapshot.workbench, lay.workbench_top),
        ("rack", snapshot.rack, lay.rack_top),
    ):
        target = out.workbench if name == "workbench" else out.rack
        for i, kind in enumerate(kinds):
            x = PAD + STRIP_LABEL_W + i * atlas.tile_w
            target.append((pygame.Rect(x, top + 2, atlas.tile_w, atlas.tile_h), kind))

    bar_y = lay.log_top + LOG_H + 4
    out.end_turn = pygame.Rect(PAD, bar_y, BUTTON_W, BUTTON_H)
    out.draw = pygame.Rect(PAD + BUTTON_W + BUTTON_GAP, bar_y, BUTTON_W, BUTTON_H)
    return out


def hit(regions: Regions, pos: tuple[int, int]) -> Hit | None:
    """Which region a click landed in. Pure: no pygame state, no display."""
    if regions.end_turn is not None and regions.end_turn.collidepoint(pos):
        return Hit(Zone.END_TURN)
    if regions.draw is not None and regions.draw.collidepoint(pos):
        return Hit(Zone.DRAW)
    for rect, kind in regions.rack:
        if rect.collidepoint(pos):
            return Hit(Zone.RACK, kind=kind)
    for rect, kind in regions.workbench:
        if rect.collidepoint(pos):
            return Hit(Zone.WORKBENCH, kind=kind)
    for rect, slot, tiles in regions.slots:
        if rect.collidepoint(pos):
            return Hit(Zone.SLOT, index=slot, tiles=tiles)
    return None


def action_for(
    cfg: RummiConfig, spot: Hit, selection: Selection, mask: np.ndarray, snapshot: GameView
) -> int | None:
    """The action a click means, or ``None`` if it is not a legal move.

    Every branch checks the mask, so this cannot produce an illegal action --
    which is what lets the UI simply ignore an unproductive click instead of
    having to explain it.
    """
    if spot.zone is Zone.END_TURN:
        return cfg.end_turn_action if mask[cfg.end_turn_action] else None
    if spot.zone is Zone.DRAW:
        return cfg.draw_action

    if spot.zone is Zone.RACK:
        action = encode_place(cfg, spot.kind)
        return action if mask[action] else None

    if spot.zone is Zone.WORKBENCH:
        return None  # selection only; handled by the caller

    if spot.zone is Zone.SLOT:
        if selection.active:
            action = encode_assign(cfg, selection.kind, spot.index)
            return action if mask[action] else None
        # Nothing in hand: clicking a set lifts a tile off it, which is how a
        # rearrangement starts. Take from the end, where a run gives up a tile
        # without breaking.
        return pick_from_slot(cfg, spot.index, spot.tiles, mask)
    return None


def pick_from_slot(
    cfg: RummiConfig, slot: int, tiles: tuple[int, ...], mask: np.ndarray
) -> int | None:
    """Lift the rightmost liftable tile out of a set."""
    for pos in reversed(range(len(tiles))):
        action = encode_pick(cfg, slot, pos)
        if mask[action]:
            return action
    return None


def legal_slots(cfg: RummiConfig, kind: int, mask: np.ndarray) -> frozenset[int]:
    """Slots that would accept the held tile. This is what the UI highlights."""
    if kind < 0:
        return frozenset()
    return frozenset(
        slot for slot in range(cfg.max_sets) if mask[encode_assign(cfg, kind, slot)]
    )


# --- drawing the interactive chrome ------------------------------------------
def draw_controls(
    window: PygameView, regions: Regions, snapshot: GameView, mask: np.ndarray, selection: Selection
) -> None:
    """Buttons and the hint line, drawn over the board."""
    import pygame

    from rummi.render.atlas import BACKGROUND, DISABLED, DROP_EDGE, PANEL, TEXT, TEXT_DIM

    surface, cfg = window._surface, window.cfg

    # Ring the tile in hand. Without it the hint line is the only clue about what
    # you picked up, which is unreadable once the workbench holds several tiles.
    if selection.active:
        for rect, kind in regions.workbench:
            if kind == selection.kind:
                pygame.draw.rect(surface, DROP_EDGE, rect.inflate(4, 4), width=3, border_radius=5)
                break

    bar = pygame.Rect(0, regions.end_turn.y - 4, window.size[0], BUTTON_H + 12)
    surface.fill(BACKGROUND, bar)

    for rect, label, enabled in (
        (regions.end_turn, "END TURN", bool(mask[cfg.end_turn_action])),
        (regions.draw, "DRAW", True),
    ):
        pygame.draw.rect(surface, PANEL if enabled else BACKGROUND, rect, border_radius=6)
        pygame.draw.rect(
            surface, DROP_EDGE if enabled else DISABLED, rect, width=2, border_radius=6
        )
        glyph = window._font.render(label, True, TEXT if enabled else DISABLED)
        surface.blit(glyph, glyph.get_rect(center=rect.center))

    if snapshot.done:
        hint = "game over -- close the window"
    elif selection.active:
        hint = f"holding {snapshot.label(selection.kind)} -- click a highlighted set, or an empty slot"
    elif snapshot.workbench:
        hint = "click a tile in the workbench to pick it up again"
    elif snapshot.needs_meld:
        hint = f"click rack tiles to build sets worth {cfg.initial_meld}+ to open"
    else:
        hint = "click a rack tile to play it, or a set to lift a tile out of it"
    surface.blit(
        window._small.render(hint, True, TEXT_DIM),
        (PAD + 2 * (BUTTON_W + BUTTON_GAP) + 6, regions.end_turn.y + 10),
    )


def redraw(window: PygameView, snapshot: GameView, mask, selection: Selection) -> Regions:
    import pygame

    window.highlight_slots = legal_slots(window.cfg, selection.kind, mask)
    window.draw(snapshot)
    regions = regions_for(window, snapshot)
    draw_controls(window, regions, snapshot, mask, selection)
    pygame.display.flip()
    return regions


# --- the loop ----------------------------------------------------------------
def play(
    cfg: RummiConfig = STANDARD,
    opponent: str = "greedy",
    seed: int = 0,
    seat: int = 0,
    opponent_delay_ms: int = 260,
) -> None:
    """Open a window and play a game. Blocks until the game ends or you close it."""
    import pygame

    state = reset(cfg, 1, seed=seed)
    rival = build(opponent, cfg)
    rival.reset(1)

    window = PygameView(cfg, headless=False, reserve_bottom=BUTTON_H + 16, caption=f"rummi -- you vs {opponent}")
    selection = Selection()
    clock = pygame.time.Clock()

    mask = legal_actions(state)
    regions = redraw(window, view(state, 0, mask), mask, selection)

    running = True
    while running:
        human_turn = int(state.current[0]) == seat and not bool(state.done[0])

        if not human_turn and not bool(state.done[0]):
            # Let the opponent play one micro-action at a time so its turn is
            # legible rather than an instant jump.
            mask = legal_actions(state)
            step(state, act_on_state(rival, state, mask), mask)
            mask = legal_actions(state)
            regions = redraw(window, view(state, 0, mask), mask, selection)
            pygame.time.wait(opponent_delay_ms)
            continue

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                selection.clear()
                regions = redraw(window, view(state, 0, mask), mask, selection)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and not state.done[0]:
                spot = hit(regions, event.pos)
                if spot is None:
                    continue
                if spot.zone is Zone.WORKBENCH:
                    selection.kind = spot.kind
                else:
                    action = action_for(cfg, spot, selection, mask, view(state, 0, mask))
                    if action is None:
                        continue
                    step(state, np.array([action]), mask)
                    # Holding the tile you just placed makes "click rack, click
                    # slot" a two-click move rather than three.
                    if spot.zone is Zone.RACK:
                        selection.kind = spot.kind
                    else:
                        selection.clear()
                        if spot.zone is Zone.SLOT and state.workbench[0].sum():
                            selection.kind = int(np.argmax(state.workbench[0] > 0))
                mask = legal_actions(state)
                regions = redraw(window, view(state, 0, mask), mask, selection)
        clock.tick(30)

    window.close()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", choices=sorted(CONFIGS), default="standard")
    p.add_argument("--opponent", default="greedy", help="any agent name from rummi.agents")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seat", type=int, default=0)
    args = p.parse_args()
    play(CONFIGS[args.config], opponent=args.opponent, seed=args.seed, seat=args.seat)


if __name__ == "__main__":
    main()
