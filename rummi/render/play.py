"""Play a hand yourself, against any bundled agent.

The point of this module, beyond being fun, is that **the mask drives the UI**.
The same ``action_mask`` an RL agent consumes decides which tiles can be picked up
and which sets light up, so an illegal move is not rejected -- it is
inexpressible. If you can drop it there, it is legal.

Tiles are dragged: press a tile to take it, release it on a set to play it.
A press takes and a release drops, which makes clicking work for free -- press and
release in the same place and the tile stays in hand until you click where it
should go. Either way the gesture is spelled out in real micro-actions, so playing
a turn here emits exactly the sequence an agent would have to emit.

Nothing here decides where anything is on screen; :mod:`rummi.render.board` does,
and it does it without touching a display, which is what lets a whole hand be
played headlessly in the tests.

    python -m rummi.render.play --opponent optimal
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from rummi.agents import build
from rummi.agents.base import act_on_state
from rummi.env.numpy.deal import reset
from rummi.env.numpy.engine import step
from rummi.env.numpy.masks import legal_actions
from rummi.env.numpy.state import BatchState
from rummi.render.board import Hit, Regions, Zone, hit
from rummi.render.pygame_view import Overlay, PygameView
from rummi.render.view_model import GameView, view
from rummi.rules.actions import decode, encode_assign, encode_pick, encode_place
from rummi.rules.config import STANDARD, TINY_GROUPS, RummiConfig

if TYPE_CHECKING:
    from pygame.event import Event

CONFIGS = {"standard": STANDARD, "tiny_groups": TINY_GROUPS}
DRAG_SLOP = 6
"""How far the pointer must travel before a press counts as a drag. Below it the
gesture is a click, and a click that took a tile must not also put it back."""
POLL_MS = 16
"""How long the loop may block at a stretch while the opponent moves. A frame at
60Hz: often enough that closing the window is answered as it is asked, coarse
enough that waiting still costs nothing."""


# --- what a click means ------------------------------------------------------
def _row(mask: np.ndarray) -> np.ndarray:
    """The single env's mask row the UI reasons about.

    ``step`` wants the batched ``(1, A)`` mask and everything here wants ``(A,)``.
    Caught here rather than left to fail deep inside drawing, with an IndexError
    that names neither shape.
    """
    if mask.ndim != 1:
        raise ValueError(f"expected a single env's mask, got shape {mask.shape}")
    return mask


def action_for(cfg: RummiConfig, spot: Hit, held: int, mask: np.ndarray) -> int | None:
    """The action a click means, or ``None`` if it is not a legal move.

    Every branch checks the mask, so this cannot produce an illegal action --
    which is what lets the UI simply ignore an unproductive click instead of
    having to explain it.
    """
    mask = _row(mask)
    if spot.zone is Zone.END_TURN:
        return cfg.end_turn_action if mask[cfg.end_turn_action] else None
    if spot.zone is Zone.DRAW:
        return cfg.draw_action
    if spot.zone is Zone.UNDO:
        # Not an action: the MDP has no "unplace", so undo is done by the caller
        # rewinding the turn and replaying it one action short.
        return None
    if spot.zone is Zone.RACK:
        action = encode_place(cfg, spot.kind)
        return action if mask[action] else None
    if spot.zone is Zone.WORKBENCH:
        return None  # taking a tile back in hand is not a move
    if spot.zone is Zone.SLOT:
        if held >= 0:
            action = encode_assign(cfg, held, spot.slot)
            return action if mask[action] else None
        return pick_action(cfg, spot, mask)
    return None


def pick_action(cfg: RummiConfig, spot: Hit, mask: np.ndarray) -> int | None:
    """Lift the tile that was pointed at.

    ``spot.pos`` is a position in the set's *stored* order even though the click
    landed on the tile as displayed -- the board translates, because a joker is
    drawn in the gap it fills while ``PICK`` indexes storage. A click that landed
    on the card but not on a tile has no position, and falls back to the rightmost
    liftable tile, where a run gives one up without breaking.
    """
    mask = _row(mask)
    if spot.pos >= 0:
        action = encode_pick(cfg, spot.slot, spot.pos)
        if mask[action]:
            return action
    for pos in reversed(range(len(spot.tiles))):
        action = encode_pick(cfg, spot.slot, pos)
        if mask[action]:
            return action
    return None


def legal_slots(cfg: RummiConfig, kind: int, mask: np.ndarray) -> frozenset[int]:
    """Slots that would accept the held tile. This is what the UI lights up."""
    mask = _row(mask)
    if kind < 0:
        return frozenset()
    return frozenset(slot for slot in range(cfg.max_sets) if mask[encode_assign(cfg, kind, slot)])


def rewind(cfg: RummiConfig, turn_start: BatchState, actions: list[int]) -> BatchState:
    """Replay a turn from its opening state, one action short.

    The action space has no "unplace": a tile leaves the workbench by being
    assigned, or by DRAW abandoning the whole turn. Rather than add an action --
    which would change the MDP every agent sees, to fix a human's mis-click --
    undo is done by rewinding to the turn's opening state and replaying. The
    engine is untouched and nothing an agent can do changes.
    """
    state = turn_start.clone()
    for action in actions:
        step(state, np.array([action]), legal_actions(state))
    return state


# --- the game, and what undo needs to know about it --------------------------
@dataclass(slots=True)
class Session:
    """One game, plus the turn's opening state and the actions taken since.

    Those two are all UNDO needs, and keeping them here rather than in the event
    loop is what lets a scripted hand exercise the same code path a player does.
    """

    cfg: RummiConfig
    state: BatchState
    seat: int = 0
    """Which seat the player holds. Everything the window shows of a rack is this
    seat's, whoever is acting."""
    taken: list[int] = field(default_factory=list, init=False)
    mask: np.ndarray = field(init=False)
    _start: BatchState = field(init=False)

    def __post_init__(self) -> None:
        self._start = self.state.clone()
        self.mask = legal_actions(self.state)

    @property
    def row(self) -> np.ndarray:
        return self.mask[0]

    @property
    def can_undo(self) -> bool:
        return bool(self.taken)

    @property
    def done(self) -> bool:
        return bool(self.state.done[0])

    def snapshot(self) -> GameView:
        return view(self.state, 0, self.mask, seat=self.seat)

    def apply(self, action: int) -> None:
        step(self.state, np.array([action]), self.mask)
        self.taken.append(int(action))
        if action in (self.cfg.end_turn_action, self.cfg.draw_action):
            self._open_turn()
        self.mask = legal_actions(self.state)

    def rival_moves(self, actions: np.ndarray) -> None:
        """Step the opponent. The baseline resets afterwards, so UNDO can never
        rewind into someone else's turn."""
        step(self.state, actions, self.mask)
        self._open_turn()
        self.mask = legal_actions(self.state)

    def undo(self) -> None:
        if not self.taken:
            return
        self.taken.pop()
        self.state = rewind(self.cfg, self._start, self.taken)
        self.mask = legal_actions(self.state)

    def _open_turn(self) -> None:
        self._start, self.taken = self.state.clone(), []


# --- the gesture -------------------------------------------------------------
@dataclass(slots=True)
class Grip:
    """What the player is holding, and the gesture in progress."""

    held: int = -1
    """Tile kind in hand -- really in the workbench -- or ``-1``."""
    press: tuple[int, int] | None = None
    """Where the button went down; ``None`` while it is up."""
    took: bool = False
    """Whether that press picked the tile up. A release near such a press is a
    click that takes, and must not immediately put the tile back."""
    at: tuple[int, int] | None = None
    """Cursor position once the press has travelled far enough to be a drag."""

    def let_go(self) -> None:
        self.press, self.at, self.took = None, None, False

    def empty(self) -> None:
        """Let go of the gesture and of the tile with it."""
        self.held = -1
        self.let_go()


def _far(a: tuple[int, int], b: tuple[int, int] | None) -> bool:
    return b is None or abs(a[0] - b[0]) + abs(a[1] - b[1]) > DRAG_SLOP


def take(session: Session, spot: Hit, held: int) -> tuple[int, bool]:
    """Apply a press. Returns the tile now in hand and whether the press took it.

    A press only ever *takes*: off the rack, out of a set, or back out of the
    workbench. Dropping is left to the release, which is what makes a drag and a
    pair of clicks end in the same place.
    """
    if spot.zone is Zone.UNDO:
        session.undo()
        return -1, False
    if spot.zone is Zone.WORKBENCH:
        return spot.kind, True
    if spot.zone in (Zone.END_TURN, Zone.DRAW):
        action = action_for(session.cfg, spot, held, session.row)
        if action is None:
            return held, False
        session.apply(action)
        return -1, False
    if spot.zone is Zone.RACK:
        action = action_for(session.cfg, spot, held, session.row)
        if action is None:
            return held, False
        session.apply(action)
        return spot.kind, True
    if spot.zone is Zone.SLOT and held < 0:
        action = pick_action(session.cfg, spot, session.row)
        if action is None:
            return held, False
        _, _, pos = decode(session.cfg, action)
        session.apply(action)
        return spot.tiles[pos], True
    return held, False


def drop(session: Session, spot: Hit | None, held: int) -> int:
    """Apply a release. Returns the tile still in hand, if any."""
    if held < 0 or spot is None or spot.zone is not Zone.SLOT:
        return held
    action = action_for(session.cfg, spot, held, session.row)
    if action is None:
        return held
    session.apply(action)
    # Keep hold of whatever is still loose: a three-tile meld is then three drags
    # rather than three pick-ups and three drops.
    loose = session.state.workbench[0]
    return int(np.argmax(loose > 0)) if loose.sum() else -1


def hint_for(snapshot: GameView, held: int) -> str:
    """One line, and only where the board cannot say it itself."""
    if snapshot.done:
        return "close the window to finish"
    if held >= 0:
        return f"drop {snapshot.label(held)} on a lit set, or on the empty slot"
    if snapshot.workbench:
        # There is no action that puts a workbench tile back, so say what does.
        return "drag the tile in the workbench onto a set -- or UNDO to take it back"
    if not snapshot.table_whole:
        # The red rings say which sets; this says why the turn will not commit.
        return "a set in red is not legal yet -- fix it before ending the turn"
    if snapshot.needs_meld:
        return f"build sets worth {snapshot.cfg.initial_meld} to open"
    return "drag a tile from your rack, or off a set to move it"


def overlay_for(
    session: Session, grip: Grip, snapshot: GameView, seat: int = 0, rival: str = ""
) -> Overlay:
    return Overlay(
        held=grip.held,
        drop_slots=legal_slots(session.cfg, grip.held, session.row),
        drag_pos=grip.at,
        you=seat,
        rival=rival,
        can_end_turn=bool(session.row[session.cfg.end_turn_action]),
        can_undo=session.can_undo,
        hint=hint_for(snapshot, grip.held),
    )


# --- the loop ----------------------------------------------------------------
def handle(event: Event, session: Session, grip: Grip, regions: Regions | None) -> bool:
    """Apply one event. ``False`` means the window has been asked to close.

    ``regions`` is what the pointer can reach in the frame on screen, and there is
    none while the opponent is playing: a press then has nothing to aim at, so it
    is dropped rather than banked for the player's next turn, where it would land
    on a board that had changed underneath it. The keys answer for the window
    rather than for the board, so they are answered wherever the loop is waiting.
    """
    import pygame

    if event.type == pygame.QUIT:
        return False
    if event.type == pygame.KEYDOWN:
        if event.key == pygame.K_ESCAPE:
            grip.held = -1  # let go; the tile stays where it is
        elif event.key == pygame.K_BACKSPACE:
            session.undo()
            grip.empty()
        return True
    if regions is None:
        return True

    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and not session.done:
        grip.press, grip.at = event.pos, None
        spot = hit(regions, event.pos)
        if spot is not None:
            grip.held, grip.took = take(session, spot, grip.held)
    elif event.type == pygame.MOUSEMOTION and grip.press is not None:
        if grip.held >= 0 and _far(event.pos, grip.press):
            grip.at = event.pos
    elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
        if not (grip.took and not _far(event.pos, grip.press)):
            grip.held = drop(session, hit(regions, event.pos), grip.held)
        grip.let_go()
    return True


def wait_out(delay_ms: int, session: Session, grip: Grip) -> bool:
    """Pace one of the opponent's micro-actions, with the window still answering.

    Blocking on ``pygame.time.wait`` instead leaves the queue unread for the whole
    of the opponent's turn -- a couple of seconds at the pace this runs at -- so
    closing the window goes unnoticed until it ends, and every press made in the
    meantime is delivered afterwards, onto a board it was never aimed at.
    """
    import pygame

    deadline = pygame.time.get_ticks() + delay_ms
    while True:
        for event in pygame.event.get():
            if not handle(event, session, grip, None):
                return False
        left = deadline - pygame.time.get_ticks()
        if left <= 0:
            return True
        pygame.time.wait(min(POLL_MS, left))


def play(
    cfg: RummiConfig = STANDARD,
    opponent: str = "greedy",
    seed: int = 0,
    seat: int = 0,
    opponent_delay_ms: int = 260,
) -> None:
    """Open a window and play a game. Blocks until the game ends or you close it."""
    import pygame

    # Checked here rather than left to the rack lookup: a negative seat indexes
    # from the end, which would quietly show the window an opponent's hand.
    if not 0 <= seat < cfg.n_players:
        raise ValueError(f"seat {seat} is not a seat in a {cfg.n_players}-player game")

    session = Session(cfg, reset(cfg, 1, seed=seed), seat=seat)
    rival = build(opponent, cfg)
    rival.reset(1)
    window = PygameView(
        cfg, headless=False, interactive=True, caption=f"rummi -- you vs {opponent}"
    )
    grip = Grip()
    clock = pygame.time.Clock()

    running = True
    while running:
        snapshot = session.snapshot()
        regions = window.draw(snapshot, overlay_for(session, grip, snapshot, seat, opponent))
        window.flip()

        if not session.done and int(session.state.current[0]) != seat:
            # One micro-action at a time, so the opponent's turn is legible rather
            # than an instant jump from one board to another.
            grip.empty()
            session.rival_moves(act_on_state(rival, session.state, session.mask))
            running = wait_out(opponent_delay_ms, session, grip)
            continue

        for event in pygame.event.get():
            if not handle(event, session, grip, regions):
                running = False
                break
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
