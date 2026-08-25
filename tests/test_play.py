"""Interactive play, driven headlessly.

The board is laid out without touching a display and hit-testing is pure --
rectangles and a click position in, an action id out -- so a whole hand can be
played here by synthesising presses and releases, with no window and no event
loop. The property that matters is that the UI cannot express an illegal move:
every gesture either maps to an action the mask allows, or to nothing.
"""

import numpy as np
import pytest

pygame = pytest.importorskip("pygame")

from rummi.env.numpy.deal import reset
from rummi.env.numpy.engine import step
from rummi.env.numpy.masks import legal_actions
from rummi.render.board import (
    Hit,
    Zone,
    hit,
    metrics_for,
    regions_for,
    storage_order,
)
from rummi.render.play import (
    Grip,
    Session,
    action_for,
    drop,
    legal_slots,
    pick_action,
    take,
)
from rummi.render.pygame_view import Overlay, PygameView
from rummi.render.view_model import view
from rummi.rules.actions import encode_assign, encode_place
from rummi.rules.config import STANDARD as C, TINY_GROUPS
from rummi.rules.encoding import kind_of

from tests.conftest import rebalance_pool, state_with

RUN_36 = [kind_of(C, 0, n) for n in (11, 12, 13)]


@pytest.fixture(autouse=True)
def _headless(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")


@pytest.fixture
def board():
    """Geometry only -- no surface, no window."""
    return metrics_for(C, interactive=True)


def regions(board, state):
    return regions_for(board, view(state, 0, legal_actions(state)))


def centre(rect):
    return rect.center


def test_every_rack_tile_has_its_own_rectangle(board):
    s = state_with(C, rack=RUN_36)
    snapshot = view(s, 0, legal_actions(s))
    r = regions_for(board, snapshot)
    assert [spot.kind for spot in r.rack] == list(snapshot.rack)
    for spot in r.rack:
        assert hit(r, centre(spot.rect)) == Hit(Zone.RACK, kind=spot.kind, pos=spot.pos)


def test_pressing_a_rack_tile_places_it(board):
    s = state_with(C, rack=RUN_36)
    mask = legal_actions(s)
    r = regions_for(board, view(s, 0, mask))
    spot = hit(r, centre(r.rack[0].rect))
    assert action_for(C, spot, -1, mask[0]) == encode_place(C, spot.kind)


def test_a_gesture_can_never_produce_an_illegal_action(board):
    """The guarantee the whole UI rests on: if you can drop it there, it is legal."""
    s = state_with(C, rack=RUN_36, table=[[kind_of(C, 1, n) for n in (1, 2, 3)]], melded=True)
    rng = np.random.default_rng(0)

    for _ in range(40):
        mask = legal_actions(s)
        snapshot = view(s, 0, mask)
        r = regions_for(board, snapshot)
        held = snapshot.workbench[0] if snapshot.workbench else -1

        width, height = board.width, board.height
        for _ in range(60):
            pos = (int(rng.integers(0, width)), int(rng.integers(0, height)))
            spot = hit(r, pos)
            if spot is None or spot.zone is Zone.WORKBENCH:
                continue
            action = action_for(C, spot, held, mask[0])
            if action is not None:
                assert mask[0, action], f"{spot.zone} produced illegal action {action}"

        step(s, np.array([rng.choice(np.flatnonzero(mask[0]))]), mask)
        if s.done[0]:
            break


def test_end_turn_is_dead_until_the_meld_is_legal(board):
    s = state_with(C, rack=RUN_36)
    mask = legal_actions(s)
    r = regions_for(board, view(s, 0, mask))
    end = hit(r, centre(r.end_turn))
    assert end.zone is Zone.END_TURN
    assert action_for(C, end, -1, mask[0]) is None, "36 points are not on the table yet"

    for action in [encode_place(C, k) for k in RUN_36] + [encode_assign(C, k, 0) for k in RUN_36]:
        step(s, np.array([action]), legal_actions(s))
    mask = legal_actions(s)
    r = regions_for(board, view(s, 0, mask))
    assert action_for(C, hit(r, centre(r.end_turn)), -1, mask[0]) == C.end_turn_action


def test_draw_is_always_clickable(board):
    s = reset(C, 1, seed=0)
    mask = legal_actions(s)
    r = regions_for(board, view(s, 0, mask))
    assert action_for(C, hit(r, centre(r.draw)), -1, mask[0]) == C.draw_action


def test_a_batched_mask_is_refused_where_a_row_is_wanted(board):
    """play() holds a (1, A) mask for step() while everything here wants one env's
    row. Passing the batched one used to fail deep inside drawing, with an
    IndexError that named neither shape."""
    s = state_with(C, rack=RUN_36)
    mask = legal_actions(s)
    assert mask.ndim == 2
    r = regions_for(board, view(s, 0, mask))
    with pytest.raises(ValueError, match="single env's mask"):
        action_for(C, hit(r, centre(r.draw)), -1, mask)
    with pytest.raises(ValueError, match="single env's mask"):
        legal_slots(C, RUN_36[0], mask)


def test_highlighted_slots_are_exactly_the_legal_drops():
    s = state_with(C, rack=RUN_36)
    step(s, np.array([encode_place(C, RUN_36[0])]), legal_actions(s))
    mask = legal_actions(s)

    highlighted = legal_slots(C, RUN_36[0], mask[0])
    assert highlighted == {slot for slot in range(C.max_sets) if mask[0, encode_assign(C, RUN_36[0], slot)]}
    assert highlighted, "a held tile with nowhere to go would be a dead end"


def test_pressing_a_tile_in_a_set_lifts_that_exact_tile(board):
    """A card draws one rectangle per tile, so a pick can take the tile that was
    pointed at rather than whichever one happens to be liftable."""
    run = [kind_of(C, 0, n) for n in (5, 6, 7, 8)]
    s = state_with(C, rack=[kind_of(C, 1, 4)], table=[run], melded=True)
    mask = legal_actions(s)
    r = regions_for(board, view(s, 0, mask))
    card = r.cards[0]
    assert [spot.kind for spot in card.spots] == sorted(run)

    for spot in card.spots:
        got = hit(r, centre(spot.rect))
        assert got == Hit(Zone.SLOT, slot=card.slot, kind=spot.kind, pos=spot.pos, tiles=card.tiles)
        action = pick_action(C, got, mask[0])
        assert action is not None and mask[0, action]

    # The one the player pointed at, not the rightmost.
    target = card.spots[1]
    trial = s.clone()
    action = pick_action(C, hit(r, centre(target.rect)), mask[0])
    step(trial, np.array([action]), mask)
    assert trial.workbench[0, target.kind] == 1


def test_a_click_on_the_card_but_not_a_tile_still_lifts_something(board):
    run = [kind_of(C, 0, n) for n in (5, 6, 7)]
    s = state_with(C, rack=[kind_of(C, 1, 4)], table=[run], melded=True)
    mask = legal_actions(s)
    r = regions_for(board, view(s, 0, mask))
    card = r.cards[0]
    # The caption strip belongs to the card and to no tile in it.
    spot = hit(r, (card.rect.x + 8, card.rect.bottom - 4))
    assert spot == Hit(Zone.SLOT, slot=card.slot, tiles=card.tiles)
    action = pick_action(C, spot, mask[0])
    assert action is not None and mask[0, action]


def test_a_joker_is_lifted_from_where_it_is_drawn(board):
    """`shown` puts the joker in the gap it fills while PICK indexes storage, so
    the position a click reports has to be translated back or the wrong tile
    comes off the table."""
    kinds = [kind_of(C, 0, n) for n in (10, 11, 13)] + [C.joker_kind]
    s = state_with(C, rack=[kind_of(C, 0, 5)], table=[kinds], melded=True)
    mask = legal_actions(s)
    snapshot = view(s, 0, mask)
    slot = snapshot.slots[0]
    assert slot.shown != slot.tiles, "this case should differ, or the test proves nothing"

    r = regions_for(board, snapshot)
    card = r.cards[0]
    joker_spot = next(spot for spot in card.spots if spot.kind == C.joker_kind)
    assert card.tiles[joker_spot.pos] == C.joker_kind, "the pick would lift a real tile"

    action = pick_action(C, hit(r, centre(joker_spot.rect)), mask[0])
    step(s, np.array([action]), mask)
    assert s.workbench[0, C.joker_kind] == 1


def test_storage_order_maps_every_displayed_tile_back():
    kinds = tuple([kind_of(C, 0, n) for n in (10, 11, 13)] + [C.joker_kind])
    s = state_with(C, rack=[kind_of(C, 0, 5)], table=[list(kinds)], melded=True)
    slot = view(s, 0).slots[0]
    order = storage_order(slot.tiles, slot.shown)
    assert sorted(order) == list(range(len(slot.tiles))), "a position used twice would lift twice"
    assert all(slot.tiles[pos] == kind for pos, kind in zip(order, slot.shown, strict=False))


def test_before_melding_the_table_cannot_be_touched(board):
    s = state_with(C, rack=RUN_36, table=[[kind_of(C, 1, n) for n in (1, 2, 3)]])
    mask = legal_actions(s)
    r = regions_for(board, view(s, 0, mask))
    card = r.cards[0]
    assert action_for(C, hit(r, centre(card.spots[0].rect)), -1, mask[0]) is None


def test_every_set_is_reachable_on_a_full_table():
    """An undrawn set has no rectangle, so it cannot be clicked -- and a real game
    reaches 22 sets."""
    rows = [[kind_of(C, c, n) for c in (0, 1, 2)] for n in range(1, 12) for _ in (0, 1)]
    s = state_with(C, rack=[C.joker_kind], table=rows, melded=True)
    s.racks[:, 1] = 0
    rebalance_pool(s)

    snapshot = view(s, 0, legal_actions(s))
    assert len(snapshot.occupied_slots) == 22

    board = metrics_for(C, interactive=True)
    r = regions_for(board, snapshot)
    reachable = {card.slot for card in r.cards}
    for slot in snapshot.occupied_slots:
        assert slot.index in reachable, f"slot {slot.index} has no clickable rect"
    for card in r.cards:
        assert board.table.contains(card.rect), f"card {card.slot} is drawn outside the table"


def test_the_worst_table_the_config_allows_still_fits():
    """The table area is sized from the config, not from what games happen to do,
    so no state can push a card out of it and out of reach."""
    from rummi.render.board import CARD_GAP, card_width

    for cfg in (C, TINY_GROUPS):
        board = metrics_for(cfg, interactive=True)
        widest = card_width(cfg.max_set_len, board.tile_w)
        # Pack the widest cards the config allows until the tiles run out.
        n_cards = min(cfg.max_sets, cfg.n_tiles // cfg.max_set_len) + 1
        x, y = board.table.x + CARD_GAP, board.table.y + CARD_GAP
        for _ in range(n_cards):
            if x > board.table.x + CARD_GAP and x + widest > board.table.right - CARD_GAP:
                x, y = board.table.x + CARD_GAP, y + board.card_h + CARD_GAP
            x += widest + CARD_GAP
        assert y + board.card_h <= board.table.bottom, f"{cfg.max_sets} widest sets overflow"


def test_a_big_hand_fills_the_rack_rather_than_overflowing_it(board):
    """A rack has two tiers, and the question they answer is what happens to a
    hand that will not fit on one. A tile drawn past the rack cannot be picked up,
    so it has to stay inside whatever the hand is."""
    kinds = [kind_of(C, c, n) for c in range(4) for n in range(1, 11)]
    s = state_with(C, rack=kinds, melded=True)
    s.racks[:, 1] = 0
    rebalance_pool(s)

    snapshot = view(s, 0, legal_actions(s))
    r = regions_for(board, snapshot)
    assert len(r.rack) == len(kinds) == 40
    assert [spot.kind for spot in r.rack] == list(snapshot.rack), "reading order, left to right"
    for spot in r.rack:
        assert board.rack.contains(spot.rect), "a tile outside the rack cannot be picked up"
        assert hit(r, spot.rect.center) == Hit(Zone.RACK, kind=spot.kind, pos=spot.pos)

    tiers = {spot.rect.y for spot in r.rack}
    assert len(tiers) == 2, "forty tiles should be spread over both tiers"
    widths = {spot.rect.x for spot in r.rack}
    assert len(widths) == 20, "and stand side by side, not fanned, while there is room"


def test_a_small_hand_sits_on_the_front_tier(board):
    s = state_with(C, rack=RUN_36)
    r = regions_for(board, view(s, 0, legal_actions(s)))
    assert len({spot.rect.y for spot in r.rack}) == 1
    assert r.rack[0].rect.bottom > board.rack.centery, "the near tier, as on a real rack"


def test_a_press_takes_and_a_release_drops():
    """One drag plays a tile: the press puts it in hand, the release assigns it."""
    board = metrics_for(C, interactive=True)
    session = Session(C, state_with(C, rack=RUN_36))
    grip = Grip()

    r = regions_for(board, session.snapshot())
    spot = hit(r, centre(next(s.rect for s in r.rack if s.kind == RUN_36[0])))
    grip.held, grip.took = take(session, spot, grip.held)
    assert grip.held == RUN_36[0] and grip.took
    assert session.state.workbench[0, RUN_36[0]] == 1, "the press moved it to the workbench"

    r = regions_for(board, session.snapshot())
    landing = next(card for card in r.cards if card.is_landing)
    grip.held = drop(session, hit(r, centre(landing.rect)), grip.held)
    assert session.state.table_sets[0, landing.slot, 0] == RUN_36[0]
    assert grip.held == -1, "nothing left loose, so nothing stays in hand"


def test_a_whole_opening_meld_can_be_played_by_dragging():
    """End to end: three drags, then END TURN."""
    board = metrics_for(C, interactive=True)
    session = Session(C, state_with(C, rack=RUN_36))
    grip = Grip()

    for _ in range(3):
        r = regions_for(board, session.snapshot())
        spot = hit(r, centre(next(s.rect for s in r.rack if s.kind in RUN_36)))
        grip.held, grip.took = take(session, spot, grip.held)

        r = regions_for(board, session.snapshot())
        drops = legal_slots(C, grip.held, session.row)
        target = next(card for card in r.cards if card.slot in drops)
        grip.held = drop(session, hit(r, centre(target.rect)), grip.held)

    r = regions_for(board, session.snapshot())
    end = hit(r, centre(r.end_turn))
    assert action_for(C, end, grip.held, session.row) == C.end_turn_action
    session.apply(C.end_turn_action)
    assert bool(session.state.melded[0, 0]) and session.state.racks[0, 0].sum() == 0
    session.state.check_invariants()


def test_a_tile_can_be_dragged_from_one_set_to_another():
    """The rearrangement gesture: press a tile in one set, release it on another."""
    board = metrics_for(C, interactive=True)
    run = [kind_of(C, 0, n) for n in (5, 6, 7, 8)]
    group = [kind_of(C, c, 9) for c in (0, 1, 2)]
    session = Session(C, state_with(C, rack=[kind_of(C, 3, 9)], table=[run, group], melded=True))
    grip = Grip()

    r = regions_for(board, session.snapshot())
    source = r.cards[0]
    grip.held, grip.took = take(session, hit(r, centre(source.spots[-1].rect)), grip.held)
    assert grip.held == run[-1] and session.state.workbench[0, run[-1]] == 1

    r = regions_for(board, session.snapshot())
    drops = legal_slots(C, grip.held, session.row)
    assert drops, "R8 must be placeable somewhere, or the drag is a dead end"
    target = next(card for card in r.cards if card.slot in drops)
    grip.held = drop(session, hit(r, centre(target.rect)), grip.held)
    assert session.state.workbench[0].sum() == 0
    session.state.check_invariants()


def test_a_press_that_took_a_tile_does_not_also_drop_it():
    """Press and release in the same place is a click that takes. Dropping on the
    release too would put the tile straight back and the gesture would do nothing."""
    from rummi.render.play import _far

    board = metrics_for(C, interactive=True)
    run = [kind_of(C, 0, n) for n in (5, 6, 7, 8)]
    session = Session(C, state_with(C, rack=[kind_of(C, 1, 4)], table=[run], melded=True))
    grip = Grip()

    r = regions_for(board, session.snapshot())
    spot = r.cards[0].spots[-1]
    at = centre(spot.rect)
    grip.press = at
    grip.held, grip.took = take(session, hit(r, at), grip.held)
    assert grip.held >= 0 and grip.took

    assert not _far(at, grip.press), "the pointer has not moved, so this is a click"
    held_before = grip.held
    if not (grip.took and not _far(at, grip.press)):  # what the loop does on release
        grip.held = drop(session, hit(r, at), grip.held)
    assert grip.held == held_before, "the tile must stay in hand"
    assert session.state.workbench[0].sum() == 1


def test_undo_rewinds_one_action_exactly():
    """The action space has no "unplace", so undo replays the turn one action
    short. It must land on precisely the state that existed before that action,
    or the board and the engine quietly disagree."""
    session = Session(C, state_with(C, rack=RUN_36))
    digests = [session.state.digest()]
    for kind in RUN_36:
        session.apply(encode_place(C, kind))
        digests.append(session.state.digest())

    while session.can_undo:
        digests.pop()
        session.undo()
        assert session.state.digest() == digests[-1], "undo did not restore the prior state"
    assert len(digests) == 1


def test_undo_can_free_a_tile_stuck_in_the_workbench():
    """The situation that motivated it: a tile placed by mistake cannot be taken
    back by any action -- only assigned somewhere, or the whole turn abandoned."""
    session = Session(C, state_with(C, rack=RUN_36))
    session.apply(encode_place(C, RUN_36[0]))
    assert session.state.workbench[0].sum() == 1
    assert session.state.racks[0, 0, RUN_36[0]] == 0, "the tile really is off the rack"

    for action in np.flatnonzero(session.row):
        if int(action) == C.draw_action:
            continue
        trial = session.state.clone()
        step(trial, np.array([int(action)]), legal_actions(trial))
        assert trial.racks[0, 0, RUN_36[0]] == 0, (
            f"action {int(action)} returned the tile to the rack, so undo is unnecessary"
        )

    session.undo()
    assert session.state.workbench[0].sum() == 0
    assert session.state.racks[0, 0, RUN_36[0]] == 1, "undo put it back on the rack"
    session.state.check_invariants()


def test_the_window_never_shows_the_opponents_rack():
    session = Session(C, state_with(C, rack=RUN_36), seat=0)
    session.apply(C.draw_action)
    assert int(session.state.current[0]) == 1, "their turn"
    snapshot = session.snapshot()
    assert snapshot.perspective == 0
    assert set(RUN_36) <= set(snapshot.rack)
    assert len(snapshot.rack) == session.state.racks[0, 0].sum()


def test_undo_cannot_rewind_into_the_opponents_turn():
    session = Session(C, state_with(C, rack=RUN_36))
    session.apply(C.draw_action)
    assert not session.can_undo, "DRAW ended the turn, so there is nothing of ours to undo"
    session.undo()  # must be a no-op rather than an error
    assert int(session.state.current[0]) == 1


def test_the_three_buttons_do_not_overlap(board):
    s = state_with(C, rack=RUN_36)
    r = regions_for(board, view(s, 0, legal_actions(s)))
    assert not r.end_turn.colliderect(r.undo)
    assert not r.undo.colliderect(r.draw)
    assert hit(r, r.undo.center).zone is Zone.UNDO
    # UNDO is not an action id -- the caller rewinds instead.
    assert action_for(C, Hit(Zone.UNDO), -1, legal_actions(s)[0]) is None


def test_a_window_with_controls_reserves_room_for_them():
    win = PygameView(C, headless=True, interactive=True)
    try:
        assert win.metrics.controls is not None
        assert win.metrics.controls.bottom < win.size[1]
        r = win.draw(view(reset(C, 1, seed=0), 0, None), Overlay(hint="hello"))
        assert r.end_turn is not None
    finally:
        win.close()

    plain = PygameView(C, headless=True)
    try:
        assert plain.metrics.controls is None
        assert plain.regions(view(reset(C, 1, seed=0), 0, None)).end_turn is None
    finally:
        plain.close()


def test_the_interactive_loop_survives_a_scripted_hand(monkeypatch):
    """Drive the real play() loop headlessly with synthetic events.

    The pure parts are well covered while the glue that wires them to the window
    is where the shape bugs live. This exercises that glue: the opponent's turns,
    presses, releases, motion and quitting.
    """
    import pygame

    from rummi.render import play as play_module

    pygame.display.quit()
    pygame.display.init()

    events = {"n": 0}
    real_get = pygame.event.get

    def scripted_events(*args, **kwargs):
        real_get(*args, **kwargs)
        events["n"] += 1
        if events["n"] > 9:
            return [pygame.event.Event(pygame.QUIT)]
        # Press a rack tile, drag it across the board, release it on the table.
        return [
            pygame.event.Event(pygame.MOUSEBUTTONDOWN, {"button": 1, "pos": (120, 300)}),
            pygame.event.Event(pygame.MOUSEMOTION, {"pos": (200, 120)}),
            pygame.event.Event(pygame.MOUSEBUTTONUP, {"button": 1, "pos": (60, 60)}),
        ]

    monkeypatch.setattr(pygame.event, "get", scripted_events)
    monkeypatch.setattr(pygame.time, "wait", lambda ms: None)

    play_module.play(TINY_GROUPS, opponent="greedy", seed=4, opponent_delay_ms=0)
    assert events["n"] > 9, "the loop should have consumed the scripted events"
