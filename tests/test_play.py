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
    PAD,
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
    play,
    take,
)
from rummi.render.pygame_view import Overlay, PygameView
from rummi.render.view_model import view
from rummi.rules.actions import encode_assign, encode_place
from rummi.rules.config import CONFIG_BY_NAME, STANDARD as C, TINY_GROUPS
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


def test_a_mixed_width_table_keeps_every_card_on_the_felt():
    """Uniform tables are not the worst case for greedy wrapping.

    A slot holding fewer than ``min_set`` tiles is still drawn ``min_set`` wide, so
    a one-tile card costs a single tile and still eats a whole card's width, while
    the widest runs the deck allows keep forcing the wrap. Five 13-runs and sixteen
    singletons -- 21 slots and 81 tiles, a state a game reaches mid-turn -- wrap
    onto a row that the same tiles laid out uniformly never reach.

    Off the felt the cards are drawn over the workbench and the button column, so a
    click meant for the set presses a button instead.
    """
    runs = [[kind_of(C, colour, n) for n in range(1, 14)] for colour in (0, 0, 1, 1, 2)]
    singles = [[kind_of(C, 2, n)] for n in range(1, 14)] + [[kind_of(C, 3, n)] for n in (1, 2, 3)]
    table: list[list[int]] = []
    for row, run in enumerate(runs[:-1]):
        table.append(run)
        table.extend(singles[4 * row : 4 * row + 4])
    table.append(runs[-1])

    s = state_with(C, rack=[kind_of(C, 3, 13)], table=table, melded=True)
    snapshot = view(s, 0, legal_actions(s))
    assert len(snapshot.occupied_slots) == 21

    for interactive in (True, False):
        m = metrics_for(C, interactive=interactive)
        r = regions_for(m, snapshot)
        for card in r.cards:
            assert m.table.contains(card.rect), f"card {card.slot} is drawn off the table"
            landed = hit(r, card.rect.center)
            assert landed is not None and landed.slot == card.slot, "the click went elsewhere"


def _slots_of(cfg, sizes: list[int]) -> list[list[int]]:
    """Real tiles for a table of these slot sizes, taken from the deck's copies.

    What a card is drawn as depends only on how many tiles the slot holds, so the
    sets need not be legal -- mid-turn they routinely are not.
    """
    from rummi.rules.encoding import tables

    deck = [k for k, n in enumerate(tables(cfg).copies) for _ in range(int(n))]
    out, at = [], 0
    for size in sizes:
        out.append(deck[at : at + size])
        at += size
    assert at <= len(deck), "the table asks for more tiles than the deck holds"
    return out


def _rows_of(cfg, m, sizes: list[int]) -> int:
    """How many rows the real layout puts these slots on."""
    s = state_with(cfg, rack=[], table=_slots_of(cfg, sizes), melded=True)
    cards = regions_for(m, view(s, 0)).cards
    assert len(cards) == len(sizes) + 1, "every slot gets a card, plus the landing one"
    return len({card.rect.y for card in cards})


@pytest.mark.parametrize("preset", sorted(CONFIG_BY_NAME))
def test_the_worst_wrapping_the_config_allows_still_fits(preset: str):
    """Hunt for the table greedy wrapping likes least, with the real layout as the
    oracle: start a row with the widest set the deck allows whenever that forces a
    wrap, and pad with one-tile slots when it does not. Sweeping uniform tables
    misses this, and the cards it finds have to have a rectangle on the felt.
    """
    cfg = CONFIG_BY_NAME[preset]
    # What one seat can put on the table: the others are still holding their deal.
    budget = cfg.n_tiles - cfg.rack_size * (cfg.n_players - 1)
    for interactive in (True, False):
        m = metrics_for(cfg, interactive=interactive)
        sizes: list[int] = []
        rows = 1
        while len(sizes) < cfg.max_sets - 1:  # one slot left for the landing card
            wide = [*sizes, cfg.max_set_len]
            wide_rows = _rows_of(cfg, m, wide) if sum(wide) <= budget else rows
            if wide_rows > rows:
                sizes, rows = wide, wide_rows
            elif sum(sizes) + 1 <= budget:
                sizes = [*sizes, 1]
                rows = _rows_of(cfg, m, sizes)
            else:
                break

        s = state_with(cfg, rack=[], table=_slots_of(cfg, sizes), melded=True)
        r = regions_for(m, view(s, 0))
        for card in r.cards:
            assert m.table.contains(card.rect), (
                f"{preset}: {len(sizes)} slots on {rows} rows push card {card.slot} off the felt"
            )


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


def test_a_seat_the_config_does_not_have_is_refused(monkeypatch):
    """`--seat` reaches a numpy index, where 2 is an IndexError naming neither the
    flag nor the seat, and -1 quietly wraps to the last seat -- which is the
    opponent, and the one thing the window must never show."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    for seat in (C.n_players, -1):
        with pytest.raises(ValueError, match="is not a seat"):
            play(C, seat=seat)


@pytest.mark.parametrize("preset", sorted(CONFIG_BY_NAME))
def test_the_button_and_hint_columns_fit_beside_the_rack(preset: str):
    """The buttons stack in the margin beside the rack, so the window has to be
    wide enough for both columns -- and a button drawn past the edge cannot be
    pressed, which is the same failure as a set with no rectangle.

    Only the horizontal direction was ever unchecked, and it is the one that
    breaks: on a reduced config the margin is narrower than a button, so the bar
    ran 45px past the edge and the hint column was painted over the rack.
    """
    cfg = CONFIG_BY_NAME[preset]
    m = metrics_for(cfg, interactive=True)
    assert m.controls is not None and m.hint is not None
    assert m.controls.right <= m.width - PAD, f"{preset}: the button bar leaves the window"
    assert m.hint.left >= PAD, f"{preset}: the hint column leaves the window"
    assert not m.controls.colliderect(m.rack), f"{preset}: buttons drawn over the rack"
    assert not m.hint.colliderect(m.rack), f"{preset}: hints drawn over the rack"

    # The rects a click actually resolves against, not just the column holding them.
    regions = regions_for(m, view(reset(cfg, 1, seed=0), 0, None))
    window = pygame.Rect(0, 0, m.width, m.height)
    for name, button in (
        ("end_turn", regions.end_turn), ("draw", regions.draw), ("undo", regions.undo)
    ):
        assert button is not None
        assert window.contains(button), f"{preset}: {name} is drawn outside the window"
        assert not button.colliderect(m.rack), f"{preset}: {name} is drawn over the rack"


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


def test_two_views_can_be_opened_and_closed_in_sequence():
    """A view owns its surfaces, and a window owns the display it opened. Closing
    one must give back exactly that: taking the whole library down instead leaves
    every other view in the process holding an uninitialised display, and drawing
    on one raises before it can say why."""
    import pygame

    pygame.display.init()
    first = PygameView(C, headless=True)
    second = PygameView(C, headless=True, interactive=True)
    snapshot = view(reset(C, 1, seed=0), 0, None)

    first.close()
    assert pygame.display.get_init(), "closing one view must not take the display down"
    assert second.rgb_array(snapshot).shape == (second.size[1], second.size[0], 3)
    second.close()

    third = PygameView(C, headless=True)
    try:
        assert third.rgb_array(snapshot).shape == (third.size[1], third.size[0], 3)
    finally:
        third.close()


def _drive(monkeypatch, on_rival, seed: int = 0, seat: int = 1):
    """Run the real loop with the opponent moving first, posting real events from
    inside its turn. Returns the actions the *player's* side of the loop applied.

    Nothing is scripted into ``pygame.event.get``: the point is what the queue does
    while the opponent plays, so the events go through the queue itself.
    """
    import pygame

    from rummi.render import play as play_module

    applied: list[int] = []
    calls: list[int] = []
    real_act = play_module.act_on_state
    real_apply = play_module.Session.apply
    real_snapshot = play_module.Session.snapshot

    def spy_act(agent, state, mask):
        calls.append(len(calls))
        on_rival(pygame)
        return real_act(agent, state, mask)

    def spy_apply(self, action):
        applied.append(int(action))
        real_apply(self, action)

    def spy_snapshot(self):
        # Once the turn is the player's the loop would otherwise sit on the queue
        # for ever; the game under test is over by then.
        if int(self.state.current[0]) == seat:
            pygame.event.post(pygame.event.Event(pygame.QUIT))
        return real_snapshot(self)

    monkeypatch.setattr(play_module, "act_on_state", spy_act)
    monkeypatch.setattr(play_module.Session, "apply", spy_apply)
    monkeypatch.setattr(play_module.Session, "snapshot", spy_snapshot)
    monkeypatch.setattr(pygame.time, "wait", lambda ms: None)

    play_module.play(TINY_GROUPS, opponent="greedy", seed=seed, seat=seat, opponent_delay_ms=0)
    return applied, calls


def test_closing_the_window_during_the_opponents_turn_is_honoured_at_once(monkeypatch):
    """The opponent's turn is seven micro-actions on this seed, each paced by a
    wait. Leaving the queue unread across them means a close request is answered
    when the turn ends, seconds later, which reads as a hung window."""
    applied, calls = _drive(monkeypatch, lambda pg: pg.event.post(pg.event.Event(pg.QUIT)))
    assert len(calls) == 1, "the loop played on after the window was asked to close"
    assert not applied, "the player never acted"


def test_a_click_made_while_the_opponent_plays_is_not_replayed_afterwards(monkeypatch):
    """A press aimed at the opponent's board must not be banked for the player's
    next turn, where it lands on a board that has changed underneath it. DRAW is
    the sharp case: it is never masked, so a stray click on it reverts and passes
    the turn it is finally delivered on."""
    import pygame

    m = metrics_for(TINY_GROUPS, interactive=True)
    r = regions_for(m, view(reset(TINY_GROUPS, 1, seed=0), 0, None))
    at = centre(r.draw)

    def click(pg):
        pg.event.post(pg.event.Event(pygame.MOUSEBUTTONDOWN, {"button": 1, "pos": at}))
        pg.event.post(pg.event.Event(pygame.MOUSEBUTTONUP, {"button": 1, "pos": at}))

    applied, calls = _drive(monkeypatch, click)
    assert len(calls) > 1, "the opponent should have taken a whole turn"
    assert TINY_GROUPS.draw_action not in applied, "a click from the opponent's turn was replayed"
