"""Interactive play, driven headlessly.

Hit-testing is pure -- rectangles and a click position in, an action id out --
so a whole hand can be played here by synthesising clicks, with no window and no
event loop. The property that matters is that the UI cannot express an illegal
move: every click either maps to an action the mask allows, or to nothing.
"""

import numpy as np
import pytest

pygame = pytest.importorskip("pygame")

from rummi.env.numpy.deal import reset
from rummi.env.numpy.engine import step
from rummi.env.numpy.masks import legal_actions
from rummi.render.play import (
    Hit,
    Regions,
    Selection,
    Zone,
    action_for,
    hit,
    legal_slots,
    pick_from_slot,
    redraw,
    regions_for,
)
from rummi.render.pygame_view import DEFAULT_CAPACITY, PygameView
from rummi.render.view_model import view
from rummi.rules.actions import encode_assign, encode_place
from rummi.rules.config import STANDARD as C
from rummi.rules.encoding import kind_of

from tests.conftest import rebalance_pool, state_with

RUN_36 = [kind_of(C, 0, n) for n in (11, 12, 13)]


@pytest.fixture(autouse=True)
def _headless(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")


@pytest.fixture
def window():
    win = PygameView(C, headless=True, reserve_bottom=50)
    yield win
    win.close()


def centre(rect):
    return rect.center


def test_regions_cover_every_rack_tile(window):
    s = state_with(C, rack=RUN_36)
    snapshot = view(s, 0, legal_actions(s))
    regions = regions_for(window, snapshot)
    assert len(regions.rack) == len(snapshot.rack)
    assert [kind for _, kind in regions.rack] == list(snapshot.rack)
    # Every rack rect must resolve back to itself.
    for rect, kind in regions.rack:
        assert hit(regions, centre(rect)) == Hit(Zone.RACK, kind=kind)


def test_clicking_a_rack_tile_places_it(window):
    s = state_with(C, rack=RUN_36)
    mask = legal_actions(s)
    snapshot = view(s, 0, mask)
    regions = regions_for(window, snapshot)

    rect, kind = regions.rack[0]
    spot = hit(regions, centre(rect))
    assert action_for(C, spot, Selection(), mask[0], snapshot) == encode_place(C, kind)


def test_a_click_can_never_produce_an_illegal_action(window):
    """The guarantee the whole UI rests on: if you can click it, it is legal."""
    s = state_with(C, rack=RUN_36, table=[[kind_of(C, 1, n) for n in (1, 2, 3)]], melded=True)
    rng = np.random.default_rng(0)

    for _ in range(40):
        mask = legal_actions(s)
        snapshot = view(s, 0, mask)
        regions = regions_for(window, snapshot)
        selection = Selection()
        if snapshot.workbench:
            selection.kind = snapshot.workbench[0]

        width, height = window.size
        for _ in range(60):
            pos = (int(rng.integers(0, width)), int(rng.integers(0, height)))
            spot = hit(regions, pos)
            if spot is None or spot.zone is Zone.WORKBENCH:
                continue
            action = action_for(C, spot, selection, mask[0], snapshot)
            if action is not None:
                assert mask[0, action], f"{spot.zone} produced illegal action {action}"

        legal = np.flatnonzero(mask[0])
        step(s, np.array([rng.choice(legal)]), mask)
        if s.done[0]:
            break


def test_end_turn_button_is_dead_until_the_meld_is_legal(window):
    s = state_with(C, rack=RUN_36)
    mask = legal_actions(s)
    snapshot = view(s, 0, mask)
    regions = regions_for(window, snapshot)
    end = hit(regions, centre(regions.end_turn))
    assert end.zone is Zone.END_TURN
    assert action_for(C, end, Selection(), mask[0], snapshot) is None, "36 points are not on the table yet"

    for action in [encode_place(C, k) for k in RUN_36] + [encode_assign(C, k, 0) for k in RUN_36]:
        step(s, np.array([action]), legal_actions(s))
    mask = legal_actions(s)
    snapshot = view(s, 0, mask)
    regions = regions_for(window, snapshot)
    assert action_for(C, hit(regions, centre(regions.end_turn)), Selection(), mask[0], snapshot) == C.end_turn_action


def test_draw_is_always_clickable(window):
    s = reset(C, 1, seed=0)
    mask = legal_actions(s)
    snapshot = view(s, 0, mask)
    regions = regions_for(window, snapshot)
    assert action_for(C, hit(regions, centre(regions.draw)), Selection(), mask[0], snapshot) == C.draw_action


def test_highlighted_slots_are_exactly_the_legal_drops(window):
    s = state_with(C, rack=RUN_36)
    step(s, np.array([encode_place(C, RUN_36[0])]), legal_actions(s))
    mask = legal_actions(s)

    highlighted = legal_slots(C, RUN_36[0], mask[0])
    expected = {slot for slot in range(C.max_sets) if mask[0, encode_assign(C, RUN_36[0], slot)]}
    assert highlighted == expected
    assert highlighted, "a held tile with nowhere to go would be a dead end"


def test_clicking_a_set_lifts_a_tile_out_of_it(window):
    """How a rearrangement starts, and it must respect the mask."""
    run = [kind_of(C, 0, n) for n in (5, 6, 7, 8)]
    s = state_with(C, rack=[kind_of(C, 1, 4)], table=[run], melded=True)
    mask = legal_actions(s)
    snapshot = view(s, 0, mask)
    regions = regions_for(window, snapshot)

    rect, slot, tiles = regions.slots[0]
    spot = hit(regions, centre(rect))
    assert spot.zone is Zone.SLOT and spot.tiles == tuple(sorted(run))

    action = action_for(C, spot, Selection(), mask[0], snapshot)
    assert action is not None and mask[0, action]
    step(s, np.array([action]), mask)
    assert s.workbench[0].sum() == 1, "a tile should now be in hand"


def test_before_melding_the_table_cannot_be_touched(window):
    s = state_with(C, rack=RUN_36, table=[[kind_of(C, 1, n) for n in (1, 2, 3)]])
    mask = legal_actions(s)
    snapshot = view(s, 0, mask)
    regions = regions_for(window, snapshot)
    rect, slot, tiles = regions.slots[0]
    assert action_for(C, hit(regions, centre(rect)), Selection(), mask[0], snapshot) is None


def test_a_whole_opening_meld_can_be_played_by_clicking(window):
    """End to end: three rack clicks and three slot clicks, then END TURN."""
    s = state_with(C, rack=RUN_36)
    selection = Selection()

    for _ in range(3):
        mask = legal_actions(s)
        snapshot = view(s, 0, mask)
        regions = regions_for(window, snapshot)

        rect, kind = next((r, k) for r, k in regions.rack if k in RUN_36)
        action = action_for(C, hit(regions, centre(rect)), selection, mask[0], snapshot)
        step(s, np.array([action]), mask)
        selection.kind = kind

        mask = legal_actions(s)
        snapshot = view(s, 0, mask)
        regions = regions_for(window, snapshot)
        drops = legal_slots(C, selection.kind, mask[0])
        target = next(r for r, slot, _ in regions.slots if slot in drops)
        action = action_for(C, hit(regions, centre(target)), selection, mask[0], snapshot)
        step(s, np.array([action]), mask)
        selection.clear()

    mask = legal_actions(s)
    snapshot = view(s, 0, mask)
    regions = regions_for(window, snapshot)
    assert action_for(C, hit(regions, centre(regions.end_turn)), selection, mask[0], snapshot) == C.end_turn_action
    step(s, np.array([C.end_turn_action]), mask)
    assert bool(s.melded[0, 0]) and s.racks[0, 0].sum() == 0
    s.check_invariants()


def test_redraw_rejects_a_batched_mask():
    """The bug this guards: play() holds a (1, A) mask for step() and the UI
    wants one env's row. Passing the batched one used to fail deep inside
    drawing, with an IndexError that named neither shape."""
    win = PygameView(C, headless=True, reserve_bottom=50)
    try:
        s = state_with(C, rack=RUN_36)
        mask = legal_actions(s)
        assert mask.ndim == 2
        with pytest.raises(ValueError, match="single env's mask"):
            redraw(win, view(s, 0, mask), mask, Selection())
        redraw(win, view(s, 0, mask), mask[0], Selection())
    finally:
        win.close()


def test_the_interactive_loop_survives_a_scripted_hand(monkeypatch):
    """Drive the real play() loop headlessly with synthetic events.

    The pure hit-testing was well covered while the glue that wires it to the
    window was not, which is exactly where the shape bug lived. This exercises
    that glue: the opponent's turns, redraws, clicks, and quitting.
    """
    import pygame

    from rummi.render import play as play_module

    pygame.display.quit()
    pygame.display.init()

    clicks = {"n": 0}
    real_get = pygame.event.get

    def scripted_events(*args, **kwargs):
        # Click the first rack tile a few times, then close the window.
        real_get(*args, **kwargs)
        clicks["n"] += 1
        if clicks["n"] > 6:
            return [pygame.event.Event(pygame.QUIT)]
        return [pygame.event.Event(pygame.MOUSEBUTTONDOWN, {"button": 1, "pos": (0, 0)})]

    monkeypatch.setattr(pygame.event, "get", scripted_events)
    monkeypatch.setattr(pygame.time, "wait", lambda ms: None)

    # tiny_groups so the opponent's turns are quick.
    from rummi.rules.config import TINY_GROUPS

    play_module.play(TINY_GROUPS, opponent="greedy", seed=4, opponent_delay_ms=0)
    assert clicks["n"] > 6, "the loop should have consumed the scripted events"


def test_every_slot_is_reachable_when_the_table_is_full():
    """An undrawn slot has no rectangle, so it cannot be clicked. The window's
    display capacity defaults below max_sets, which would silently make part of a
    busy table unplayable -- and a real game reaches 22 sets."""
    cfg = C
    # Two groups per number from three colours: 11 numbers gives 22 sets, using
    # both copies of each tile so nothing collides with a dealt rack.
    rows = [[kind_of(cfg, c, n) for c in (0, 1, 2)] for n in range(1, 12) for _ in (0, 1)]
    s = state_with(cfg, rack=[cfg.joker_kind], table=rows, melded=True)
    s.racks[:, 1] = 0
    rebalance_pool(s)

    snapshot = view(s, 0, legal_actions(s))
    assert len(snapshot.occupied_slots) == 22 > DEFAULT_CAPACITY, "must exceed the display default"

    columns = max(2, -(-cfg.max_sets // 12))
    win = PygameView(cfg, headless=True, columns=columns, capacity=cfg.max_sets, reserve_bottom=50)
    try:
        visible, hidden = win.visible_slots(snapshot)
        assert hidden == 0, f"{hidden} slots would be invisible and unclickable"

        regions = regions_for(win, snapshot)
        reachable = {slot for _, slot, _ in regions.slots}
        for slot in snapshot.occupied_slots:
            assert slot.index in reachable, f"slot {slot.index} has no clickable rect"
    finally:
        win.close()

    # And the default capacity really would have hidden some, so this matters.
    small = PygameView(cfg, headless=True, reserve_bottom=50)
    try:
        assert small.visible_slots(snapshot)[1] > 0
    finally:
        small.close()
