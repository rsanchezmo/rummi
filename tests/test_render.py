"""Renderers: correctness of the shared view model, dirty tracking, and degradation."""

import io
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from rummi.rules.actions import encode_assign, encode_place
from rummi.rules.config import STANDARD, TINY_GROUPS
from rummi.env.numpy.deal import reset
from rummi.rules.encoding import kind_of
from rummi.env.numpy.masks import legal_actions
from rummi.render.driver import RenderMode, Renderer
from rummi.render.text import Palette, TerminalView, frame
from rummi.render.view_model import SlotShape, view

from tests.conftest import play, state_with

C = STANDARD
RUN_36 = [kind_of(C, 0, n) for n in (11, 12, 13)]


@pytest.fixture(autouse=True)
def _headless(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")


def test_view_classifies_slot_shapes():
    s = state_with(
        C,
        rack=[kind_of(C, 0, 5)],
        table=[
            [kind_of(C, 0, n) for n in (1, 2, 3)],      # run
            [kind_of(C, c, 4) for c in (1, 2, 3)],      # group
            [kind_of(C, 0, 7), kind_of(C, 0, 8)],       # too short, still completable
            [kind_of(C, 0, 7), kind_of(C, 0, 7)],       # duplicate: never completable
        ],
        melded=True,
    )
    v = view(s, 0, legal_actions(s))
    assert [sl.shape for sl in v.slots[:4]] == [
        SlotShape.RUN,
        SlotShape.GROUP,
        SlotShape.PARTIAL,
        SlotShape.BROKEN,
    ]
    assert not v.table_whole
    assert v.slots[2].blocks_end_turn and v.slots[3].blocks_end_turn
    assert v.slots[0].value == 6 and v.slots[1].value == 12


def test_view_tracks_meld_progress_and_seat():
    s = state_with(C, rack=RUN_36)
    v = view(s, 0)
    assert v.needs_meld and v.meld_progress == 0
    play(s, [encode_place(C, k) for k in RUN_36] + [encode_assign(C, k, 0) for k in RUN_36])
    v = view(s, 0)
    assert v.meld_progress == 36
    assert v.current_player == 0
    assert v.slots[0].is_new


def test_a_view_can_be_pinned_to_one_seat():
    """The play window shows your rack whoever is acting. Following the acting seat
    instead deals the opponent's hand face up every time they take a turn."""
    s = state_with(C, rack=RUN_36)
    play(s, [C.draw_action])  # hand the turn over
    assert int(s.current[0]) == 1

    theirs, mine = view(s, 0), view(s, 0, seat=0)
    assert theirs.perspective == 1 and len(theirs.rack) == s.racks[0, 1].sum()
    assert mine.perspective == 0 and mine.current_player == 1
    assert set(RUN_36) <= set(mine.rack), "your tiles, not theirs"
    assert mine.rack != theirs.rack
    assert mine.needs_meld, "you have not opened; whether they have is their business"
    assert mine.meld_progress == 0, "their turn's progress is not yours to read"


def test_frame_is_plain_text_without_color():
    s = state_with(C, rack=RUN_36, table=[[kind_of(C, 0, n) for n in (1, 2, 3)]], melded=True)
    text = frame(view(s, 0, legal_actions(s)), Palette(False))
    assert "\x1b[" not in text, "monochrome frame must contain no escape codes"
    assert "R11" in text and "R1" in text and "table" in text


def test_frame_marks_a_blocked_end_turn():
    s = state_with(C, rack=[kind_of(C, 0, 5)], table=[[kind_of(C, 0, 7), kind_of(C, 0, 8)]], melded=True)
    text = frame(view(s, 0, legal_actions(s)), Palette(False))
    assert "PARTIAL" in text and "blocks END_TURN" in text


def test_terminal_view_degrades_on_a_non_tty(monkeypatch):
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    s = reset(C, 1, seed=0)
    stream = io.StringIO()  # StringIO has no isatty() returning True
    tv = TerminalView(stream=stream)
    assert not tv.color and not tv.live
    tv.render(view(s, 0))
    assert "\x1b[" not in stream.getvalue()


def test_no_color_env_var_is_respected(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setenv("FORCE_COLOR", "1")
    stream = io.StringIO()
    assert not TerminalView(stream=stream).color


def test_live_terminal_only_rewrites_changed_lines(monkeypatch):
    monkeypatch.setenv("FORCE_COLOR", "1")
    s = state_with(C, rack=RUN_36)
    stream = io.StringIO()
    tv = TerminalView(stream=stream, color=True, live=True)
    tv.render(view(s, 0))
    first = len(stream.getvalue())

    stream.truncate(0), stream.seek(0)
    tv.render(view(s, 0))  # identical frame
    unchanged = stream.getvalue()
    assert "\x1b[2K" not in unchanged, "an unchanged frame must clear no lines"
    assert len(unchanged) < first


def test_the_same_state_draws_the_same_pixels():
    """Frames are painted whole rather than patched, so a repeat draw has to be
    identical -- that is what makes the generated figures byte-stable."""

    from rummi.render.pygame_view import PygameView

    s = state_with(C, rack=RUN_36)
    win = PygameView(C, headless=True)
    try:
        first = win.rgb_array(view(s, 0)).copy()
        np.testing.assert_array_equal(win.rgb_array(view(s, 0)), first)

        play(s, [encode_place(C, RUN_36[0])])
        assert not np.array_equal(win.rgb_array(view(s, 0)), first), "PLACE must show"
    finally:
        win.close()


def test_rgb_array_shape_and_dtype_are_stable():
    from rummi.render.pygame_view import PygameView

    s = reset(C, 2, seed=0)
    win = PygameView(C, headless=True)
    try:
        a = win.rgb_array(view(s, 0))
        b = win.rgb_array(view(s, 1))
        assert a.dtype == np.uint8 and a.ndim == 3 and a.shape[2] == 3
        assert a.shape == b.shape
    finally:
        win.close()


def test_atlas_builds_headless_for_every_config():
    from rummi.render.atlas import Variant, build

    for cfg in (STANDARD, TINY_GROUPS):
        atlas = build(cfg, tile_w=20, tile_h=28)
        w, h = atlas.surface.get_size()
        assert w == cfg.n_kinds * 20
        assert h == len(Variant) * 28
        assert atlas.rect(cfg.joker_kind, Variant.NEW).y == 28


def test_render_mode_none_is_a_no_op():
    s = reset(C, 1, seed=0)
    r = Renderer(C, RenderMode.NONE)
    assert r.render(s) is None
    assert r._terminal is None and r._window is None, "nothing should be constructed"


def test_explicit_frame_ignores_the_throttle():
    s = reset(C, 1, seed=0)
    r = Renderer(C, RenderMode.RGB_ARRAY, every=100)
    try:
        assert r.render(s) is None, "the step loop must respect the throttle"
        assert r.frame(s) is not None, "an explicit render() must always draw"
    finally:
        r.close()


def test_throttling_skips_frames():
    s = reset(C, 1, seed=0)
    drawn = []
    r2 = Renderer(C, RenderMode.ANSI, every=3)
    r2._terminal = type("Spy", (), {"render": lambda self, v: drawn.append(v), "close": lambda self: None})()
    for _ in range(9):
        r2.render(s)
    assert len(drawn) == 3


def test_recording_replays_to_an_identical_state(tmp_path: Path):
    from rummi.render.record import Recorder, replay

    cfg = TINY_GROUPS
    path = tmp_path / "game.jsonl"
    from rummi.render.record import play as play_game

    with Recorder(path, cfg, seed=5) as rec:
        play_game(cfg, "greedy", 5, Renderer(cfg, RenderMode.NONE), recorder=rec)

    live = reset(cfg, 1, seed=5)
    from rummi.bench.fuzz import make_policy
    from rummi.env.numpy.engine import step

    policy = make_policy(cfg, "greedy", 5)
    while not live.done.all():
        m = legal_actions(live)
        step(live, policy(live, m), m)

    replayed = replay(path, Renderer(cfg, RenderMode.NONE))
    np.testing.assert_array_equal(replayed.table_sets, live.table_sets)
    np.testing.assert_array_equal(replayed.racks, live.racks)
    assert int(replayed.winner[0]) == int(live.winner[0])
    assert int(replayed.turn_count[0]) == int(live.turn_count[0])


def test_action_history_survives_render_throttling():
    """The log lives in the state, so it is complete regardless of how often the
    renderer is allowed to draw."""
    from rummi.env.numpy.state import HISTORY_LEN

    s = state_with(C, rack=RUN_36)
    actions = [encode_place(C, k) for k in RUN_36] + [encode_assign(C, k, 0) for k in RUN_36]
    play(s, actions)
    v = view(s, 0)
    assert v.history[0] == "ASSIGN(R13, slot=0)"
    assert len(v.history) == min(len(actions), HISTORY_LEN)
    # Most recent first.
    assert v.history[1] == "ASSIGN(R12, slot=0)"


def test_touched_slot_tracks_the_last_action():
    s = state_with(C, rack=RUN_36)
    play(s, [encode_place(C, RUN_36[0])])
    assert view(s, 0).touched_slot is None, "PLACE touches no slot"
    play(s, [encode_assign(C, RUN_36[0], 0)])
    assert view(s, 0).touched_slot == 0


def test_every_set_on_the_table_gets_a_card():
    """A set with no rectangle is a set that cannot be clicked."""
    from rummi.render.pygame_view import PygameView

    rows = [[kind_of(C, c, n) for c in (0, 1, 2)] for n in (1, 2, 3, 4, 5, 6)]
    s = state_with(C, rack=[kind_of(C, 0, 8)], table=rows, melded=True)
    win = PygameView(C, headless=True)
    try:
        snapshot = view(s, 0)
        regions = win.draw(snapshot)
        assert {card.slot for card in regions.cards if not card.is_landing} == {
            slot.index for slot in snapshot.occupied_slots
        }
        assert sum(card.is_landing for card in regions.cards) == 1, "one place to start a set"
        for card in regions.cards:
            assert win.metrics.table.contains(card.rect)
            assert len(card.spots) == len(card.tiles)
    finally:
        win.close()


@pytest.mark.parametrize(
    "numbers,expected",
    [
        ((10, 11, 13), ["R10", "R11", "*", "R13"]),   # joker fills the gap
        ((10, 11, 12), ["R10", "R11", "R12", "*"]),   # nothing missing: it extends
        ((11, 12, 13), ["*", "R11", "R12", "R13"]),   # no 14 exists, so it is the 10
    ],
)
def test_a_joker_is_shown_where_it_belongs(numbers, expected):
    """`R10 R11 R13 *` is a legal run with the joker standing for 12, but the
    joker sorts last in storage, so unmodified it reads as if the run skipped a
    number."""
    kinds = [kind_of(C, 0, n) for n in numbers] + [C.joker_kind]
    s = state_with(C, rack=[kind_of(C, 0, 5)], table=[kinds], melded=True)
    slot = view(s, 0).slots[0]
    assert slot.shape is SlotShape.RUN
    assert [view(s, 0).label(k) for k in slot.shown] == expected
    assert sorted(slot.shown) == sorted(slot.tiles), "reordering must not add or drop a tile"


def test_storage_order_is_left_alone_for_pick():
    """PICK indexes the stored order, so it must stay as the engine wrote it --
    reordering it for looks would lift the wrong tile."""
    kinds = [kind_of(C, 0, n) for n in (10, 11, 13)] + [C.joker_kind]
    s = state_with(C, rack=[kind_of(C, 0, 5)], table=[kinds], melded=True)
    slot = view(s, 0).slots[0]
    assert list(slot.tiles) == sorted(kinds)
    assert slot.shown != slot.tiles, "this case should differ, or the test proves nothing"


def _render_docs():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    import render_docs

    return render_docs


def test_a_recording_deals_its_agents_round_the_seats():
    """One name per seat, cycled: `optimal,frugal` has to mean alternating seats at
    three and four as well as two, or a recording of a 4p game would be two agents
    and two empty chairs."""
    from rummi.rules.config import STANDARD_3P, STANDARD_4P

    seat_agents = _render_docs().seat_agents
    for cfg, expected in (
        (STANDARD, ["greedy", "rearrange"]),
        (STANDARD_3P, ["greedy", "rearrange", "greedy"]),
        (STANDARD_4P, ["greedy", "rearrange", "greedy", "rearrange"]),
    ):
        assert [a.name for a in seat_agents(cfg, ["greedy", "rearrange"])] == expected
    assert [a.name for a in seat_agents(STANDARD_3P, ["greedy"])] == ["greedy"] * 3


def test_a_recording_samples_one_frame_per_committed_turn():
    """What makes the figures readable: a turn is the unit of play, so a frame is a
    board that changed rather than a tile that moved."""
    rd = _render_docs()
    cap = 8
    frames = list(rd.game_frames(TINY_GROUPS, 5, rd.seat_agents(TINY_GROUPS, ["greedy"]), cap))

    turns = [f.turn for f in frames]
    assert turns[0] == 0, "the opening position is the first frame"
    # A frame per committed turn, and the last turn repeats when the game ends on it.
    assert {b - a for a, b in pairwise(turns)} <= {0, 1}
    assert frames[-1].done or turns[-1] == cap
