"""Renderers: correctness of the shared view model, dirty tracking, and degradation."""

import io
import os
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


def test_pygame_dirty_set_is_exactly_what_changed():
    from rummi.render.pygame_view import PygameView

    s = state_with(C, rack=RUN_36)
    win = PygameView(C, headless=True)
    try:
        assert len(win.draw(view(s, 0))) > 0, "first draw must paint everything"
        assert win.draw(view(s, 0)) == [], "redrawing the same state must be free"

        play(s, [encode_place(C, RUN_36[0])])
        dirty = win.draw(view(s, 0))
        # PLACE changes the status line, the rack and the workbench -- no slots.
        assert 1 <= len(dirty) <= 4
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


def test_a_table_larger_than_the_window_reports_the_overflow():
    """Truncating silently would make a partial table read as a complete one."""
    from rummi.render.pygame_view import PygameView

    rows = [[kind_of(C, 0, n) for n in (1, 2, 3)]] * 6
    s = state_with(C, rack=[kind_of(C, 0, 5)], table=rows, melded=True)
    win = PygameView(C, headless=True, capacity=4)
    try:
        visible, hidden = win._visible(view(s, 0))
        assert len(visible) == 4
        assert hidden == 3, "6 sets plus a landing slot into a capacity of 4"
    finally:
        win.close()
