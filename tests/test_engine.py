"""Effects of each action family, turn commitment, termination and reward."""

import numpy as np
import pytest

from rummi.rules.actions import encode_assign, encode_dissolve, encode_place
from rummi.rules.config import STANDARD, RewardMode, RummiConfig
from rummi.env.numpy.deal import reset
from rummi.rules.encoding import EMPTY, kind_of
from rummi.env.numpy.engine import step
from rummi.env.numpy.masks import legal_actions

from tests.conftest import drain_pool, play, rebalance_pool, state_with

C = STANDARD
RUN_36 = [kind_of(C, 0, n) for n in (11, 12, 13)]
GROUP_4 = [kind_of(C, c, 4) for c in (1, 2, 3)]


def meld_plan(cfg, kinds, slot=0):
    return [encode_place(cfg, k) for k in kinds] + [
        encode_assign(cfg, k, slot) for k in kinds
    ]


def test_place_moves_a_tile_to_the_workbench():
    s = state_with(C, rack=RUN_36)
    play(s, [encode_place(C, RUN_36[0])])
    assert s.racks[0, 0, RUN_36[0]] == 0
    assert s.workbench[0, RUN_36[0]] == 1
    assert s.placed_rack[0, RUN_36[0]] == 1
    s.check_invariants()


def test_assign_fills_the_slot_in_canonical_order_and_marks_it_new():
    s = state_with(C, rack=RUN_36)
    play(s, meld_plan(C, RUN_36))
    row = s.table_sets[0, 0]
    assert list(row[:3]) == sorted(RUN_36)
    assert (row[3:] == EMPTY).all()
    assert bool(s.slot_new[0, 0])
    s.check_invariants()


def test_dissolve_returns_a_whole_set_to_the_workbench():
    s = state_with(C, table=[GROUP_4], rack=[kind_of(C, 0, 7)], melded=True)
    play(s, [encode_dissolve(C, 0)])
    assert (s.table_sets[0, 0] == EMPTY).all()
    for k in GROUP_4:
        assert s.workbench[0, k] == 1
    s.check_invariants()


def test_pick_takes_one_tile_and_recanonicalises():
    from rummi.rules.actions import encode_pick

    s = state_with(C, table=[GROUP_4], rack=[kind_of(C, 0, 7)], melded=True)
    taken = int(s.table_sets[0, 0, 0])
    play(s, [encode_pick(C, 0, 0)])
    assert s.workbench[0, taken] == 1
    row = s.table_sets[0, 0]
    assert (row[:2] >= 0).all() and (row[2:] == EMPTY).all(), "EMPTY must sort to the end"
    s.check_invariants()


def test_end_turn_commits_and_passes_play():
    s = state_with(C, rack=RUN_36)
    play(s, [*meld_plan(C, RUN_36), C.end_turn_action])
    assert bool(s.melded[0, 0])
    assert s.current[0] == 1
    assert s.turn_count[0] == 1
    assert s.micro_count[0] == 0
    assert s.workbench[0].sum() == 0
    assert s.placed_rack[0].sum() == 0
    assert not s.slot_new[0].any()
    np.testing.assert_array_equal(s.table_snapshot[0], s.table_sets[0])
    s.check_invariants()


def test_draw_reverts_the_whole_turn_exactly():
    s = state_with(C, table=[GROUP_4], rack=RUN_36, melded=True)
    before_table = s.table_sets[0].copy()
    before_rack = s.racks[0, 0].copy()
    before_pool = int(s.pool_size[0])

    # Stage a mess: tiles out of the rack and a table set taken apart.
    play(s, [encode_place(C, RUN_36[0]), encode_dissolve(C, 0)])
    assert not np.array_equal(s.table_sets[0], before_table)

    play(s, [C.draw_action])
    np.testing.assert_array_equal(s.table_sets[0], before_table)
    assert s.racks[0, 0].sum() == before_rack.sum() + 1, "should have drawn exactly one tile"
    assert s.pool_size[0] == before_pool - 1
    assert s.workbench[0].sum() == 0 and s.placed_rack[0].sum() == 0
    s.check_invariants()


def test_draw_on_an_empty_pool_is_a_pass():
    s = state_with(C, rack=RUN_36)
    drain_pool(s, to_player=1)
    before = s.racks[0, 0].sum()
    play(s, [C.draw_action])
    assert s.racks[0, 0].sum() == before
    assert s.current[0] == 1


def test_emptying_the_rack_wins():
    s = state_with(C, rack=RUN_36, melded=True)
    play(s, [*meld_plan(C, RUN_36), C.end_turn_action])
    assert bool(s.done[0]) and not bool(s.truncated[0])
    assert int(s.winner[0]) == 0
    s.check_invariants()


def test_stalemate_ends_the_game_on_lowest_rack():
    cfg = RummiConfig(n_players=2, reward_mode=RewardMode.WIN_LOSS)
    s = state_with(cfg, rack=[kind_of(cfg, 0, 1)])
    s.racks[0, 1] = 0
    s.racks[0, 1, kind_of(cfg, 0, 13)] = 1  # seat 1 holds more value
    rebalance_pool(s)
    drain_pool(s, to_player=1)
    s.consecutive_draws[:] = cfg.n_players - 1

    play(s, [cfg.draw_action])
    assert bool(s.done[0])
    assert int(s.winner[0]) == 0, "lowest rack value should win a stalled game"


def test_running_out_of_turns_truncates_without_a_winner():
    cfg = RummiConfig(n_players=2, max_turns=1)
    s = state_with(cfg, rack=[kind_of(cfg, 0, 1)])
    result = None
    mask = legal_actions(s)
    result = step(s, np.array([cfg.draw_action]), mask)
    assert bool(s.done[0]) and bool(s.truncated[0])
    assert bool(result.truncated[0]) and not bool(result.terminated[0])
    assert (result.rewards == 0).all(), "an artificial cutoff should not pay out"


def test_win_loss_reward_is_zero_sum():
    cfg = RummiConfig(n_players=3, reward_mode=RewardMode.WIN_LOSS)
    kinds = [kind_of(cfg, 0, n) for n in (11, 12, 13)]
    s = state_with(cfg, rack=kinds, melded=True)
    for a in meld_plan(cfg, kinds):
        step(s, np.array([a]), legal_actions(s))
    result = step(s, np.array([cfg.end_turn_action]), legal_actions(s))
    assert int(s.winner[0]) == 0
    assert result.rewards[0, 0] == pytest.approx(1.0)
    assert result.rewards[0].sum() == pytest.approx(0.0)


def test_score_reward_pays_the_winner_the_losers_racks():
    cfg = RummiConfig(n_players=2, reward_mode=RewardMode.SCORE)
    kinds = [kind_of(cfg, 0, n) for n in (11, 12, 13)]
    s = state_with(cfg, rack=kinds, melded=True)
    s.racks[0, 1] = 0
    s.racks[0, 1, kind_of(cfg, 1, 7)] = 1
    s.racks[0, 1, cfg.joker_kind] = 1
    rebalance_pool(s)
    for a in meld_plan(cfg, kinds):
        step(s, np.array([a]), legal_actions(s))
    result = step(s, np.array([cfg.end_turn_action]), legal_actions(s))
    owed = 7 + cfg.joker_penalty
    assert result.rewards[0, 1] == pytest.approx(-owed)
    assert result.rewards[0, 0] == pytest.approx(owed)


def test_illegal_action_is_rejected_loudly():
    s = reset(C, 1, seed=0)
    mask = legal_actions(s)
    with pytest.raises(ValueError, match="illegal action"):
        step(s, np.array([C.end_turn_action]), mask)


def test_done_envs_ignore_further_actions():
    s = state_with(C, rack=RUN_36, melded=True)
    play(s, [*meld_plan(C, RUN_36), C.end_turn_action])
    assert bool(s.done[0])
    frozen = s.clone()
    step(s, np.array([C.draw_action]), legal_actions(s))
    np.testing.assert_array_equal(s.racks, frozen.racks)
    np.testing.assert_array_equal(s.table_sets, frozen.table_sets)
    assert s.turn_count[0] == frozen.turn_count[0]


def test_end_turn_canonicalises_slot_order():
    # Two sets laid in descending order must come back sorted, so the same table
    # always has the same representation.
    low = [kind_of(C, 0, n) for n in (1, 2, 3)]
    high = [kind_of(C, 0, n) for n in (11, 12, 13)]
    s = state_with(C, rack=high + low, melded=True)
    play(s, meld_plan(C, high, slot=0))
    play(s, meld_plan(C, low, slot=1))
    play(s, [C.end_turn_action])
    assert int(s.table_sets[0, 0, 0]) == min(low), "lowest set should sort first"
