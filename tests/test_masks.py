"""Mask rules: what may be done, and the invariants the mask must guarantee."""

import numpy as np
import pytest

from rummi.rules.actions import encode_assign, encode_dissolve, encode_pick, encode_place
from rummi.rules.config import STANDARD, TINY_GROUPS, RummiConfig
from rummi.env.numpy.deal import reset
from rummi.rules.encoding import kind_of
from rummi.env.numpy.engine import step
from rummi.env.numpy.masks import legal_actions, meld_value
from rummi.env.numpy.sets import evaluate_slots

from tests.conftest import rebalance_pool, state_with


def _state_with_rack(cfg: RummiConfig, kinds, batch_size=1, seed=0):
    return state_with(cfg, rack=kinds, batch_size=batch_size, seed=seed)


def _play(cfg, state, actions):
    for a in actions:
        mask = legal_actions(state)
        assert mask[0, a], f"expected action {a} to be legal"
        step(state, np.array([a] * state.batch_size), mask)
    return legal_actions(state)


def test_draw_is_always_legal_so_the_mask_is_never_empty():
    s = reset(STANDARD, 8, seed=1)
    mask = legal_actions(s)
    assert mask[:, STANDARD.draw_action].all()
    assert mask.any(-1).all()


def test_place_is_gated_on_holding_the_tile():
    c = STANDARD
    held = kind_of(c, 0, 5)
    s = _state_with_rack(c, [held])
    mask = legal_actions(s)
    assert mask[0, encode_place(c, held)]
    assert not mask[0, encode_place(c, kind_of(c, 0, 6))]


def test_end_turn_needs_an_empty_workbench():
    c = STANDARD
    kinds = [kind_of(c, 0, n) for n in (11, 12, 13)]
    s = _state_with_rack(c, kinds)
    mask = _play(c, s, [encode_place(c, k) for k in kinds])
    assert s.workbench[0].sum() == 3
    assert not mask[0, c.end_turn_action]


def test_end_turn_unlocks_once_the_meld_is_worth_enough():
    c = STANDARD
    kinds = [kind_of(c, 0, n) for n in (11, 12, 13)]
    s = _state_with_rack(c, kinds)
    plan = [encode_place(c, k) for k in kinds] + [encode_assign(c, k, 0) for k in kinds]
    mask = _play(c, s, plan)
    ev = evaluate_slots(c, s.table_sets)
    assert int(meld_value(s, ev.value)[0]) == 11 + 12 + 13
    assert mask[0, c.end_turn_action]


def test_a_cheap_meld_is_refused():
    c = STANDARD
    kinds = [kind_of(c, 0, n) for n in (1, 2, 3)]  # worth 6, needs 30
    s = _state_with_rack(c, kinds)
    plan = [encode_place(c, k) for k in kinds] + [encode_assign(c, k, 0) for k in kinds]
    mask = _play(c, s, plan)
    assert bool(evaluate_slots(c, s.table_sets).is_valid[0, 0])
    assert not mask[0, c.end_turn_action], "6 points must not satisfy a 30-point meld"


def test_the_existing_table_is_untouchable_before_melding():
    c = STANDARD
    s = _state_with_rack(c, [kind_of(c, 0, 5)])
    # Put a valid set on the table that the player did not create this turn.
    for i, n in enumerate((1, 2, 3)):
        s.table_sets[0, 0, i] = kind_of(c, 1, n)
    s.table_snapshot[:] = s.table_sets
    rebalance_pool(s)

    mask = legal_actions(s)
    assert not mask[0, encode_pick(c, 0, 0)]
    assert not mask[0, encode_dissolve(c, 0)]

    s.melded[0, 0] = True
    mask = legal_actions(s)
    assert mask[0, encode_pick(c, 0, 0)]
    assert mask[0, encode_dissolve(c, 0)]


def test_only_the_lowest_empty_slot_is_offered():
    c = STANDARD
    kind = kind_of(c, 0, 5)
    s = _state_with_rack(c, [kind])
    mask = _play(c, s, [encode_place(c, kind)])
    offered = [i for i in range(c.max_sets) if mask[0, encode_assign(c, kind, i)]]
    assert offered == [0], f"expected one empty slot on offer, got {offered}"


def test_assign_is_refused_when_it_would_kill_the_set():
    c = STANDARD
    # R5 and B7 can never share a set: different colour and different number.
    s = _state_with_rack(c, [kind_of(c, 0, 5), kind_of(c, 1, 7)])
    mask = _play(
        c, s, [encode_place(c, kind_of(c, 0, 5)), encode_place(c, kind_of(c, 1, 7))]
    )
    mask = _play(c, s, [encode_assign(c, kind_of(c, 0, 5), 0)])
    assert not mask[0, encode_assign(c, kind_of(c, 1, 7), 0)]


def test_budget_exhaustion_leaves_only_draw():
    c = STANDARD
    s = _state_with_rack(c, [kind_of(c, 0, 5)])
    s.micro_count[:] = c.max_micro_per_turn
    mask = legal_actions(s)
    assert mask[0].sum() == 1 and mask[0, c.draw_action]


@pytest.mark.parametrize("cfg", [STANDARD, TINY_GROUPS], ids=["standard", "tiny_groups"])
def test_mask_shape(cfg: RummiConfig):
    s = reset(cfg, 5, seed=2)
    assert legal_actions(s).shape == (5, cfg.n_actions)
