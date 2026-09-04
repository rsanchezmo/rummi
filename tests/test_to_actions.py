"""The plan translator, and through it the completeness of the action space.

If every table CP-SAT proposes can be reached by PLACE/PICK/DISSOLVE/ASSIGN within
the per-turn budget, then the micro-action decomposition loses nothing: no turn the
rules permit is inexpressible in the MDP.

The other half is length, which is why the three renderings below are pinned to an
exact action count: a standing set the target only lengthens or only shortens is
morphed in place, and a plan that dissolved it would be legal and several times
longer -- the difference a student imitating these sequences is taught.
"""

from collections import Counter

import numpy as np
import pytest

pytest.importorskip("ortools")

from rummi.rules.config import STANDARD, TINY, TINY_GROUPS, RummiConfig
from rummi.env.numpy.deal import reset
from rummi.rules.encoding import EMPTY, kind_of
from rummi.env.numpy.engine import step
from rummi.env.numpy.masks import legal_actions
from rummi.agents import OptimalAgent
from rummi.agents.base import act_on_state
from rummi.solver.ilp import solve_turn
from rummi.solver.to_actions import plan, slot_contents

C = STANDARD


def _contents(table):
    return Counter(c for c in slot_contents(table) if c)


def _rack(cfg, kinds):
    r = np.zeros(cfg.n_kinds, dtype=np.int64)
    for k in kinds:
        r[k] += 1
    return r


def _table(cfg, rows):
    t = np.full((cfg.max_sets, cfg.max_set_len), EMPTY, dtype=np.int16)
    for i, row in enumerate(rows):
        t[i, : len(row)] = row
    return t


def test_untouched_sets_are_left_alone():
    """Stillness is not cosmetic: each kept set is two fewer actions."""
    table = _table(C, [[kind_of(C, 0, n) for n in (1, 2, 3)], [kind_of(C, 1, n) for n in (5, 6, 7)]])
    target = [
        tuple(kind_of(C, 0, n) for n in (1, 2, 3)),
        tuple(kind_of(C, 1, n) for n in (5, 6, 7)),
        tuple(kind_of(C, 2, n) for n in (9, 10, 11)),
    ]
    played = np.zeros(C.n_kinds, dtype=np.int16)
    for n in (9, 10, 11):
        played[kind_of(C, 2, n)] = 1

    actions = plan(C, table, target, played)
    assert not [a for a in actions if C.dissolve_offset <= a < C.assign_offset], (
        "nothing needed dissolving"
    )
    assert sum(1 for a in actions if a < C.n_kinds) == 3


def _play(cfg, table_rows, rack_kinds, target, played):
    """Replay a plan on a real state and hand back the table it reached."""
    from tests.conftest import rebalance_pool

    s = reset(cfg, 1, seed=0)
    s.racks[:, 0] = 0
    s.racks[0, 0] = _rack(cfg, rack_kinds)
    s.table_sets[0] = _table(cfg, table_rows)
    s.table_snapshot[:] = s.table_sets
    s.melded[:] = True
    rebalance_pool(s)

    actions = plan(cfg, s.table_sets[0], target, played)
    for action in actions:
        mask = legal_actions(s)
        assert mask[0, action], f"planned action {action} was illegal"
        step(s, np.array([action]), mask)
    s.check_invariants()
    return actions, s


def test_a_lay_off_extends_the_standing_set_rather_than_rebuilding_it():
    """The commonest move in the game: PLACE, ASSIGN onto the slot, END_TURN.

    Dissolving the receiving set and laying it out again reaches the same table in
    seven actions where the mask allows three.
    """
    run = [kind_of(C, 0, n) for n in (1, 2, 3)]
    played = np.zeros(C.n_kinds, dtype=np.int16)
    played[kind_of(C, 0, 4)] = 1

    actions, s = _play(
        C, [run], [kind_of(C, 0, 4)],
        [tuple(kind_of(C, 0, n) for n in (1, 2, 3, 4))], played,
    )
    assert len(actions) == 3
    assert not [a for a in actions if C.dissolve_offset <= a < C.assign_offset]
    assert not [a for a in actions if C.pick_offset <= a < C.dissolve_offset]
    assert _contents(s.table_sets[0]) == Counter(
        {tuple(sorted(kind_of(C, 0, n) for n in (1, 2, 3, 4))): 1}
    )


def test_a_shortened_set_is_picked_from_rather_than_dissolved():
    """A steal leaves the donor shorter, and PICK is the action that says so."""
    donor = [kind_of(C, 0, n) for n in (5, 6, 7, 8)]
    rack = [kind_of(C, 1, 8), kind_of(C, 2, 8)]
    played = np.zeros(C.n_kinds, dtype=np.int16)
    for kind in rack:
        played[kind] = 1
    target = [
        tuple(kind_of(C, 0, n) for n in (5, 6, 7)),
        tuple(sorted((kind_of(C, 0, 8), *rack))),
    ]

    actions, s = _play(C, [donor], rack, target, played)
    picks = [a for a in actions if C.pick_offset <= a < C.dissolve_offset]
    assert len(picks) == 1
    assert not [a for a in actions if C.dissolve_offset <= a < C.assign_offset]
    assert len(actions) == 7
    assert _contents(s.table_sets[0]) == Counter(tuple(sorted(x)) for x in target)


def test_a_target_that_is_neither_longer_nor_shorter_still_dissolves():
    """Morphing only reaches one direction at a time, so a different set rebuilds.

    The mask has no action that swaps a tile, and picking one out and assigning
    another in would cost what the dissolve costs anyway.
    """
    run = [kind_of(C, 0, n) for n in (5, 6, 7)]
    rack = [kind_of(C, 0, 4), kind_of(C, 0, 8), kind_of(C, 0, 9)]
    played = np.zeros(C.n_kinds, dtype=np.int16)
    for kind in rack:
        played[kind] = 1
    target = [
        tuple(kind_of(C, 0, n) for n in (4, 5, 6)),
        tuple(kind_of(C, 0, n) for n in (7, 8, 9)),
    ]

    actions, s = _play(C, [run], rack, target, played)
    assert [a for a in actions if C.dissolve_offset <= a < C.assign_offset]
    assert not [a for a in actions if C.pick_offset <= a < C.dissolve_offset]
    assert _contents(s.table_sets[0]) == Counter(tuple(sorted(x)) for x in target)


def test_an_unbalanced_plan_is_rejected():
    """A target that does not account for every tile is a solver bug, and must
    fail here rather than as a mystery illegal action later."""
    table = _table(C, [[kind_of(C, 0, n) for n in (1, 2, 3)]])
    with pytest.raises(ValueError, match="does not balance"):
        plan(C, table, [tuple(kind_of(C, 0, n) for n in (1, 2, 3, 4))], np.zeros(C.n_kinds, np.int16))


def test_the_split_manoeuvre_executes_legally_and_lands_the_target():
    rack = _rack(C, [kind_of(C, 1, 4), kind_of(C, 0, 7), kind_of(C, 2, 7)])
    s = reset(C, 1, seed=0)
    s.racks[:, 0] = 0
    s.racks[0, 0] = rack
    s.table_sets[0] = _table(C, [[kind_of(C, 1, n) for n in (5, 6, 7)]])
    s.table_snapshot[:] = s.table_sets
    s.melded[:] = True
    from tests.conftest import rebalance_pool

    rebalance_pool(s)

    sol = solve_turn(C, rack, s.table_sets[0], True)
    actions = plan(C, s.table_sets[0], sol.sets, sol.played)
    assert len(actions) <= C.max_micro_per_turn

    for action in actions:
        mask = legal_actions(s)
        assert mask[0, action], f"planned action {action} was illegal"
        step(s, np.array([action]), mask)

    assert _contents(s.table_sets[0]) == Counter(tuple(sorted(x)) for x in sol.sets)
    assert s.racks[0, 0].sum() == 0, "all three tiles should have been played"
    s.check_invariants()


@pytest.mark.parametrize(
    "cfg", [TINY, TINY_GROUPS, STANDARD], ids=["tiny", "tiny_groups", "standard"]
)
def test_every_solver_target_is_reachable(cfg: RummiConfig):
    """The completeness claim, checked over real play on every config."""
    plans_checked = 0
    plans_with_rearrangement = 0
    # Reduced configs finish in a handful of turns, so gather plans across several
    # games rather than leaning on one long one.
    for seed in range(6):
        policy = OptimalAgent(cfg)
        policy.reset(2)
        state = reset(cfg, 2, seed=13 + seed)
        for _ in range(250):
            for env in np.flatnonzero(state.micro_count == 0):
                if state.done[env]:
                    continue
                player = int(state.current[env])
                melded = bool(state.melded[env, player])
                sol = solve_turn(
                    cfg, state.racks[env, player].astype(np.int64), state.table_sets[env], melded
                )
                if not sol.plays_anything:
                    continue
                target = list(sol.sets)
                if not melded:
                    target += [c for c in slot_contents(state.table_sets[env]) if c]

                actions = plan(cfg, state.table_sets[env], target, sol.played)
                assert len(actions) <= cfg.max_micro_per_turn, (
                    f"plan of {len(actions)} exceeds the per-turn budget "
                    f"{cfg.max_micro_per_turn}"
                )
                # Replay it on a copy so the real game is unaffected.
                probe = state.select(env)
                for action in actions:
                    mask = legal_actions(probe)
                    assert mask[0, action], f"unreachable target: action {action} illegal"
                    step(probe, np.array([action]), mask)
                assert _contents(probe.table_sets[0]) == Counter(
                    tuple(sorted(x)) for x in target
                )
                probe.check_invariants()
                plans_checked += 1
                plans_with_rearrangement += any(
                    cfg.dissolve_offset <= a < cfg.assign_offset for a in actions
                )

            mask = legal_actions(state)
            step(state, act_on_state(policy, state, mask), mask)
            if state.done.all():
                break

    assert plans_checked >= 5, f"only {plans_checked} plans were exercised"
    if cfg is STANDARD:
        # Reaching a target that leaves the table alone is the easy half. The claim
        # only means something if plans that take sets apart are covered too.
        assert plans_with_rearrangement > 0, "no plan exercised a dissolve"


def test_optimal_policy_plays_whole_games_and_wins_by_emptying_a_rack():
    policy = OptimalAgent(C)
    policy.reset(2)
    state = reset(C, 2, seed=31)
    for _ in range(4000):
        mask = legal_actions(state)
        step(state, act_on_state(policy, state, mask), mask)
        state.check_invariants()
        if state.done.all():
            break
    assert state.done.all()
    assert (state.racks.sum(-1).min(axis=-1) == 0).all(), (
        "optimal play should finish by emptying a rack, not by stalling out"
    )
