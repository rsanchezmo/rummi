"""The plan translator, and through it the completeness of the action space.

If every table CP-SAT proposes can be reached by PLACE/DISSOLVE/ASSIGN within the
per-turn budget, then the micro-action decomposition loses nothing: no turn the
rules permit is inexpressible in the MDP.
"""

from collections import Counter

import numpy as np
import pytest

pytest.importorskip("ortools")

from rummi.core.config import STANDARD, TINY, TINY_GROUPS, RummiConfig
from rummi.core.deal import reset
from rummi.core.encoding import EMPTY, kind_of
from rummi.core.engine import step
from rummi.core.masks import legal_actions
from rummi.core.sets import evaluate_slots
from rummi.policies.optimal_policy import OptimalPolicy
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
        policy = OptimalPolicy(cfg)
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
            step(state, policy.act(state, mask), mask)
            if state.done.all():
                break

    assert plans_checked >= 5, f"only {plans_checked} plans were exercised"
    if cfg is STANDARD:
        # Reaching a target that leaves the table alone is the easy half. The claim
        # only means something if plans that take sets apart are covered too.
        assert plans_with_rearrangement > 0, "no plan exercised a dissolve"


def test_optimal_policy_plays_whole_games_and_wins_by_emptying_a_rack():
    policy = OptimalPolicy(C)
    state = reset(C, 2, seed=31)
    for _ in range(4000):
        mask = legal_actions(state)
        step(state, policy.act(state, mask), mask)
        state.check_invariants()
        if state.done.all():
            break
    assert state.done.all()
    assert (state.racks.sum(-1).min(axis=-1) == 0).all(), (
        "optimal play should finish by emptying a rack, not by stalling out"
    )
