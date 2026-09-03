"""The CP-SAT solver, checked against brute force and against greedy."""

import numpy as np
import pytest

pytest.importorskip("ortools")

from rummi.rules.config import STANDARD, TINY, TINY_GROUPS, RummiConfig
from rummi.env.numpy.deal import reset
from rummi.rules.encoding import EMPTY, kind_of
from rummi.env.numpy.engine import step
from rummi.env.numpy.masks import legal_actions
from rummi.env.numpy.sets import evaluate_slots
from rummi.env.numpy.state import counts_of
from rummi.agents.greedy_agent import plan_turn as greedy_plan
from rummi.solver import brute_force
from rummi.solver.ilp import Objective, solve_turn

C = STANDARD


def rack_of(cfg, kinds):
    r = np.zeros(cfg.n_kinds, dtype=np.int64)
    for k in kinds:
        r[k] += 1
    return r


def table_of(cfg, rows):
    t = np.full((cfg.max_sets, cfg.max_set_len), EMPTY, dtype=np.int16)
    for i, row in enumerate(rows):
        t[i, : len(row)] = row
    return t


def test_opening_meld_is_found_when_it_exists():
    sol = solve_turn(C, rack_of(C, [kind_of(C, 0, n) for n in (11, 12, 13)]), table_of(C, []), False)
    assert sol.feasible and sol.tiles_played == 3
    assert sol.meld_value == 36 >= C.initial_meld


def test_opening_meld_is_refused_when_too_cheap():
    """1-2-3 is a legal set worth 6; the opening rule needs 30."""
    sol = solve_turn(C, rack_of(C, [kind_of(C, 0, n) for n in (1, 2, 3)]), table_of(C, []), False)
    assert not sol.feasible


def test_opening_ignores_the_table_it_may_not_touch():
    """A tile on the table must not help the opening meld, even if it would fit."""
    rack = rack_of(C, [kind_of(C, 0, 12), kind_of(C, 0, 13)])
    table = table_of(C, [[kind_of(C, 0, n) for n in (9, 10, 11)]])
    assert not solve_turn(C, rack, table, has_melded=False).feasible
    # Once melded, the same tiles combine into 9-13 for a five-tile run.
    after = solve_turn(C, rack, table, has_melded=True)
    assert after.feasible and after.tiles_played == 2


def test_it_finds_a_rearrangement_greedy_cannot():
    """The canonical Rummikub manoeuvre, which needs a split to see.

    Table holds the run B5-B6-B7 and the rack holds B4, R7, Y7. Two loose sevens
    cannot make a group on their own, so the only way to play them is to extend
    the run with B4 and then take B7 out of it: B4-B5-B6 stays legal and
    B7-R7-Y7 becomes a group. Greedy appends B4 and stops, because it never
    splits an existing set.
    """
    rack = rack_of(C, [kind_of(C, 1, 4), kind_of(C, 0, 7), kind_of(C, 2, 7)])
    table = table_of(C, [[kind_of(C, 1, n) for n in (5, 6, 7)]])

    sol = solve_turn(C, rack, table, has_melded=True)
    assert sol.feasible and sol.tiles_played == 3, "all three rack tiles are playable"

    sevens = {kind_of(C, c, 7) for c in range(C.n_colors)}
    assert any(len(s) >= C.min_set and set(s) <= sevens for s in sol.sets), (
        f"expected a group of sevens, got {sol.sets}"
    )

    greedy = greedy_plan(C, rack, table, True)
    greedy_tiles = sum(1 for a in greedy if a < C.n_kinds)
    assert greedy_tiles == 1, f"greedy should manage only the B4 append, got {greedy_tiles}"
    assert sol.tiles_played > greedy_tiles, "this is the case that separates the two"


def test_every_returned_set_is_legal_and_uses_the_whole_table():
    s = reset(C, 1, seed=5)
    s.melded[:] = True
    for _ in range(12):
        sol = solve_turn(C, s.racks[0, 0].astype(np.int64), s.table_sets[0], True)
        if not sol.feasible:
            break
        padded = table_of(C, sol.sets)
        assert evaluate_slots(C, padded[: len(sol.sets)]).is_valid.all()
        # Tiles already on the table cannot be removed, only rearranged.
        before = counts_of(C, s.table_sets[None, 0])[0]
        after = counts_of(C, padded[None])[0]
        np.testing.assert_array_equal(after, before + sol.played)
        m = legal_actions(s)
        step(s, np.array([np.argmax(m[0])]), m)


@pytest.mark.parametrize("cfg", [TINY, TINY_GROUPS], ids=["tiny", "tiny_groups"])
def test_matches_brute_force_optimum_on_reduced_configs(cfg: RummiConfig):
    """Exhaustive check that "optimal" means it: brute force tries every subset of
    the rack against every partition of the result."""
    checked = 0
    for trial in range(25):
        state = reset(cfg, 1, seed=trial)
        state.melded[:] = True
        rack = state.racks[0, 0].astype(np.int64)
        table_counts = np.zeros(cfg.n_kinds, dtype=np.int64)

        expected = brute_force.best_turn(cfg, rack, table_counts)
        sol = solve_turn(cfg, rack, table_of(cfg, []), has_melded=True)
        actual = sol.tiles_played if sol.feasible else 0
        assert actual == expected, (
            f"{cfg.n_colors}x{cfg.n_numbers}: solver played {actual}, brute force says {expected}"
        )
        checked += 1
    assert checked == 25


def test_never_does_worse_than_greedy_over_real_play():
    from rummi.agents import GreedyAgent
    from rummi.agents.base import act_on_state

    pol = GreedyAgent(C)
    pol.reset(2)
    s = reset(C, 2, seed=23)
    better = tied = 0
    for _ in range(120):
        for env in np.flatnonzero(s.micro_count == 0):
            if s.done[env]:
                continue
            player = int(s.current[env])
            melded = bool(s.melded[env, player])
            greedy = greedy_plan(C, s.racks[env, player], s.table_sets[env], melded)
            greedy_tiles = sum(1 for a in greedy if a < C.n_kinds)
            sol = solve_turn(C, s.racks[env, player].astype(np.int64), s.table_sets[env], melded)
            optimal_tiles = sol.tiles_played if sol.feasible else 0
            assert optimal_tiles >= greedy_tiles, (
                f"optimal played {optimal_tiles} where greedy managed {greedy_tiles}"
            )
            better += optimal_tiles > greedy_tiles
            tied += optimal_tiles == greedy_tiles
        m = legal_actions(s)
        step(s, act_on_state(pol, s, m), m)
        if s.done.all():
            break
    assert better > 0, "the comparison found no case where rearrangement helps"


def test_max_value_objective_prefers_shedding_penalty():
    # A joker is worth 30 on the rack; a low run is worth little.
    rack = rack_of(C, [C.joker_kind, kind_of(C, 0, 1), kind_of(C, 0, 2)])
    table = table_of(C, [[kind_of(C, 1, n) for n in (5, 6, 7)]])
    by_value = solve_turn(C, rack, table, True, objective=Objective.MAX_VALUE)
    assert by_value.feasible
    assert by_value.played[C.joker_kind] == 1, "should get rid of the joker"


def test_the_no_good_cut_walks_the_tables_that_shed_the_same_count():
    """Red 1-2-3-4 in hand sheds three tiles two ways, and only two ways."""
    rack = rack_of(C, [kind_of(C, 0, n) for n in (1, 2, 3, 4)])
    table = table_of(C, [])
    found, exclude = [], []
    for _ in range(4):
        sol = solve_turn(C, rack, table, True, tiles_min=3, tiles_cap=3, exclude=exclude)
        if not sol.feasible:
            break
        found.append(sol.sets)
        exclude.append(sol.set_counts)
    assert sorted(found) == [
        ((kind_of(C, 0, 1), kind_of(C, 0, 2), kind_of(C, 0, 3)),),
        ((kind_of(C, 0, 2), kind_of(C, 0, 3), kind_of(C, 0, 4)),),
    ]


def test_freezing_the_table_forbids_the_steal_but_still_allows_a_lay_off():
    """Red 3-4-5-6 gives up its 3 to a group of 3s, unless the table is frozen."""
    table = table_of(C, [[kind_of(C, 0, n) for n in (3, 4, 5, 6)]])
    steal = rack_of(C, [kind_of(C, 1, 3), kind_of(C, 2, 3)])
    assert solve_turn(C, steal, table, True).tiles_played == 2
    assert not solve_turn(C, steal, table, True, freeze_table=True).feasible
    # The same set may still grow, which is the difference between frozen and
    # untouched.
    lay_off = rack_of(C, [kind_of(C, 0, 7)])
    frozen = solve_turn(C, lay_off, table, True, freeze_table=True)
    assert frozen.feasible and frozen.sets == (
        tuple(kind_of(C, 0, n) for n in (3, 4, 5, 6, 7)),
    )


def test_freezing_the_table_pins_a_joker_that_is_already_on_it():
    """The canonical joker steal: two 7s and a joker give the joker up to complete
    a run, unless the table is frozen -- and then the group grows instead."""
    table = table_of(
        C,
        [
            [kind_of(C, 0, 7), kind_of(C, 1, 7), C.joker_kind],
            [kind_of(C, 0, n) for n in (1, 2, 3)],
        ],
    )
    rack = rack_of(C, [kind_of(C, 2, 7), kind_of(C, 3, 7)])
    steal = solve_turn(C, rack, table, True)
    assert steal.tiles_played == 2
    assert tuple(kind_of(C, c, 7) for c in range(4)) in steal.sets

    frozen = solve_turn(C, rack, table, True, freeze_table=True)
    assert frozen.tiles_played == 1
    kept = next(s for s in frozen.sets if C.joker_kind in s)
    assert kept == (kind_of(C, 0, 7), kind_of(C, 1, 7), kind_of(C, 3, 7), C.joker_kind)
