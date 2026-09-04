"""The CP-SAT solver, checked against brute force and against greedy."""

from dataclasses import replace

from collections import Counter

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
from rummi.agents import build
from rummi.agents.base import act_on_state
from rummi.agents.greedy_agent import plan_turn as greedy_plan
from rummi.solver import brute_force
from rummi.solver.ilp import Objective, solve_turn
from rummi.solver.to_actions import allocate_slots, slot_contents
from tests.conftest import state_with

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


def test_an_unknown_objective_is_refused_rather_than_silently_maximising_tiles():
    """`Objective` was a bare string namespace compared with `==`, so any typo --
    `max_tilez` -- fell through to the tile-maximising branch and returned a
    plausible answer to a question nobody asked."""
    with pytest.raises(ValueError, match="max_tilez"):
        solve_turn(C, rack_of(C, [kind_of(C, 0, 1)]), table_of(C, []), True, objective="max_tilez")
    # The accepted spellings still work, by value and by member.
    rack = rack_of(C, [kind_of(C, 0, n) for n in (11, 12, 13)])
    assert solve_turn(C, rack, table_of(C, []), True, objective="max_value").feasible
    assert solve_turn(C, rack, table_of(C, []), True, objective=Objective.MAX_TILES).feasible


def test_a_relaxed_opening_is_credited_the_way_the_engine_credits_it():
    """`strict_initial_meld=False` changes two rules at once and `solve_turn` read
    neither: the table becomes touchable before melding, and the opening is
    credited as the face value of what left the rack -- with the joker at **zero**,
    because its worth is positional and no set has claimed it yet.

    The solver instead priced the joker into the set it built, so it believed 36
    where the engine credits 23, and the plan it produced hit a masked `END_TURN`
    after six micro-actions, reverting the whole turn.
    """
    cfg = replace(C, strict_initial_meld=False)
    rack = rack_of(cfg, [kind_of(cfg, 0, 11), kind_of(cfg, 0, 12), cfg.joker_kind])
    sol = solve_turn(cfg, rack, table_of(cfg, []), has_melded=False)
    # 11 + 12 + 0 = 23 < 30, so there is no opening here at all.
    assert not sol.feasible, f"solved a meld the engine will not credit: {sol.sets}"

    # One that clears the bar on face value alone must still be found.
    good = rack_of(cfg, [kind_of(cfg, 0, n) for n in (11, 12, 13)])
    assert solve_turn(cfg, good, table_of(cfg, []), has_melded=False).feasible


def test_a_relaxed_opening_plays_through_the_engine_to_a_legal_end_turn():
    """The other half of the same flag: `may_touch_table` is unconditional when the
    meld is not strict, so a pre-meld turn may lay off onto standing sets.

    This rack forms no set on its own, so laying off is the *only* way to reach 30
    -- which is what makes the strict arm infeasible rather than merely worse. Run
    through the real mask, because the solver and the agent had to agree about
    whose sets the target already contains: the agent appended the standing sets on
    the assumption the solver never saw them.
    """
    cfg = replace(C, strict_initial_meld=False)
    rows = [
        [kind_of(cfg, 0, n) for n in (5, 6, 7)],
        [kind_of(cfg, 1, n) for n in (10, 11, 12)],
    ]
    held = [kind_of(cfg, 0, 4), kind_of(cfg, 0, 8), kind_of(cfg, 1, 9), kind_of(cfg, 1, 13)]

    assert not solve_turn(C, rack_of(C, held), table_of(C, rows), has_melded=False).feasible

    state = state_with(cfg, rack=held, table=rows)
    agent = build("optimal", cfg)
    agent.reset(1)
    actions = act_on_state(agent, state, legal_actions(state))
    assert actions[0] != cfg.draw_action, "the agent drew instead of opening"

    # Play the whole turn out; every action must be legal and the turn must commit.
    committed = False
    for _ in range(cfg.max_micro_per_turn + 2):
        mask = legal_actions(state)
        action = int(act_on_state(agent, state, mask)[0])
        assert mask[0, action], f"the agent proposed an illegal action {action}"
        assert action != cfg.draw_action, "the turn was abandoned, not committed"
        step(state, np.full(1, action), mask)
        state.check_invariants()
        if action == cfg.end_turn_action:
            committed = True
            break
    assert committed, "the plan never reached END_TURN"
    assert bool(state.melded[0, 0]), "the opening was not credited"


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


def test_freezing_the_table_pins_a_real_tile_as_well_as_a_joker():
    """The steal `freeze_table` missed: a joker takes a real tile's *position* in a
    frozen set and frees the tile for another one, leaving the multiset of sets
    intact while both sets are physically taken apart.

    Two runs sharing colour 1: (9, 10, J, 12) and (11, 12, 13). Swapping the joker
    for the real 11 gives (9, 10, 11, 12) and (12, 13, J) -- both still legal, both
    still "the same sets", and `allocate_slots` has to dissolve both to build them.
    """
    c1 = lambda n: kind_of(C, 1, n)  # noqa: E731
    rows = [[c1(9), c1(10), c1(12), C.joker_kind], [c1(11), c1(12), c1(13)]]
    rack = rack_of(C, [kind_of(C, 3, n) for n in (5, 6, 7)])

    frozen = solve_turn(C, rack, table_of(C, rows), True, freeze_table=True)
    assert frozen.feasible

    standing = [tuple(sorted(r)) for r in rows]
    target = [tuple(sorted(s)) for s in frozen.sets]
    for content in standing:
        assert any(Counter(content) <= Counter(s) for s in target), (
            f"the frozen set {content} is not inside any target set: {target}"
        )
    # And the plan must not have to take one apart to get there.
    alloc = allocate_slots(slot_contents(table_of(C, rows)), list(frozen.sets))
    assert not alloc.dissolve, f"a frozen slot is dissolved: {alloc.dissolve}"


def test_freezing_the_table_refuses_to_buy_a_joker_back():
    """Buying a joker back -- swapping it for the real tile it stands in for -- is a
    legal Rummikub move, and it is rearrangement, so a frozen table refuses it.

    Without the pinning it was the same hole as the tile steal: the joker leaves,
    the set still reads as the same set, and the plan has to take it apart to get
    there. The group here can only be completed by displacing the joker, so a
    frozen solve has nothing to do at all.
    """
    fours = [kind_of(C, c, 4) for c in range(4)]
    table = table_of(C, [[fours[0], fours[2], fours[3], C.joker_kind]])
    rack = rack_of(C, [fours[1], kind_of(C, 0, 9), kind_of(C, 0, 10)])

    assert not solve_turn(C, rack, table, True, freeze_table=True).feasible
    # Unfrozen, the same rack does displace it -- which is what makes this a
    # restriction rather than an impossibility.
    assert solve_turn(C, rack, table, True).feasible


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


def test_the_optimum_reads_the_multiset_and_not_the_arrangement():
    """Unrestricted rearrangement means a solved turn is a function of the tiles the
    table holds, not of how they are grouped.

    Nine tiles that partition either as three runs or as three groups are the same
    problem post-meld, and `tools/oracle_regret.py` prices endgame denial on exactly
    this: what an opponent can do against a table it may repartition cannot be
    changed by rearranging the table it is handed.
    """
    runs = table_of(C, [[kind_of(C, c, n) for n in (5, 6, 7)] for c in (0, 1, 2)])
    groups = table_of(C, [[kind_of(C, c, n) for c in (0, 1, 2)] for n in (5, 6, 7)])
    assert (counts_of(C, runs[None]) == counts_of(C, groups[None])).all()

    for rack in ([kind_of(C, 0, 8)], [kind_of(C, 3, 5), kind_of(C, 3, 6)], []):
        as_runs = solve_turn(C, rack_of(C, rack), runs, has_melded=True)
        as_groups = solve_turn(C, rack_of(C, rack), groups, has_melded=True)
        assert as_runs.feasible == as_groups.feasible
        assert as_runs.tiles_played == as_groups.tiles_played
        assert as_runs.value_played == as_groups.value_played
