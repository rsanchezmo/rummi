"""What the oracle-regret harness records beside an outcome, and what checks it.

The tables in `tools/oracle_regret.py` that ask whether a deviation was *targeted*
rest on three records, and each one is worth more as a test than as an argument.

*The observable proxy is the denial arm's own metric.* Permeability is imported
from that tool rather than recomposed; the door count printed beside it is the same
matrix read unweighted, and if the two disagree one of the numbers is wrong.

*The oracle reply bounds what actually happened.* `opp_shed` is CP-SAT's answer to
"how many tiles could this opponent shed against this table". At two seats nothing
happens between the turn and the reply, so the opponent can never shed more than
that on its next turn -- and where it is zero the opponent has to draw.

*The two readings of a next turn are independent.* A rollout carries its own
boundaries; the played turn's continuation is read off the baseline game's boundary
list instead. `--check` compares them at every decision, and the end-to-end run
below is that comparison on a config small enough to test.
"""

from pathlib import Path
import sys

import numpy as np
import pytest

pytest.importorskip("ortools")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import denial_ab
import oracle_regret as orr

from rummi.rules.config import STANDARD, TINY_GROUPS
from rummi.rules.encoding import EMPTY, kind_of
from rummi.env.numpy.deal import reset as deal_reset
from rummi.env.observation import encode

from tests.conftest import rebalance_pool

C = STANDARD


def _table(cfg, rows):
    t = np.full((cfg.max_sets, cfg.max_set_len), EMPTY, dtype=np.int16)
    for i, row in enumerate(rows):
        t[i, : len(row)] = sorted(row)
    return t


def _state_with(cfg, rows, opp_rack, melded=True):
    """A dealt state holding ``rows`` on the table and ``opp_rack`` at seat 1."""
    state = deal_reset(cfg, 1, seed=0)
    state.table_sets[0] = _table(cfg, rows)
    state.table_snapshot[:] = state.table_sets
    state.racks[0, 1] = 0
    for kind in opp_rack:
        state.racks[0, 1, kind] += 1
    state.melded[:] = melded
    rebalance_pool(state)
    return state


def test_the_door_count_is_the_permeability_matrix_read_unweighted():
    """Two numbers off one matrix: a kind counts once for the count and once per
    unseen copy for the level."""
    unseen = np.full(C.n_kinds, 2, dtype=np.int16)
    run = _table(C, [[kind_of(C, 0, n) for n in (5, 6, 7)]])
    group = _table(C, [[kind_of(C, c, 7) for c in (0, 1, 2)]])

    for rows in (run, group):
        doors = orr.appendable(C, rows, unseen).any(0)
        assert float(denial_ab.permeability(C, rows, unseen)) == float(
            (doors * unseen).sum()
        )
    # The motivating example: a run takes both ends, a group one colour, and the
    # joker lays off onto either.
    assert int(orr.appendable(C, run, unseen).any(0).sum()) == 3
    assert int(orr.appendable(C, group, unseen).any(0).sum()) == 2


def test_unseen_is_what_the_observation_merges():
    """The feature reads it off the state because the turn is over and `current` has
    moved on -- so it has to agree with the encoder where a seat is still acting."""
    state = deal_reset(C, 3, seed=7)
    obs = encode(state)
    for env in range(3):
        seat = int(state.current[env])
        assert (orr.unseen_for(state, env, seat) == obs["unseen"][env]).all()


def test_the_oracle_reply_reads_the_table_it_is_given():
    """Same one tile shed, two tables: one leaves the door the opponent's last tile
    needs and one does not, which is the whole hypothesis in one assertion."""
    red4 = kind_of(C, 0, 4)
    open_table = [[kind_of(C, 0, n) for n in (5, 6, 7)]]
    shut_table = [[kind_of(C, c, 7) for c in (0, 1, 2)]]

    opened = orr.board_features(
        C, _state_with(C, open_table, [red4]), 0, 0, [1], time_limit=2.0
    )
    shut = orr.board_features(
        C, _state_with(C, shut_table, [red4]), 0, 0, [1], time_limit=2.0
    )
    assert opened.opp_shed == [1] and not opened.ended
    assert shut.opp_shed == [0]
    assert opened.perm > shut.perm and opened.doors > shut.doors


def test_the_reply_is_the_maximum_the_opponent_could_shed():
    """One solve answers both questions, so it has to be the max and not merely
    feasible: an opponent holding a whole run must read as going out, not as one
    lay-off."""
    rack = [kind_of(C, 1, n) for n in (9, 10, 11)]
    left = orr.board_features(
        C, _state_with(C, [[kind_of(C, 0, n) for n in (5, 6, 7)]], rack), 0, 0, [1], 2.0
    )
    assert left.opp_shed == [3]


def test_the_reply_is_invariant_to_how_the_same_tiles_are_arranged():
    """Why the denial cell is nearly empty, as a two-line demonstration.

    A melded opponent may repartition the whole table, so its best reply is a
    function of the tile *multiset* the table holds and not of the arrangement --
    three runs and three groups of the same nine tiles offer it exactly as much. The
    observable proxy is not invariant, which is the gap the targeted tables measure:
    permeability moves where the oracle cannot.
    """
    runs = [[kind_of(C, c, n) for n in (5, 6, 7)] for c in (0, 1, 2)]
    groups = [[kind_of(C, c, n) for c in (0, 1, 2)] for n in (5, 6, 7)]
    rack = [kind_of(C, 0, 8)]

    as_runs = orr.board_features(C, _state_with(C, runs, rack), 0, 0, [1], 2.0)
    as_groups = orr.board_features(C, _state_with(C, groups, rack), 0, 0, [1], 2.0)
    assert as_runs.opp_shed == as_groups.opp_shed == [1]
    assert as_runs.perm != as_groups.perm


def test_first_turn_reads_the_boundary_it_is_pointed_at():
    """`skip` exists for the zero-deviation candidate, which still has the played
    turn in front of it."""
    trace = orr.Trace.blank(1, 4, 2)
    trace.n[0] = 4
    trace.seat[0] = [1, 0, 1, 0]
    trace.racks[0] = [[5, 3], [5, 2], [4, 2], [4, 1]]

    assert trace.first_turn(0, 1, winner=-1, skip=0) == (1, False)
    assert trace.first_turn(0, 0, winner=-1, skip=0) == (1, False)
    # From one boundary in, seat 1's first turn is its *second* one.
    assert trace.first_turn(0, 1, winner=-1, skip=2) == (1, False)
    # A seat that never acts inside the recorded window has no next turn.
    assert trace.first_turn(0, 0, winner=-1, skip=2) == (None, False)

    # A boundary that is already terminal is the turn before it ending the game.
    trace.ended[0, 2] = True
    assert trace.first_turn(0, 0, winner=0, skip=1) == (1, True)
    assert trace.first_turn(0, 0, winner=1, skip=1) == (1, False)


def test_a_drawn_turn_reads_as_a_negative_shed():
    """A rack that grew is not a rack that shed nothing, and the tables split on the
    sign."""
    trace = orr.Trace.blank(1, 3, 2)
    trace.n[0] = 3
    trace.seat[0] = [1, 0, 1]
    trace.racks[0] = [[5, 3], [5, 4], [4, 4]]
    assert trace.first_turn(0, 1, winner=-1, skip=0) == (-1, False)


@pytest.fixture(scope="module")
def tiny_run():
    """One whole deal on a config small enough to test, with every check on."""
    games = [
        orr.run_game((seed, TINY_GROUPS, 5, 2, 1.0, 16, 2_000, True, True, True))
        for seed in range(4)
    ]
    assert not any("error" in g for g in games), [g.get("error") for g in games]
    return games


def test_the_two_readings_of_a_next_turn_agree(tiny_run):
    """What `--check` asserts: the rollout's own boundaries and the baseline game's
    boundary list are separate derivations of the same shed counts."""
    assert [f for g in tiny_run for f in g["failures"]] == []
    assert sum(g["continuations"] for g in tiny_run) > 0


def test_the_two_readings_agree_with_one_deal_per_rollout():
    """The same check at `--chunk 1`, where nothing else keeps the loop running.

    A batch hides this: an env that finishes early is re-recorded on every later
    step, so its terminal boundary arrives for free while a slower env keeps the
    rollout going. Alone, the loop stops on the step that boundary is on -- and a
    turn whose successor was never recorded is unpriceable, which the two
    derivations then disagree about rather than both reporting as absent.
    """
    games = [
        orr.run_game((seed, TINY_GROUPS, 5, 2, 1.0, 1, 2_000, True, True, True))
        for seed in range(4)
    ]
    assert not any("error" in g for g in games), [g.get("error") for g in games]
    assert [f for g in games for f in g["failures"]] == []
    assert sum(g["continuations"] for g in games) > 0


def test_the_oracle_reply_bounds_what_the_opponent_actually_did(tiny_run):
    """At two seats nothing intervenes between the turn and the reply, so CP-SAT's
    maximum is a bound on the rollout -- and a table that offers nothing has to
    leave the opponent drawing."""
    compared = playable = 0
    for game in tiny_run:
        for d in game["decisions"]:
            nxt = d["opps"].index((d["seat"] + 1) % TINY_GROUPS.n_players)
            for turn in (d["base"], *d["alts"]):
                shed = turn["next_shed"][nxt]
                if turn["ended"] or shed is None:
                    continue
                compared += 1
                playable += turn["opp_shed"][nxt] > 0
                assert shed <= turn["opp_shed"][nxt]
                if turn["opp_shed"][nxt] == 0:
                    assert shed <= 0, "the table offered nothing yet a tile was shed"
    assert compared > 20 and playable > 0, "the check found nothing to compare"


def test_every_turn_records_the_features_the_tables_read(tiny_run):
    """The schema the analysis assumes, asserted where it is written rather than
    where it is aggregated."""
    for game in tiny_run:
        for d in game["decisions"]:
            assert d["opps"] == [p for p in range(TINY_GROUPS.n_players) if p != d["seat"]]
            assert d["opp_sizes"] == [sum(n for _, n in r) for r in d["opp_racks"]]
            for turn in (d["base"], *d["alts"]):
                assert len(turn["opp_shed"]) == len(d["opps"])
                assert len(turn["next_shed"]) == len(d["opps"])
                assert turn["perm"] >= 0.0 and turn["doors"] >= 0


def _decision(**over):
    d = {
        "turn": 4,
        "seat": 0,
        "pre_meld": False,
        "own": 6,
        "opp": 2,
        "pool": 30,
        "base_tiles": 2,
        "base_won": False,
        "opps": [1],
        "opp_sizes": [2],
        "opp_racks": [[[3, 2]]],
        "base": {
            "perm": 10.0,
            "doors": 5,
            "opp_shed": [2],
            "ended": False,
            "next_shed": [2],
            "next_win": [True],
            "own_shed": None,
        },
        "alts": [],
    }
    d.update(over)
    return d


def _alt(**over):
    a = {
        "kind": orr.SHAPED,
        "tiles": 2,
        "won": 1,
        "perm": 4.0,
        "doors": 2,
        "opp_shed": [1],
        "ended": False,
        "next_shed": [1],
        "next_win": [False],
        "own_shed": 3,
    }
    a.update(over)
    return a


def test_a_pair_reads_a_closed_exit_off_the_two_replies():
    """The headline cell: the opponent could shed its whole rack against the played
    table and cannot against this one."""
    pairs, dropped = orr.denial_pairs(
        [{"game": 0, "decisions": [_decision(alts=[_alt()])]}]
    )
    assert dropped == 0
    (pair,) = pairs
    assert pair.closed_out and pair.d_out == -1 and pair.d_play == 0
    assert pair.d_shed == -1 and pair.delta == 1.0
    assert pair.d_perm == -6.0 and pair.d_doors == -3 and pair.leader


def test_a_turn_that_ends_the_game_is_dropped_rather_than_scored():
    """It would read as a large positive delta with nothing to do with the table."""
    _, dropped = orr.denial_pairs(
        [{"game": 0, "decisions": [_decision(alts=[_alt(ended=True)])]}]
    )
    assert dropped == 1
    pairs, _ = orr.denial_pairs(
        [
            {
                "game": 0,
                "decisions": [_decision(base={**_decision()["base"], "ended": True}, alts=[_alt()])],
            }
        ]
    )
    assert pairs == []


def test_a_next_turn_that_never_happened_is_not_a_shed_of_zero():
    """None on either side of the pair means the split has nothing to say, and the
    `unchanged` column must not swallow it."""
    pairs, _ = orr.denial_pairs(
        [{"game": 0, "decisions": [_decision(alts=[_alt(next_shed=[None])])]}]
    )
    assert pairs[0].d_shed is None
    assert not any(inside(pairs[0]) for _, inside in orr.SHED_COLUMNS)


def test_the_interval_is_clustered_on_the_deal():
    """Two deals of five identical alternatives are two observations, not ten."""
    one = orr._cluster([(0, 1.0)] * 5 + [(1, 0.0)] * 5)
    two = orr._cluster([(0, 1.0), (1, 0.0)])
    assert one[0] == 10 and two[0] == 2
    assert one[1] == two[1] == 0.5
    assert one[2] == pytest.approx(two[2])


def test_the_headline_counts_every_seat_and_not_only_the_ones_that_played() -> None:
    """A seat that never ends a turn is still a seat that lost.

    The count used to be inferred from the decisions on record, so a seat that
    never chose anything -- it can still win through the stalemate branch -- was
    dropped from both the numerator and the denominator, and a run whose every
    game came back an error had no maximum to take at all.
    """
    game = {"winner": 1, "decisions": [{"seat": 0, "base_won": False, "alts": [
        {"won": 1, "kind": "swap"}
    ]}]}
    head = orr.headline([game], seats=3)
    assert head.seats == 3
    assert (head.wins, head.losses, head.rescued) == (1, 2, 1)

    empty = orr.headline([], seats=2)
    assert (empty.seats, empty.wins, empty.losses) == (2, 0, 0)
    assert orr.summarise([], 2), "a run that measured nothing still has to report"
