"""The denial tie-break: what makes it a refinement, and what makes it measurable.

Three claims carry the experiment in `tools/denial_ab.py`, and each is worth more as
a test than as an argument.

*The arm is `by_value` plus one rule.* Its ranking is recomposed from the public
tables rather than read out of the chooser, which only exposes its argmax -- so the
recomposition is pinned to that argmax over random masks, and the arm is held to
playing exactly what `by_value` plays wherever the ranking has a single best play.

*The table it scores is the table that happens.* Permeability is read off a predicted
afterstate, so the prediction is replayed against the env: the expansion of the same
macro must leave the sets the prediction named.

*The metric says what it claims.* A run leaves two ends open and a group leaves one
colour, and the number has to reflect that or the arm is ranking noise.
"""

from collections import Counter
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import denial_ab

from rummi.agents.base import turn_starting
from rummi.agents.macro import (
    MacroAgent,
    by_value,
    extend_offset,
    repartition_offset,
    steal_offset,
)
from rummi.rules.config import STANDARD, TINY_GROUPS
from rummi.rules.encoding import EMPTY, kind_of
from rummi.env.numpy.deal import reset as deal_reset
from rummi.env.numpy.engine import step as engine_step
from rummi.env.numpy.masks import legal_actions
from rummi.env.numpy.sets import summarize
from rummi.env.observation import encode
from rummi.solver.to_actions import slot_contents

C = STANDARD


def _table(cfg, rows):
    t = np.full((cfg.max_sets, cfg.max_set_len), EMPTY, dtype=np.int16)
    for i, row in enumerate(rows):
        t[i, : len(row)] = sorted(row)
    return t


def _melded(flag: bool):
    """The only field `by_value` reads besides its argument."""
    return {"melded": np.array([[int(flag), 0]], dtype=np.int16)}


def test_the_recomputed_ranking_reproduces_by_value():
    """`rank_tables` is the one thing here that duplicates a decision it cannot call."""
    points, tiles = denial_ab.rank_tables(C)
    ranked = repartition_offset(C)
    assert len(points) == ranked and len(tiles) == ranked

    choose = by_value(C)
    rng = np.random.default_rng(0)
    for melded in (False, True):
        obs = _melded(melded)
        rank = tiles if melded else points
        for _ in range(200):
            legal = rng.random(ranked + 2) < 0.02
            legal[-1] = True  # DRAW is never masked
            options = np.flatnonzero(legal[:ranked])
            if not options.size:
                continue
            assert choose(obs, 0, legal) == int(options[np.argmax(rank[options])])


def test_permeability_counts_a_run_as_more_permeable_than_a_group():
    """The motivating example, as a number.

    Blue 5-6-7 takes a 4 and an 8; three 7s take the one missing colour. Same three
    tiles shed, and the run leaves twice the doors -- which is exactly what
    `by_value`'s post-meld "most tiles" rule cannot see.
    """
    unseen = np.full(C.n_kinds, 2, dtype=np.int16)
    run = _table(C, [[kind_of(C, 0, n) for n in (5, 6, 7)]])
    group = _table(C, [[kind_of(C, c, 7) for c in (0, 1, 2)]])

    # A joker lays off onto either, so the doors that separate them are the numbered
    # ones: two ends against one colour.
    assert float(denial_ab.permeability(C, run, unseen)) > float(
        denial_ab.permeability(C, group, unseen)
    )
    assert float(denial_ab.permeability(C, group, unseen)) > 0.0


def test_permeability_ignores_a_door_no_unseen_tile_can_walk_through():
    """`unseen` is the weight because a door for a kind wholly on the table is shut."""
    run = _table(C, [[kind_of(C, 0, n) for n in (5, 6, 7)]])
    unseen = np.full(C.n_kinds, 2, dtype=np.int16)
    shut = unseen.copy()
    for n in (4, 8):
        shut[kind_of(C, 0, n)] = 0
    assert float(denial_ab.permeability(C, run, shut)) < float(
        denial_ab.permeability(C, run, unseen)
    )


def test_the_steal_term_reads_what_a_set_can_spare():
    """A four-long run can give up either end; a three-long one cannot spare a tile."""
    long_run = _table(C, [[kind_of(C, 0, n) for n in (5, 6, 7, 8)]])
    short_run = _table(C, [[kind_of(C, 0, n) for n in (5, 6, 7)]])
    unseen = np.zeros(C.n_kinds, dtype=np.int16)  # doors weigh nothing, so this is the term
    assert float(denial_ab.permeability(C, long_run, unseen, steal_weight=1.0)) == 2.0
    assert float(denial_ab.permeability(C, short_run, unseen, steal_weight=1.0)) == 0.0


def test_permeability_broadcasts_over_a_batch_of_tables():
    """The arm scores a whole tie set in one call, so the batched answer must be the
    per-table one."""
    unseen = np.full(C.n_kinds, 2, dtype=np.int16)
    tables = [
        _table(C, [[kind_of(C, 0, n) for n in (5, 6, 7)]]),
        _table(C, [[kind_of(C, c, 7) for c in (0, 1, 2)]]),
        _table(C, [[kind_of(C, 0, n) for n in (1, 2, 3)], [kind_of(C, 1, n) for n in (9, 10, 11)]]),
    ]
    batched = denial_ab.permeability(C, np.stack(tables), unseen, steal_weight=1.0)
    alone = [float(denial_ab.permeability(C, t, unseen, steal_weight=1.0)) for t in tables]
    assert batched.shape == (3,)
    assert np.allclose(batched, alone)


def _states(cfg, games: int, steps: int, seed: int = 0):
    """Real mid-game states, played by `by_value` itself.

    Random play never reaches a meld on this config, so the states the arm actually
    decides in have to be reached by something that plays.
    """
    state = deal_reset(cfg, games, seed=seed)
    agent = MacroAgent(cfg, choose=by_value(cfg))
    agent.reset(games)
    for _ in range(steps):
        if state.done.all():
            break
        summary = summarize(cfg, state.table_sets)
        mask = legal_actions(state, summary)
        obs = encode(state, summary)
        yield state, obs, mask
        engine_step(state, agent.act(obs, mask), mask)


def _contents(rows):
    return Counter(c for c in slot_contents(rows) if c)


def test_the_predicted_table_is_the_one_the_env_reaches():
    """`after_rows` mirrors `expand`'s target, so the env has to agree with it.

    Replayed on a clone per macro: the point is what one macro leaves behind, and the
    real game carries on from whatever the driver chose.

    Counted per block, because `after_rows` has three branches and the legal macros of
    a state are overwhelmingly the first one -- a sample not stratified by block would
    leave the steal branch, the only one that dissolves a set, untested.
    """
    cfg = C
    agent = MacroAgent(cfg, choose=by_value(cfg))
    ends = (0, extend_offset(cfg), steal_offset(cfg), repartition_offset(cfg))
    checked = [0, 0, 0]
    for state, obs, _mask in _states(cfg, 8, 220, seed=3):
        fresh = turn_starting(obs)
        for env in range(state.batch_size):
            if state.done[env] or not fresh[env]:
                continue
            legal = agent.legal_macros(obs, env)
            board = np.asarray(obs["table_sets"][env])
            hand = np.asarray(obs["rack"][env]).astype(np.int64)
            for block in range(3):
                offered = np.flatnonzero(legal[ends[block] : ends[block + 1]]) + ends[block]
                for macro in offered[:: max(1, len(offered) // 2)].tolist():
                    actions = agent.expand(obs, env, int(macro))
                    if not actions:
                        continue
                    predicted = denial_ab.after_rows(cfg, board, hand, int(macro))
                    replay = state.clone()
                    for action in actions:
                        replay_mask = legal_actions(replay, summarize(cfg, replay.table_sets))
                        assert replay_mask[env, action], "the expansion left the mask behind"
                        step = np.full(replay.batch_size, cfg.draw_action, dtype=np.int64)
                        step[env] = action
                        engine_step(replay, step, replay_mask)
                    assert _contents(replay.table_sets[env]) == _contents(predicted), (
                        f"macro {macro} landed a different table than predicted"
                    )
                    checked[block] += 1
    assert all(checked), f"a branch went unexercised: {checked}"
    assert sum(checked) > 100, f"only {sum(checked)} expansions exercised"


def test_the_arm_plays_by_value_wherever_nothing_is_tied():
    """The refinement property, on the states the arm decides in.

    A tie-break that changed an untied decision would be a different agent, and the
    head-to-head would be measuring two changes at once.
    """
    cfg = C
    base = by_value(cfg)
    for arm in (denial_ab.TieBreak(cfg), denial_ab.TieBreak(cfg, mode="null")):
        points, tiles = arm.points, arm.tiles
        ranked = arm.ranked
        agent = MacroAgent(cfg, choose=arm)
        decided = 0
        for state, obs, _mask in _states(cfg, 12, 400, seed=7):
            fresh = turn_starting(obs)
            for env in range(state.batch_size):
                if state.done[env] or not fresh[env]:
                    continue
                legal = agent.legal_macros(obs, env)
                options = np.flatnonzero(legal[:ranked])
                chosen = arm(obs, env, legal)
                wanted = base(obs, env, legal)
                if not options.size:
                    assert chosen == wanted
                    continue
                rank = tiles if bool(obs["melded"][env, 0]) else points
                tied = options[rank[options] == rank[options].max()]
                decided += 1
                if tied.size == 1:
                    assert chosen == wanted, "an untied decision moved"
                # A refinement may only reorder within the tie, never leave it.
                assert chosen in tied.tolist()
                assert rank[chosen] == rank[wanted]
        assert decided > 200, f"only {decided} rankings exercised"


def test_the_null_control_breaks_the_same_ties_as_the_arm():
    """Same code path, same tie set: only the key inside it differs.

    That is what makes the control able to say whether the harness manufactures an
    effect -- a control that fired at a different rate would not.
    """
    cfg = C
    real, null = denial_ab.TieBreak(cfg), denial_ab.TieBreak(cfg, mode="null")
    agent = MacroAgent(cfg, choose=by_value(cfg))
    for state, obs, _mask in _states(cfg, 8, 260, seed=11):
        fresh = turn_starting(obs)
        for env in range(state.batch_size):
            if state.done[env] or not fresh[env]:
                continue
            legal = agent.legal_macros(obs, env)
            before = real.stats.ties, null.stats.ties
            real(obs, env, legal)
            null(obs, env, legal)
            assert real.stats.ties - before[0] == null.stats.ties - before[1]
    assert real.stats.ties > 20, "no decision tied, so this asserts nothing"


def test_the_null_control_is_reproducible_across_processes():
    """Seeded from the state's bytes, not from `hash`, whose seed varies per run."""
    cfg = C
    board = _table(cfg, [[kind_of(cfg, 0, n) for n in (5, 6, 7)]])
    rack = np.zeros(cfg.n_kinds, dtype=np.int64)
    rack[kind_of(cfg, 1, 4)] = 1
    arm = denial_ab.TieBreak(cfg, mode="null")
    first = arm._coin(board, rack, 6)
    assert np.array_equal(first, arm._coin(board, rack, 6))
    assert not np.array_equal(first, arm._coin(board, rack + 1, 6))


def test_a_head_to_head_pairs_every_deal_across_both_seats():
    """The rotation is what cancels turn order, and `by_value` mirrored against itself
    is the check that it does: exactly half the wins, and a score of exactly zero."""
    cfg = TINY_GROUPS
    make = lambda c: MacroAgent(c, choose=by_value(c))  # noqa: E731
    wins, scores = denial_ab.head_to_head(cfg, make, make, deals=8, seed_base=1234, batch=8)
    assert wins.shape == (8,) and scores.shape == (8,)
    assert float(wins.mean()) == 0.5
    assert float(scores.mean()) == 0.0


def test_a_mirrored_arm_leaves_exactly_what_it_meets():
    """The same rotation, so the mechanism reading has the same exactness the score has.

    An agent mirrored against itself hands over the tables it is handed, deal for deal,
    and `left - met` must be exactly zero. A drift there is a broken attribution --
    turns credited to the wrong side -- not noise, and it would make a small real gap
    unreadable.
    """
    cfg = TINY_GROUPS
    make = lambda c: MacroAgent(c, choose=by_value(c))  # noqa: E731
    watcher = denial_ab.shape_left(cfg, make, make, deals=8, seed_base=1234, batch=8)
    left, met, difference = watcher.paired()
    assert difference.size == 8
    assert left > 0.0 and met > 0.0
    assert np.array_equal(difference, np.zeros(8))


def _template_index(cfg, kinds):
    """Which template row is exactly this multiset of kinds."""
    want = np.zeros(cfg.n_kinds, dtype=np.int16)
    for k in kinds:
        want[k] += 1
    rows = np.flatnonzero((denial_ab.set_templates(cfg) == want).all(-1))
    assert rows.size == 1
    return int(rows[0])


def test_the_positive_control_sheds_the_dearer_of_two_equal_plays():
    """`value` is what says the measurement resolves anything at all.

    Post-meld `by_value` ranks a tie by tile count and is blind to face value inside
    it, so red 1-2-3 and red 11-12-13 are the same play to it -- 6 rack points against
    36. A tie-break that takes the second is a real improvement of exactly the shape
    denial has, which is what makes it the control a null needs.
    """
    cfg = C
    cheap = [kind_of(cfg, 0, n) for n in (1, 2, 3)]
    dear = [kind_of(cfg, 0, n) for n in (11, 12, 13)]
    rack = np.zeros(cfg.n_kinds, dtype=np.int16)
    for k in (*cheap, *dear):
        rack[k] += 1
    obs = {
        "rack": rack[None],
        "workbench": np.zeros((1, cfg.n_kinds), dtype=np.int16),
        "table_sets": _table(cfg, [])[None],
        "unseen": np.full((1, cfg.n_kinds), 2, dtype=np.int16),
        "melded": np.array([[1, 0]], dtype=np.int16),
    }
    low, high = _template_index(cfg, cheap), _template_index(cfg, dear)
    legal = np.zeros(repartition_offset(cfg) + 2, dtype=bool)
    legal[[low, high]] = True
    legal[-1] = True

    assert by_value(cfg)(obs, 0, legal) == min(low, high), "the tie is real"
    assert denial_ab.TieBreak(cfg, mode="value")(obs, 0, legal) == high
