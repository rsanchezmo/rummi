"""The rack-potential objective: what makes it an override, and what makes it exact.

Four claims carry the experiment in `tools/rack_potential_ab.py`, and each is worth
more as a test than as an argument.

*`w = 0` is `by_value`, exactly.* The control and the correctness check are the same
thing here, and it is a stronger property than the denial tie-break needed: that arm
was a refinement by construction, this one adds a term to the score and has to be held
to vanishing when the term is switched off.

*And a positive `w` is not a tie-break.* It must be able to take a play the ranking
puts **below** the best one, or the whole experiment sits inside the ~1.6pp bound
`tools/denial_ab.py` already measured over `by_value`'s indifference set.

*The potential is what it says.* `shortfall` is recomputed as a matmul, which is exact
only because template rows are binary, so it is pinned to the reference. Beyond that:
one draw serving four near-runs counts once, a door no unseen copy can walk through
counts zero, and a held joker is what makes a one-away set ready.

*The rack it scores is the rack the env reaches.* `played` predicts what leaves the
hand, so the prediction is replayed against the engine, per macro block.
"""

from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import denial_ab
import rack_potential_ab as rp

from rummi.agents.base import turn_starting
from rummi.agents.macro import (
    MacroAgent,
    by_value,
    extend_offset,
    playable,
    repartition_offset,
    set_templates,
    shortfall,
    steal_offset,
)
from rummi.rules.config import STANDARD, TINY_GROUPS
from rummi.rules.encoding import EMPTY, kind_of
from rummi.env.numpy.deal import reset as deal_reset
from rummi.env.numpy.engine import step as engine_step
from rummi.env.numpy.masks import current_rack, legal_actions
from rummi.env.numpy.sets import summarize
from rummi.env.observation import encode

C = STANDARD
WEIGHTED = tuple(m for m, mode in rp.MODES.items() if not mode.joker_hold)


def _table(cfg, rows):
    t = np.full((cfg.max_sets, cfg.max_set_len), EMPTY, dtype=np.int16)
    for i, row in enumerate(rows):
        t[i, : len(row)] = sorted(row)
    return t


def _rack(cfg, kinds):
    out = np.zeros(cfg.n_kinds, dtype=np.int16)
    for k in kinds:
        out[k] += 1
    return out


def _obs(cfg, rack, rows=(), unseen=2, melded=True):
    """The five fields the objective reads, and nothing else."""
    return {
        "rack": rack[None],
        "workbench": np.zeros((1, cfg.n_kinds), dtype=np.int16),
        "table_sets": _table(cfg, rows)[None],
        "unseen": np.full((1, cfg.n_kinds), unseen, dtype=np.int16),
        "melded": np.array([[int(melded), 0]], dtype=np.int16),
    }


def _legal(cfg, macros):
    out = np.zeros(rp.repartition_offset(cfg) + 2, dtype=bool)
    out[list(macros)] = True
    out[-1] = True  # DRAW is never masked
    return out


def _template_index(cfg, kinds):
    """Which template row is exactly this multiset of kinds."""
    want = np.zeros(cfg.n_kinds, dtype=np.int16)
    for k in kinds:
        want[k] += 1
    rows = np.flatnonzero((set_templates(cfg) == want).all(-1))
    assert rows.size == 1
    return int(rows[0])


def _states(cfg, games: int, steps: int, seed: int = 0):
    """Real mid-game states, played by `by_value` itself.

    Random play never reaches a meld on this config, so the states the objective
    actually decides in have to be reached by something that plays.
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


def test_the_matmul_shortfall_is_the_reference_one():
    """Exact, not an approximation -- and only because template rows are binary.

    A run holds distinct kinds and a group distinct colours of one number, so
    `max(template - rack, 0)` collapses to `template & (rack == 0)`. If a config ever
    admitted a repeated kind the collapse would silently under-count, so the equality
    is asserted rather than assumed.
    """
    for cfg in (C, TINY_GROUPS):
        pot = rp.RackPotential(cfg)
        rng = np.random.default_rng(0)
        racks = rng.integers(0, 3, (64, cfg.n_kinds))
        lack = (racks == 0).astype(np.float32)
        assert np.array_equal((lack @ pot.rows.T).astype(np.int64), shortfall(cfg, racks))


def test_ready_is_the_largest_set_the_remaining_rack_can_lay():
    """`ready` is `macro.playable` -- the predicate `legal_macros` gates a new set on
    -- read for its biggest member, so the term is in tiles and `w = 1` means "a tile I
    can shed next is worth a tile shed now"."""
    pot = rp.RackPotential(C)
    rng = np.random.default_rng(1)
    racks = rng.integers(0, 2, (64, C.n_kinds))
    sizes = set_templates(C).sum(-1)
    wanted = np.where(playable(C, racks), sizes[None], 0).max(-1)
    ready, _ = pot.parts(racks, np.full(C.n_kinds, 2))
    assert np.array_equal(ready, wanted.astype(np.float64))


def test_a_held_joker_is_what_makes_a_one_away_set_ready():
    """The sharpest instance of the hypothesis, as a number: the same two tiles are
    worth a whole set with a joker behind them and nothing without one."""
    pot = rp.RackPotential(C)
    two = [kind_of(C, 0, n) for n in (5, 6)]
    unseen = np.full(C.n_kinds, 2)
    bare, _ = pot.parts(_rack(C, two)[None], unseen)
    held, _ = pot.parts(_rack(C, [*two, C.joker_kind])[None], unseen)
    assert float(bare[0]) == 0.0
    assert float(held[0]) == 3.0


def test_reach_counts_a_completing_kind_once_however_many_sets_it_finishes():
    """One tile serving four near-runs is one draw.

    Red 5-6-7-8 is one tile away from several templates -- 4-5-6, 4-5-6-7, 4-5-6-7-8,
    7-8-9, 6-7-8-9 -- but only *two* draws finish anything, the 4 and the 9. Counting
    templates would score those two doors as five.
    """
    pot = rp.RackPotential(C)
    rack = _rack(C, [kind_of(C, 0, n) for n in (5, 6, 7, 8)])
    unseen = np.full(C.n_kinds, 2)
    short = (rack == 0).astype(np.float32) @ pot.rows.T
    assert int((short == 1).sum()) > 2, "the undeduplicated count has to be larger"

    _, reach = pot.parts(rack[None], unseen)
    doors = unseen[kind_of(C, 0, 4)] + unseen[kind_of(C, 0, 9)]
    assert float(reach[0]) == pytest.approx(doors / unseen.sum())


def test_reach_ignores_a_kind_no_unseen_copy_can_supply():
    """A one-away set whose missing tile is all on the table is not a chance."""
    pot = rp.RackPotential(C)
    rack = _rack(C, [kind_of(C, 0, n) for n in (5, 6, 7, 8)])
    unseen = np.full(C.n_kinds, 2)
    shut = unseen.copy()
    shut[kind_of(C, 0, 4)] = 0
    _, open_reach = pot.parts(rack[None], unseen)
    _, shut_reach = pot.parts(rack[None], shut)
    assert float(shut_reach[0]) < float(open_reach[0])
    assert float(shut_reach[0]) == pytest.approx(unseen[kind_of(C, 0, 9)] / shut.sum())


def test_the_played_counts_are_the_tiles_the_env_takes_out_of_the_rack():
    """`played` is what the potential is computed on, so the engine has to agree.

    Counted per block, because the legal macros of a state are overwhelmingly the
    first one -- a sample not stratified by block would leave the steal branch, the
    only one that takes a tile *off* the table, untested.
    """
    cfg = C
    agent = MacroAgent(cfg, choose=by_value(cfg))
    pot = rp.RackPotential(cfg)
    ends = (0, extend_offset(cfg), steal_offset(cfg), repartition_offset(cfg))
    checked = [0, 0, 0]
    for state, obs, _mask in _states(cfg, 8, 220, seed=3):
        fresh = turn_starting(obs)
        for env in range(state.batch_size):
            if state.done[env] or not fresh[env]:
                continue
            legal = agent.legal_macros(obs, env)
            hand = np.asarray(obs["rack"][env]).astype(np.int64)
            for block in range(3):
                offered = np.flatnonzero(legal[ends[block] : ends[block + 1]]) + ends[block]
                for macro in offered[:: max(1, len(offered) // 2)].tolist():
                    actions = agent.expand(obs, env, int(macro))
                    if not actions:
                        continue
                    predicted = pot.played(np.array([macro]), hand)[0]
                    replay = state.clone()
                    for action in actions:
                        replay_mask = legal_actions(replay, summarize(cfg, replay.table_sets))
                        assert replay_mask[env, action], "the expansion left the mask behind"
                        step = np.full(replay.batch_size, cfg.draw_action, dtype=np.int64)
                        step[env] = action
                        engine_step(replay, step, replay_mask)
                    left = current_rack(replay)[env].astype(np.int64)
                    assert np.array_equal(hand - left, predicted), (
                        f"macro {macro} took different tiles than predicted"
                    )
                    checked[block] += 1
    assert all(checked), f"a branch went unexercised: {checked}"
    assert sum(checked) > 100, f"only {sum(checked)} expansions exercised"


def test_a_zero_weight_is_by_value_exactly():
    """The control is the same code as the arm, so it has to reduce to the baseline.

    Over random masks and both meld phases, which is where `by_value`'s two rankings
    differ -- and `argmax`'s first-maximum order is part of the claim, not an
    accident: two plays of equal rank must still go the baseline's way.
    """
    ranked = repartition_offset(C)
    choose = by_value(C)
    rng = np.random.default_rng(0)
    for mode in WEIGHTED:
        arm = rp.WeightedRack(C, mode=mode, w=0.0)
        for melded in (False, True):
            for _ in range(60):
                rack = rng.integers(0, 2, C.n_kinds).astype(np.int16)
                obs = _obs(C, rack, melded=melded)
                legal = rng.random(ranked + 2) < 0.02
                legal[-1] = True
                if not legal[:ranked].any():
                    continue
                assert arm(obs, 0, legal) == choose(obs, 0, legal), mode


def test_a_zero_weight_plays_by_value_in_the_states_the_arm_decides_in():
    """The same claim on real states, where the mask is the engine's and not random."""
    cfg = C
    base = by_value(cfg)
    arms = [rp.WeightedRack(cfg, mode=mode, w=0.0) for mode in WEIGHTED]
    agent = MacroAgent(cfg, choose=base)
    decided = 0
    for state, obs, _mask in _states(cfg, 12, 400, seed=7):
        fresh = turn_starting(obs)
        for env in range(state.batch_size):
            if state.done[env] or not fresh[env]:
                continue
            legal = agent.legal_macros(obs, env)
            wanted = base(obs, env, legal)
            for arm in arms:
                assert arm(obs, env, legal) == wanted
            decided += 1
    assert decided > 200, f"only {decided} decisions exercised"


def test_a_positive_weight_overrules_the_ranking():
    """What makes this an objective and not a tie-break, on the motivating hand.

    Red 1-2-3-4 with the blue and green 4s: `by_value` post-meld takes the four-tile
    run and strands a pair, while the three-tile run leaves 4-4-4 whole -- three
    tiles shed now against three shed now and three ready, and the ranking cannot
    express that because it only counts the first three.
    """
    run4 = [kind_of(C, 0, n) for n in (1, 2, 3, 4)]
    group = [kind_of(C, c, 4) for c in (0, 1, 2)]
    rack = _rack(C, [*run4, *group[1:]])
    obs = _obs(C, rack)
    long_run = _template_index(C, run4)
    short_run = _template_index(C, run4[:3])
    legal = _legal(C, (long_run, short_run, _template_index(C, group)))

    assert by_value(C)(obs, 0, legal) == long_run, "the ranking prefers the longer run"
    assert rp.WeightedRack(C, mode="ready", w=1.0)(obs, 0, legal) == short_run


def test_the_joker_rule_withholds_only_a_joker_that_does_not_finish():
    """A joker spent to win is not a joker lost, and the rule has to say so."""
    arm = rp.WeightedRack(C, mode="joker")
    reds = [kind_of(C, 0, n) for n in (5, 6, 7)]
    gapped = _template_index(C, [kind_of(C, 0, n) for n in (5, 6, 7, 8)])
    whole = _template_index(C, reds)
    other = _template_index(C, [kind_of(C, 1, n) for n in (1, 2, 3)])

    spare = _rack(C, [*reds, C.joker_kind, *(kind_of(C, 1, n) for n in (1, 2, 3))])
    legal = _legal(C, (gapped, whole, other))
    assert by_value(C)(_obs(C, spare), 0, legal) == gapped, "four tiles outrank three"
    assert arm(_obs(C, spare), 0, legal) == whole, "the joker should have been held"

    # Nothing else in hand, so the gapped run empties the rack: the rule must let it.
    last = _rack(C, [*reds, C.joker_kind])
    legal = _legal(C, (gapped, whole))
    assert arm(_obs(C, last), 0, legal) == gapped


def test_the_null_control_perturbs_at_a_comparable_rate():
    """A control that fired far less often would answer a different question.

    It cannot be exact -- the null permutes the arm's own potentials across the
    candidates rather than reordering an identical tie set -- so what is asserted is
    that both act, on the same decisions, at the same order of magnitude.
    """
    cfg = C
    agent = MacroAgent(cfg, choose=by_value(cfg))
    real = rp.WeightedRack(cfg, mode="both", w=1.0)
    null = rp.WeightedRack(cfg, mode="both", w=1.0, null=True)
    for state, obs, _mask in _states(cfg, 16, 460, seed=11):
        fresh = turn_starting(obs)
        for env in range(state.batch_size):
            if state.done[env] or not fresh[env]:
                continue
            legal = agent.legal_macros(obs, env)
            real(obs, env, legal)
            null(obs, env, legal)
    assert real.stats.decisions == null.stats.decisions
    assert real.stats.moved > 20, "the arm never acted, so this asserts nothing"
    assert 0.4 < null.stats.moved / real.stats.moved < 2.5


def test_the_null_control_is_reproducible_across_processes():
    """Seeded from the state's bytes, not from `hash`, whose seed varies per run."""
    board = _table(C, [[kind_of(C, 0, n) for n in (5, 6, 7)]])
    rack = _rack(C, [kind_of(C, 1, 4)]).astype(np.int64)
    first = rp.state_rng(board, rack).permutation(6)
    assert np.array_equal(first, rp.state_rng(board, rack).permutation(6))
    assert not np.array_equal(first, rp.state_rng(board, rack + 1).permutation(6))


def test_the_arm_at_zero_weight_mirrors_the_baseline_exactly():
    """End to end: the control is not merely the same choice, it is the same games.

    Anything less and the head-to-head's exact 50.00% would be measuring the harness
    rather than the objective.
    """
    cfg = TINY_GROUPS
    stats = rp.MoveStats()
    arm = lambda c: rp.build_arm(c, "both", 0.0, False, stats)  # noqa: E731
    base = lambda c: MacroAgent(c, choose=by_value(c))  # noqa: E731
    wins, scores = denial_ab.head_to_head(cfg, arm, base, deals=8, seed_base=1234, batch=8)
    assert float(wins.mean()) == 0.5
    assert float(scores.mean()) == 0.0
    assert stats.decisions > 0, "the arm never decided anything"
    assert stats.moved == 0


def test_the_outcome_change_bound_is_zero_for_a_mirror_and_positive_for_an_arm():
    """The ceiling the win rate cannot give, and it has to be exact at both ends.

    A deal the arm played identically is a deal it cannot have won differently, so a
    mirrored arm must register no change at all -- and an arm that overrides the
    ranking must register some, or the diagnostic is blind.
    """
    # The standard config, not `tiny`: a five-number deck gives the objective almost
    # nothing to choose between, and the arm changed no outcome there at all.
    cfg = C
    base = lambda c: MacroAgent(c, choose=by_value(c))  # noqa: E731
    stats = rp.MoveStats()
    same = rp.outcome_changes(cfg, base, base, deals=5, seed_base=1234, batch=8)
    assert not same.any()
    arm = lambda c: rp.build_arm(c, "stop", 4.0, False, stats)  # noqa: E731
    moved = rp.outcome_changes(cfg, arm, base, deals=5, seed_base=1234, batch=8)
    assert moved.any()
