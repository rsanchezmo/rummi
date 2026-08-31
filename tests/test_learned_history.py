"""Opponent history: attribution against the state, and no effect on play.

The tracker reconstructs the opponent's turn from the learner's own side of the
table -- what the table gained while the learner was not acting. The state knows
whose turn it really was and what that seat was holding, so every check here is
against *that*, never against another reading of the same observation.

Which is also why these run the seats by hand rather than through
:class:`~rummi.env.fixed_opponent.FixedOpponentEnv`: that class hides the
opponent's whole turn inside one step, and the thing under test is precisely the
reconstruction of what happened in there.

Three hazards, one test each: slots re-sorted at ``END_TURN``, a ``DRAW`` that
reverts the learner's own turn, and an env recycled by autoreset.
"""

from __future__ import annotations

import numpy as np
import pytest

from rummi.agents.base import Observation, act_by_seat
from rummi.agents.greedy_agent import GreedyAgent
from rummi.agents.learned.history import (
    H_DRAWS,
    HistoryMacroAgent,
    OpponentHistory,
    history_dim,
    history_scale,
)
from rummi.agents.macro import MacroAgent, by_value
from rummi.env.numpy.deal import reset
from rummi.env.numpy.engine import step
from rummi.env.numpy.masks import legal_actions
from rummi.env.numpy.sets import summarize
from rummi.env.observation import encode
from rummi.rules.config import STANDARD, TINY_GROUPS, RummiConfig

LEARNER, OPPONENT = 0, 1
ENVS = 8


def table_counts(cfg: RummiConfig, table_sets: np.ndarray) -> np.ndarray:
    """`(B, K)` kinds on each table, counted the slow obvious way on purpose."""
    out = np.zeros((table_sets.shape[0], cfg.n_kinds), dtype=np.int64)
    for env in range(table_sets.shape[0]):
        row = table_sets[env].reshape(-1)
        for kind in row[row >= 0]:
            out[env, int(kind)] += 1
    return out


def first_set_then_draw(cfg: RummiConfig):
    """Build the best set available, then throw the turn away.

    ``DRAW`` reverts everything the turn placed, so the table the opponent
    inherits is the one the turn *began* on. A tracker that snapshotted at the
    decision instead would hand the opponent the set the learner just lost.
    """
    end = MacroAgent(cfg).end_macro
    order = by_value(cfg)

    def choose(obs: Observation, env: int, legal: np.ndarray) -> int:
        return order(obs, env, legal) if legal[:end].any() else end + 1

    return choose


def drive(cfg: RummiConfig, chooser, decay: float = 0.0, steps: int = 500) -> dict:
    """Play a batch out, checking the tracker against the state at every boundary.

    Returns what the run exercised, so a test can refuse to pass on a batch where
    nothing interesting happened.
    """
    history = OpponentHistory(cfg, decay=decay)
    learner = HistoryMacroAgent(cfg, history, choose=chooser)
    seats = [learner, GreedyAgent(cfg)]
    state = reset(cfg, ENVS, seed=7)
    for seat in seats:
        seat.reset(ENVS)

    # State-side truth: what the opponent put on the table, summed over the turns
    # `state.current` says were its own.
    truth = np.zeros((ENVS, cfg.n_kinds), dtype=np.int64)
    opponent_table: list[np.ndarray | None] = [None] * ENVS
    opponent_rack: list[np.ndarray | None] = [None] * ENVS
    folded: list[np.ndarray | None] = [None] * ENVS
    checks = {"attributions": 0, "declines": 0, "quiet_turns": 0}

    for _ in range(steps):
        if state.done.all():
            break
        summary = summarize(cfg, state.table_sets)
        mask = legal_actions(state, summary)
        obs = encode(state, summary)
        now = table_counts(cfg, state.table_sets)

        at_boundary = []
        for env in range(ENVS):
            if state.done[env] or state.micro_count[env] != 0:
                continue
            if state.current[env] == OPPONENT:
                opponent_table[env] = now[env].copy()
                opponent_rack[env] = state.racks[env, OPPONENT].copy()
                continue
            started = opponent_table[env]
            if started is None:
                continue
            played = np.maximum(now[env] - started, 0)
            truth[env] += played
            folded[env] = opponent_rack[env]
            opponent_table[env] = None
            at_boundary.append(env)
            checks["quiet_turns"] += int(played.sum() == 0)

        actions, illegal = act_by_seat(seats, cfg, state.current, state.done, mask, obs)
        assert illegal == 0

        for env in at_boundary:
            checks["attributions"] += 1
            assert np.array_equal(history._played[env].astype(np.int64), truth[env]), (
                env, history._played[env], truth[env]
            )
            held = folded[env]
            assert held is not None
            # At decay 0 the counter is this turn's verdict alone, so a nonzero
            # column is a claim about the rack the opponent was holding.
            for kind in np.flatnonzero(history._declined[env] > 0):
                assert held[kind] == 0, (env, int(kind), held)
                checks["declines"] += 1

        step(state, actions, mask)

    return checks


@pytest.mark.parametrize("cfg", [TINY_GROUPS, STANDARD], ids=["tiny_groups", "standard"])
def test_the_scale_covers_every_column(cfg: RummiConfig):
    scale = history_scale(cfg, 0.8)
    assert scale.shape == (history_dim(cfg),)
    assert (scale > 0).all()


def test_it_refuses_more_than_one_opponent():
    """Three seats and one table delta cannot say which of them played what."""
    from dataclasses import replace

    with pytest.raises(ValueError, match="two seats"):
        OpponentHistory(replace(TINY_GROUPS, n_players=3))


def test_what_it_attributes_is_what_the_opponent_played():
    """Cumulative per-kind plays, against the turns ``state.current`` says were the
    opponent's.

    Slots are re-sorted into canonical order at every ``END_TURN``, so slot *i*
    before a boundary and slot *i* after it are unrelated sets: a per-slot diff
    fails this on the first turn that lands a set.
    """
    checks = drive(TINY_GROUPS, by_value(TINY_GROUPS))
    assert checks["attributions"] > 15, checks


def test_a_reverted_turn_is_attributed_to_nobody():
    """A learner that builds a set and then draws must not see it come back as the
    opponent's play."""
    cfg = TINY_GROUPS
    checks = drive(cfg, first_set_then_draw(cfg))
    assert checks["attributions"] > 15, checks


def test_a_declined_layoff_means_the_opponent_held_none_of_that_kind():
    """The sharp claim, and the whole reason the block is worth its width.

    ``greedy`` lays off every tile the table would take, so a kind the table
    accepted and the opponent did not play is a kind it does not hold. Checked
    against the rack the state says it was holding -- which no agent may read, and
    which is exactly why the inference has to come from history instead.
    """
    cfg = STANDARD
    checks = drive(cfg, by_value(cfg), steps=260)
    assert checks["declines"] > 0, "no lay-off was ever declined, so nothing was tested"


def test_the_tracker_never_changes_what_is_played():
    """It is read-only on the game, which is what makes a flag-off arm trustworthy:
    a tracker that perturbed the action stream would compare two experiments."""
    cfg = TINY_GROUPS
    plain = MacroAgent(cfg, choose=by_value(cfg))
    watched = HistoryMacroAgent(
        cfg, OpponentHistory(cfg), choose=by_value(cfg)
    )

    def transcript(learner) -> list[int]:
        seats = [learner, GreedyAgent(cfg)]
        state = reset(cfg, ENVS, seed=11)
        for seat in seats:
            seat.reset(ENVS)
        out: list[int] = []
        for _ in range(500):
            if state.done.all():
                break
            summary = summarize(cfg, state.table_sets)
            mask = legal_actions(state, summary)
            actions, _ = act_by_seat(
                seats, cfg, state.current, state.done, mask, encode(state, summary)
            )
            out.extend(actions.tolist())
            step(state, actions, mask)
        return out

    assert transcript(plain) == transcript(watched)


def test_a_recycled_env_starts_from_nothing():
    """Autoreset re-deals an env, and nothing the tracker holds may cross that.

    The env re-deals on the step *after* it reports done, so exactly one
    observation -- the terminal one -- reaches the agent in between. Dropping that
    one is what stops a turn boundary being read across two episodes.
    """
    pytest.importorskip("gymnasium")
    from rummi.env.fixed_opponent import FixedOpponentEnv

    cfg = TINY_GROUPS
    history = OpponentHistory(cfg)
    learner = HistoryMacroAgent(cfg, history, choose=by_value(cfg))
    env = FixedOpponentEnv(num_envs=ENVS, cfg=cfg, seed=3, opponent="greedy")
    learner.reset(ENVS)
    obs, info = env.reset()

    resets = 0
    for _ in range(400):
        actions = learner.act(obs, np.asarray(info["action_mask"]))
        obs, _, term, trunc, info = env.step(actions)
        done = np.asarray(term) | np.asarray(trunc)
        history.clear(done)
        resets += int(done.sum())
        for e in np.flatnonzero(done):
            assert history._played[e].sum() == 0
            assert history._scalars[e].sum() == 0
        # Nothing is played more often than the deck holds it, and a count carried
        # across a re-deal breaks exactly this.
        assert (history._played <= cfg.n_copies).all()
    env.close()
    assert resets > 0, "no episode finished, so the boundary was never exercised"


def test_a_turn_that_played_nothing_reads_as_a_draw():
    """A draw is invisible in ``unseen`` -- the pool and the opponent's rack are
    merged there -- so the empty table delta is the only evidence of one."""
    cfg = TINY_GROUPS
    history = OpponentHistory(cfg)
    learner = HistoryMacroAgent(cfg, history, choose=by_value(cfg))
    seats = [learner, GreedyAgent(cfg)]
    state = reset(cfg, ENVS, seed=5)
    for seat in seats:
        seat.reset(ENVS)

    for _ in range(400):
        if state.done.all():
            break
        summary = summarize(cfg, state.table_sets)
        mask = legal_actions(state, summary)
        actions, _ = act_by_seat(
            seats, cfg, state.current, state.done, mask, encode(state, summary)
        )
        step(state, actions, mask)

    drawn = history._scalars[:, H_DRAWS]
    assert (drawn > 0).any(), "greedy never passed a turn, so nothing was counted"
    assert (drawn <= history._scalars[:, 4]).all(), "more draws than turns seen"
