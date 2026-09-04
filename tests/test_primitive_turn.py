"""The primitive turn decoder, held to the engine it has to be replayed into.

Everything `rummi/agents/learned/primitive_turn.py` claims rests on one property:
what comes out of the beam is a turn the env will accept, action for action. That
cannot be checked by inspection -- the search runs on a *simulated* state, and the
whole point is that the simulation and the real thing agree -- so it is checked by
replaying every decoded plan against `legal_actions` on the real state it was
decoded from.

Both tests are driven by a scorer that hands back a recorded action, for the reason
`tests/test_repartition_net.py` gives: an untrained net commits a turn far too
rarely to say anything about the loop it came out of, and a test that passes because
nothing happened is worth nothing.
"""

from __future__ import annotations

import numpy as np
import pytest

from rummi.agents.base import act_by_seat, turn_starting
from rummi.agents.frugal_agent import FrugalAgent
from rummi.agents.greedy_agent import GreedyAgent
from rummi.env.numpy.deal import reset
from rummi.env.numpy.engine import step as engine_step
from rummi.env.numpy.masks import legal_actions
from rummi.env.observation import encode
from rummi.rules.config import STANDARD, RummiConfig
from tests.conftest import play

pytest.importorskip("ortools")
pytest.importorskip("torch")

from rummi.agents.learned.primitive_turn import (
    PrimitiveRepartition,
    PrimitiveTurnAgent,
    Scorer,
    decode_turns,
)
from rummi.agents.learned.torch_net import TorchPolicy
from rummi.agents.learned.turn_sim import TurnStart


class Replay:
    """Scores a recorded plan's next action, keyed by the position it belongs to.

    `group` is what makes this exact: the beam drops rows as they finish, so a row
    index says nothing about which turn it is decoding and only the group does. An
    empty plan scores `DRAW`, which is how the teacher declines.
    """

    def __init__(self, cfg: RummiConfig, plans: list[list[int]]) -> None:
        self.cfg = cfg
        self.plans = plans
        self.depth = 0

    def __call__(self, obs, mask: np.ndarray, group: np.ndarray) -> np.ndarray:
        out = np.zeros(mask.shape, dtype=np.float32)
        for row, which in enumerate(group.tolist()):
            plan = self.plans[which]
            wanted = plan[self.depth] if self.depth < len(plan) else self.cfg.draw_action
            out[row, wanted] = 10.0
        self.depth += 1
        return out


def _teacher_turns(
    cfg: RummiConfig, envs: int = 3, seed: int = 11, max_steps: int = 700
) -> list[tuple]:
    """`(state at the boundary, TurnStart, the turn frugal played)` for real games."""
    state = reset(cfg, envs, seed=seed)
    seats = [FrugalAgent(cfg), GreedyAgent(cfg)]
    for agent in seats:
        agent.reset(envs)

    out: list[tuple] = []
    open_turns: dict[int, tuple] = {}
    for _ in range(max_steps):
        if state.done.all():
            break
        mask = legal_actions(state)
        obs = encode(state)
        actions, _ = act_by_seat(seats, cfg, state.current, state.done, mask, obs)
        fresh = turn_starting(obs)
        for env in range(envs):
            if state.done[env] or state.current[env] != 0:
                continue
            if fresh[env]:
                open_turns[env] = (state.select([env]), TurnStart.from_obs(obs, env), [])
            entry = open_turns.get(env)
            if entry is None:
                continue
            action = int(actions[env])
            if action == cfg.draw_action:
                open_turns.pop(env)
                continue
            entry[2].append(action)
            if action == cfg.end_turn_action:
                out.append(entry)
                open_turns.pop(env)
        engine_step(state, actions, mask)
    return out


def test_a_teacher_driven_decode_replays_into_the_real_env() -> None:
    cfg = STANDARD
    records = _teacher_turns(cfg)
    assert len(records) > 30, "the fixture has to reach enough committed turns to mean something"

    starts = TurnStart.stack([row[1] for row in records])
    plans = [row[2] for row in records]
    decoded = decode_turns(cfg, Replay(cfg, plans), starts, beam=1)

    for (real, _, plan), found in zip(records, decoded, strict=True):
        assert list(found.actions) == plan
        assert found.actions[-1] == cfg.end_turn_action
        # The reward the beam ranks on is the rack delta, and a PLACE is the only
        # action that moves a tile out of the rack.
        assert found.tiles == sum(1 for a in plan if a < cfg.pick_offset)
        # The decode ran on a simulated state; this is the real one it came from.
        play(real, plan)


def test_a_wider_beam_never_loses_a_turn() -> None:
    """Beam 4 searches a superset of beam 1, so it cannot come back with less."""
    cfg = STANDARD
    records = _teacher_turns(cfg, envs=2, max_steps=400)
    starts = TurnStart.stack([row[1] for row in records])
    plans = [row[2] for row in records]
    wide = decode_turns(cfg, Replay(cfg, plans), starts, beam=4)
    for (_, _, plan), found in zip(records, wide, strict=True):
        assert found.tiles >= sum(1 for a in plan if a < cfg.pick_offset)


class _Oracle(PrimitiveTurnAgent):
    """Arm B, decoding whatever `greedy` would play from the same boundary.

    A stand-in for a trained policy, and a legitimate one: `GreedyAgent.plan` reads
    only the rack, the table and the meld flag, so at a turn boundary it is a
    function of the observation exactly as a network is.
    """

    def plan_batch(self, obs, envs: np.ndarray) -> list[list[int]]:
        self.scorer = Replay(self.cfg, GreedyAgent(self.cfg).plan_batch(obs, envs))
        return super().plan_batch(obs, envs)


def test_arm_b_commits_whole_turns_and_otherwise_draws() -> None:
    cfg = STANDARD
    envs = 3
    state = reset(cfg, envs, seed=17)
    learner = _Oracle(cfg, Scorer(TorchPolicy(cfg, seed=0)))
    seats = [learner, GreedyAgent(cfg)]
    for agent in seats:
        agent.reset(envs)

    turns: dict[int, list[int]] = {}
    finished: list[list[int]] = []
    for _ in range(700):
        if state.done.all():
            break
        mask = legal_actions(state)
        obs = encode(state)
        actions, illegal = act_by_seat(seats, cfg, state.current, state.done, mask, obs)
        assert illegal == 0, "arm B proposed an action the mask refuses"
        for env in range(envs):
            if state.done[env] or state.current[env] != 0:
                continue
            turns.setdefault(env, []).append(int(actions[env]))
            if actions[env] in (cfg.end_turn_action, cfg.draw_action):
                finished.append(turns.pop(env))
        engine_step(state, actions, mask)

    assert len(finished) > 30
    committed = [turn for turn in finished if turn[-1] == cfg.end_turn_action]
    assert committed, "the oracle-driven arm has to actually play some turns"
    for turn in finished:
        # Either the whole turn was played out, or nothing of it was: a half-built
        # turn abandoned into DRAW is what the primitive space costs and what this
        # agent must never do.
        assert turn == [cfg.draw_action] or cfg.draw_action not in turn


def test_the_repartition_arm_falls_through_when_the_decode_declines() -> None:
    """Arm A with an untrained net still plays legally: no answer means `by_value`.

    The gate is `MacroAgent`'s, unchanged, so a decode that comes back with nothing
    has to leave the agent exactly where a declining CP-SAT solve leaves it.
    """
    cfg = STANDARD
    envs = 2
    state = reset(cfg, envs, seed=23)
    learner = PrimitiveRepartition(cfg, Scorer(TorchPolicy(cfg, seed=0)), beam=1)
    seats = [learner, GreedyAgent(cfg)]
    for agent in seats:
        agent.reset(envs)

    for _ in range(400):
        if state.done.all():
            break
        mask = legal_actions(state)
        actions, illegal = act_by_seat(seats, cfg, state.current, state.done, mask, encode(state))
        assert illegal == 0
        engine_step(state, actions, mask)

    assert learner.asked > 0, "the gate never fired, so nothing was tested"
    assert learner.answered == 0, "an untrained net is not expected to construct a repartition"


def test_the_decode_says_whether_it_declined_or_ran_out_of_room() -> None:
    """A failed decode returns no plan either way, so the fields have to separate them.

    With `DRAW` labelled, declining is a decision the decoder can take at the first
    step, and it has to be distinguishable from a hypothesis the mask killed at the
    turn's micro budget -- the diagnostic that says which of the two the space costs.
    """
    cfg = STANDARD
    records = _teacher_turns(cfg, envs=2, max_steps=400)[:10]
    assert len(records) == 10
    starts = TurnStart.stack([row[1] for row in records])
    plans = [row[2] for row in records]

    played = decode_turns(cfg, Replay(cfg, plans), starts, beam=1)
    for plan, found in zip(plans, played, strict=True):
        # `END_TURN` finishes the hypothesis rather than extending it.
        assert found.depth == len(plan) - 1
        assert not found.declined

    # An empty plan makes the teacher emit DRAW at the first step, which is exactly
    # a decline: nothing explored, and the reason on record.
    declined = decode_turns(cfg, Replay(cfg, [[] for _ in plans]), starts, beam=1)
    for found in declined:
        assert not found.plays
        assert found.declined
        assert found.depth == 0
