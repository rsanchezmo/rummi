"""The simulated own-turn state must be the one the env is actually in.

`rummi/agents/learned/turn_sim.py` rebuilds a position as a `BatchState` from the
observation alone, so that a decoder searching over primitive actions can ask the
env's own mask and encoder about turns it has not played. Everything downstream
rests on that reconstruction being exact rather than close: a beam that searched a
slightly different table would emit a plan the real env refuses, and it would refuse
it three actions in, after the turn was already half spent.

So the contract is measured against real play. A game per env is played out, and at
every step of whichever seat is acting the simulated mask and every field of the
simulated observation are held to the ones the env reports -- carried forward
through the whole turn, not re-snapshotted per step, because carrying it forward is
what a decode does.
"""

from __future__ import annotations

import numpy as np
import pytest

from rummi.agents.base import act_by_seat, turn_starting
from rummi.agents.greedy_agent import GreedyAgent
from rummi.agents.frugal_agent import FrugalAgent
from rummi.agents.learned.turn_sim import TurnStart, to_state
from rummi.env.numpy.deal import reset
from rummi.env.numpy.engine import step as engine_step
from rummi.env.numpy.masks import legal_actions
from rummi.env.numpy.sets import summarize
from rummi.env.observation import encode
from rummi.rules.config import STANDARD, RummiConfig

pytest.importorskip("ortools")

CHECKED = (
    "rack",
    "table_sets",
    "slot_features",
    "workbench",
    "placed_this_turn",
    "unseen",
    "rack_sizes",
    "melded",
    "scalars",
)
"""Every field the encoder produces. Wider than the ones a network reads, because
`table_sets` and `scalars` are where a mismatch shows up first and a drift that a
policy would merely learn around is still a drift."""


def _walk(cfg: RummiConfig, envs: int = 4, seed: int = 3, max_steps: int = 900) -> int:
    """Play, carrying a simulated copy of each acting seat's turn beside the real one."""
    state = reset(cfg, envs, seed=seed)
    seats = [FrugalAgent(cfg), GreedyAgent(cfg)]
    for agent in seats:
        agent.reset(envs)

    sims: dict[int, object] = {}
    checked = 0
    for _ in range(max_steps):
        if state.done.all():
            break
        summary = summarize(cfg, state.table_sets)
        mask = legal_actions(state, summary)
        obs = encode(state, summary)
        actions, illegal = act_by_seat(seats, cfg, state.current, state.done, mask, obs)
        assert illegal == 0
        fresh = turn_starting(obs)

        for env in range(envs):
            if state.done[env]:
                sims.pop(env, None)
                continue
            if fresh[env] or env not in sims:
                sims[env] = to_state(cfg, TurnStart.from_obs(obs, env))
            sim = sims[env]
            sim_summary = summarize(cfg, sim.table_sets)
            np.testing.assert_array_equal(legal_actions(sim, sim_summary)[0], mask[env])
            simulated = encode(sim, sim_summary)
            for name in CHECKED:
                np.testing.assert_array_equal(
                    np.asarray(simulated[name][0]),
                    np.asarray(obs[name][env]),
                    err_msg=f"{name} drifted in env {env}",
                )
            sim.check_invariants()
            checked += 1

            action = int(actions[env])
            if action in (cfg.end_turn_action, cfg.draw_action):
                # Both commit the turn and hand over to a seat whose rack this
                # cannot know, so the simulation ends where the turn does.
                sims.pop(env)
            else:
                engine_step(sim, np.array([action]), legal_actions(sim, sim_summary))

        engine_step(state, actions, mask)
    return checked


def test_the_simulated_turn_matches_the_env_at_every_step() -> None:
    assert _walk(STANDARD) > 800


def test_a_wrong_pool_count_is_caught() -> None:
    """The negative control: the comparison has to be able to fail.

    `pool_size` is the one quantity the merged `unseen` vector cannot supply, so it
    is carried over -- and carrying the wrong one has to show up rather than being
    absorbed into a dummy opponent's rack.
    """
    cfg = STANDARD
    state = reset(cfg, 1, seed=5)
    obs = encode(state)
    start = TurnStart.from_obs(obs, 0)
    off_by_one = TurnStart(
        **{
            **{f: getattr(start, f) for f in TurnStart.__slots__},
            "pool_size": start.pool_size + 1,
        }
    )
    wrong = encode(to_state(cfg, off_by_one))
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(
            np.asarray(wrong["scalars"][0]), np.asarray(obs["scalars"][0])
        )
