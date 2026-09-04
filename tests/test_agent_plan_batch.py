"""`plan_batch` has to answer for the batch it is given, including the empty one.

`PlanningAgent.act` only calls it where some env's turn is starting, so an empty
batch never reaches it from there -- which is exactly why an override that cannot
take one goes unnoticed. Anything that plans turn boundaries itself (a probe, a
dataset collector, a search) selects its own rows, and one of them being empty is
an ordinary step of a rollout rather than an error.
"""

from __future__ import annotations

import numpy as np
import pytest

from rummi.agents.greedy_agent import GreedyAgent
from rummi.agents.rearrange_agent import RearrangeAgent
from rummi.env.numpy.deal import reset
from rummi.env.observation import encode
from rummi.rules.config import STANDARD

PLANNERS = ("greedy", "rearrange")


@pytest.mark.parametrize("name", PLANNERS)
def test_planning_no_envs_at_all_returns_no_plans(name: str) -> None:
    cfg = STANDARD
    obs = encode(reset(cfg, 4, seed=1))
    agent = GreedyAgent(cfg) if name == "greedy" else RearrangeAgent(cfg)
    assert agent.plan_batch(obs, np.zeros(0, dtype=np.int64)) == []
