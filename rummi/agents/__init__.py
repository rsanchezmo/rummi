"""Agents, and the interface they implement.

One way to write an agent: implement :class:`~rummi.agents.base.Agent`, or
subclass :class:`~rummi.agents.base.PlanningAgent` if you decide a whole turn at
once. Agents see an observation and a legal-action mask and nothing else -- see
``base.py`` for why that restriction is load-bearing.

The bundled agents run from weakest to strongest: ``random`` (which on the
standard config plays no differently from passing), ``greedy``, ``rearrange``,
and ``optimal``.
"""

from rummi.agents.base import Agent, Observation, PlanningAgent
from rummi.agents.greedy_agent import GreedyAgent
from rummi.agents.optimal_agent import OptimalAgent
from rummi.agents.random_agent import RandomAgent, WeightedRandomAgent
from rummi.agents.rearrange_agent import RearrangeAgent
from rummi.rules.config import RummiConfig

REGISTRY: dict[str, type] = {
    "random": RandomAgent,
    "weighted-random": WeightedRandomAgent,
    "greedy": GreedyAgent,
    "rearrange": RearrangeAgent,
    "optimal": OptimalAgent,
}


def build(name: str, cfg: RummiConfig, **kwargs) -> Agent:
    if name not in REGISTRY:
        raise ValueError(f"unknown agent {name!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[name](cfg, **kwargs)


__all__ = [
    "Agent",
    "GreedyAgent",
    "Observation",
    "OptimalAgent",
    "PlanningAgent",
    "RandomAgent",
    "RearrangeAgent",
    "REGISTRY",
    "WeightedRandomAgent",
    "build",
]
