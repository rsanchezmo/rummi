"""The reference agents, driven purely by observation.

Each is a thin adapter over a planner that already exists in ``rummi.policies``
or ``rummi.solver``. The planners take ``(rack, table, has_melded)`` -- all three
of which the observation carries -- so no logic is duplicated, and the adapters
double as evidence that an agent needs nothing beyond the observation.
"""

from __future__ import annotations

import numpy as np

from rummi.agents.base import Agent, Observation, PlanningAgent, has_melded, table
from rummi.rules.config import RummiConfig


class RandomAgent:
    """Uniform over legal actions. The floor: anything that loses to this is broken."""

    name = "random"

    def __init__(self, cfg: RummiConfig, seed: int = 0) -> None:
        self.cfg = cfg
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def reset(self, n_envs: int) -> None:
        # Re-seeded so a suite is reproducible however many times it is run.
        self.rng = np.random.default_rng(self.seed)

    def act(
        self, obs: Observation, mask: np.ndarray, active: np.ndarray | None = None
    ) -> np.ndarray:
        scores = np.where(mask, self.rng.random(mask.shape), -1.0)
        return np.argmax(scores, axis=-1)


class WeightedRandomAgent(RandomAgent):
    """Random, but biased towards finishing what it starts.

    Flat-uniform play almost never assembles a legal opening meld by chance -- in
    ten million steps of fuzzing it managed four -- so it never reaches most of
    the game. Weighting the action families gives a floor that at least plays.
    """

    name = "weighted-random"

    def __init__(self, cfg: RummiConfig, seed: int = 0) -> None:
        super().__init__(cfg, seed)
        from rummi.policies.random_policy import action_weights

        self.weights = action_weights(cfg)

    def act(
        self, obs: Observation, mask: np.ndarray, active: np.ndarray | None = None
    ) -> np.ndarray:
        scores = self.rng.exponential(size=mask.shape) / self.weights
        return np.argmax(np.where(mask, -scores, -np.inf), axis=-1)


class GreedyAgent(PlanningAgent):
    """Appends to existing sets and lays new ones, but never rearranges the table.

    The interesting baseline precisely because of that limit: the gap to
    :class:`OptimalAgent` is the value of rearrangement.
    """

    name = "greedy"

    def plan(self, obs: Observation, env: int) -> list[int]:
        from rummi.policies.greedy_policy import plan_turn

        return plan_turn(self.cfg, obs["rack"][env], table(obs)[env], bool(has_melded(obs)[env]))


class OptimalAgent(PlanningAgent):
    """Plays the best possible turn, via OR-Tools CP-SAT.

    Optimal for the turn in front of it, not over the game: no opponent model and
    no lookahead. Even so it beats greedy in every head-to-head measured, because
    unrestricted rearrangement is on the table. Treat it as the ceiling to aim at,
    and note it is far too slow to sit inside a training loop.
    """

    name = "optimal"

    def __init__(self, cfg: RummiConfig, time_limit: float = 2.0) -> None:
        super().__init__(cfg)
        self.time_limit = time_limit
        self.solves = 0
        self.no_play = 0

    def plan(self, obs: Observation, env: int) -> list[int]:
        from rummi.solver.ilp import solve_turn
        from rummi.solver.to_actions import plan, slot_contents

        cfg = self.cfg
        rack = obs["rack"][env].astype(np.int64)
        board = table(obs)[env]
        melded = bool(has_melded(obs)[env])

        solution = solve_turn(cfg, rack, board, melded, time_limit=self.time_limit)
        self.solves += 1
        if not solution.plays_anything:
            self.no_play += 1
            return []

        target = list(solution.sets)
        if not melded:
            # Pre-meld the table is untouchable, so the solved sets are additions.
            target += [c for c in slot_contents(board) if c]
        return plan(cfg, board, target, solution.played)


REGISTRY: dict[str, type] = {
    "random": RandomAgent,
    "weighted-random": WeightedRandomAgent,
    "greedy": GreedyAgent,
    "optimal": OptimalAgent,
}


def build(name: str, cfg: RummiConfig, **kwargs) -> Agent:
    if name not in REGISTRY:
        raise ValueError(f"unknown agent {name!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[name](cfg, **kwargs)
