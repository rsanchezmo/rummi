"""Optimal single-turn play: CP-SAT chooses the table, a translator reaches it.

Optimal *per turn*, not over the game -- no opponent model and no lookahead. Even
so it beats greedy in every head-to-head measured, because unrestricted
rearrangement is on the table. Treat it as the ceiling to aim at.

Far too slow for a training loop: tens of milliseconds per turn, per env rather
than per batch. It is an opponent, a reference score, and a source of expert
trajectories.
"""

from __future__ import annotations

import numpy as np

from rummi.agents.base import Observation, PlanningAgent, has_melded, table
from rummi.rules.config import RummiConfig


class OptimalAgent(PlanningAgent):
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
        board = table(obs)[env]
        melded = bool(has_melded(obs)[env])

        solution = solve_turn(
            cfg, obs["rack"][env].astype(np.int64), board, melded, time_limit=self.time_limit
        )
        self.solves += 1
        if not solution.plays_anything:
            self.no_play += 1
            return []

        target = list(solution.sets)
        if not melded:
            # Pre-meld the table is untouchable, so the solved sets are additions.
            target += [c for c in slot_contents(board) if c]
        return plan(cfg, board, target, solution.played)
