"""Optimal single-turn play: CP-SAT chooses the table, a translator reaches it.

Optimal *per turn*, not over the game -- no opponent model and no lookahead. Even
so it beats greedy in every head-to-head measured, because unrestricted
rearrangement is on the table. Treat it as the ceiling to aim at.

Far too slow for a training loop: tens of milliseconds per turn, per env rather
than per batch. It is an opponent, a reference score, and a source of expert
trajectories.
"""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from rummi.agents.base import Observation, PlanningAgent, has_melded, table
from rummi.rules.config import RummiConfig


class OptimalAgent(PlanningAgent):
    name = "optimal"

    def __init__(
        self, cfg: RummiConfig, time_limit: float = 2.0, workers: int | None = None
    ) -> None:
        super().__init__(cfg)
        self.time_limit = time_limit
        self.workers = workers if workers is not None else min(8, os.cpu_count() or 1)
        self.solves = 0
        self.no_play = 0
        self._tally = threading.Lock()

    def plan(self, obs: Observation, env: int) -> list[int]:
        from rummi.solver.ilp import solve_turn
        from rummi.solver.to_actions import plan, slot_contents

        cfg = self.cfg
        board = table(obs)[env]
        melded = bool(has_melded(obs)[env])

        solution = solve_turn(
            cfg, obs["rack"][env].astype(np.int64), board, melded, time_limit=self.time_limit
        )
        with self._tally:
            self.solves += 1
            self.no_play += not solution.plays_anything
        if not solution.plays_anything:
            return []

        # `plays_anything` implies a played vector, but only at runtime.
        if solution.played is None:
            return []

        target = list(solution.sets)
        if not melded and cfg.strict_initial_meld:
            # Only the strict rule locks the table, and only then does the solver
            # hide it -- so only then are the solved sets additions to it. With the
            # rule relaxed the solver saw the table and `sets` is the whole target
            # already; appending the standing sets would double them.
            target += [c for c in slot_contents(board) if c]
        return plan(cfg, board, target, solution.played)

    def plan_batch(self, obs: Observation, envs: np.ndarray) -> list[list[int]]:
        """One solve per env, run in a thread pool.

        Each solve builds its own model and is held to a single CP-SAT worker so a
        baseline stays reproducible, which also makes them independent: threading
        changes the wall clock and nothing else, because OR-Tools drops the GIL
        inside `Solve`, where the time goes. Measured 2.9x, and it plateaus there --
        building the model is Python, and that part does not parallelise.

        With one caveat, and it is the wall clock: `solve_turn`'s limit is
        `max_time_in_seconds`, so a solve that *reaches* it returns whichever
        incumbent it had, which depends on the CPU it got. Solves run in tens of
        milliseconds against a two-second limit, so this does not bite -- but it is
        why the parity claim above is about finished solves and not about all of
        them.
        """
        if self.workers <= 1 or envs.size < 2:
            return super().plan_batch(obs, envs)
        with ThreadPoolExecutor(self.workers) as pool:
            return list(pool.map(lambda env: self.plan(obs, int(env)), envs))
