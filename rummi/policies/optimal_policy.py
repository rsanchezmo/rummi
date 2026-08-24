"""Optimal single-turn play: CP-SAT chooses the table, the translator reaches it.

Optimal *per turn*, not over the game -- it maximises what it can shed now with
no model of the opponent and no lookahead. That is still far stronger than
greedy, because unrestricted rearrangement is on the table, and it is the right
reference point for judging a learned agent.

Solving is far too slow for an RL inner loop (tens of milliseconds a turn, and
per env rather than per batch). This is an opponent, a benchmark, and a source of
expert trajectories.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from rummi.rules.actions import action_name
from rummi.rules.config import RummiConfig
from rummi.env.numpy.state import BatchState
from rummi.solver.ilp import DEFAULT_TIME_LIMIT, Objective, solve_turn
from rummi.solver.to_actions import plan


class OptimalPolicy:
    def __init__(
        self,
        cfg: RummiConfig,
        objective: str = Objective.MAX_TILES,
        time_limit: float = DEFAULT_TIME_LIMIT,
        strict: bool = True,
    ) -> None:
        self.cfg = cfg
        self.objective = objective
        self.time_limit = time_limit
        self.strict = strict
        self._plans: dict[int, deque[int]] = {}
        self.solves = 0
        self.infeasible = 0

    def plan_turn(self, state: BatchState, env: int) -> list[int]:
        cfg = self.cfg
        player = int(state.current[env])
        rack = state.racks[env, player].astype(np.int64)
        table = state.table_sets[env]
        has_melded = bool(state.melded[env, player])

        solution = solve_turn(
            cfg,
            rack,
            table,
            has_melded,
            objective=self.objective,
            time_limit=self.time_limit,
        )
        self.solves += 1
        if not solution.plays_anything:
            self.infeasible += 1
            return []

        # Pre-meld the existing table is untouched, so the solved sets are
        # additions to it rather than a repartition of it.
        target = list(solution.sets)
        if not has_melded:
            target += [c for c in _occupied(table)]
        return plan(cfg, table, target, solution.played)

    def act(
        self, state: BatchState, mask: np.ndarray, envs: np.ndarray | None = None
    ) -> np.ndarray:
        """``envs`` restricts which envs this policy owns.

        It must be honoured rather than ignored: plans are cached per env and
        consumed one action at a time, so touching an env another policy controls
        would desynchronise that env's plan.
        """
        cfg = self.cfg
        out = np.full(state.batch_size, cfg.draw_action, dtype=np.int64)
        for env in range(state.batch_size):
            if state.done[env] or (envs is not None and not envs[env]):
                continue
            if state.micro_count[env] == 0:
                self._plans[env] = deque(self.plan_turn(state, env))
            plan_queue = self._plans.get(env)
            if not plan_queue:
                continue
            action = plan_queue.popleft()
            if not mask[env, action]:
                if self.strict:
                    raise AssertionError(
                        f"optimal planned an illegal {action_name(cfg, action)} in env {env}"
                    )
                plan_queue.clear()
                continue
            out[env] = action
        return out


def _occupied(table: np.ndarray) -> list[tuple[int, ...]]:
    from rummi.solver.to_actions import slot_contents

    return [c for c in slot_contents(table) if c]
