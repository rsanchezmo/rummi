"""The agent interface the benchmark evaluates.

An agent sees an **observation and a legal-action mask, and nothing else**. That
is deliberate and it is the integrity property the whole benchmark rests on: the
observation exposes only what a player is entitled to know -- their own rack, the
table, and ``unseen`` (the pool and opponents' racks combined, indistinguishable
from each other). An agent handed the raw state could read individual opponents'
racks, and every score would be meaningless.

The reference agents obey this too, including the CP-SAT one. That is worth more
than a convention: a solver that plays *optimally* from the observation alone is
a proof that the observation is sufficient to play the game.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from rummi.rules.config import RummiConfig
from rummi.env.observation import MICRO_COUNT

Observation = dict[str, np.ndarray]
"""Batched observation as produced by :func:`rummi.env.observation.encode`.
Every entry has a leading env dimension. Per-seat fields are rotated so index 0
is always the acting seat, which is what lets one agent play every seat."""


@runtime_checkable
class Agent(Protocol):
    """Implement this to enter the benchmark.

    ``act`` is called once per env-step, and a turn spans several steps, so an
    agent that plans a whole turn should cache its plan and consume it. Use
    :func:`turn_starting` to detect when to re-plan.
    """

    name: str

    def reset(self, n_envs: int) -> None:
        """Called before a suite starts, and whenever episodes are re-dealt.

        Any per-env memory must be cleared here -- envs are recycled.
        """

    def act(
        self,
        obs: Observation,
        mask: np.ndarray,
        active: np.ndarray | None = None,
    ) -> np.ndarray:
        """Choose one action per env.

        ``mask`` is ``(n_envs, n_actions)`` bool and is never all-zero: ``DRAW``
        is always legal. ``active`` marks the envs this agent controls in a
        multi-seat match; an agent holding per-env plans **must** honour it, or it
        will consume the plan of an env another agent is playing.

        Returning a masked-out action is a hard failure, not a penalty: the
        benchmark records it and disqualifies the run.
        """


def turn_starting(obs: Observation) -> np.ndarray:
    """``(n_envs,)`` true where a fresh turn is about to begin."""
    return obs["scalars"][:, MICRO_COUNT] == 0


def own_rack(obs: Observation) -> np.ndarray:
    """``(n_envs, K)`` the acting seat's rack counts."""
    return obs["rack"]


def table(obs: Observation) -> np.ndarray:
    """``(n_envs, S, L)`` the table as kind ids with ``EMPTY`` padding."""
    return obs["table_sets"]


def has_melded(obs: Observation) -> np.ndarray:
    """``(n_envs,)`` whether the acting seat has already opened.

    Per-seat fields are seat-rotated, so column 0 is always the actor.
    """
    return obs["melded"][:, 0].astype(bool)


class PlanningAgent:
    """Base class for agents that decide a whole turn at once.

    Subclasses implement :meth:`plan`, returning the action ids for one env's
    turn. This handles the bookkeeping that is easy to get wrong: re-planning only
    at turn boundaries, honouring ``active``, and falling back to ``DRAW`` when a
    plan is exhausted or empty.
    """

    name = "planning"

    def __init__(self, cfg: RummiConfig) -> None:
        self.cfg = cfg
        self._plans: dict[int, list[int]] = {}

    def reset(self, n_envs: int) -> None:
        self._plans = {}

    def plan(self, obs: Observation, env: int) -> list[int]:
        raise NotImplementedError

    def act(
        self, obs: Observation, mask: np.ndarray, active: np.ndarray | None = None
    ) -> np.ndarray:
        cfg = self.cfg
        n = mask.shape[0]
        out = np.full(n, cfg.draw_action, dtype=np.int64)
        fresh = turn_starting(obs)

        for env in range(n):
            if active is not None and not active[env]:
                continue
            if fresh[env]:
                self._plans[env] = self.plan(obs, env)
            plan = self._plans.get(env)
            if not plan:
                continue
            action = plan.pop(0)
            # A plan that has gone stale is abandoned rather than forced through:
            # DRAW reverts the turn cleanly and is always legal.
            if not mask[env, action]:
                plan.clear()
                continue
            out[env] = action
        return out
