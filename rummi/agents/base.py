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

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np

from rummi.rules.config import RummiConfig
from rummi.rules.observation import MICRO_COUNT

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


def act_on_state(agent: Agent, state, mask: np.ndarray, active=None) -> np.ndarray:
    """Drive an agent from a raw ``BatchState`` by encoding the observation.

    The single bridge between the simulator and the agent interface. Anything
    that holds a state and wants an action goes through here, so agents never
    receive the state itself and cannot accidentally read an opponent's rack.
    """
    from rummi.env.observation import encode

    return agent.act(encode(state), mask, active)


def act_by_seat(
    seats: Sequence[Agent | None],
    cfg: RummiConfig,
    current: np.ndarray,
    done: np.ndarray,
    mask: np.ndarray,
    obs: Observation,
    actions: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    """Fill in one action per env from the agent seated at whichever seat is acting.

    ``seats[i]`` plays every env whose ``current`` seat is ``i``; a seat mapped to
    ``None`` keeps whatever ``actions`` already holds, which is how a caller drives
    one seat itself. Masked-out proposals are replaced by ``DRAW`` and counted
    rather than raised -- a run that reports its illegal attempts is worth more
    than one that dies half-way with no diagnosis.

    The single multi-seat counterpart to :func:`act_on_state`, and it takes no
    state at all: the caller passes the observation, which is what an agent is
    entitled to see, plus the two host-side vectors saying who is acting and who is
    finished. That is also what lets a caller on a device backend use it -- it
    converts those three and keeps its state where it is.
    """
    n = mask.shape[0]
    if actions is None:
        actions = np.full(n, cfg.draw_action, dtype=np.int64)
    rows = np.arange(n)
    illegal_attempts = 0

    for seat, agent in enumerate(seats):
        if agent is None:
            continue
        active = (current == seat) & ~done
        if not active.any():
            continue
        proposed = np.asarray(agent.act(obs, mask, active))
        illegal = active & ~mask[rows, proposed]
        illegal_attempts += int(illegal.sum())
        actions[active] = np.where(illegal[active], cfg.draw_action, proposed[active])

    return actions, illegal_attempts


def oracle_actions(agent: PlanningAgent, obs: Observation, mask: np.ndarray) -> np.ndarray:
    """What would ``agent`` do *from this state*, with no cached plan.

    :meth:`PlanningAgent.act` deliberately replans only at a turn boundary and
    replays the plan in between -- correct when that agent is the one playing, and
    wrong when it is being asked for a label. DAgger queries states the *student*
    steered into, where a plan made for a different state is stale, and a stale plan
    that hits a masked action falls back to ``DRAW``.

    Measured, that is not a small effect: with the student driving, **76% of the
    labels came back ``DRAW``** against 26% with the teacher driving. Training on
    those teaches the student to draw and pass, which is what it learned.

    So: replan every state and take the first action. It costs a plan per step
    instead of one per turn -- about 2.7x more on the standard config -- and it is
    the difference between an oracle and an echo.
    """
    cfg = agent.cfg
    n = mask.shape[0]
    out = np.full(n, cfg.draw_action, dtype=np.int64)
    for env in range(n):
        plan = agent.plan(obs, env)
        if plan and mask[env, plan[0]]:
            out[env] = plan[0]
    return out


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

    needs_mask_per_action = False
    """A plan is decided at the turn boundary, and :meth:`act` reads the mask after
    that only to abandon one that has gone stale. So a caller driving this agent may
    skip rebuilding the mask mid-turn -- which is the most expensive thing in a step.
    A subclass that overrides :meth:`act` to *choose* from the mask must set this."""

    def __init__(self, cfg: RummiConfig) -> None:
        self.cfg = cfg
        self._plans: dict[int, list[int]] = {}

    def reset(self, n_envs: int) -> None:
        self._plans = {}

    def plan(self, obs: Observation, env: int) -> list[int]:
        raise NotImplementedError

    def plan_batch(self, obs: Observation, envs: np.ndarray) -> list[list[int]]:
        """A plan for each of ``envs``, whose turns all start on this step.

        One :meth:`plan` call each by default. An agent whose planning vectorises
        overrides this instead and gets the whole batch in one pass -- which is
        most of what a planner costs, since the work per turn is small and there
        is one per env.
        """
        return [self.plan(obs, int(env)) for env in envs]

    def act(
        self, obs: Observation, mask: np.ndarray, active: np.ndarray | None = None
    ) -> np.ndarray:
        cfg = self.cfg
        n = mask.shape[0]
        out = np.full(n, cfg.draw_action, dtype=np.int64)
        fresh = turn_starting(obs)
        if active is not None:
            fresh = fresh & np.asarray(active)
        starting = np.flatnonzero(fresh)
        if starting.size:
            planned = self.plan_batch(obs, starting)
            for index, plan in zip(starting.tolist(), planned, strict=True):
                self._plans[index] = plan

        for env in range(n):
            if active is not None and not active[env]:
                continue
            queued = self._plans.get(env)
            if not queued:
                continue
            action = queued.pop(0)
            # A plan that has gone stale is abandoned rather than forced through:
            # DRAW reverts the turn cleanly and is always legal.
            if not mask[env, action]:
                queued.clear()
                continue
            out[env] = action
        return out
