"""Two choices per turn: play the turn a planner found, or hold and draw.

This exists to test whether **cross-turn strategy exists** in this game. Every
bundled agent maximises the turn in front of it, and the table is shared -- tiles
you place become material any opponent may rearrange -- so always playing the best
available turn gives structure away for free. If holding a playable turn ever
beats playing it, that gap is strategy no single-turn agent can express.

The action space is **two**, and all the combinatorial work is delegated to
whichever planner is passed in. That is what makes a whole-turn action space
tractable here: deciding which turns are *achievable* is the NP-hard partition
problem, so this never enumerates them -- it asks the planner for one turn and
decides whether to take it.

`decide` is handed the candidate turn, not only the observation. Judging "this
play exposes too much" needs to know what the play is, and with `optimal` inside,
the solve costs the same whichever way the decision goes.

Not in `REGISTRY`: the bundled ladder is `random` through `optimal`, and a rung
has to earn its place by beating one of them. `delegate` with `always_play` *is*
its inner agent, which is the sanity check, not a new rung.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable

import numpy as np

from rummi.agents.base import Agent, Observation, PlanningAgent
from rummi.rules.actions import decode_batch
from rummi.rules.config import RummiConfig


@dataclass(frozen=True, slots=True)
class PlanSummary:
    """What a candidate turn would do, for a policy to judge it by."""

    tiles: int
    """Tiles leaving the rack. The turn's immediate gain, and what it hands over."""
    dissolves: int
    """Sets broken open: how much of the shared table the turn rearranges."""
    length: int
    """Micro-actions the turn costs."""


def summarise(cfg: RummiConfig, plan: list[int]) -> PlanSummary:
    """Read the plan through `rules.actions` rather than re-deriving the offsets."""
    d = decode_batch(cfg, np.asarray(plan, dtype=np.int64))
    return PlanSummary(
        tiles=int(d.is_place.sum()), dissolves=int(d.is_dissolve.sum()), length=len(plan)
    )


Decide = Callable[[Observation, int, PlanSummary], bool]
"""`(obs, env, summary) -> play it?`"""


def always_play(obs: Observation, env: int, summary: PlanSummary) -> bool:
    """The baseline: reproduces the inner agent exactly, so it is what to beat."""
    return True


def tiles_at_least(k: int) -> Decide:
    """Hold any turn that sheds fewer than `k` tiles.

    A hand-written stand-in for a learned policy, and the cheapest probe of the
    whole question: a valid turn always plays at least one tile from the rack, so
    `k=1` is `always_play`. If any `k > 1` scores better, holding pays.
    """

    def decide(obs: Observation, env: int, summary: PlanSummary) -> bool:
        return summary.tiles >= k

    return decide


class DelegatingAgent(PlanningAgent):
    name = "delegate"

    def __init__(
        self, cfg: RummiConfig, inner: str | Agent = "optimal", decide: Decide | None = None,
        **inner_kwargs,
    ) -> None:
        super().__init__(cfg)
        if isinstance(inner, str):
            from rummi.agents import build  # deferred: `agents` imports this module

            inner = build(inner, cfg, **inner_kwargs)
        self.inner = inner
        self.decide = decide if decide is not None else always_play
        self.played = 0
        self.held = 0

    def reset(self, n_envs: int) -> None:
        super().reset(n_envs)
        self.inner.reset(n_envs)
        # `played`/`held` deliberately survive: they are run diagnostics, not the
        # per-env memory the protocol requires `reset` to clear, and a suite resets
        # once per batch -- zeroing them here would report only the last batch.

    def plan(self, obs: Observation, env: int) -> list[int]:
        candidate = self.inner.plan(obs, env)  # type: ignore[attr-defined]
        if not candidate:
            # Nothing playable, so there is no decision to make: an empty plan is
            # already DRAW.
            return []
        if self.decide(obs, env, summarise(self.cfg, candidate)):
            self.played += 1
            return candidate
        self.held += 1
        return []
