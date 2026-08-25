"""A vector env where one seat is the learner and the rest are bundled agents.

*Why a subclass and not a wrapper.* An opponent's whole turn has to run inside a
single ``step``, and :class:`~rummi.env.vector_env.RummiVectorEnv` re-deals a
finished episode on the *following* step. Driving it from outside would start a
fresh deal partway through a macro step and the learner would never see the
terminal observation. Owning the autoreset boundary is the entire difference, so
this inherits rather than wraps.

``RummiVectorEnv`` is left as it was -- seat-cycling self-play with no
data-dependent loop in ``step``. That design is not weakened here; this is the
class that opts into the loop on purpose.

*What a step is.* One learner micro-action, followed by however many actions the
other seats need to hand control back. So a step is still one decision, but the
observation the learner sees next is always its own turn again, and the reward
covers the opponents' replies rather than arriving on a step it could not act on.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from rummi.agents import build
from rummi.agents.base import Agent, act_by_seat
from rummi.env.observation import encode
from rummi.env.vector_env import RummiVectorEnv


class FixedOpponentEnv(RummiVectorEnv):
    """Seat ``learner_seat`` is yours; every other seat plays ``opponent``.

    ``opponent`` is any name from :data:`rummi.agents.REGISTRY`. Note that
    ``optimal`` costs a CP-SAT solve per opponent turn per env and cannot batch,
    which makes it an evaluation opponent rather than a training one.
    """

    def __init__(
        self,
        *args,
        opponent: str = "greedy",
        learner_seat: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if not self.wants_mask:
            raise ValueError("the opponents choose from the mask, so it cannot be turned off")
        if not 0 <= learner_seat < self.cfg.n_players:
            raise ValueError(
                f"learner_seat {learner_seat} is not a seat in a "
                f"{self.cfg.n_players}-player game"
            )
        self.opponent = opponent
        self.learner_seat = learner_seat
        # One instance per seat: two seats sharing a PlanningAgent would collide
        # on its per-env plan keys and consume each other's turns.
        self._seats: list[Agent | None] = [
            None if seat == learner_seat else build(opponent, self.cfg)
            for seat in range(self.cfg.n_players)
        ]
        self._illegal = 0

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        obs, info = super().reset(seed=seed, options=options)
        self._reset_seats()
        # A fresh deal starts at seat 0, so this only does work when the learner
        # sits elsewhere -- but it is what makes learner_seat mean anything.
        rewards_all = self._run_opponents()
        return encode(self.state), self._info(rewards_all)

    def step(self, actions):
        actions = self._check_actions(actions)
        just_reset = self._autoreset()
        if just_reset.any():
            self._reset_seats()

        result = self._advance(actions, active=~just_reset & ~self.state.done)
        rewards_all = result.rewards + self._run_opponents()

        # Read the flags off the state rather than off the first advance: the
        # learner's own move may end the game, and so may an opponent's reply.
        terminated = self.state.done & ~self.state.truncated
        truncated = self.state.truncated.copy()
        self._pending_reset = terminated | truncated

        rewards = rewards_all[:, self.learner_seat]
        return encode(self.state), rewards, terminated, truncated, self._info(rewards_all)

    # --- the opponents -------------------------------------------------------
    def _reset_seats(self) -> None:
        """Clear every opponent's per-env memory.

        Safe to do wholesale even though only some envs were re-dealt: an
        opponent's turn begins and ends inside one macro step, so none of them is
        ever holding a plan that has to survive this call.
        """
        for agent in self._seats:
            if agent is not None:
                agent.reset(self.num_envs)

    def _run_opponents(self) -> np.ndarray:
        """Play the other seats until every live env is back on the learner.

        Returns the ``(N, P)`` reward accumulated over the advance, which the
        caller adds to the learner's own -- a macro step pays out for everything
        that happened inside it.
        """
        cfg = self.cfg
        total = np.zeros((self.num_envs, cfg.n_players), np.float32)
        self._illegal = 0
        # An opponent turn is at most `max_micro_per_turn` actions plus the one
        # that commits it, and at most P-1 seats reply before the learner is up.
        budget = (cfg.n_players - 1) * (cfg.max_micro_per_turn + 2)

        for _ in range(budget):
            waiting = ~self.state.done & (self.state.current != self.learner_seat)
            if not waiting.any():
                return total
            actions, illegal = act_by_seat(self._seats, self.state, self.required_mask)
            self._illegal += illegal
            total += self._advance(actions, active=waiting).rewards

        raise RuntimeError(
            f"seat {self.learner_seat} did not get control back within {budget} "
            f"actions: {self.opponent!r} is not committing its turn. DRAW is never "
            "masked, so a well-behaved agent always ends one."
        )

    def _info(self, rewards_all: np.ndarray) -> dict[str, Any]:
        info = super()._info(rewards_all)
        # A bundled opponent never proposes one; a custom one might, and silently
        # substituting DRAW would look like a weak opponent rather than a bug.
        info["opponent_illegal"] = self._illegal
        return info
