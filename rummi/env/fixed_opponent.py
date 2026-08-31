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

from collections.abc import Sequence
from typing import Any

import numpy as np

from rummi.agents import build
from rummi.agents.base import Agent, Observation, act_by_seat
from rummi.env.vector_env import RummiVectorEnv

Opponent = str | Agent
"""A name from :data:`rummi.agents.REGISTRY`, or an agent already built -- a
learned one, say, whose weights the caller refreshes between training updates."""


class _Pool:
    """One seat, played by a different agent depending on which env it is.

    :func:`~rummi.agents.base.act_by_seat` dispatches per *seat*, so a batch with
    mixed opponents needs the per-env split to happen inside a seat: each member is
    called with ``active`` narrowed to the envs assigned to it, which the ``Agent``
    protocol already requires it to honour. A member with no env of its own this
    step is not called at all, so a pool holding ``optimal`` pays for its solves
    only on its own share of the batch.
    """

    name = "pool"

    def __init__(
        self, members: Sequence[Agent], assignment: np.ndarray, draw_action: int
    ) -> None:
        self.members = tuple(members)
        self.needs_mask_per_action = any(
            getattr(member, "needs_mask_per_action", True) for member in members
        )
        self.assignment = assignment
        self.draw_action = draw_action

    def reset(self, n_envs: int) -> None:
        for member in self.members:
            member.reset(n_envs)

    def act(
        self, obs: Observation, mask: np.ndarray, active: np.ndarray | None = None
    ) -> np.ndarray:
        out = np.full(mask.shape[0], self.draw_action, dtype=np.int64)
        for index, member in enumerate(self.members):
            mine = self.assignment == index
            if active is not None:
                mine = mine & active
            if not mine.any():
                continue
            out = np.where(mine, np.asarray(member.act(obs, mask, mine)), out)
        return out


def _as_pool(opponent: Opponent | Sequence[Opponent]) -> tuple[Opponent, ...]:
    """The opponents as a tuple, whether one was passed or several."""
    if isinstance(opponent, str) or not isinstance(opponent, Sequence):
        return (opponent,)
    if not opponent:
        raise ValueError("the opponent pool is empty; pass a name, an agent, or a list of them")
    return tuple(opponent)


class FixedOpponentEnv(RummiVectorEnv):
    """Seat ``learner_seat`` is yours; every other seat plays ``opponent``.

    ``opponent`` is a name from :data:`rummi.agents.REGISTRY`, an already-built
    :class:`~rummi.agents.base.Agent`, or a sequence mixing the two -- a pool, from
    which each env draws one opponent by round-robin over its index. That is an
    even, fixed split: a member's share of the batch does not drift, and an env
    slot's opponent is the same across re-deals, so a metric read per env index
    means something. Nothing is sampled, so nothing here needs a seed of its own.

    Note that ``optimal`` costs a CP-SAT solve per opponent turn per env and cannot
    batch, which makes it an evaluation opponent rather than a training one.

    *What the opponents cost.* A mask is the most expensive thing in a step, and an
    opponent that plans whole turns cannot learn anything from one built mid-plan --
    it decided at the turn boundary and is replaying. So when every seat says so
    (:attr:`~rummi.agents.base.PlanningAgent.needs_mask_per_action`), the mask is
    rebuilt only where a seat takes over, and the actions between are taken on
    trust: nothing checks them, and ``opponent_illegal`` counts only what a current
    mask covers. Every bundled planner is exact mid-turn -- nobody else touches the
    table during a turn -- and `test_the_opponents_play_the_same_games_either_way`
    holds the two paths to identical games -- for `MacroAgent` too, which is the
    harder case: it decides again mid-turn, every time an expansion runs out, and
    may still skip the mask because `can_end_turn` reads the one bit it wanted out
    of the observation. An opponent that *chooses* from the mask, such as the hybrid
    one or a pool holding it, keeps a mask before each action.
    """

    def __init__(
        self,
        *args,
        opponent: Opponent | Sequence[Opponent] = "greedy",
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
        self.opponent_pool = _as_pool(opponent)
        self._pool_index = np.arange(self.num_envs) % len(self.opponent_pool)
        self._seats: list[Agent | None] = [
            None if seat == learner_seat else self._seat_agent()
            for seat in range(self.cfg.n_players)
        ]
        # Rebuilding the mask before every opponent action is the most expensive
        # thing in the loop, and an opponent that plans whole turns cannot learn
        # anything from one built mid-plan. Each seat answers for itself; anything
        # that does not answer is taken to choose per action.
        self._mask_per_action = any(
            getattr(agent, "needs_mask_per_action", True)
            for agent in self._seats
            if agent is not None
        )
        self._illegal = 0

    @property
    def pool_index(self) -> np.ndarray:
        """``(N,)`` which pool member each env faces, by index into ``opponent``.

        Public because the split is only useful if a caller can read it: metrics
        pooled over a mixed batch cannot say whether the learner improved or its
        opponent got worse.
        """
        return self._pool_index

    def _seat_agent(self) -> Agent:
        """The agent for one seat, built fresh so no two seats share per-env memory.

        Two seats sharing a ``PlanningAgent`` would collide on its per-env plan keys
        and consume each other's turns, which is why a *name* is built again for
        every seat. An instance the caller handed in cannot be, so it plays every
        opponent seat: safe because per-env memory is keyed by env and re-planned at
        each turn boundary, and one env is only ever on one seat at a time.
        """
        members = [
            build(member, self.cfg) if isinstance(member, str) else member
            for member in self.opponent_pool
        ]
        if len(members) == 1:
            return members[0]
        return _Pool(members, self._pool_index, self.cfg.draw_action)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        obs, info = super().reset(seed=seed, options=options)
        self._reset_seats()
        # A fresh deal starts at seat 0, so this only does work when the learner
        # sits elsewhere -- but it is what makes learner_seat mean anything.
        rewards_all = self._run_opponents()
        return self._encode(), self._info(rewards_all)

    def step(self, actions):
        actions = self._check_actions(actions)
        just_reset = self._autoreset()
        if just_reset.any():
            self._reset_seats()

        live = ~just_reset & ~self.backend.to_numpy(self.state.done)
        result = self._advance(actions, active=live)
        rewards_all = result.rewards + self._run_opponents()

        # Read the flags off the state rather than off the first advance: the
        # learner's own move may end the game, and so may an opponent's reply.
        done, truncated = self.backend.to_numpy(self.state.done), self._host_truncated()
        terminated = done & ~truncated
        self._pending_reset = terminated | truncated

        rewards = rewards_all[:, self.learner_seat]
        return self._encode(), rewards, terminated, truncated, self._info(rewards_all)

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

        Where every seat plans, the mask is rebuilt only where a seat takes over --
        which is where a planner decides; the rest of a turn is the plan it already
        made, replayed. See the class docstring for what that gives up.
        """
        cfg = self.cfg
        total = np.zeros((self.num_envs, cfg.n_players), np.float32)
        self._illegal = 0
        # An opponent turn is at most `max_micro_per_turn` actions plus the one
        # that commits it, and at most P-1 seats reply before the learner is up.
        budget = (cfg.n_players - 1) * (cfg.max_micro_per_turn + 2)

        for _ in range(budget):
            to_numpy = self.backend.to_numpy
            current, done = to_numpy(self.state.current), to_numpy(self.state.done)
            waiting = ~done & (current != self.learner_seat)
            if not waiting.any():
                # The learner is up next and chooses from the mask, so it must be
                # the mask of this state.
                self._refresh_mask()
                return total
            deciding = self._mask_per_action or bool(
                (waiting & (to_numpy(self.state.micro_count) == 0)).any()
            )
            if deciding:
                self._refresh_mask()
                mask = self.required_mask
            else:
                # No seat decides this step -- every acting env is mid-plan -- so the
                # mask is only what a planner abandons a stale plan against. The one
                # in hand describes the state before the last action, where an ASSIGN
                # is still illegal because its PLACE had not happened yet: it would
                # abandon nearly every plan. Say "unchecked" rather than something
                # false, which is what a planning opponent asked for.
                mask = np.broadcast_to(np.True_, self.required_mask.shape)
            # The env encoded this state when the last advance left it; the
            # opponents read the same observation the learner would.
            actions, illegal = act_by_seat(
                self._seats, self.cfg, current, done, self._host(mask), self._host_obs()
            )
            # An illegal proposal is only visible against a mask of this state.
            self._illegal += illegal if deciding else 0
            total += self._advance(actions, active=waiting, mask=deciding).rewards

        raise RuntimeError(
            f"seat {self.learner_seat} did not get control back within {budget} "
            f"actions: {self.opponent!r} is not committing its turn. DRAW is never "
            "masked, so a well-behaved agent always ends one."
        )

    def _host(self, value):
        """``value`` as NumPy, which is what the opponents read.

        Free on the NumPy backend and on JAX, whose arrays are already host memory;
        a real copy on a device backend, which is the price of playing NumPy agents
        against it -- and small beside what the device saves on the step itself.
        """
        return self.backend.to_numpy(value)

    def _host_obs(self) -> Observation:
        return {key: self._host(value) for key, value in self._encode().items()}

    def _host_truncated(self) -> np.ndarray:
        return np.array(self.backend.to_numpy(self.state.truncated), copy=True)

    def _refresh_mask(self) -> None:
        """Bring the mask up to date if an advance was told to leave it behind."""
        if not self._mask_fresh:
            self._mask = self.backend.legal_actions(self.cfg, self.state)
            self._mask_fresh = True

    def _info(self, rewards_all: np.ndarray) -> dict[str, Any]:
        info = super()._info(rewards_all)
        # A bundled opponent never proposes one; a custom one might, and silently
        # substituting DRAW would look like a weak opponent rather than a bug.
        info["opponent_illegal"] = self._illegal
        return info
