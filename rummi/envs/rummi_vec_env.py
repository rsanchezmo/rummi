"""Gymnasium vector environment over the batched simulator.

Design notes worth knowing before using this.

*One micro-action per env per step.* A player's turn spans several steps, so
``turn_count`` and ``step`` count different things. ``info["micro_count"]`` says
how far into the current turn each env is.

*Seat-cycling self-play.* The observation is always the acting seat's view and
``info["current_player"]`` says whose it is, so one shared policy plays every
seat and the batch stays in lockstep -- no data-dependent opponent loop inside
``step``.

*Reward shape.* Gymnasium's contract is one reward per env, so ``step`` returns
the reward credited to the seat that just acted. The full ``(num_envs, P)``
matrix is in ``info["rewards_all"]``, which is what a multi-agent trainer needs
for correct credit assignment across seats.

*Autoreset.* Follows the Gymnasium 1.x next-step convention: the step that
terminates an env returns that episode's final observation, and the *following*
step returns the reset observation with zero reward and no termination flags,
ignoring whatever action was passed for it.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    from gymnasium.vector import VectorEnv
    from gymnasium.vector.utils import batch_space
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError(
        "the Gymnasium env needs the optional extra: pip install 'rummi[env]'"
    ) from exc

from rummi.core.config import STANDARD, RummiConfig
from rummi.core.deal import reset as deal_reset
from rummi.core.deal import reset_envs
from rummi.core.engine import step as engine_step
from rummi.core.masks import legal_actions
from rummi.core.state import allocate
from rummi.envs.observation import encode, observation_space
from rummi.render.driver import RenderMode, Renderer


class RummiVectorEnv(VectorEnv):
    metadata = {"render_modes": [m.value for m in RenderMode], "autoreset_mode": "next-step"}

    def __init__(
        self,
        num_envs: int = 8,
        cfg: RummiConfig = STANDARD,
        seed: int = 0,
        render_mode: str | RenderMode = RenderMode.NONE,
        render_env_index: int = 0,
        render_fps: float | None = None,
        render_every: int = 1,
        validate_actions: bool = True,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_envs = num_envs
        self.validate_actions = validate_actions

        self.single_action_space = spaces.Discrete(cfg.n_actions)
        self.single_observation_space = observation_space(cfg)
        self.action_space = batch_space(self.single_action_space, num_envs)
        self.observation_space = batch_space(self.single_observation_space, num_envs)

        self.render_mode = RenderMode(render_mode).value
        self._renderer = Renderer(
            cfg,
            self.render_mode,
            env_index=render_env_index,
            fps=render_fps,
            every=render_every,
        )

        self._base_seed = seed
        self._seeds = np.random.SeedSequence(seed)
        self.state = allocate(cfg, num_envs)
        self._mask = np.zeros((num_envs, cfg.n_actions), dtype=bool)
        # Envs that terminated on the previous step and must be re-dealt now.
        self._pending_reset = np.zeros(num_envs, dtype=bool)

    # --- Gymnasium API -------------------------------------------------------
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        if seed is not None:
            self._base_seed = seed
            self._seeds = np.random.SeedSequence(seed)
        self.state = deal_reset(self.cfg, self.num_envs, seed=self._base_seed)
        self._pending_reset[:] = False
        self._mask = legal_actions(self.state)
        self._renderer.render(self.state, self._mask)
        return encode(self.state), self._info(np.zeros((self.num_envs, self.cfg.n_players), np.float32))

    def step(self, actions):
        actions = np.asarray(actions, dtype=np.int64)
        if actions.shape != (self.num_envs,):
            raise ValueError(f"expected {(self.num_envs,)} actions, got {actions.shape}")

        # Next-step autoreset: re-deal whatever finished last step, then hold
        # those envs out of this step so the action supplied for them is
        # discarded rather than played on the fresh episode.
        just_reset = self._pending_reset.copy()
        if just_reset.any():
            reset_envs(self.state, np.flatnonzero(just_reset), self._seeds.spawn(int(just_reset.sum())))
            self._pending_reset[:] = False

        self._mask = legal_actions(self.state)
        result = engine_step(
            self.state,
            actions,
            self._mask if self.validate_actions else None,
            active=~just_reset,
        )

        rewards_all = result.rewards
        acting = (self.state.current - 1) % self.cfg.n_players
        rewards = rewards_all[np.arange(self.num_envs), acting]

        terminated = result.terminated.copy()
        truncated = result.truncated.copy()
        self._pending_reset = terminated | truncated

        self._mask = legal_actions(self.state)
        self._renderer.render(self.state, self._mask)
        return encode(self.state), rewards, terminated, truncated, self._info(rewards_all)

    def render(self):
        """Always draws, ignoring the throttle: an explicit call wants a frame."""
        return self._renderer.frame(self.state, self._mask)

    def close(self, **kwargs) -> None:
        self._renderer.close()

    # --- extras --------------------------------------------------------------
    def _info(self, rewards_all: np.ndarray) -> dict[str, Any]:
        return {
            "action_mask": self._mask,
            "current_player": self.state.current.copy(),
            "rewards_all": rewards_all,
            "micro_count": self.state.micro_count.copy(),
            "turn_count": self.state.turn_count.copy(),
            "winner": self.state.winner.copy(),
        }

    @property
    def action_mask(self) -> np.ndarray:
        return self._mask
