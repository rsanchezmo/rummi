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
    from gymnasium.vector import AutoresetMode, VectorEnv
    from gymnasium.vector.utils import batch_space
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError(
        "the Gymnasium env needs the optional extra: pip install 'rummi[env]'"
    ) from exc

from rummi.rules.config import STANDARD, RummiConfig
from rummi.env.numpy.deal import reset as deal_reset
from rummi.env.numpy.deal import reset_envs
from rummi.env.numpy.engine import step as engine_step
from rummi.env.numpy.masks import legal_actions
from rummi.env.numpy.state import allocate
from rummi.env.observation import encode, observation_space
from rummi.render.driver import RenderMode, Renderer


class RummiVectorEnv(VectorEnv):
    metadata = {
        "render_modes": [m.value for m in RenderMode],
        "autoreset_mode": AutoresetMode.NEXT_STEP,
    }

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
        actions = self._check_actions(actions)
        just_reset = self._autoreset()
        result = self._advance(actions, active=~just_reset)

        rewards_all = result.rewards
        acting = (self.state.current - 1) % self.cfg.n_players
        rewards = rewards_all[np.arange(self.num_envs), acting]

        terminated = result.terminated.copy()
        truncated = result.truncated.copy()
        self._pending_reset = terminated | truncated
        return encode(self.state), rewards, terminated, truncated, self._info(rewards_all)

    # --- pieces a subclass drives differently --------------------------------
    def _check_actions(self, actions) -> np.ndarray:
        actions = np.asarray(actions, dtype=np.int64)
        if actions.shape != (self.num_envs,):
            raise ValueError(f"expected {(self.num_envs,)} actions, got {actions.shape}")
        return actions

    def _autoreset(self) -> np.ndarray:
        """Re-deal whatever finished last step; returns which envs those were.

        Next-step autoreset: the caller then holds those envs out of this step, so
        the action supplied alongside a fresh deal is discarded rather than played.
        """
        just_reset = self._pending_reset.copy()
        if just_reset.any():
            reset_envs(self.state, np.flatnonzero(just_reset), self._seeds.spawn(int(just_reset.sum())))
            self._pending_reset[:] = False
        return just_reset

    def _advance(self, actions: np.ndarray, active: np.ndarray, mask=None):
        """Apply one micro-action to every ``active`` env.

        ``mask`` short-circuits recomputing what the caller already holds: after
        any advance ``self._mask`` is the current mask, so a loop of advances pays
        for :func:`legal_actions` once per step rather than twice. Only a re-deal
        invalidates it, which is why :meth:`step` lets it default.
        """
        self._mask = legal_actions(self.state) if mask is None else mask
        result = engine_step(
            self.state,
            actions,
            self._mask if self.validate_actions else None,
            active=active,
        )
        self._mask = legal_actions(self.state)
        self._renderer.render(self.state, self._mask)
        return result

    def render(self):
        """Always draws, ignoring the throttle: an explicit call wants a frame.

        Returns a *tuple* of frames under ``rgb_array``, which is the Gymnasium
        vector convention and what ``gymnasium.wrappers.vector.RecordVideo``
        expects to concatenate. Only ``render_env_index`` is drawn -- rendering
        all ``num_envs`` games would cost N times as much to show N games of the
        same thing -- so the tuple holds one frame regardless of batch size.
        """
        frame = self._renderer.frame(self.state, self._mask)
        if self.render_mode == RenderMode.RGB_ARRAY.value and frame is not None:
            return (frame,)
        return frame

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
