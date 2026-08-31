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
    from gymnasium import spaces
    from gymnasium.vector import AutoresetMode, VectorEnv
    from gymnasium.vector.utils import batch_space
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError(
        "the Gymnasium env needs the optional extra: pip install 'rummi[env]'"
    ) from exc

from rummi.rules.config import STANDARD, RummiConfig
from rummi.env.api import get_backend
from rummi.env.observation import observation_space
from rummi.render.driver import RenderMode, Renderer


class RummiVectorEnv(VectorEnv):
    metadata = {  # noqa: RUF012 -- VectorEnv declares this as an instance variable
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
        action_mask: bool = True,
        backend: str = "numpy",
    ) -> None:
        super().__init__()
        if validate_actions and not action_mask:
            raise ValueError(
                "validate_actions needs the mask it validates against; pass "
                "validate_actions=False as well to assert your actions are legal"
            )
        self.backend = get_backend(backend)
        if backend != "numpy" and RenderMode(render_mode) is not RenderMode.NONE:
            raise ValueError(
                f"rendering reads a NumPy BatchState, so it cannot draw the "
                f"{self.backend.name} backend's state"
            )
        self.cfg = cfg
        self.num_envs = num_envs
        self.validate_actions = validate_actions
        self.wants_mask = action_mask

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
        # Re-deal seeds derive from (seed, step, env) rather than from a spawned
        # SeedSequence: that is the scheme every backend shares, so an autoresetting
        # run replays identically whichever one is driving it.
        self._steps = 0
        self.state = self.backend.reset(cfg, num_envs, seed=seed)
        self._mask: np.ndarray | None = np.zeros((num_envs, cfg.n_actions), dtype=bool)
        self._obs: dict[str, Any] | None = None
        # Whether `_mask` describes the state as it stands. Only a re-deal (or a
        # not-yet-reset env) makes it stale, so a step never has to recompute the
        # mask its predecessor already produced.
        self._mask_fresh = False
        # Envs that terminated on the previous step and must be re-dealt now.
        self._pending_reset = np.zeros(num_envs, dtype=bool)

    # --- Gymnasium API -------------------------------------------------------
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        if seed is not None:
            self._base_seed = seed
        self._steps = 0
        self.state = self.backend.reset(self.cfg, self.num_envs, seed=self._base_seed)
        self._pending_reset[:] = False
        self._observe()
        self._render()
        return self._encode(), self._info(
            np.zeros((self.num_envs, self.cfg.n_players), np.float32)
        )

    def step(self, actions):
        actions = self._check_actions(actions)
        just_reset = self._autoreset()
        result = self._advance(actions, active=~just_reset)

        rewards_all = result.rewards
        # NumPy, because it indexes the NumPy reward matrix: a device tensor here
        # would drag `rewards_all` back through `__array__` and fail on MPS.
        acting = (self.backend.to_numpy(self.state.current) - 1) % self.cfg.n_players
        rewards = rewards_all[np.arange(self.num_envs), acting]

        terminated = result.terminated.copy()
        truncated = result.truncated.copy()
        self._pending_reset = terminated | truncated
        return self._encode(), rewards, terminated, truncated, self._info(rewards_all)

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
            self.state = self.backend.reset_envs(
                self.cfg, self.state, np.flatnonzero(just_reset), self._base_seed, self._steps
            )
            self._pending_reset[:] = False
            self._mask_fresh = False
            self._obs = None
        return just_reset

    def _advance(self, actions: np.ndarray, active: np.ndarray):
        """Apply one micro-action to every ``active`` env.

        The mask an advance leaves behind is the mask its successor needs, so this
        recomputes one only after a re-deal. Doing it unconditionally costs a
        second :func:`legal_actions` per step, which measures **83% slower** on the
        standard config, for a value already in hand.
        """
        if self.wants_mask and not self._mask_fresh:
            self._mask = self.backend.legal_actions(self.cfg, self.state)
            self._mask_fresh = True
        self.state, result = self.backend.step(
            self.cfg,
            self.state,
            actions,
            self._mask if self.validate_actions else None,
            active,
        )
        self._steps += 1
        self._observe()
        self._render()
        return result

    def _observe(self) -> None:
        """Refresh the mask and the observation together.

        Both describe the state as it now stands and both are built on the same
        summary of the table, so a backend computing them in one call does that
        work once. The step that follows consumes the mask this leaves behind.
        """
        if self.wants_mask:
            self._mask, self._obs = self.backend.observe(self.cfg, self.state)
        else:
            self._mask, self._obs = None, self.backend.encode(self.cfg, self.state)
        self._mask_fresh = True

    def _encode(self) -> dict[str, Any]:
        """The observation in the backend's own array type.

        Whatever :meth:`_observe` last produced: it is the observation of the
        current state, and re-encoding would repeat the whole table summary.

        Not converted: a device backend that copied its observation to the host
        every step would give back the speedup it was chosen for. Gymnasium's
        ``wrappers.vector`` conversions are the boundary -- see the note in
        CLAUDE.md.
        """
        if self._obs is None:
            self._obs = self.backend.encode(self.cfg, self.state)
        return self._obs

    def _render(self) -> None:
        if self.render_mode != RenderMode.NONE.value:
            self._renderer.render(self.state, self._mask)

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
        """Telemetry is NumPy whatever the backend is.

        These are `(N,)` bookkeeping vectors a caller reads on the host anyway --
        which seat is up, who won -- and copying them costs about 1% even on a
        device backend. The observation is the field where staying native matters.
        """
        to_numpy = self.backend.to_numpy
        info: dict[str, Any] = {
            "current_player": to_numpy(self.state.current).copy(),
            "rewards_all": rewards_all,
            "micro_count": to_numpy(self.state.micro_count).copy(),
            "turn_count": to_numpy(self.state.turn_count).copy(),
            "winner": to_numpy(self.state.winner).copy(),
        }
        if self.wants_mask:
            info["action_mask"] = self._mask
        return info

    @property
    def action_mask(self) -> np.ndarray | None:
        return self._mask

    @property
    def required_mask(self) -> np.ndarray:
        """The mask, for the internals that cannot work without one.

        `action_mask` is the public, nullable view; this is the one that says so
        rather than failing later on a `None`.
        """
        if self._mask is None:
            raise RuntimeError("this env was built with action_mask=False")
        return self._mask
