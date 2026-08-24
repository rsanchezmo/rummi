"""One uniform surface over the three backends.

The implementations are deliberately *not* written against a shared abstraction:
each is idiomatic for its framework, so the benchmark compares implementations
rather than the cost of a common layer. That leaves three genuinely different
call signatures, forced by the frameworks themselves:

* ``cfg`` rides on the state in NumPy and torch, but must be a static argument in
  JAX or ``jit`` cannot specialise the shapes.
* NumPy and torch mutate the state and return only the step result; JAX is
  functional and returns a new state alongside it.
* NumPy and torch validate the chosen action inline; JAX cannot, because reading
  a boolean off the device would break the trace.

This module reconciles those at the *boundary*, where one extra Python call per
step is immaterial. It normalises on the functional form -- ``step`` always
returns ``(state, result)`` -- because that is the superset: a mutating backend
simply hands back the same object.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from rummi.rules.config import RummiConfig


@dataclass(frozen=True, slots=True)
class StepOut:
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray


class Backend(Protocol):
    """Uniform driver for one implementation of the simulator."""

    name: str
    supports_inline_validation: bool

    def reset(self, cfg: RummiConfig, batch_size: int, seed: int = 0) -> Any: ...

    def reset_envs(self, cfg: RummiConfig, state: Any, envs, base_seed: int, step_index: int) -> Any:
        """Re-deal ``envs``, seeding from ``(base_seed, step_index, env)`` so a
        recorded trajectory replays identically on any backend."""

    def legal_actions(self, cfg: RummiConfig, state: Any) -> Any: ...

    def step(
        self, cfg: RummiConfig, state: Any, actions, mask=None, active=None
    ) -> tuple[Any, StepOut]: ...

    def digest(self, state: Any) -> str: ...

    def to_numpy(self, value) -> np.ndarray: ...


class NumpyBackend:
    name = "numpy"
    supports_inline_validation = True

    def reset(self, cfg, batch_size, seed=0):
        from rummi.env.numpy.deal import reset

        return reset(cfg, batch_size, seed=seed)

    def reset_envs(self, cfg, state, envs, base_seed, step_index):
        from rummi.env.numpy.deal import derived_seeds, reset_envs

        envs = np.asarray(envs)
        reset_envs(state, envs, derived_seeds(base_seed, step_index, envs))
        return state

    def legal_actions(self, cfg, state):
        from rummi.env.numpy.masks import legal_actions

        return legal_actions(state)

    def step(self, cfg, state, actions, mask=None, active=None):
        from rummi.env.numpy.engine import step

        out = step(state, np.asarray(actions), mask, active)
        return state, StepOut(out.rewards, out.terminated, out.truncated)

    def digest(self, state):
        return state.digest()

    def to_numpy(self, value):
        return np.asarray(value)


class TorchBackend:
    name = "torch"
    supports_inline_validation = True

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self.name = f"torch-{device}"

    def _dev(self):
        import torch

        return torch.device(self.device)

    def reset(self, cfg, batch_size, seed=0):
        from rummi.env.torch import sim

        return sim.reset(cfg, batch_size, seed=seed, device=self._dev())

    def reset_envs(self, cfg, state, envs, base_seed, step_index):
        import torch

        from rummi.env.torch import sim

        envs = list(np.asarray(envs).tolist())
        sim.reset_envs(
            state,
            torch.as_tensor(envs, device=self._dev()),
            sim.derived_deck_orders(cfg, base_seed, step_index, envs, self._dev()),
        )
        return state

    def legal_actions(self, cfg, state):
        from rummi.env.torch import sim

        return sim.legal_actions(state)

    def step(self, cfg, state, actions, mask=None, active=None):
        import torch

        from rummi.env.torch import sim

        out = sim.step(state, torch.as_tensor(np.asarray(actions), device=self._dev()), mask, active)
        return state, StepOut(
            self.to_numpy(out.rewards), self.to_numpy(out.terminated), self.to_numpy(out.truncated)
        )

    def digest(self, state):
        return state.digest()

    def to_numpy(self, value):
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)


class JaxBackend:
    name = "jax"
    supports_inline_validation = False
    """JAX validates host-side instead; see :meth:`check_actions`."""

    def reset(self, cfg, batch_size, seed=0):
        from rummi.env.jax import sim

        return sim.reset(cfg, batch_size, seed=seed)

    def reset_envs(self, cfg, state, envs, base_seed, step_index):
        import jax.numpy as jnp

        from rummi.env.jax import sim

        envs = list(np.asarray(envs).tolist())
        return sim.reset_envs(
            cfg, state, jnp.asarray(envs),
            sim.derived_deck_orders(cfg, base_seed, step_index, envs),
        )

    def legal_actions(self, cfg, state):
        from rummi.env.jax import sim

        return sim.legal_actions(cfg, state)

    def step(self, cfg, state, actions, mask=None, active=None):
        import jax.numpy as jnp

        from rummi.env.jax import sim

        if mask is not None:
            live = ~np.asarray(state.done) if active is None else np.asarray(active)
            sim.check_actions(mask, jnp.asarray(np.asarray(actions)), live)
        state, out = sim.step(cfg, state, jnp.asarray(np.asarray(actions)), active)
        return state, StepOut(
            self.to_numpy(out.rewards), self.to_numpy(out.terminated), self.to_numpy(out.truncated)
        )

    def digest(self, state):
        from rummi.env.jax import sim

        return sim.digest(state)

    def to_numpy(self, value):
        return np.asarray(value)


def available() -> list[str]:
    """Backend names that can actually be constructed on this machine."""
    names = ["numpy"]
    try:
        import torch

        names.append("torch")
        if torch.backends.mps.is_available():
            names.append("torch-mps")
        if torch.cuda.is_available():
            names.append("torch-cuda")
    except ModuleNotFoundError:
        pass
    try:
        import jax  # noqa: F401

        names.append("jax")
    except ModuleNotFoundError:
        pass
    return names


def get_backend(name: str = "numpy") -> Backend:
    if name == "numpy":
        return NumpyBackend()
    if name == "jax":
        return JaxBackend()
    if name == "torch":
        return TorchBackend("cpu")
    if name.startswith("torch-"):
        return TorchBackend(name.removeprefix("torch-"))
    raise ValueError(f"unknown backend {name!r}; available: {available()}")
