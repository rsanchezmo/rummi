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

    def encode(self, cfg: RummiConfig, state: Any) -> dict[str, Any]:
        """The observation of SPEC.md section 8, in the backend's own array type.

        Not converted to NumPy: a device backend that copied its observation to
        the host every step would give back the speedup it was chosen for.
        Gymnasium's ``wrappers.vector`` conversions are the boundary.
        """

    def observe(self, cfg: RummiConfig, state: Any) -> tuple[Any, dict[str, Any]]:
        """The mask and the observation of one state, in a single call.

        Every step needs both, and both are built on the same per-slot summary of
        the table -- the most expensive thing in either. Asking for them together
        is what lets a backend compute that once.
        """

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

    def encode(self, cfg, state):
        from rummi.env.observation import encode

        return encode(state)

    def observe(self, cfg, state):
        from rummi.env.numpy.masks import legal_actions
        from rummi.env.numpy.sets import summarize
        from rummi.env.observation import encode

        summary = summarize(cfg, state.table_sets)
        return legal_actions(state, summary), encode(state, summary)

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

    def __init__(self, device: str = "cpu", compiled: bool = False) -> None:
        self.device = device
        self.compiled = compiled
        self.name = f"torch-{device}" + ("+compile" if compiled else "")
        # Compiled, validation moves out of the step; see `step` below.
        self.supports_inline_validation = not compiled
        self._graphs: dict[str, Any] = {}

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

    def _graph(self, name: str, fn):
        """``fn``, compiled once and kept.

        ``torch.compile`` caches per callable, so a wrapper made fresh each step
        would compile fresh each step. ``dynamic=False`` because the batch size of
        a vector env never changes, and specialising on it is the point.
        """
        if not self.compiled:
            return fn
        hit = self._graphs.get(name)
        if hit is None:
            import torch

            hit = self._graphs[name] = torch.compile(fn, dynamic=False)
        return hit

    def legal_actions(self, cfg, state):
        from rummi.env.torch import sim
        from rummi.env.torch.observation import observe

        if self.compiled:
            # Through `observe`, discarding the observation. Only a re-deal asks for
            # a bare mask, and compiling a second graph for one call in a hundred
            # costs more than the encode it saves -- inside a benchmark's timed
            # region, it would also be the only thing measured.
            return self._graph("observe", observe)(state)[0]
        return sim.legal_actions(state)

    def encode(self, cfg, state):
        from rummi.env.torch.observation import encode

        return self._graph("encode", encode)(state)

    def observe(self, cfg, state):
        from rummi.env.torch.observation import observe

        return self._graph("observe", observe)(state)

    def step(self, cfg, state, actions, mask=None, active=None):
        import torch

        from rummi.env.torch import sim

        dev = self._dev()
        actions = torch.as_tensor(np.asarray(actions), device=dev)
        # `active` needs converting as much as `actions` does: a NumPy mask reaching
        # the torch sim makes `state.done |= ...` a byte-into-bool cast and raises.
        if active is not None:
            active = torch.as_tensor(np.asarray(active), device=dev, dtype=torch.bool)
        if self.compiled and mask is not None:
            # Validation branches on a device boolean, which would split the graph
            # around it. Run it here instead, the way the JAX backend does.
            sim.check_actions(state, actions, mask, active)
            mask = None
        out = self._graph("step", sim.step)(state, actions, mask, active)
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

    def encode(self, cfg, state):
        from rummi.env.jax.observation import encode

        return encode(cfg, state)

    def observe(self, cfg, state):
        from rummi.env.jax.observation import observe

        return observe(cfg, state)

    def step(self, cfg, state, actions, mask=None, active=None):
        import jax.numpy as jnp

        from rummi.env.jax import sim

        actions = jnp.asarray(np.asarray(actions))
        if mask is not None:
            # The same predicate the other two backends validate under: `done` and
            # `active` are independent, and an env the step will ignore for either
            # reason must not be judged on the action passed alongside it.
            live = ~np.asarray(state.done)
            if active is not None:
                live = live & np.asarray(active, dtype=bool)
            sim.check_actions(mask, actions, live)
        if active is not None:
            active = jnp.asarray(np.asarray(active, dtype=bool))
        state, out = sim.step(cfg, state, actions, active)
        return state, StepOut(
            self.to_numpy(out.rewards), self.to_numpy(out.terminated), self.to_numpy(out.truncated)
        )

    def digest(self, state):
        from rummi.env.jax import sim

        return sim.digest(state)

    def to_numpy(self, value):
        return np.asarray(value)


def available() -> list[str]:
    """Backend names that can actually be constructed on this machine.

    The ``+compile`` torch variants are constructible too but are deliberately not
    listed: compiling costs seconds per graph, and this is the list that
    benchmarks and the conformance tests sweep by default.
    """
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
    """Build a backend by name.

    A torch name may carry a ``+compile`` suffix -- ``torch+compile``,
    ``torch-mps+compile`` -- which puts the mask, the observation and the step
    through ``torch.compile``.
    """
    if name == "numpy":
        return NumpyBackend()
    if name == "jax":
        return JaxBackend()
    base, _, suffix = name.partition("+")
    if suffix not in ("", "compile"):
        raise ValueError(f"unknown backend suffix {suffix!r} in {name!r}; only '+compile' exists")
    if base == "torch":
        return TorchBackend("cpu", compiled=bool(suffix))
    if base.startswith("torch-"):
        return TorchBackend(base.removeprefix("torch-"), compiled=bool(suffix))
    raise ValueError(f"unknown backend {name!r}; available: {available()}")
