"""The bridge from a network to :class:`rummi.agents.base.Agent`.

A learned policy is the easy case for the agent contract. `PlanningAgent` needs
careful `active` bookkeeping because it caches a plan per env and would otherwise
consume the plan of an env someone else is playing; a network holds nothing
between calls, so it can score the whole batch and let the caller use the rows it
owns.

Deterministic by default. A score has to be reproducible, and `evaluate` pins
every seed precisely so two people get the same number -- a sampling policy would
put that back in play.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from rummi.rules.config import RummiConfig
from rummi.agents.base import Observation

Chooser = Callable[[Observation, np.ndarray], np.ndarray]
"""`(obs, mask) -> (n_envs,)` actions. Every returned action must be legal."""


class LearnedAgent:
    """Framework-agnostic: it only ever calls the chooser it was handed."""

    def __init__(self, choose: Chooser, name: str = "learned") -> None:
        self.choose = choose
        self.name = name

    def reset(self, n_envs: int) -> None:
        """Nothing to clear -- the policy carries no per-env state."""

    def act(
        self, obs: Observation, mask: np.ndarray, active: np.ndarray | None = None
    ) -> np.ndarray:
        return np.asarray(self.choose(obs, mask), dtype=np.int64)


def torch_agent(net, name: str = "learned-torch", sample: bool = False, seed: int = 0):
    """`net` is a :class:`~rummi.agents.learned.torch_net.TorchPolicy`."""
    import torch

    from rummi.agents.learned import torch_net

    generator = torch.Generator().manual_seed(seed)

    def choose(obs: Observation, mask: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            logits, _ = net(
                {k: torch.as_tensor(np.asarray(v)) for k, v in obs.items()},
                torch.as_tensor(np.asarray(mask)),
            )
            picked = (
                torch_net.sample(logits, generator) if sample else torch.argmax(logits, dim=-1)
            )
        return picked.cpu().numpy()

    return LearnedAgent(choose, name=name)


def jax_agent(
    cfg: RummiConfig,
    params,
    arch=None,
    name: str = "learned-jax",
    sample: bool = False,
    seed: int = 0,
):
    import jax
    import jax.numpy as jnp

    from rummi.agents.learned.architecture import Architecture
    from rummi.agents.learned import jax_net

    arch = arch or Architecture()
    key = jax.random.PRNGKey(seed)

    def choose(obs: Observation, mask: np.ndarray) -> np.ndarray:
        nonlocal key
        logits, _ = jax_net.apply(
            cfg,
            arch,
            params,
            {k: jnp.asarray(np.asarray(v)) for k, v in obs.items()},
            jnp.asarray(np.asarray(mask)),
        )
        if not sample:
            return np.asarray(jnp.argmax(logits, axis=-1))
        key, sub = jax.random.split(key)
        return np.asarray(jax_net.sample(logits, sub))

    return LearnedAgent(choose, name=name)
