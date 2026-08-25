"""The network's shape and its initial weights, as plain NumPy.

Both implementations are built *from this dict* rather than each initialising
itself, which makes cross-framework parity a thing you can test rather than hope
for: same `seed`, same weights, same logits.

The trunk is an MLP. That is a deliberate baseline, not an ambition -- see
`features.py` for what a set-aware encoder would do instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from rummi.rules.config import RummiConfig
from rummi.agents.learned.features import feature_dim

HIDDEN: tuple[int, ...] = (256, 256)


@dataclass(frozen=True, slots=True)
class Architecture:
    hidden: tuple[int, ...] = HIDDEN

    def layer_sizes(self, cfg: RummiConfig) -> list[tuple[int, int]]:
        dims = [feature_dim(cfg), *self.hidden]
        return list(pairwise(dims))


def param_names(arch: Architecture) -> list[str]:
    names = [f"{p}{i}" for i in range(len(arch.hidden)) for p in ("w", "b")]
    return [*names, "w_pi", "b_pi", "w_v", "b_v"]


def init_params(
    cfg: RummiConfig, arch: Architecture | None = None, seed: int = 0
) -> dict[str, np.ndarray]:
    """Orthogonal init, PPO's usual gains.

    The gains are the part that matters: `sqrt(2)` through the trunk, **0.01 on
    the policy head** so the first policy is near-uniform over legal actions
    rather than confidently wrong, and 1.0 on the value head. A policy head at
    full gain starts out committed to arbitrary actions, which with 1.5% of the
    action space legal is a slow thing to unlearn.
    """
    arch = arch or Architecture()
    rng = np.random.default_rng(seed)
    params: dict[str, np.ndarray] = {}

    for i, (fan_in, fan_out) in enumerate(arch.layer_sizes(cfg)):
        params[f"w{i}"] = _orthogonal(rng, fan_in, fan_out, np.sqrt(2.0))
        params[f"b{i}"] = np.zeros(fan_out, dtype=np.float32)

    last = arch.hidden[-1]
    params["w_pi"] = _orthogonal(rng, last, cfg.n_actions, 0.01)
    params["b_pi"] = np.zeros(cfg.n_actions, dtype=np.float32)
    params["w_v"] = _orthogonal(rng, last, 1, 1.0)
    params["b_v"] = np.zeros(1, dtype=np.float32)

    assert set(params) == set(param_names(arch))
    return params


def _orthogonal(rng: np.random.Generator, fan_in: int, fan_out: int, gain: float) -> np.ndarray:
    """`(fan_in, fan_out)` orthogonal matrix, scaled by `gain`.

    The sign correction on the diagonal of R is what makes this deterministic:
    without it `qr` may return either of two valid factorisations and the same
    seed would not give the same weights on every LAPACK.
    """
    a = rng.standard_normal((max(fan_in, fan_out), min(fan_in, fan_out)))
    q, r = np.linalg.qr(a)
    q = q * np.sign(np.diag(r))
    if fan_in < fan_out:
        q = q.T
    return (gain * q[:fan_in, :fan_out]).astype(np.float32)
