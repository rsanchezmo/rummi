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
    activation: str = "relu"
    """`relu` or `tanh`. `trunk_gain` is matched to this: `sqrt(2)` is ReLU's gain,
    and carrying it into `tanh` drives a deep trunk towards saturation."""
    head: str = "flat"
    """`flat` or `bilinear`.

    Flat is 2400 independent logits. But `ASSIGN(kind, slot)` and `PICK(slot, pos)`
    are 96% of the action space and both are row-major over two indices, so a flat
    head has to learn "kind k fits slot s" as 1855 unrelated weights. `bilinear`
    scores each pair as a dot product of a kind representation and a slot
    representation, so what it learns about one pair transfers to the others.
    """
    factored: bool = False
    """Normalise per action family instead of over all 2400 actions at once.

    Flat, `END_TURN` is one logit against the 2,310 of `ASSIGN` and `PICK`, so a
    uniform epsilon across those blocks outweighs it by 2,310x and trunk noise
    off-distribution drowns the one action that commits a turn. Factored, the
    policy picks a family and then an argument within it, so `END_TURN` competes
    six ways and noise inside `ASSIGN` redistributes inside `ASSIGN`.

    It reassembles to 2400 logits -- `log p(family) + log p(arg | family)` -- so a
    global softmax over them is the factored distribution and nothing downstream
    changes.
    """
    head_dim: int = 16
    """Width of the `bilinear` head's per-index representations, and so the rank of
    the `(K, S)` score matrix it factors: it bounds how many independent ways a
    kind can relate to a slot. Unused by the flat head."""

    def layer_sizes(self, cfg: RummiConfig) -> list[tuple[int, int]]:
        dims = [feature_dim(cfg), *self.hidden]
        return list(pairwise(dims))

    @property
    def trunk_gain(self) -> float:
        # torch.nn.init.calculate_gain's values for the two we support.
        return float(np.sqrt(2.0)) if self.activation == "relu" else 5.0 / 3.0


def family_bounds(cfg: RummiConfig) -> tuple[int, ...]:
    """The seven boundaries of the six action families, in `ActionKind` order.

    Read off `cfg`'s own offsets rather than recomputing each block's width, so a
    family cannot drift from the layout `rules.actions` encodes against.
    """
    return (
        cfg.place_offset,
        cfg.pick_offset,
        cfg.dissolve_offset,
        cfg.assign_offset,
        cfg.end_turn_action,
        cfg.draw_action,
        cfg.n_actions,
    )


def family_sizes(cfg: RummiConfig) -> tuple[int, ...]:
    """Width of each family's block. `END_TURN` and `DRAW` are singletons."""
    return tuple(hi - lo for lo, hi in pairwise(family_bounds(cfg)))


def param_names(arch: Architecture) -> list[str]:
    names = [f"{p}{i}" for i in range(len(arch.hidden)) for p in ("w", "b")]
    if arch.head == "bilinear":
        head = [f"{p}_{n}" for n in ("kind", "slot", "pos", "flat") for p in ("w", "b")]
    else:
        head = ["w_pi", "b_pi"]
    if arch.factored:
        head += ["w_fam", "b_fam"]
    return [*names, *head, "w_v", "b_v"]


def init_params(
    cfg: RummiConfig, arch: Architecture | None = None, seed: int = 0
) -> dict[str, np.ndarray]:
    """Orthogonal init, PPO's usual gains.

    The gains are the part that matters: `arch.trunk_gain` through the trunk --
    matched to the activation, which is the bug an earlier version had -- **0.01 on
    the policy head** so the first policy is near-uniform over legal actions
    rather than confidently wrong, and 1.0 on the value head. A policy head at
    full gain starts out committed to arbitrary actions, which with 1.5% of the
    action space legal is a slow thing to unlearn.
    """
    arch = arch or Architecture()
    rng = np.random.default_rng(seed)
    params: dict[str, np.ndarray] = {}

    for i, (fan_in, fan_out) in enumerate(arch.layer_sizes(cfg)):
        params[f"w{i}"] = _orthogonal(rng, fan_in, fan_out, arch.trunk_gain)
        params[f"b{i}"] = np.zeros(fan_out, dtype=np.float32)

    last = arch.hidden[-1]
    if arch.head == "bilinear":
        d = arch.head_dim
        # Small gains for the same reason the flat head uses 0.01: a bilinear score
        # is a product of two of these, so each wants to start smaller still.
        for name, rows in (
            ("kind", cfg.n_kinds), ("slot", cfg.max_sets), ("pos", cfg.max_set_len)
        ):
            params[f"w_{name}"] = _orthogonal(rng, last, rows * d, 0.1)
            params[f"b_{name}"] = np.zeros(rows * d, dtype=np.float32)
        # PLACE, DISSOLVE, END_TURN and DRAW stay flat -- 4% of the space, and
        # neither PLACE(kind) nor DISSOLVE(slot) has a second index to factor over.
        flat = cfg.n_kinds + cfg.max_sets + 2
        params["w_flat"] = _orthogonal(rng, last, flat, 0.01)
        params["b_flat"] = np.zeros(flat, dtype=np.float32)
    else:
        params["w_pi"] = _orthogonal(rng, last, cfg.n_actions, 0.01)
        params["b_pi"] = np.zeros(cfg.n_actions, dtype=np.float32)
    if arch.factored:
        params["w_fam"] = _orthogonal(rng, last, len(family_sizes(cfg)), 0.01)
        # Zero, so the first family distribution is uniform over the families that
        # hold a legal action and `p(END_TURN)` starts at ~1/5 against the flat
        # head's 1/n_legal.
        #
        # Biasing by `log(width)` instead -- to start where the flat head starts --
        # does the opposite: it hands `ASSIGN` mass for all 1855 of its ids when a
        # handful are legal, measured p(END_TURN) 0.0018 against the flat head's
        # 0.059 on `standard`. Only the *legal* width would be the right weight and
        # that is state-dependent, so no bias can encode it.
        params["b_fam"] = np.zeros(len(family_sizes(cfg)), dtype=np.float32)
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
