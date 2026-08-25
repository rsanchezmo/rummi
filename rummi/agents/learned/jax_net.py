"""JAX implementation of the reference policy.

Functional, so the parameters are an argument rather than state: `apply(params,
obs, mask)`. That is what lets the whole forward pass sit inside one `jit`, and
what an optax update expects to be handed.

Built from the same NumPy dict as the torch version, so
`test_learned.py::test_the_two_frameworks_agree_on_logits` is a real check rather
than two initialisers that happen to look similar.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from rummi.rules.config import RummiConfig
from rummi.agents.learned.architecture import Architecture, init_params
from rummi.agents.learned.features import FEATURE_FIELDS, feature_dim, feature_scale

MASKED = -1e8
"""Finite for the same reason as in the torch version: `0 * -inf` is NaN."""

Params = dict[str, jax.Array]


def init(
    cfg: RummiConfig, arch: Architecture | None = None, seed: int = 0
) -> Params:
    return {k: jnp.asarray(v) for k, v in init_params(cfg, arch, seed).items()}


def slot_counts(cfg: RummiConfig, table_sets: jax.Array) -> jax.Array:
    """`(B, S, K)` count of each kind per slot. See `features.slot_counts_numpy`.

    `EMPTY` is masked rather than clamped: clamping would place a phantom tile of
    kind 0 in every empty position.
    """
    valid = (table_sets >= 0).astype(jnp.float32)
    idx = jnp.clip(table_sets, 0, cfg.n_kinds - 1).astype(jnp.int32)
    return (jax.nn.one_hot(idx, cfg.n_kinds) * valid[..., None]).sum(-2)


def features(cfg: RummiConfig, obs: dict[str, jax.Array]) -> jax.Array:
    batch = obs["rack"].shape[0]
    flat = jnp.concatenate(
        [jnp.asarray(obs[f]).reshape(batch, -1) for f in FEATURE_FIELDS], axis=-1
    ).astype(jnp.float32)
    assert flat.shape[-1] == feature_dim(cfg), flat.shape
    # A NumPy constant, never a traced array: a table cached from inside a trace
    # would leak tracers into every later call.
    return flat / jnp.asarray(feature_scale(cfg))


@partial(jax.jit, static_argnums=(0, 1, 2))
def apply(
    cfg: RummiConfig,
    n_hidden: int,
    activation: str,
    params: Params,
    obs: dict[str, jax.Array],
    mask: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """`(masked_logits, value)`.

    `n_hidden` and `activation` are static and separate from `params`: the depth
    sets a Python loop bound, which a traced dict cannot report, and the activation
    picks the function being traced.
    """
    act = jax.nn.relu if activation == "relu" else jnp.tanh
    x = features(cfg, obs)
    for i in range(n_hidden):
        x = act(x @ params[f"w{i}"] + params[f"b{i}"])
    logits = x @ params["w_pi"] + params["b_pi"]
    value = (x @ params["w_v"] + params["b_v"]).squeeze(-1)
    return jnp.where(mask.astype(bool), logits, MASKED), value


def sample(logits: jax.Array, key: jax.Array) -> jax.Array:
    return jax.random.categorical(key, logits, axis=-1)


def log_prob(logits: jax.Array, actions: jax.Array) -> jax.Array:
    logp = jax.nn.log_softmax(logits, axis=-1)
    return jnp.take_along_axis(logp, actions[:, None].astype(jnp.int32), axis=-1)[:, 0]


def entropy(logits: jax.Array) -> jax.Array:
    logp = jax.nn.log_softmax(logits, axis=-1)
    return -(jnp.exp(logp) * logp).sum(-1)


def to_numpy(params: Params) -> dict[str, np.ndarray]:
    return {k: np.asarray(v) for k, v in params.items()}
