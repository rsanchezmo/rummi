"""The reference learned agent: feature layout, masking, and framework parity.

The parity test is the one worth having. Two implementations of a network are
easy to write and hard to tell apart by eye, so both are built from the same
NumPy parameter dict and required to produce the same logits -- the same trick
`tests/test_backends.py` plays on the simulator.
"""

import numpy as np
import pytest

from rummi.agents.base import Agent
from rummi.agents.learned.architecture import Architecture, init_params, param_names
from rummi.agents.learned.features import (
    FEATURE_FIELDS,
    feature_dim,
    feature_scale,
    slot_counts_numpy,
)
from rummi.env.numpy.deal import reset
from rummi.env.numpy.engine import step
from rummi.env.numpy.masks import legal_actions
from rummi.env.observation import encode
from rummi.rules.config import STANDARD, STANDARD_4P, TINY_GROUPS, RummiConfig

CONFIGS = [TINY_GROUPS, STANDARD, STANDARD_4P]
SMALL = Architecture(hidden=(32, 32))


def played(cfg: RummiConfig, batch: int = 6, steps: int = 30, seed: int = 0):
    """A state with a table and a workbench: a fresh deal leaves most features zero."""
    from rummi.agents import build
    from rummi.agents.base import act_on_state

    state = reset(cfg, batch, seed=seed)
    agent = build("greedy", cfg)
    for _ in range(steps):
        mask = legal_actions(state)
        step(state, act_on_state(agent, state, mask), mask)
    return state, encode(state), legal_actions(state)


# --- the feature layout ------------------------------------------------------
def flat_features(cfg: RummiConfig, obs) -> np.ndarray:
    """The vector both networks build. `slot_counts` is not in it -- see the
    `features` module docstring for the measurement that decided that."""
    batch = obs["rack"].shape[0]
    return np.concatenate([obs[f].reshape(batch, -1) for f in FEATURE_FIELDS], -1)


@pytest.mark.parametrize("cfg", CONFIGS, ids=lambda c: f"{c.n_kinds}k-{c.n_players}p")
def test_feature_dim_matches_what_the_fields_actually_flatten_to(cfg: RummiConfig):
    _, obs, _ = played(cfg)
    assert flat_features(cfg, obs).shape[-1] == feature_dim(cfg)
    assert feature_scale(cfg).shape == (feature_dim(cfg),)


@pytest.mark.parametrize("cfg", CONFIGS, ids=lambda c: f"{c.n_kinds}k-{c.n_players}p")
def test_slot_counts_account_for_the_table_exactly(cfg: RummiConfig):
    """Not in the baseline's features (measured: no gain), but kept correct and
    tested, because a factored action head will want per-slot representations."""
    state, obs, _ = played(cfg)
    counts = slot_counts_numpy(cfg, obs["table_sets"])
    np.testing.assert_array_equal(
        counts.sum(1).astype(np.int64), state.table_counts().astype(np.int64)
    )
    assert counts.sum() > 0, "the probe state has an empty table, so this proved nothing"
    # EMPTY padding must contribute nothing: clamping it into kind 0 instead of
    # masking would add S*L phantom tiles of the lowest kind.
    empty_slots = obs["table_sets"].max(-1) < 0
    assert counts[empty_slots].sum() == 0


@pytest.mark.parametrize("cfg", CONFIGS, ids=lambda c: f"{c.n_kinds}k-{c.n_players}p")
def test_scaling_lands_every_feature_in_a_sane_range(cfg: RummiConfig):
    """The scales exist so no column arrives orders of magnitude larger than its
    neighbours; a divisor that was wrong by a factor of the deck size would show
    up here rather than as a slow-training mystery."""
    _, obs, _ = played(cfg)
    scaled = flat_features(cfg, obs) / feature_scale(cfg)
    assert np.isfinite(scaled).all()
    assert scaled.max() <= 1.5, f"max feature {scaled.max():.2f}"
    assert scaled.min() >= -1.5, f"min feature {scaled.min():.2f}"


def test_the_same_seed_gives_the_same_weights():
    a = init_params(TINY_GROUPS, SMALL, seed=3)
    b = init_params(TINY_GROUPS, SMALL, seed=3)
    c = init_params(TINY_GROUPS, SMALL, seed=4)
    assert set(a) == set(param_names(SMALL))
    assert all(np.array_equal(a[k], b[k]) for k in a)
    assert not all(np.array_equal(a[k], c[k]) for k in a)


def test_the_policy_head_starts_small_and_the_trunk_does_not():
    """PPO's usual gains. A policy head at full gain starts out confidently wrong
    about which of ~1.5% legal actions to take, which is slow to unlearn."""
    p = init_params(STANDARD, SMALL, seed=0)
    assert p["w_pi"].std() < 0.01 * p["w0"].std()


# --- torch -------------------------------------------------------------------
@pytest.mark.parametrize("cfg", CONFIGS, ids=lambda c: f"{c.n_kinds}k-{c.n_players}p")
def test_torch_never_prefers_an_illegal_action(cfg: RummiConfig):
    torch = pytest.importorskip("torch")
    from rummi.agents.learned import torch_net

    _, obs, mask = played(cfg)
    net = torch_net.TorchPolicy(cfg, SMALL, init_params(cfg, SMALL, seed=1))
    logits, value = net({k: torch.as_tensor(v) for k, v in obs.items()}, torch.as_tensor(mask))

    assert logits.shape == (mask.shape[0], cfg.n_actions)
    assert value.shape == (mask.shape[0],)
    assert (logits.detach().numpy()[~mask] == torch_net.MASKED).all()
    rows = np.arange(mask.shape[0])
    assert mask[rows, logits.argmax(-1).numpy()].all(), "argmax chose an illegal action"

    g = torch.Generator().manual_seed(0)
    for _ in range(40):
        a = torch_net.sample(logits, g).numpy()
        assert mask[rows, a].all(), "sampling chose an illegal action"


def test_entropy_and_log_prob_stay_finite_with_almost_everything_masked():
    """`MASKED` is finite rather than `-inf` precisely so this holds: `0 * -inf`
    in the entropy term would be NaN, and a single NaN ends a training run."""
    torch = pytest.importorskip("torch")
    from rummi.agents.learned import torch_net

    cfg = STANDARD
    _, obs, _ = played(cfg)
    n = obs["rack"].shape[0]
    # The pathological case: only DRAW legal, which the mask guarantees is possible.
    mask = np.zeros((n, cfg.n_actions), dtype=bool)
    mask[:, cfg.draw_action] = True

    net = torch_net.TorchPolicy(cfg, SMALL, init_params(cfg, SMALL, seed=2))
    logits, _ = net({k: torch.as_tensor(v) for k, v in obs.items()}, torch.as_tensor(mask))
    ent = torch_net.entropy(logits)
    lp = torch_net.log_prob(logits, torch.full((n,), cfg.draw_action))
    assert torch.isfinite(ent).all() and torch.isfinite(lp).all()
    assert float(ent.detach().max()) < 1e-4, "one legal action should mean ~zero entropy"


@pytest.mark.parametrize("cfg", CONFIGS, ids=lambda c: f"{c.n_kinds}k-{c.n_players}p")
def test_a_fresh_policy_is_near_uniform_over_the_legal_actions(cfg: RummiConfig):
    """What the 0.01 policy-head gain buys, stated as a number: entropy should sit
    on `log(n_legal)`, not below it."""
    torch = pytest.importorskip("torch")
    from rummi.agents.learned import torch_net

    _, obs, mask = played(cfg)
    net = torch_net.TorchPolicy(cfg, SMALL, init_params(cfg, SMALL, seed=0))
    logits, _ = net({k: torch.as_tensor(v) for k, v in obs.items()}, torch.as_tensor(mask))
    ent = torch_net.entropy(logits).detach().numpy()
    np.testing.assert_allclose(ent, np.log(mask.sum(-1)), rtol=0.02)


# --- parity ------------------------------------------------------------------
@pytest.mark.parametrize("cfg", CONFIGS, ids=lambda c: f"{c.n_kinds}k-{c.n_players}p")
def test_the_two_frameworks_agree_on_logits(cfg: RummiConfig):
    torch = pytest.importorskip("torch")
    pytest.importorskip("jax")
    import jax.numpy as jnp

    from rummi.agents.learned import jax_net, torch_net

    p = init_params(cfg, SMALL, seed=5)
    _, obs, mask = played(cfg)

    t_logits, t_value = torch_net.TorchPolicy(cfg, SMALL, p)(
        {k: torch.as_tensor(v) for k, v in obs.items()}, torch.as_tensor(mask)
    )
    j_logits, j_value = jax_net.apply(
        cfg,
        len(SMALL.hidden),
        SMALL.activation,
        {k: jnp.asarray(v) for k, v in p.items()},
        {k: jnp.asarray(v) for k, v in obs.items()},
        jnp.asarray(mask),
    )

    np.testing.assert_allclose(
        t_logits.detach().numpy(), np.asarray(j_logits), atol=1e-4,
        err_msg="the two networks diverged on logits",
    )
    np.testing.assert_allclose(
        t_value.detach().numpy(), np.asarray(j_value), atol=1e-4,
        err_msg="the two networks diverged on values",
    )
    np.testing.assert_allclose(
        torch_net.entropy(t_logits).detach().numpy(), np.asarray(jax_net.entropy(j_logits)),
        atol=1e-4,
    )


# --- the agent contract ------------------------------------------------------
def _agents(cfg: RummiConfig):
    import jax.numpy as jnp

    from rummi.agents.learned.agent import jax_agent, torch_agent
    from rummi.agents.learned.torch_net import TorchPolicy

    p = init_params(cfg, SMALL, seed=7)
    return [
        ("torch", torch_agent(TorchPolicy(cfg, SMALL, p))),
        ("jax", jax_agent(cfg, {k: jnp.asarray(v) for k, v in p.items()}, SMALL)),
    ]


def test_both_adapters_satisfy_the_agent_protocol():
    pytest.importorskip("torch")
    pytest.importorskip("jax")
    for _, agent in _agents(TINY_GROUPS):
        assert isinstance(agent, Agent)
        assert agent.name
        agent.reset(4)


def test_both_adapters_score_identically_through_the_frozen_protocol():
    """An untrained net is weak, but it must be *legal* and it must be
    reproducible: the same weights through either framework is the same score."""
    pytest.importorskip("torch")
    pytest.importorskip("jax")
    from rummi.evaluate.protocol import SUITE_BY_NAME, evaluate

    suite = SUITE_BY_NAME["tiny"]
    results = {}
    for label, agent in _agents(TINY_GROUPS):
        r = evaluate(f"untrained-{label}", suite, build_agent=lambda c, a=agent: a, games=8)
        assert r.illegal_attempts == 0, f"{label} proposed a masked-out action"
        assert not r.disqualified
        results[label] = (r.win_rate, r.mean_score, r.wins, r.losses)
    assert results["torch"] == results["jax"], results
