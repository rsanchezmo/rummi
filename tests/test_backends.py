"""Cross-backend conformance through the uniform adapter.

Every available backend is driven through the same code path here, so adding one
means adding a name rather than another near-copy of this file. The
framework-specific checks -- ``torch.compile`` agreement, JAX trace-ability,
exhaustive kernel comparison -- stay in the per-backend test modules.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from rummi.env.api import available, get_backend
from rummi.env.numpy import masks as np_masks
from rummi.rules.config import STANDARD, TINY, TINY_GROUPS
from rummi.env.numpy.deal import derived_seeds
from rummi.env.numpy.deal import reset as np_reset
from rummi.env.numpy.deal import reset_envs as np_reset_envs
from rummi.env.numpy.engine import step as np_step

CONFIGS = {"tiny": TINY, "tiny_groups": TINY_GROUPS, "standard": STANDARD}
GOLDEN = Path(__file__).parent / "golden"
BACKENDS = available()


def _payload(name: str) -> dict:
    return json.loads((GOLDEN / f"{name}.json").read_text())


@pytest.mark.parametrize("backend_name", BACKENDS)
@pytest.mark.parametrize("config", sorted(CONFIGS))
def test_backend_reproduces_the_golden_trajectory(backend_name: str, config: str):
    """The contract: same seeded actions in, same state digests out."""
    backend = get_backend(backend_name)
    cfg = CONFIGS[config]
    payload = _payload(config)
    resets = {r[0]: r[1] for r in payload["resets"]}

    state = backend.reset(cfg, payload["batch_size"], seed=payload["seed"])
    seen = [backend.digest(state)]
    for i, actions in enumerate(payload["actions"]):
        state, _ = backend.step(cfg, state, np.asarray(actions))
        if (i + 1) % payload["digest_every"] == 0:
            seen.append(backend.digest(state))
        if i in resets:
            state = backend.reset_envs(cfg, state, resets[i], payload["reset_seed"], i)
    seen.append(backend.digest(state))

    bad = next((i for i, (a, b) in enumerate(zip(seen, payload["digests"], strict=False)) if a != b), None)
    assert bad is None, f"{backend.name} on {config}: diverged at digest {bad}"


@pytest.mark.parametrize("backend_name", BACKENDS)
@pytest.mark.parametrize("config", sorted(CONFIGS))
def test_masks_and_rewards_match_the_reference(backend_name: str, config: str):
    """Digests could in principle hide a mask that differs only on actions the
    trajectory never took, and rewards are outside the digest entirely."""
    backend = get_backend(backend_name)
    cfg = CONFIGS[config]
    payload = _payload(config)
    resets = {r[0]: r[1] for r in payload["resets"]}

    ref = np_reset(cfg, payload["batch_size"], seed=payload["seed"])
    state = backend.reset(cfg, payload["batch_size"], seed=payload["seed"])

    for i, actions in enumerate(payload["actions"][:100]):
        ref_mask = np_masks.legal_actions(ref)
        mask = backend.legal_actions(cfg, state)
        np.testing.assert_array_equal(
            backend.to_numpy(mask), ref_mask,
            err_msg=f"{backend.name}/{config}: masks differ at step {i}",
        )
        ref_out = np_step(ref, np.asarray(actions), ref_mask)
        state, out = backend.step(cfg, state, np.asarray(actions), mask)
        np.testing.assert_allclose(
            out.rewards, ref_out.rewards, atol=1e-6,
            err_msg=f"{backend.name}/{config}: rewards differ at step {i}",
        )
        np.testing.assert_array_equal(out.terminated, ref_out.terminated)
        np.testing.assert_array_equal(out.truncated, ref_out.truncated)
        if i in resets:
            np_reset_envs(ref, np.asarray(resets[i]), derived_seeds(payload["reset_seed"], i, resets[i]))
            state = backend.reset_envs(cfg, state, resets[i], payload["reset_seed"], i)
        assert backend.digest(state) == ref.digest(), f"{backend.name}: diverged at step {i}"


def test_backends_are_interchangeable():
    """The swappability claim: the same driver code, only the name changed, must
    produce byte-identical outcomes."""
    cfg = TINY_GROUPS
    digests, rewards = {}, {}

    for name in BACKENDS:
        backend = get_backend(name)
        state = backend.reset(cfg, 6, seed=101)
        total = np.zeros((6, cfg.n_players), dtype=np.float64)
        for _ in range(60):
            mask = backend.legal_actions(cfg, state)
            actions = backend.to_numpy(mask).argmax(-1)
            state, out = backend.step(cfg, state, actions, mask)
            total += out.rewards
        digests[backend.name] = backend.digest(state)
        rewards[backend.name] = total

    assert len(set(digests.values())) == 1, f"backends diverged: {digests}"
    reference = rewards[next(iter(rewards))]
    for name, value in rewards.items():
        np.testing.assert_allclose(value, reference, atol=1e-6, err_msg=name)


@pytest.mark.parametrize("backend_name", BACKENDS)
def test_illegal_actions_are_rejected(backend_name: str):
    """Uniform even where the mechanism differs: torch and NumPy check inline,
    JAX checks host-side because a device read would break its trace."""
    backend = get_backend(backend_name)
    cfg = TINY_GROUPS
    state = backend.reset(cfg, 2, seed=0)
    mask = backend.legal_actions(cfg, state)
    with pytest.raises(ValueError, match="illegal action"):
        backend.step(cfg, state, np.full(2, cfg.end_turn_action), mask)


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError, match="unknown backend"):
        get_backend("tensorflow")


def test_numpy_is_always_available():
    assert "numpy" in available()
