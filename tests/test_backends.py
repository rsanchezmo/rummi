"""Cross-backend conformance through the uniform adapter.

Every available backend is driven through the same code path here, so adding one
means adding a name rather than another near-copy of this file. The
framework-specific checks -- ``torch.compile`` agreement, JAX trace-ability,
exhaustive kernel comparison -- stay in the per-backend test modules.
"""

import json
from dataclasses import replace
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
from rummi.env.observation import encode as np_encode

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


# --- reward shaping ----------------------------------------------------------
# Every golden fixture leaves these at zero, so the nine branches implementing
# them -- three terms in each of three backends -- are the one part of the reward
# no other test reaches. Each is exercised on its own, so a term that silently
# does nothing cannot hide behind the other two.
SHAPING_TERMS = {
    "tiles_placed_bonus": 0.5,
    "rack_value_delta": 0.25,
    "micro_step_cost": 0.01,
}


def _greedy_actions(cfg, batch_size: int, steps: int, seed: int) -> list[np.ndarray]:
    """A recorded trajectory every backend can replay.

    Greedy rather than random: random play never assembles a legal opening meld,
    so it never reaches ``END_TURN`` and the meld-time shaping terms would stay at
    zero no matter how wrong they were.
    """
    from rummi.agents import build
    from rummi.agents.base import act_on_state

    state = np_reset(cfg, batch_size, seed=seed)
    agent = build("greedy", cfg)
    recorded = []
    for _ in range(steps):
        mask = np_masks.legal_actions(state)
        actions = act_on_state(agent, state, mask)
        recorded.append(np.asarray(actions).copy())
        np_step(state, actions, mask)
    return recorded


def _replay_rewards(backend, cfg, actions: list[np.ndarray], batch_size: int, seed: int):
    state = backend.reset(cfg, batch_size, seed=seed)
    total = np.zeros((batch_size, cfg.n_players), dtype=np.float64)
    per_step = []
    for step_actions in actions:
        mask = backend.legal_actions(cfg, state)
        state, out = backend.step(cfg, state, step_actions, mask)
        rewards = backend.to_numpy(out.rewards)
        per_step.append(rewards.copy())
        total += rewards
    return total, per_step


@pytest.mark.parametrize("term", sorted(SHAPING_TERMS))
@pytest.mark.parametrize("backend_name", BACKENDS)
def test_each_shaping_term_matches_the_reference_and_actually_fires(backend_name: str, term: str):
    backend = get_backend(backend_name)
    cfg = replace(TINY_GROUPS, **{term: SHAPING_TERMS[term]})
    batch_size, steps, seed = 6, 120, 17

    actions = _greedy_actions(cfg, batch_size, steps, seed)
    plain_total, _ = _replay_rewards(get_backend("numpy"), TINY_GROUPS, actions, batch_size, seed)
    ref_total, ref_steps = _replay_rewards(get_backend("numpy"), cfg, actions, batch_size, seed)
    got_total, got_steps = _replay_rewards(backend, cfg, actions, batch_size, seed)

    for i, (got, ref) in enumerate(zip(got_steps, ref_steps, strict=True)):
        np.testing.assert_allclose(
            got, ref, atol=1e-6,
            err_msg=f"{backend.name}: {term} reward differs at step {i}",
        )
    assert not np.allclose(ref_total, plain_total, atol=1e-9), (
        f"{term} changed no reward over {steps} steps, so this proved nothing"
    )
    np.testing.assert_allclose(got_total, ref_total, atol=1e-6)


@pytest.mark.parametrize("backend_name", BACKENDS)
def test_shaping_is_credited_only_to_the_seat_that_acted(backend_name: str):
    """Shaping is a per-seat signal, so a term leaking into another seat's column
    would quietly reward a player for an opponent's move."""
    backend = get_backend(backend_name)
    cfg = replace(TINY_GROUPS, **SHAPING_TERMS)
    batch_size, steps, seed = 4, 120, 23

    actions = _greedy_actions(cfg, batch_size, steps, seed)
    state = backend.reset(cfg, batch_size, seed=seed)
    ref = np_reset(cfg, batch_size, seed=seed)
    shaped_steps = 0

    for i, step_actions in enumerate(actions):
        acting = ref.current.copy()
        done_before = ref.done.copy()
        mask = backend.legal_actions(cfg, state)
        state, out = backend.step(cfg, state, step_actions, mask)
        np_step(ref, step_actions, np_masks.legal_actions(ref))
        rewards = backend.to_numpy(out.rewards)

        for env in range(batch_size):
            if done_before[env] or ref.done[env]:
                continue  # a terminal payout is zero-sum across every seat
            others = [p for p in range(cfg.n_players) if p != acting[env]]
            assert not rewards[env, others].any(), (
                f"{backend.name}: step {i} paid a seat that did not act"
            )
            shaped_steps += int(rewards[env, acting[env]] != 0)

    assert shaped_steps, "no shaped step was observed, so this proved nothing"


# --- observation -------------------------------------------------------------
@pytest.mark.parametrize("config", sorted(CONFIGS))
@pytest.mark.parametrize("backend_name", BACKENDS)
def test_the_observation_matches_the_reference_field_for_field(backend_name: str, config: str):
    """SPEC.md section 8 is the contract each backend's encoder was written
    against; this is what makes that claim checkable.

    Replayed over a recorded trajectory rather than a fresh deal, because a fresh
    deal has an empty table -- every `slot_features` column, the workbench and
    `placed_this_turn` would all be zero and any divergence in them invisible.
    """
    backend = get_backend(backend_name)
    cfg = CONFIGS[config]
    payload = _payload(config)

    ref = np_reset(cfg, payload["batch_size"], seed=payload["seed"])
    state = backend.reset(cfg, payload["batch_size"], seed=payload["seed"])
    nonzero_seen = set()

    for i, actions in enumerate(payload["actions"][:80]):
        ref_obs = np_encode(ref)
        obs = backend.encode(cfg, state)
        assert set(obs) == set(ref_obs), f"{backend.name}: field names differ"
        for name, want in ref_obs.items():
            got = backend.to_numpy(obs[name])
            assert got.shape == want.shape, f"{backend.name}/{name}: {got.shape} != {want.shape}"
            assert got.dtype == want.dtype, f"{backend.name}/{name}: {got.dtype} != {want.dtype}"
            np.testing.assert_array_equal(
                got, want, err_msg=f"{backend.name}/{config}: {name} differs at step {i}",
            )
            if want.any():
                nonzero_seen.add(name)

        mask = backend.legal_actions(cfg, state)
        state, _ = backend.step(cfg, state, np.asarray(actions), mask)
        np_step(ref, np.asarray(actions), np_masks.legal_actions(ref))

    # Otherwise a trajectory that never touched the table would pass vacuously.
    assert "table_sets" in nonzero_seen or "workbench" in nonzero_seen, (
        f"{config}: the trajectory never put a tile in play, so this proved little"
    )


@pytest.mark.parametrize("backend_name", BACKENDS)
def test_active_opts_envs_out_of_a_step(backend_name: str):
    """`active` is in the Backend protocol and no golden replay uses it, so until
    the vector env grew a `backend=` argument nothing converted it -- a NumPy mask
    reaching the torch sim raised on `state.done |= ...`.

    Half the batch is held out and must be byte-identical to a batch that never
    took the step at all.
    """
    backend = get_backend(backend_name)
    cfg = TINY_GROUPS
    n = 8
    active = np.zeros(n, dtype=bool)
    active[::2] = True

    stepped = backend.reset(cfg, n, seed=7)
    held = backend.reset(cfg, n, seed=7)
    ref = np_reset(cfg, n, seed=7)

    for _ in range(12):
        mask = backend.legal_actions(cfg, stepped)
        actions = backend.to_numpy(mask).argmax(-1)
        stepped, _ = backend.step(cfg, stepped, actions, mask, active)
        np_step(ref, actions, np_masks.legal_actions(ref), active)

    assert backend.digest(stepped) == ref.digest(), f"{backend.name}: diverged under active"

    # The held-out half never moved: same racks as a state that took no steps.
    moved = backend.to_numpy(stepped.racks if hasattr(stepped, "racks") else stepped[0])
    untouched = backend.to_numpy(held.racks if hasattr(held, "racks") else held[0])
    np.testing.assert_array_equal(
        moved[~active], untouched[~active],
        err_msg=f"{backend.name}: an inactive env was stepped anyway",
    )
