"""Whole-system properties: conservation, determinism, and batch equivalence."""

import numpy as np
import pytest

from rummi.bench.fuzz import fuzz, make_policy
from rummi.rules.config import CONFIG_BY_NAME, STANDARD, RummiConfig
from rummi.env.numpy.deal import reset
from rummi.env.numpy.engine import step
from rummi.env.numpy.masks import legal_actions

# Driven off the preset map rather than a list: an invariant that only holds on
# the configs someone remembered to add is not an invariant. Golden fixtures are
# the opposite case -- see CONFIG_BY_NAME.
CONFIGS = [(cfg, name) for name, cfg in CONFIG_BY_NAME.items()]


@pytest.mark.parametrize("cfg,name", CONFIGS, ids=[n for _, n in CONFIGS])
@pytest.mark.parametrize("policy", ["random", "greedy"])
def test_fuzz_holds_every_invariant(cfg: RummiConfig, name: str, policy: str):
    games = 4 if name.startswith("standard") else 20
    stats = fuzz(cfg, games=games, batch_size=8, seed=0, policy_name=policy)
    assert stats.games >= games
    assert stats.steps > 0


@pytest.mark.parametrize("policy", ["random", "greedy"])
def test_greedy_reaches_the_paths_random_cannot(policy: str):
    """Random play essentially never assembles a 30-point opening meld, so it
    cannot cover END_TURN, melding or winning. Greedy must."""
    stats = fuzz(STANDARD, games=6, batch_size=8, seed=1, policy_name=policy)
    if policy == "greedy":
        assert stats.end_turns > 0 and stats.melds > 0
    else:
        assert stats.steps > 0  # still exercises masks and the mid-turn machinery


@pytest.mark.parametrize("cfg,name", CONFIGS, ids=[n for _, n in CONFIGS])
def test_same_seed_gives_an_identical_trajectory(cfg: RummiConfig, name: str):
    def rollout():
        policy = make_policy(cfg, "greedy", 0)
        state = reset(cfg, 4, seed=7)
        for _ in range(60):
            mask = legal_actions(state)
            step(state, policy(state, mask), mask)
        return state

    a, b = rollout(), rollout()
    np.testing.assert_array_equal(a.table_sets, b.table_sets)
    np.testing.assert_array_equal(a.racks, b.racks)
    np.testing.assert_array_equal(a.turn_count, b.turn_count)
    np.testing.assert_array_equal(a.deck_order, b.deck_order)


@pytest.mark.parametrize("cfg,name", CONFIGS, ids=[n for _, n in CONFIGS])
def test_a_batched_rollout_equals_the_same_envs_run_alone(cfg: RummiConfig, name: str):
    """The property that makes batching safe -- and the reason the step function
    holds no RNG: each env's deck is fixed at reset, so nothing couples them."""
    batch_size, steps = 6, 40

    batched = reset(cfg, batch_size, seed=3)
    singles = [batched.select(i) for i in range(batch_size)]

    batch_policy = make_policy(cfg, "greedy", 0)
    for _ in range(steps):
        mask = legal_actions(batched)
        step(batched, batch_policy(batched, mask), mask)

    for i, single in enumerate(singles):
        solo_policy = make_policy(cfg, "greedy", 0)
        for _ in range(steps):
            mask = legal_actions(single)
            step(single, solo_policy(single, mask), mask)
        np.testing.assert_array_equal(
            single.table_sets[0], batched.table_sets[i], err_msg=f"table diverged in env {i}"
        )
        np.testing.assert_array_equal(
            single.racks[0], batched.racks[i], err_msg=f"racks diverged in env {i}"
        )
        assert int(single.turn_count[0]) == int(batched.turn_count[i])
        assert bool(single.done[0]) == bool(batched.done[i])


@pytest.mark.parametrize("cfg,name", CONFIGS, ids=[n for _, n in CONFIGS])
def test_every_masked_in_action_is_accepted(cfg: RummiConfig, name: str):
    """Exhaustive per-step check that the mask never offers something the engine
    would refuse -- the contract that lets step() trust its preconditions."""
    rng = np.random.default_rng(0)
    state = reset(cfg, 4, seed=11)
    for _ in range(50):
        mask = legal_actions(state)
        for env in range(state.batch_size):
            legal = np.flatnonzero(mask[env])
            assert legal.size, "no legal action"
            probe = state.select(env)
            for action in rng.choice(legal, size=min(6, legal.size), replace=False):
                trial = probe.clone()
                step(trial, np.array([action]), legal_actions(trial))
                trial.check_invariants()
        actions = np.argmax(np.where(mask, rng.random(mask.shape), -1.0), axis=-1)
        step(state, actions, mask)


@pytest.mark.parametrize("cfg,name", CONFIGS, ids=[n for _, n in CONFIGS])
def test_the_mask_is_never_all_zero_even_after_termination(cfg: RummiConfig, name: str):
    policy = make_policy(cfg, "greedy", 0)
    state = reset(cfg, 8, seed=2)
    saw_done = False
    for _ in range(400):
        mask = legal_actions(state)
        assert mask.any(-1).all(), "a policy sampling an all-zero row would produce NaNs"
        assert mask[:, cfg.draw_action].all()
        step(state, policy(state, mask), mask)
        saw_done |= bool(state.done.any())
        if state.done.all():
            break
    assert saw_done, "test never reached a terminated env"
    assert legal_actions(state).any(-1).all()
