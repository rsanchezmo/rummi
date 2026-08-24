"""Conformance of the JAX backend against the NumPy reference.

Same ladder as the torch backend: the validity kernel on exhaustively enumerated
slots, then the mask step for step, then the state digest after replaying the
golden trajectories. The digest is the contract.
"""

import json
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from rummi.core import masks as np_masks
from rummi.core import sets as np_sets
from rummi.core.config import STANDARD, TINY, TINY_GROUPS, RummiConfig
from rummi.core.deal import derived_seeds
from rummi.core.deal import reset as np_reset
from rummi.core.deal import reset_envs as np_reset_envs
from rummi.core.engine import step as np_step
from rummi.core.sets import pad_slot

from rummi.backends.jax_backend import kernel as j_kernel
from rummi.backends.jax_backend import sim as j_sim

CONFIGS = {"tiny": TINY, "tiny_groups": TINY_GROUPS, "standard": STANDARD}
GOLDEN = Path(__file__).parent / "golden"


def _all_contents(cfg: RummiConfig):
    from itertools import combinations_with_replacement

    for length in range(cfg.max_set_len + 1):
        for content in combinations_with_replacement(range(cfg.n_kinds), length):
            counts = {}
            for k in content:
                counts[k] = counts.get(k, 0) + 1
            if max(counts.values(), default=0) <= 2:
                yield content


@pytest.mark.parametrize("cfg", [TINY, TINY_GROUPS], ids=["tiny", "tiny_groups"])
def test_validity_kernel_agrees_exhaustively(cfg: RummiConfig):
    contents = list(_all_contents(cfg))
    slots = np.stack([pad_slot(cfg, c) for c in contents])
    want = np_sets.evaluate_slots(cfg, slots)
    got = j_kernel.evaluate_slots(cfg, jnp.asarray(slots))

    for field in (
        "is_empty", "run_valid", "group_valid", "is_valid",
        "run_open", "group_open", "is_extendable",
    ):
        np.testing.assert_array_equal(
            np.asarray(getattr(got, field)), getattr(want, field), err_msg=field
        )
    np.testing.assert_array_equal(np.asarray(got.value), want.value, err_msg="value")


@pytest.mark.parametrize("cfg", [TINY, TINY_GROUPS], ids=["tiny", "tiny_groups"])
def test_assign_predicate_agrees_exhaustively(cfg: RummiConfig):
    contents = [c for c in _all_contents(cfg) if len(c) < cfg.max_set_len]
    slots = np.stack([pad_slot(cfg, c) for c in contents])
    want = np_sets.assign_open(cfg, np_sets.slot_stats(cfg, slots))
    got = j_kernel.assign_open(cfg, j_kernel.slot_stats(cfg, jnp.asarray(slots)))
    np.testing.assert_array_equal(np.asarray(got), want)


@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_dealing_agrees(name: str):
    cfg = CONFIGS[name]
    ref = np_reset(cfg, 4, seed=101)
    got = j_sim.reset(cfg, 4, seed=101)
    np.testing.assert_array_equal(np.asarray(got.deck_order), ref.deck_order)
    np.testing.assert_array_equal(np.asarray(got.racks), ref.racks)
    assert j_sim.digest(got) == ref.digest()


@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_golden_trajectory_is_reproduced(name: str):
    payload = json.loads((GOLDEN / f"{name}.json").read_text())
    cfg = CONFIGS[name]
    state = j_sim.reset(cfg, payload["batch_size"], seed=payload["seed"])
    seen = [j_sim.digest(state)]
    resets = {r[0]: r[1] for r in payload["resets"]}

    for i, actions in enumerate(payload["actions"]):
        state, _ = j_sim.step(cfg, state, jnp.asarray(actions))
        if (i + 1) % payload["digest_every"] == 0:
            seen.append(j_sim.digest(state))
        if i in resets:
            envs = resets[i]
            state = j_sim.reset_envs(
                cfg, state, jnp.asarray(envs),
                j_sim.derived_deck_orders(cfg, payload["reset_seed"], i, envs),
            )
    seen.append(j_sim.digest(state))

    first_bad = next((i for i, (a, b) in enumerate(zip(seen, payload["digests"])) if a != b), None)
    assert first_bad is None, f"{name}: diverged at digest {first_bad}"


@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_masks_and_rewards_agree_step_for_step(name: str):
    """A digest match could hide a mask that differs on actions never taken, and
    rewards are outside the digest entirely, so compare both explicitly."""
    cfg = CONFIGS[name]
    payload = json.loads((GOLDEN / f"{name}.json").read_text())
    ref = np_reset(cfg, payload["batch_size"], seed=payload["seed"])
    got = j_sim.reset(cfg, payload["batch_size"], seed=payload["seed"])
    resets = {r[0]: r[1] for r in payload["resets"]}

    for i, actions in enumerate(payload["actions"][:120]):
        ref_mask = np_masks.legal_actions(ref)
        got_mask = j_sim.legal_actions(cfg, got)
        np.testing.assert_array_equal(
            np.asarray(got_mask), ref_mask, err_msg=f"{name}: masks differ at step {i}"
        )
        ref_out = np_step(ref, np.asarray(actions), ref_mask)
        got, got_out = j_sim.step(cfg, got, jnp.asarray(actions))
        np.testing.assert_allclose(
            np.asarray(got_out.rewards), ref_out.rewards, atol=1e-6,
            err_msg=f"{name}: rewards differ at step {i}",
        )
        np.testing.assert_array_equal(np.asarray(got_out.terminated), ref_out.terminated)
        np.testing.assert_array_equal(np.asarray(got_out.truncated), ref_out.truncated)
        j_sim.check_invariants(cfg, got)
        if i in resets:
            envs = resets[i]
            np_reset_envs(ref, np.asarray(envs), derived_seeds(payload["reset_seed"], i, envs))
            got = j_sim.reset_envs(
                cfg, got, jnp.asarray(envs),
                j_sim.derived_deck_orders(cfg, payload["reset_seed"], i, envs),
            )
        assert j_sim.digest(got) == ref.digest(), f"{name}: state diverged at step {i}"


def test_illegal_actions_are_caught_host_side():
    """Validation cannot live inside the jitted step -- it would read a device
    boolean and break the trace -- so it is a separate host-side call."""
    cfg = TINY_GROUPS
    state = j_sim.reset(cfg, 2, seed=0)
    mask = j_sim.legal_actions(cfg, state)
    with pytest.raises(ValueError, match="illegal action"):
        j_sim.check_actions(mask, jnp.full((2,), cfg.end_turn_action), ~state.done)
    j_sim.check_actions(mask, jnp.full((2,), cfg.draw_action), ~state.done)


def test_step_is_jitted_and_donates_nothing_surprising():
    """A traced step must not need a host round-trip; if it did, the whole point
    of the port would be lost."""
    cfg = TINY_GROUPS
    state = j_sim.reset(cfg, 8, seed=3)
    mask = j_sim.legal_actions(cfg, state)
    actions = jnp.argmax(mask.astype(jnp.int8), axis=-1)
    lowered = jax.jit(j_sim.step, static_argnums=0).lower(cfg, state, actions)
    assert lowered.compile() is not None
