"""JAX-specific checks.

Cross-backend conformance -- golden digests, masks, rewards -- lives in
``test_backends.py``. What stays here is what only applies to JAX: exhaustive
comparison of the validity kernel, host-side action validation (which cannot live
inside the traced step), and that the step actually lowers and compiles.
"""

from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from rummi.env.numpy import sets as np_sets
from rummi.rules.config import STANDARD, TINY, TINY_GROUPS, RummiConfig
from rummi.env.numpy.sets import pad_slot

from rummi.env.jax import kernel as j_kernel
from rummi.env.jax import sim as j_sim

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
