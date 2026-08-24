"""Torch-specific checks.

Cross-backend conformance -- golden digests, masks, rewards -- lives in
``test_backends.py``, which drives every backend through one code path. What
stays here is what only applies to torch: exhaustive comparison of the validity
kernel, and agreement between the eager and ``torch.compile`` paths, since the
benchmark's headline figure depends on the latter.
"""

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rummi.core import masks as np_masks
from rummi.core import sets as np_sets
from rummi.core.config import STANDARD, TINY, TINY_GROUPS, RummiConfig
from rummi.core.deal import derived_seeds
from rummi.core.deal import reset as np_reset
from rummi.core.deal import reset_envs as np_reset_envs
from rummi.core.engine import step as np_step
from rummi.core.sets import pad_slot

from rummi.backends.torch_backend import kernel as t_kernel
from rummi.backends.torch_backend import sim as t_sim

CONFIGS = {"tiny": TINY, "tiny_groups": TINY_GROUPS, "standard": STANDARD}
GOLDEN = Path(__file__).parent / "golden"
DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])


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
    got = t_kernel.evaluate_slots(cfg, torch.as_tensor(slots, dtype=torch.int64))

    for field in ("is_empty", "run_valid", "group_valid", "is_valid", "run_open", "group_open", "is_extendable"):
        np.testing.assert_array_equal(
            getattr(got, field).cpu().numpy(), getattr(want, field), err_msg=field
        )
    np.testing.assert_array_equal(got.value.cpu().numpy(), want.value, err_msg="value")


@pytest.mark.parametrize("cfg", [TINY, TINY_GROUPS], ids=["tiny", "tiny_groups"])
def test_assign_predicate_agrees_exhaustively(cfg: RummiConfig):
    contents = [c for c in _all_contents(cfg) if len(c) < cfg.max_set_len]
    slots = np.stack([pad_slot(cfg, c) for c in contents])
    want = np_sets.assign_open(cfg, np_sets.slot_stats(cfg, slots))
    got = t_kernel.assign_open(cfg, t_kernel.slot_stats(cfg, torch.as_tensor(slots, dtype=torch.int64)))
    np.testing.assert_array_equal(got.cpu().numpy(), want)


@pytest.mark.parametrize("device", DEVICES)
def test_invariants_hold_and_rewards_match(device: str):
    """Rewards are excluded from the digest, so check them explicitly."""
    cfg = TINY_GROUPS
    payload = json.loads((GOLDEN / "tiny_groups.json").read_text())
    dev = torch.device(device)
    ref = np_reset(cfg, payload["batch_size"], seed=payload["seed"])
    got = t_sim.reset(cfg, payload["batch_size"], seed=payload["seed"], device=dev)
    resets = {r[0]: r[1] for r in payload["resets"]}

    for i, actions in enumerate(payload["actions"]):
        ref_out = np_step(ref, np.asarray(actions), np_masks.legal_actions(ref))
        got_out = t_sim.step(got, torch.as_tensor(actions, device=dev), t_sim.legal_actions(got))
        got.check_invariants()
        np.testing.assert_allclose(
            got_out.rewards.cpu().numpy(), ref_out.rewards, atol=1e-6,
            err_msg=f"rewards differ at step {i}",
        )
        np.testing.assert_array_equal(got_out.terminated.cpu().numpy(), ref_out.terminated)
        np.testing.assert_array_equal(got_out.truncated.cpu().numpy(), ref_out.truncated)
        if i in resets:
            envs = resets[i]
            np_reset_envs(ref, np.asarray(envs), derived_seeds(payload["reset_seed"], i, envs))
            t_sim.reset_envs(
                got, torch.as_tensor(envs, device=dev),
                t_sim.derived_deck_orders(cfg, payload["reset_seed"], i, envs, dev),
            )


@pytest.mark.parametrize("device", DEVICES)
def test_compiled_masks_agree_with_eager(device: str):
    """The benchmark's headline figure comes from torch.compile, so the compiled
    path has to be verified, not assumed."""
    cfg = TINY_GROUPS
    payload = json.loads((GOLDEN / "tiny_groups.json").read_text())
    dev = torch.device(device)
    compiled = torch.compile(t_sim.legal_actions, dynamic=False)

    state = t_sim.reset(cfg, payload["batch_size"], seed=payload["seed"], device=dev)
    resets = {r[0]: r[1] for r in payload["resets"]}
    for i, actions in enumerate(payload["actions"][:80]):
        eager = t_sim.legal_actions(state)
        fused = compiled(state)
        assert torch.equal(eager, fused), f"compiled mask differs at step {i}"
        t_sim.step(state, torch.as_tensor(actions, device=dev), eager)
        if i in resets:
            envs = resets[i]
            t_sim.reset_envs(
                state, torch.as_tensor(envs, device=dev),
                t_sim.derived_deck_orders(cfg, payload["reset_seed"], i, envs, dev),
            )
    assert state.digest()
