"""Golden trajectories: the cross-backend contract.

Each fixture is a seeded action sequence plus periodic state digests. Any
implementation -- this NumPy reference, or the torch and JAX ports -- must
reproduce the digests exactly. That is what makes the eventual benchmark a
comparison of speed rather than of behaviour.

Regenerate deliberately, never to make a failure go away: a digest changing means
the rules changed, and that is either a bug or a decision worth recording.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from rummi.bench.fuzz import make_policy
from rummi.core.config import STANDARD, TINY, TINY_GROUPS
from rummi.core.deal import derived_seeds, reset, reset_envs
from rummi.core.engine import step
from rummi.core.masks import legal_actions

CONFIGS = {"tiny": TINY, "tiny_groups": TINY_GROUPS, "standard": STANDARD}
GOLDEN = Path(__file__).parent / "golden"


def replay(payload):
    """Re-drive the recorded actions and return the digests observed."""
    cfg = CONFIGS[payload["config"]]
    state = reset(cfg, payload["batch_size"], seed=payload["seed"])
    seen = [state.digest()]
    resets = {r[0]: r[1] for r in payload["resets"]}

    for i, actions in enumerate(payload["actions"]):
        mask = legal_actions(state)
        step(state, np.asarray(actions), mask)
        if (i + 1) % payload["digest_every"] == 0:
            seen.append(state.digest())
        if i in resets:
            envs = np.asarray(resets[i])
            reset_envs(state, envs, derived_seeds(payload["reset_seed"], i, envs))
    seen.append(state.digest())
    return state, seen


@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_golden_trajectory_is_reproduced(name: str):
    payload = json.loads((GOLDEN / f"{name}.json").read_text())
    _, seen = replay(payload)
    assert seen == payload["digests"], (
        f"{name}: state diverged from the recorded trajectory at digest "
        f"{next(i for i, (a, b) in enumerate(zip(seen, payload['digests'])) if a != b)}"
    )


@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_the_recorded_actions_are_what_the_policy_still_chooses(name: str):
    """Catches a silent behaviour change in the baseline itself, which would
    otherwise sit undetected behind a passing digest replay."""
    payload = json.loads((GOLDEN / f"{name}.json").read_text())
    cfg = CONFIGS[name]
    policy = make_policy(cfg, "greedy", 0)
    state = reset(cfg, payload["batch_size"], seed=payload["seed"])
    resets = {r[0]: r[1] for r in payload["resets"]}

    for i, expected in enumerate(payload["actions"]):
        mask = legal_actions(state)
        chosen = policy(state, mask)
        assert [int(x) for x in chosen] == expected, f"{name}: policy diverged at step {i}"
        step(state, chosen, mask)
        if i in resets:
            envs = np.asarray(resets[i])
            reset_envs(state, envs, derived_seeds(payload["reset_seed"], i, envs))


def test_digests_distinguish_states():
    """A digest that collided would make every conformance test vacuous."""
    a = reset(STANDARD, 2, seed=1)
    b = reset(STANDARD, 2, seed=2)
    assert a.digest() != b.digest()
    assert a.digest() == reset(STANDARD, 2, seed=1).digest()

    mutated = a.clone()
    mutated.racks[0, 0, 0] += 1
    assert mutated.digest() != a.digest()
