"""The observation contract from SPEC.md section 8.

Shapes and dtypes, but mostly the two properties a port can satisfy every shape
and still get wrong: the seat rotation, and that individual opponent racks are
not recoverable from what an agent is handed.
"""

import numpy as np
import pytest

from rummi.env.numpy.deal import reset
from rummi.env.numpy.engine import step
from rummi.env.numpy.masks import legal_actions
from rummi.env.observation import encode
from rummi.rules.observation import SLOT_FEATURES
from rummi.rules.config import STANDARD_4P, TINY_GROUPS, RummiConfig

CONFIGS = [TINY_GROUPS, STANDARD_4P]

# Exactly the table in SPEC.md section 8.
FIELDS = {
    "rack": ("K",),
    "table_sets": ("S", "L"),
    "slot_features": ("S", "F"),
    "workbench": ("K",),
    "placed_this_turn": ("K",),
    "unseen": ("K",),
    "rack_sizes": ("P",),
    "melded": ("P",),
    "scalars": ("N",),
}
DTYPES = {
    "rack": np.int16, "table_sets": np.int16, "slot_features": np.int32,
    "workbench": np.int16, "placed_this_turn": np.int16, "unseen": np.int16,
    "rack_sizes": np.int16, "melded": np.int8, "scalars": np.int32,
}


def advanced(cfg: RummiConfig, batch: int = 6, steps: int = 40, seed: int = 0):
    """A state with a table, a workbench and some seats melded -- a fresh deal
    exercises almost none of the observation."""
    from rummi.agents import build
    from rummi.agents.base import act_on_state

    state = reset(cfg, batch, seed=seed)
    agent = build("greedy", cfg)
    for _ in range(steps):
        mask = legal_actions(state)
        step(state, act_on_state(agent, state, mask), mask)
    return state


@pytest.mark.parametrize("cfg", CONFIGS, ids=lambda c: f"{c.n_players}p")
def test_every_documented_field_has_its_documented_shape_and_dtype(cfg: RummiConfig):
    state = advanced(cfg)
    obs = encode(state)
    dims = {
        "K": cfg.n_kinds, "S": cfg.max_sets, "L": cfg.max_set_len,
        "P": cfg.n_players, "F": SLOT_FEATURES, "N": 4,
    }
    assert set(obs) == set(FIELDS), "the observation gained or lost a field"
    for name, axes in FIELDS.items():
        expected = (state.batch_size, *(dims[a] for a in axes))
        assert obs[name].shape == expected, f"{name}: {obs[name].shape} != {expected}"
        assert obs[name].dtype == DTYPES[name], f"{name}: {obs[name].dtype}"


@pytest.mark.parametrize("cfg", CONFIGS, ids=lambda c: f"{c.n_players}p")
def test_per_seat_fields_start_at_the_acting_seat(cfg: RummiConfig):
    """Index `i` is seat `(current + i) mod P`. Without this a shared policy would
    have to learn a convention per seat."""
    state = advanced(cfg)
    obs = encode(state)
    sizes = state.racks.sum(-1)
    for env in range(state.batch_size):
        current = int(state.current[env])
        for i in range(cfg.n_players):
            seat = (current + i) % cfg.n_players
            assert obs["rack_sizes"][env, i] == sizes[env, seat], (env, i)
            assert bool(obs["melded"][env, i]) == bool(state.melded[env, seat]), (env, i)
        np.testing.assert_array_equal(obs["rack"][env], state.racks[env, current])


@pytest.mark.parametrize("cfg", CONFIGS, ids=lambda c: f"{c.n_players}p")
def test_unseen_is_every_tile_the_actor_cannot_locate(cfg: RummiConfig):
    state = advanced(cfg)
    obs = encode(state)
    for env in range(state.batch_size):
        current = int(state.current[env])
        others = [p for p in range(cfg.n_players) if p != current]
        hidden = state.pool[env] + state.racks[env, others].sum(0)
        np.testing.assert_array_equal(obs["unseen"][env], hidden)


def test_shuffling_tiles_between_opponents_changes_nothing_an_agent_sees():
    """The integrity property, tested at its weakest point rather than by
    inspection: swap which kind two opponents hold, leaving every seat's tile
    count and the pool untouched. Any leak of an *individual* rack would move the
    observation. Nothing may."""
    cfg = STANDARD_4P
    state = advanced(cfg, batch=4, steps=30, seed=3)
    before = encode(state)

    swaps = 0
    for env in range(state.batch_size):
        current = int(state.current[env])
        others = [p for p in range(cfg.n_players) if p != current]
        a, b = others[0], others[1]
        kind_a = next((k for k in range(cfg.n_kinds) if state.racks[env, a, k] > 0), None)
        kind_b = next(
            (k for k in range(cfg.n_kinds) if state.racks[env, b, k] > 0 and k != kind_a), None
        )
        if kind_a is None or kind_b is None:
            continue
        state.racks[env, a, kind_a] -= 1
        state.racks[env, a, kind_b] += 1
        state.racks[env, b, kind_b] -= 1
        state.racks[env, b, kind_a] += 1
        swaps += 1

    assert swaps, "no swap was possible, so this proved nothing"
    state.check_invariants()
    after = encode(state)
    for name in before:
        np.testing.assert_array_equal(
            after[name], before[name],
            err_msg=f"{name} leaked which opponent holds which tile",
        )


@pytest.mark.parametrize("cfg", CONFIGS, ids=lambda c: f"{c.n_players}p")
def test_meld_remaining_is_zero_once_the_seat_has_opened(cfg: RummiConfig):
    state = advanced(cfg)
    obs = encode(state)
    progress, remaining = obs["scalars"][:, 1], obs["scalars"][:, 2]
    opened = state.melded[np.arange(state.batch_size), state.current]
    assert (remaining[opened] == 0).all()
    expected = np.maximum(0, cfg.initial_meld - progress[~opened])
    np.testing.assert_array_equal(remaining[~opened], expected)


def test_the_space_accepts_what_encode_produces():
    """`observation_space` and `encode` are written separately, so a bound that
    does not actually hold is a live possibility."""
    pytest.importorskip("gymnasium")
    from rummi.env.observation import observation_space

    for cfg in CONFIGS:
        state = advanced(cfg, steps=60)
        space = observation_space(cfg)
        obs = encode(state)
        for env in range(state.batch_size):
            single = {k: v[env] for k, v in obs.items()}
            assert space.contains(single), f"{cfg.n_players}p env {env} outside the space"
