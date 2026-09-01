"""The `learned` rung: it loads, it refuses what it cannot play, it repeats itself.

The rung ships weights, which is what makes these worth asserting. A rung built
from code is either present or a syntax error; one built from a file can be
present, load, and quietly be the wrong net -- so construction is checked on every
preset the package ships weights for, and a preset with none has to say so rather
than reshape itself into a net that would score meaninglessly.
"""

import numpy as np
import pytest

from rummi.agents import build
from rummi.agents.base import Agent, act_on_state
from rummi.agents.learned.clone import WEIGHTS, preset_name
from rummi.agents.macro import n_macros, repartition_offset
from rummi.env.numpy.deal import reset
from rummi.env.numpy.engine import step
from rummi.env.numpy.masks import legal_actions
from rummi.rules.config import STANDARD, STANDARD_3P, STANDARD_4P, TINY, RummiConfig

# The package imports without torch on purpose -- the rung loads it only when one
# is built -- so it is the construction that has to be skipped, not the module.
pytest.importorskip("torch")

BUNDLED = [STANDARD, STANDARD_3P, STANDARD_4P]
by_preset = pytest.mark.parametrize("cfg", BUNDLED, ids=lambda c: f"{c.n_players}p")


@by_preset
def test_the_rung_loads_its_bundled_weights(cfg: RummiConfig):
    agent = build("learned", cfg)
    assert isinstance(agent, Agent)
    assert agent.name == "learned"
    # The solver-backed macro is what the rung is for, and it widens the head by
    # exactly one action -- so a net loaded without it would be a different agent
    # that happened to fit.
    assert agent.repartition_macro == repartition_offset(cfg)
    assert agent.net.action_bias.shape[0] == n_macros(cfg, True)


def test_weights_are_shipped_for_every_preset_the_rung_claims():
    """Guards the guard: without this, deleting a file would silently shrink the
    parametrisation above to whatever is left."""
    assert {p.stem for p in WEIGHTS.glob("*.pt")} == {preset_name(c) for c in BUNDLED}


def test_a_config_with_no_weights_is_refused_by_name():
    with pytest.raises(ValueError, match="no bundled weights for 'tiny'"):
        build("learned", TINY)


@by_preset
def test_two_of_them_play_the_same_game(cfg: RummiConfig):
    """Deterministic, not merely seeded: the chooser takes the mode of the masked
    logits, so two agents handed one state owe the same action every step."""
    state = reset(cfg, 4, seed=0)
    one, two = build("learned", cfg), build("learned", cfg)
    one.reset(4)
    two.reset(4)
    for _ in range(40):
        mask = legal_actions(state)
        chosen = act_on_state(one, state, mask)
        assert np.array_equal(chosen, act_on_state(two, state, mask))
        step(state, chosen, mask)
