"""What the afterstate trainer stores, and what it counts as a decision.

Two rules decide what the value net is trained on, and neither is visible in the
rollout loop that applies them.

*An episode the clock cut off has no outcome.* The engine pays nothing on
truncation (SPEC.md section 7), so storing a cut-off episode as terminated
regresses every row of it towards zero -- V is taught that a whole game it was
half-way through winning was worth nothing. There is no successor row to bootstrap
from either, so the transition is not a transition.

*A decision the solver answered is still a decision.* `REPARTITION` carries no
afterstate to value -- `afterstate.py` refuses to build one -- but leaving it out
of the counters conditions every other block's rate, and `dec/s`, on the decisions
where the solver was not reached.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("gymnasium")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import train_afterstate as tool

from rummi.agents.macro import by_value, first_legal
from rummi.rules.config import TINY_GROUPS

DIM = 4


def _row(value: float) -> np.ndarray:
    return np.full(DIM, value, dtype=np.float32)


def _episodes(n_envs: int = 1) -> tool.Episodes:
    return tool.Episodes(n_envs, len(tool.BLOCKS))


def test_a_terminated_episode_is_stored_with_the_outcome_it_reached() -> None:
    """The control for the test below: a real ending is still a real ending."""
    eps = _episodes()
    eps.record(0, _row(1.0), block=0)
    eps.accrued[0] = 0.25
    eps.record(0, _row(2.0), block=0)
    eps.accrued[0] = 0.75

    stored = eps.close(0, terminal=True)
    assert len(stored) == 2
    # Two transitions: the first is followed by the second, the second ends it.
    # Monte Carlo, so the first row's outcome is every reward after it.
    outcomes = [outcome for _, outcome, _, _, _ in stored]
    assert outcomes == pytest.approx([1.0, 0.75])
    assert [terminal for *_, terminal in stored] == [False, True]
    assert eps.open_row[0] is None and eps.episode[0] == []


def test_a_cut_off_episode_is_dropped_rather_than_scored_as_worthless() -> None:
    eps = _episodes()
    eps.record(0, _row(1.0), block=0)
    eps.accrued[0] = 0.25
    eps.record(0, _row(2.0), block=0)
    eps.accrued[0] = 0.75

    assert eps.close(0, terminal=False) == []
    assert eps.dropped == 1
    # And it is forgotten either way, or the next episode inherits its transitions.
    assert eps.open_row[0] is None and eps.episode[0] == []
    assert float(eps.accrued[0]) == 0.0


def test_the_env_that_is_re_dealing_records_nothing() -> None:
    """The step after `done` carries an action the env discards."""
    eps = _episodes()
    eps.blocked[0] = True
    eps.record(0, _row(1.0), block=0)
    eps.count(0, block=3)
    assert eps.open_row[0] is None
    assert int(eps.blocks.sum()) == 0


def _learner(repartition: bool) -> tool.Learner:
    return tool.Learner(
        TINY_GROUPS,
        _episodes(),
        lambda rows: np.zeros(len(rows), dtype=np.float32),
        by_value(TINY_GROUPS),
        np.random.default_rng(0),
        repartition,
    )


def test_a_solver_answered_decision_is_counted_as_a_decision() -> None:
    """It is offered only where nothing else plays, and it is still a turn taken."""
    learner = _learner(repartition=True)
    agent = learner.agent
    assert agent.repartition_macro is not None

    legal = np.zeros(agent.n_macros, dtype=bool)
    legal[agent.repartition_macro] = True
    legal[agent.draw_macro] = True
    # `choose` short-circuits before it reads the observation, which is what makes
    # the macro answerable with no state at all.
    assert learner.choose({}, 0, legal) == agent.repartition_macro

    blocks = learner.episodes.blocks
    assert int(blocks[tool.BLOCKS.index("repart")]) == 1
    # `decisions` is `blocks.sum()`, so this is also what stops every other rate
    # being a rate over the decisions the solver was not reached on.
    assert int(blocks.sum()) == 1
    # No afterstate was stored: there is no row for V to fit.
    assert learner.episodes.open_row[0] is None


def test_the_repartition_block_is_only_reachable_where_the_macro_exists() -> None:
    """With the macro off, nothing maps to its block and nothing counts into it."""
    learner = _learner(repartition=False)
    assert learner.agent.repartition_macro is None
    assert not (learner.block == tool.BLOCKS.index("repart")).any()


def test_a_ranked_decision_records_the_afterstate_it_chose() -> None:
    """The ordinary path, so the short-circuit above is not the only one tested."""
    learner = _learner(repartition=False)
    agent = learner.agent
    agent.choose = first_legal
    legal = np.zeros(agent.n_macros, dtype=bool)
    legal[agent.end_macro] = True
    legal[agent.draw_macro] = True

    from rummi.env.numpy.deal import reset
    from rummi.env.observation import encode

    obs = encode(reset(TINY_GROUPS, 1, seed=4))
    chosen = learner.choose(obs, 0, legal)
    assert chosen in (agent.end_macro, agent.draw_macro)
    assert learner.episodes.open_row[0] is not None
    assert int(learner.episodes.blocks.sum()) == 1
