"""The analytic afterstate must be the observation the env actually reports.

`rummi/agents/learned/afterstate.py` predicts a macro's successor without stepping
anything, by mirroring where `MacroAgent.expand` puts the tiles and where
`to_actions.plan` puts the sets. A mirror can drift, and the drift is silent: every
field keeps its shape, and a value net trained on a table whose slots are one
permutation out would simply learn less. So the contract is measured against real
play -- a game per env against `greedy`, every prediction held to the observation
the next decision in the same turn produces -- with the per-family unit tests below
it to say *which* macro broke when it does.
"""

from __future__ import annotations

import numpy as np
import pytest

from rummi.agents.learned.afterstate import (
    KIND_DRAW,
    KIND_END,
    KIND_PLAY,
    afterstate_batch,
    afterstate_dim,
    afterstate_obs,
)
from rummi.agents.macro import (
    MacroAgent,
    by_value,
    extend_offset,
    set_templates,
    steal_offset,
)
from rummi.env.observation import encode
from rummi.rules.config import STANDARD, TINY_GROUPS, RummiConfig
from rummi.rules.encoding import kind_of, kinds_to_counts
from tests.conftest import play, state_with

pytest.importorskip("gymnasium")

CHECKED = (
    "rack",
    "workbench",
    "placed_this_turn",
    "unseen",
    "slot_features",
    "rack_sizes",
    "melded",
    "scalars",
    "table_sets",
)
"""Every field the prediction claims, `table_sets` and `scalars` included.

Wider than the fields a network reads: `micro_count` and the slot layout are what a
mismatch shows up in first, and they are the two the mirror is most able to get
wrong."""


def _same(predicted: dict[str, np.ndarray], actual: dict[str, np.ndarray], where: str) -> None:
    for name in CHECKED:
        np.testing.assert_array_equal(
            np.asarray(predicted[name][0]),
            np.asarray(actual[name][0]),
            err_msg=f"{name} after {where}",
        )


def _template(cfg: RummiConfig, kinds: list[int]) -> int:
    """The macro index of the template holding exactly ``kinds``."""
    rows = set_templates(cfg)
    match = np.flatnonzero((rows == kinds_to_counts(cfg, kinds)).all(-1))
    assert match.size == 1, f"no single template for {kinds}"
    return int(match[0])


def _expand_and_compare(
    cfg: RummiConfig, state, agent: MacroAgent, macro: int, where: str
) -> None:
    """Predict, then actually play the expansion, then hold the two together."""
    obs = encode(state)
    assert agent.legal_macros(obs, 0)[macro], f"{where} is not legal in this state"
    predicted, kinds = afterstate_obs(cfg, obs, 0, [macro], agent)
    assert kinds.tolist() == [KIND_PLAY]

    actions = agent.expand(obs, 0, macro)
    assert actions, f"{where} expanded to nothing"
    play(state, actions)
    _same(predicted, encode(state), where)


def test_a_new_set_from_the_rack_lands_where_the_prediction_says() -> None:
    cfg = STANDARD
    run = [kind_of(cfg, 0, n) for n in (5, 6, 7)]
    state = state_with(cfg, rack=[*run, kind_of(cfg, 1, 9)], melded=True)
    _expand_and_compare(cfg, state, MacroAgent(cfg), _template(cfg, run), "a new run")


def test_a_joker_stands_in_for_the_tile_the_rack_is_missing() -> None:
    cfg = STANDARD
    run = [kind_of(cfg, 0, n) for n in (5, 6, 7)]
    # The rack holds two of the three, so the template lands with a joker in it and
    # the joker is what leaves the rack.
    state = state_with(cfg, rack=[run[0], run[2], cfg.joker_kind], melded=True)
    _expand_and_compare(cfg, state, MacroAgent(cfg), _template(cfg, run), "a joker set")


def test_a_lay_off_grows_the_slot_that_takes_it() -> None:
    cfg = STANDARD
    run = [kind_of(cfg, 0, n) for n in (5, 6, 7)]
    state = state_with(cfg, rack=[kind_of(cfg, 0, 8)], table=[run], melded=True)
    _expand_and_compare(
        cfg, state, MacroAgent(cfg), extend_offset(cfg) + kind_of(cfg, 0, 8), "an EXTEND"
    )


def test_a_steal_leaves_the_donor_short_and_the_new_set_whole() -> None:
    cfg = STANDARD
    donor = [kind_of(cfg, 0, n) for n in (5, 6, 7, 8)]
    # Three eights of other colours: the group needs a fourth, and the run can spare
    # its top tile.
    group = [kind_of(cfg, c, 8) for c in (1, 2, 3)]
    state = state_with(cfg, rack=group, table=[donor], melded=True)
    macro = steal_offset(cfg) + _template(cfg, [*group, kind_of(cfg, 0, 8)])
    _expand_and_compare(cfg, state, MacroAgent(cfg), macro, "a STEAL")


def test_ending_and_drawing_carry_the_state_they_were_offered_in() -> None:
    cfg = STANDARD
    agent = MacroAgent(cfg)
    state = state_with(cfg, rack=[kind_of(cfg, 0, n) for n in (5, 6, 7)], melded=True)
    obs = encode(state)

    fields, kinds = afterstate_obs(cfg, obs, 0, [agent.end_macro, agent.draw_macro], agent)
    assert kinds.tolist() == [KIND_END, KIND_DRAW]
    for row in range(2):
        for name in CHECKED:
            np.testing.assert_array_equal(
                np.asarray(fields[name][row]),
                np.asarray(obs[name][0]),
                err_msg=f"{name} under a committing macro",
            )

    rows = afterstate_batch(cfg, obs, 0, [agent.end_macro, agent.draw_macro], agent)
    assert rows.shape == (2, afterstate_dim(cfg))
    # Identical states, so only the flag can tell the two apart.
    np.testing.assert_array_equal(rows[0, :-3], rows[1, :-3])
    np.testing.assert_array_equal(rows[:, -3:], np.eye(3, dtype=np.float32)[[KIND_END, KIND_DRAW]])


def test_a_batch_of_macros_is_each_one_scored_alone() -> None:
    cfg = STANDARD
    agent = MacroAgent(cfg)
    run = [kind_of(cfg, 0, n) for n in (5, 6, 7)]
    state = state_with(cfg, rack=[*run, kind_of(cfg, 0, 8)], table=[run], melded=True)
    obs = encode(state)

    legal = np.flatnonzero(agent.legal_macros(obs, 0)).tolist()
    assert len(legal) > 3, "this state was meant to offer a choice"
    batch = afterstate_batch(cfg, obs, 0, legal, agent)
    for i, macro in enumerate(legal):
        np.testing.assert_array_equal(
            batch[i], afterstate_batch(cfg, obs, 0, [macro], agent)[0],
            err_msg=f"macro {macro} scored differently in a batch",
        )


def test_the_repartition_macro_is_refused_rather_than_guessed() -> None:
    cfg = STANDARD
    agent = MacroAgent(cfg, repartition=True)
    state = state_with(cfg, rack=[kind_of(cfg, 0, n) for n in (5, 6, 7)], melded=True)
    with pytest.raises(ValueError, match="REPARTITION"):
        afterstate_obs(cfg, encode(state), 0, [agent.repartition_macro], agent)


@pytest.mark.parametrize(
    ("cfg", "n_envs", "steps", "least"),
    [(STANDARD, 8, 500, 300), (TINY_GROUPS, 8, 500, 60)],
)
def test_the_prediction_matches_the_env_across_real_games(
    cfg: RummiConfig, n_envs: int, steps: int, least: int
) -> None:
    """Play against `greedy` and hold every prediction to what the env reports.

    Random play never melds, so it never reaches a second decision inside a turn --
    which is the only place this can be checked at all. `by_value` drives for the
    same reason `greedy` does elsewhere in the suite.
    """
    from rummi.env.fixed_opponent import FixedOpponentEnv

    env = FixedOpponentEnv(num_envs=n_envs, cfg=cfg, seed=7, opponent="greedy")
    agent = MacroAgent(cfg, choose=by_value(cfg))
    rank = by_value(cfg)
    # Per env: the fields predicted for the macro just chosen, and the turn it
    # belongs to. A turn is the seat plus the env's turn counter, read off the state
    # rather than inferred -- a prediction is only claimed about the *next decision
    # in the same turn*, and an abandoned expansion or a committed turn is neither.
    pending: dict[int, tuple[dict[str, np.ndarray], tuple[int, int]]] = {}
    checked = 0

    def turn_of(e: int) -> tuple[int, int]:
        return int(env.state.current[e]), int(env.state.turn_count[e])

    def probe(obs, e: int, legal: np.ndarray) -> int:
        nonlocal checked
        held = pending.pop(e, None)
        if held is not None and held[1] == turn_of(e):
            _same(held[0], {k: np.asarray(v)[e][None] for k, v in obs.items()}, "real play")
            checked += 1
        macro = rank(obs, e, legal)
        if macro not in (agent.end_macro, agent.draw_macro):
            pending[e] = (afterstate_obs(cfg, obs, e, [macro], agent)[0], turn_of(e))
        return macro

    agent.choose = probe
    obs, info = env.reset()
    agent.reset(n_envs)
    for _ in range(steps):
        actions = agent.act(obs, np.asarray(info["action_mask"]))
        obs, _, term, trunc, info = env.step(actions)
        for e in np.flatnonzero(np.asarray(term) | np.asarray(trunc)):
            pending.pop(int(e), None)
    env.close()

    assert checked >= least, f"only {checked} predictions were checked; nothing was proved"
