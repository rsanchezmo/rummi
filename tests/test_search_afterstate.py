"""The turn the search imagines must be the turn the env plays.

`tools/search_afterstate.py` completes a turn by recursion over `afterstate_obs`:
the observation predicted for one macro is handed back to
:meth:`MacroAgent.legal_macros`, and *its* macros are predicted from that. So the
prediction is now load-bearing twice over -- once as a row the network scores, and
once as the state the next question is asked of. `tests/test_afterstate.py` holds
one step of it to the env; a chain can drift where a step does not, because an
error at depth 1 is invisible until depth 2 asks the wrong question of it.

So this plays real games with the search **committed** to its plan: the completion
is decided once, replayed macro by macro, and every real decision inside the turn
is compared field-exact -- the same fields the drift test compares -- against the
view the search predicted for it. The scoring agent re-searches at every decision
instead, which cannot check anything past depth 1.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("gymnasium")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import search_afterstate as tool

from rummi.agents.learned.afterstate import afterstate_dim
from rummi.agents.macro import MacroAgent, extend_offset, steal_offset
from rummi.env.fixed_opponent import FixedOpponentEnv
from rummi.rules.config import STANDARD, RummiConfig
from tests.test_afterstate import CHECKED


def _linear_value(cfg: RummiConfig, seed: int) -> tool.Value:
    """A deterministic, non-constant V. What is under test is the simulation.

    A trained net would tie the test to a checkpoint and to whatever that
    checkpoint happens to prefer; random weights rank positions arbitrarily, which
    is what drives the search down continuations a good policy would never take.
    """
    weights = np.random.default_rng(seed).normal(size=afterstate_dim(cfg)).astype(np.float32)
    return lambda rows: np.tanh(rows @ weights)


def _family(cfg: RummiConfig, macro: int) -> str:
    if macro >= steal_offset(cfg):
        return "steal"
    if macro >= extend_offset(cfg):
        return "extend"
    return "new_set"


def _same(predicted: dict[str, np.ndarray], obs: dict[str, np.ndarray], env: int) -> None:
    for name in CHECKED:
        np.testing.assert_array_equal(
            np.asarray(predicted[name][0]),
            np.asarray(obs[name])[env],
            err_msg=f"{name} predicted by the search",
        )


def test_the_committed_turn_is_the_turn_the_env_plays() -> None:
    cfg = STANDARD
    n_envs, steps = 8, 400
    env = FixedOpponentEnv(num_envs=n_envs, cfg=cfg, seed=17, opponent="greedy")
    agent = MacroAgent(cfg)
    search = tool.TurnSearch(cfg, agent, _linear_value(cfg, 0), beam=1)

    # Per env: the rest of the committed completion, and the one prediction waiting
    # to be checked. A turn is the seat plus the env's turn counter, read off the
    # state -- a prediction is claimed only about the next decision in the *same*
    # turn, and a committed turn or an abandoned expansion is neither.
    plan: dict[int, list[tuple[int, dict[str, np.ndarray]]]] = {}
    pending: dict[int, tuple[dict[str, np.ndarray], tuple[int, int], str, bool]] = {}
    verified: Counter[str] = Counter()
    chained = 0
    lengths: Counter[int] = Counter()

    def turn_of(e: int) -> tuple[int, int]:
        return int(env.state.current[e]), int(env.state.turn_count[e])

    def choose(obs, e: int, legal: np.ndarray) -> int:
        nonlocal chained
        key = turn_of(e)
        held = pending.pop(e, None)
        following = held is not None and held[1] == key
        if following:
            assert held is not None
            _same(held[0], obs, e)
            verified[held[2]] += 1
            chained += int(held[3])

        queued = plan.get(e) if following else None
        if queued:
            macro, view = queued.pop(0)
            replayed = True
        else:
            plan.pop(e, None)
            macro, completion = search.choose(obs, e, legal)
            view, replayed = None, False
            if completion is not None:
                lengths[len(completion.macros)] += 1
                rest = list(zip(completion.macros, completion.views, strict=True))
                macro, view = rest.pop(0)
                plan[e] = rest

        assert legal[macro], "the search offered a macro the mask does not"
        if view is not None:
            pending[e] = (view, key, _family(cfg, macro), replayed)
        return macro

    agent.choose = choose
    obs, info = env.reset()
    agent.reset(n_envs)
    for _ in range(steps):
        actions = agent.act(obs, np.asarray(info["action_mask"]))
        obs, _, term, trunc, info = env.step(actions)
        for e in np.flatnonzero(np.asarray(term) | np.asarray(trunc)):
            pending.pop(int(e), None)
            plan.pop(int(e), None)
    env.close()

    total = sum(verified.values())
    assert total >= 300, f"only {total} predictions were checked; nothing was proved"
    assert chained >= 40, f"only {chained} of them were past depth 1: the chain is untested"
    for name in ("new_set", "extend", "steal"):
        assert verified[name] >= 5, f"{name} continuations were never exercised: {verified}"
    assert max(lengths) >= 3, f"no completion ran past two macros: {sorted(lengths.items())}"


def test_a_flat_value_still_plays_legally_and_terminates() -> None:
    """With every afterstate scored alike the search has nothing to rank by.

    It must still stop, and the micro budget is the only thing that makes it: every
    tile-playing macro spends at least one, and a continuation that would leave none
    for the trailing `END_TURN` is not offered. It must also still return a macro the
    mask allows -- a search that walks off the legal set in simulation would come
    back with one that does not exist in the real state.
    """
    cfg = STANDARD
    n_envs, steps = 8, 300
    env = FixedOpponentEnv(num_envs=n_envs, cfg=cfg, seed=5, opponent="greedy")
    agent = MacroAgent(cfg)
    search = tool.TurnSearch(cfg, agent, lambda rows: np.zeros(len(rows), np.float32))
    decisions, longest = 0, 0

    def choose(obs, e: int, legal: np.ndarray) -> int:
        nonlocal decisions, longest
        macro, completion = search.choose(obs, e, legal)
        # In range as well as legal: `legal[-1]` is DRAW's column and always true,
        # so a sentinel escaping the ranking would pass a mask check on its own.
        assert 0 <= macro < agent.n_macros, "the search returned no macro at all"
        assert legal[macro], "the search offered a macro the mask does not"
        decisions += 1
        if completion is not None:
            longest = max(longest, len(completion.macros))
            for played in completion.macros:
                assert 0 <= played < agent.n_macros
        return macro

    agent.choose = choose
    obs, info = env.reset()
    agent.reset(n_envs)
    for _ in range(steps):
        actions = agent.act(obs, np.asarray(info["action_mask"]))
        obs, _, _, _, info = env.step(actions)
    env.close()

    assert decisions >= 200, f"only {decisions} decisions; the game barely moved"
    assert longest <= cfg.max_micro_per_turn, "a completion outran the budget that bounds it"


def test_a_value_that_ranks_nothing_still_returns_a_legal_macro() -> None:
    """A diverged net scores every row NaN, and NaN satisfies no comparison.

    The ranking is a strict-improvement loop, so nothing is ever chosen and whatever
    it was seeded with is what comes back. That has to be one of the options: `-1`
    reads as DRAW to a mask check and as the last template to `expand`, so a search
    that returned it would play a real set nobody chose.
    """
    cfg = STANDARD
    agent = MacroAgent(cfg)
    search = tool.TurnSearch(cfg, agent, lambda rows: np.full(len(rows), np.nan, np.float32))
    env = FixedOpponentEnv(num_envs=4, cfg=cfg, seed=7, opponent="greedy")
    decided = 0

    def choose(obs, e: int, legal: np.ndarray) -> int:
        nonlocal decided
        macro, _ = search.choose(obs, e, legal)
        assert 0 <= macro < agent.n_macros, f"the ranking came back with {macro}"
        assert legal[macro], "the search offered a macro the mask does not"
        decided += 1
        return macro

    agent.choose = choose
    obs, info = env.reset()
    agent.reset(4)
    try:
        for _ in range(60):
            obs, _, _, _, info = env.step(agent.act(obs, np.asarray(info["action_mask"])))
    finally:
        env.close()
    assert decided > 20, "no decision was reached, so nothing was tested"
