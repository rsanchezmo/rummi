"""The solver-free composition: a learned chooser over a learned constructor.

Two claims, and neither is about how well it plays -- that is what
`tools/eval_solver_free.py` scores. What has to hold whatever the weights say is
that the agent stays inside the rules: every action it proposes is one the mask
allows, and a turn is committed whole or not at all, because `REPARTITION` is the
longest expansion in the macro space and overrunning the micro budget mid-rebuild
leaves the env offering only `DRAW` -- which reverts everything the expansion did.

The weights are random on purpose. Checkpoints are not in the repo, and a test
pinned to one would assert what that checkpoint happens to prefer. What drives the
game instead is a hand-written value: fewest tiles left in the rack, which plays
whatever it can and then ends -- enough to meld, fill the table and reach the stuck
states the picker exists for, which uniform-random play never does.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("gymnasium")

import torch

from rummi.agents.learned.afterstate_net import Value
from rummi.agents.learned.solver_free import PickerMacroAgent, SolverFreeAgent
from rummi.agents.learned.two_phase_net import TwoPhaseNet, TwoPhaseScorer, stop_break
from rummi.agents.macro import MacroAgent, by_value
from rummi.env.fixed_opponent import FixedOpponentEnv
from rummi.env.numpy.sets import evaluate_slots
from rummi.evaluate.protocol import SUITE_BY_NAME
from rummi.rules.config import STANDARD, RummiConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import eval_solver_free as tool

ENVS = 8
STEPS = 500


def picker(cfg: RummiConfig, seed: int = 0) -> TwoPhaseScorer:
    """An untrained decoder. Every finished decode is legal by construction, so
    what the weights change is which repartition is found, not whether it is one."""
    torch.manual_seed(seed)
    return TwoPhaseScorer(TwoPhaseNet(cfg, hidden=32, key=16))


def fewest_tiles_left(cfg: RummiConfig) -> Value:
    """Rank an afterstate by the rack it leaves. `rack` is the first block of the
    row (`features.FEATURE_FIELDS`), and `END_TURN` and `DRAW` carry the rack
    unplayed -- so this plays whatever it can and then commits."""
    return lambda rows: -rows[:, : cfg.n_kinds].sum(-1)


class NeverBreaks(TwoPhaseScorer):
    """Phase A stopping before it dissolves anything.

    The decode is still valid -- keeping every set where it is always is -- and it
    frees nothing, so the cover has nothing to place and the repartition plays no
    tile. That is the one failure the backend has to report as *no plan*.
    """

    def __init__(self, cfg: RummiConfig, net: TwoPhaseNet) -> None:
        super().__init__(net)
        self.stop = stop_break(cfg)

    def brk(
        self, state: np.ndarray, static: np.ndarray, dynamic: np.ndarray, legal: np.ndarray
    ) -> np.ndarray:
        out = np.full(legal.shape, -1e9, dtype=np.float32)
        out[:, self.stop] = 0.0
        return out


def play(agent: MacroAgent, cfg: RummiConfig, steps: int = STEPS) -> list[np.ndarray]:
    """One batch of games against `greedy`, returning every action taken.

    Legality is asserted here rather than left to the env: `validate_actions`
    raises on an illegal action, but the mask is what an agent is contractually
    reading, so the check belongs beside the choice.
    """
    env = FixedOpponentEnv(num_envs=ENVS, cfg=cfg, seed=3, opponent="greedy")
    obs, info = env.reset()
    agent.reset(ENVS)
    taken: list[np.ndarray] = []
    try:
        for _ in range(steps):
            mask = np.asarray(info["action_mask"])
            actions = np.asarray(agent.act(obs, mask))
            assert mask[np.arange(ENVS), actions].all(), "proposed a masked-out action"
            taken.append(actions.copy())
            obs, _, _, _, info = env.step(actions)
            verdict = evaluate_slots(cfg, env.state.table_sets)
            whole = (verdict.is_valid | verdict.is_empty).all(-1)
            assert whole[env.state.micro_count == 0].all(), "broken set at a turn boundary"
    finally:
        env.close()
    return taken


def test_the_composition_plays_whole_games_inside_the_mask():
    cfg = STANDARD
    agent = SolverFreeAgent(cfg, picker(cfg), fewest_tiles_left(cfg), beam=2, monotone=False)
    taken = play(agent, cfg)

    commits = sum(int((step == cfg.end_turn_action).sum()) for step in taken)
    # Without these the run proves only that an agent which passes every turn is
    # legal, which is the trap `CLAUDE.md` names: random play never gets here.
    assert commits > 100, f"only {commits} turns were committed; nothing was proved"
    assert agent.asked > 0, "the repartition gate never fired, so the backend is untested"


def test_the_duel_refuses_a_suite_it_cannot_seat() -> None:
    """`head_to_head` seats exactly two agents, and its `--suite` choices do not.

    A three-seat suite left the third seat with no agent at all, so `act_by_seat`
    filled it with `DRAW` -- a game where one player passes for ever, reported as a
    win rate like any other. The duel has to say it cannot run instead.
    """
    # Shortened so the pre-fix behaviour -- playing the whole thing out with a seat
    # that only ever draws -- is cheap enough to assert against.
    suite = dataclasses.replace(SUITE_BY_NAME["standard-3p"], max_steps=20)
    cfg = suite.cfg
    with pytest.raises(ValueError, match="two"):
        tool.head_to_head(
            suite, MacroAgent(cfg, choose=by_value(cfg)), MacroAgent(cfg), 2, 0
        )


def test_a_decode_that_finds_nothing_plays_the_game_without_it():
    """No plan must mean *no plan*: the turn falls through to what the gate offers.

    Held to the strongest form available -- the same chooser with the backend
    removed entirely plays the identical game, action for action.
    """
    cfg = STANDARD
    blind = PickerMacroAgent(
        cfg, NeverBreaks(cfg, TwoPhaseNet(cfg, hidden=32, key=16)), choose=by_value(cfg)
    )
    with_backend = play(blind, cfg)

    assert blind.asked > 100, "the gate never fired, so the fall-through is untested"
    assert blind.answered == 0, "a decode that breaks nothing cannot play a tile"

    without = play(MacroAgent(cfg, choose=by_value(cfg)), cfg)
    assert all(np.array_equal(a, b) for a, b in zip(with_backend, without, strict=True))
