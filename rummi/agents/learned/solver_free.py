"""Both halves learned: a value net ranks the ordinary moves, a picker builds the
repartition, and no solver runs at inference.

The two pieces were measured apart and never composed. `tools/train_afterstate.py`
learns V over afterstates from outcomes alone -- no imitation past a five-update
critic warmup -- and playing its argmax reaches the heuristic band. The two-phase
template picker (`two_phase_net.py`) is cloned from CP-SAT's own solutions and RL
fine-tuned, and answering the `REPARTITION` macro with it recovers ~90% of the
solver's points. `frugal` is the pair the other way round: a hand-written chooser
and a real solve.

So this is the one cell neither run filled -- learned chooser, learned constructor
-- and it is a cost claim as much as a strength one, since the solve is the only
millisecond-scale thing either agent does.

Two things it does *not* claim. Nothing here is learned from scratch by RL: the
constructor is a clone of CP-SAT and the chooser's first rollouts are the
teacher's. And the *gate* is still the rules' -- `MacroAgent.legal_macros` decides
where a repartition is even asked for, exactly as it does for the solver, so the
arms differ in the backend and in nothing else.
"""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

import numpy as np

from rummi.agents.base import Observation, table
from rummi.agents.macro import Choose, MacroAgent
from rummi.rules.config import RummiConfig
from rummi.rules.observation import MICRO_COUNT

if TYPE_CHECKING:
    from rummi.agents.learned.afterstate_net import Value
    from rummi.agents.learned.two_phase_net import TwoPhaseScorer


class PickerMacroAgent(MacroAgent):
    """`MacroAgent` with the stuck-state solve answered by the two-phase decode.

    `choose` is the caller's, because what this replaces is the *backend*: a
    decode that comes back invalid, plays nothing, or overruns the turn's micro
    budget returns no actions, which is the same fall-through to `END_TURN`/`DRAW`
    that a declining solve produces.
    """

    name = "picker"

    def __init__(
        self,
        cfg: RummiConfig,
        scorer: TwoPhaseScorer,
        choose: Choose | None = None,
        beam: int = 1,
        monotone: bool = True,
    ) -> None:
        super().__init__(cfg, choose=choose, repartition=True)
        self.scorer = scorer
        self.beam = beam
        self.monotone = monotone
        self.asked = 0
        self.answered = 0

    def _repartition(self, obs: Observation, env: int) -> list[int]:
        from rummi.agents.learned.two_phase_net import decode_two_phase
        from rummi.solver.to_actions import plan

        cfg = self.cfg
        board = np.asarray(table(obs)[env])
        rack = np.asarray(obs["rack"][env]).astype(np.int64)
        self.asked += 1
        found = decode_two_phase(cfg, self.scorer, rack, board, self.beam, self.monotone)
        if found is None or found.tiles_played < 1:
            return []
        actions = plan(cfg, board, list(found.sets), found.played)
        spent = int(np.asarray(obs["scalars"])[env, MICRO_COUNT])
        if len(actions) > cfg.max_micro_per_turn - spent:
            return []
        actions.pop()  # dropped for the reason `expand` drops it from every macro
        self.answered += 1
        return actions


class SolverFreeAgent(PickerMacroAgent):
    """The composition: argmax over learned afterstate values, picker underneath.

    Torch is imported when one is built, not when the package is -- the rule
    `clone.py` and `optimal` follow for their own extras. :func:`solver_free`
    builds one from two checkpoints, which is the usual way in.
    """

    name = "solver_free"

    def __init__(
        self,
        cfg: RummiConfig,
        scorer: TwoPhaseScorer,
        value_of: Value,
        beam: int = 1,
        monotone: bool = True,
    ) -> None:
        super().__init__(cfg, scorer, beam=beam, monotone=monotone)
        from rummi.agents.learned.afterstate_net import argmax_chooser

        # Bound after the base class, because the chooser reads back the macro
        # layout it is ranking -- `repartition_macro` above all, which it takes
        # rather than scores.
        self.choose = argmax_chooser(cfg, self, value_of)


def load_picker(cfg: RummiConfig, path: pathlib.Path) -> tuple[TwoPhaseScorer, bool]:
    """The picker's scorer, and the mask order it was trained under.

    The order comes back beside the net because a decode run under the other one
    is a different search over the same weights, and no caller should have to
    remember which flag a checkpoint was produced with.
    """
    import torch

    from rummi.agents.learned.two_phase_net import TwoPhaseScorer, two_phase_from_checkpoint

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    scorer = TwoPhaseScorer(two_phase_from_checkpoint(cfg, checkpoint))
    return scorer, bool(checkpoint["monotone"])


def solver_free(
    cfg: RummiConfig,
    value: pathlib.Path,
    picker: pathlib.Path,
    beam: int = 1,
) -> SolverFreeAgent:
    """Both checkpoints off disk and the agent that runs them."""
    from rummi.agents.learned.afterstate_net import load_value_net, value_fn

    net, _ = load_value_net(value, cfg)
    scorer, monotone = load_picker(cfg, picker)
    return SolverFreeAgent(cfg, scorer, value_fn(net), beam=beam, monotone=monotone)
