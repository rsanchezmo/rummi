"""A whole turn constructed out of primitive actions, one step at a time.

The template picker (`learned/repartition_net.py`) established the recipe: a
construction is a *sequence*, every step is legal by mask, the search keeps several
candidates and ranks the finished ones by what they shed. This is that recipe over
the env's own vocabulary -- `PLACE`, `PICK`, `DISSOLVE`, `ASSIGN`, `END_TURN` --
so the only thing that differs from the picker is what the decoder emits.

Legality comes from the engine rather than from arithmetic over count vectors:
`learned/turn_sim.py` rebuilds the position as a `BatchState`, so the mask at every
step of the search is :func:`~rummi.env.numpy.masks.legal_actions` itself and the
network reads the observation the env would have handed it. A finished hypothesis
is therefore a turn the env will accept, action for action.

Two properties of the space shape the loop.

**`END_TURN` is the only exit worth taking.** It is masked until the workbench is
empty, the table whole and something played, so a hypothesis that reaches it *is* a
committable turn -- exactly as `STOP` is masked until the picker's table is covered.
`DRAW` is legal everywhere and reverts the turn, so a hypothesis that emits it has
declined; it is dropped rather than followed, and a position where every hypothesis
declines returns nothing.

**Finished hypotheses are ranked by tiles committed, not by likelihood.** Every one
of them is legal, so likelihood has nothing left to discriminate -- the picker
measured this and it carries. Log-probability is the tie-break, which keeps the
model's own preference among equal turns.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np
import torch

from rummi.agents.base import Observation, PlanningAgent
from rummi.agents.learned.torch_net import TorchPolicy
from rummi.agents.learned.turn_sim import TurnStart, to_state
from rummi.agents.macro import MacroAgent, by_value
from rummi.env.numpy.engine import step as engine_step
from rummi.env.numpy.masks import legal_actions
from rummi.env.numpy.sets import summarize
from rummi.env.observation import encode
from rummi.rules.config import RummiConfig


Score = Callable[[Observation, np.ndarray, np.ndarray], np.ndarray]
"""`(obs, mask, group) -> (rows, n_actions)` logits. `group` says which position
each row is decoding, which only a teacher needs -- see :class:`Scorer`."""


@dataclass(frozen=True, slots=True)
class Decoded:
    """One position's answer: the turn to play, or nothing at all."""

    actions: tuple[int, ...]
    """Every primitive of the turn, `END_TURN` last. Empty means decline."""
    tiles: int
    """Tiles this decode moved out of the rack, which is the reward."""
    logp: float
    depth: int = 0
    """Primitives the deepest hypothesis of this position reached, committed or not."""
    declined: bool = False
    """Some hypothesis emitted `DRAW`. With the budget still open that is a decision
    to decline; at the budget it is all the mask has left, which is running out."""

    @property
    def plays(self) -> bool:
        return bool(self.actions) and self.tiles >= 1


DECLINED = Decoded((), 0, 0.0)


class Scorer:
    """A trained policy behind the one call the decode makes, with no tape.

    `group` says which position each row belongs to. A network has no use for it --
    every row is a state and states are what it reads -- but a *teacher* does, and
    that is what makes a teacher-driven decode testable: the test in
    `tests/test_primitive_turn.py` hands back the recorded action, which an untrained
    net reaches far too rarely to say anything about the loop it came out of.
    """

    def __init__(self, net: TorchPolicy) -> None:
        self.net = net.eval()

    def __call__(
        self, obs: Observation, mask: np.ndarray, group: np.ndarray
    ) -> np.ndarray:
        with torch.no_grad():
            logits, _ = self.net(obs, torch.from_numpy(mask))
        return logits.numpy()


def log_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(-1, keepdims=True))


def _best_per_group(groups: np.ndarray, scores: np.ndarray, keep: int) -> np.ndarray:
    """Indices of the `keep` highest-scoring rows of each group.

    One lexsort rather than a pass per position: a beam over a whole vector env's
    turn boundaries has as many groups as envs, and the decode is already paying a
    Python loop per row for the mask.
    """
    order = np.lexsort((-scores, groups))
    sorted_groups = groups[order]
    first = np.flatnonzero(np.r_[True, sorted_groups[1:] != sorted_groups[:-1]])
    counts = np.diff(np.r_[first, len(order)])
    rank = np.arange(len(order)) - np.repeat(first, counts)
    return order[rank < keep]


def decode_turns(
    cfg: RummiConfig,
    score: Score,
    starts: TurnStart,
    beam: int = 1,
    max_depth: int | None = None,
) -> list[Decoded]:
    """A turn per position, searched `beam` hypotheses wide. Beam 1 is greedy.

    Every position advances in lockstep on one `BatchState`, so a batch of turn
    boundaries costs one forward pass per depth rather than one per position --
    which is what makes this affordable as an agent rather than only as a probe.

    The depth cap is the turn's own micro budget and needs no arithmetic here: a
    spent budget masks everything but `DRAW`, so a hypothesis that overruns dies on
    the mask exactly as the env would kill it.
    """
    n = len(starts)
    state = to_state(cfg, starts)
    group = np.arange(n)
    logp = np.zeros(n)
    trail: list[tuple[int, ...]] = [() for _ in range(n)]
    opening = starts.placed.sum(-1).astype(np.int64)
    best: list[Decoded] = [DECLINED] * n
    # How far each position got and whether anything in it declined: the two things
    # that separate a decode that chose not to play from one that wandered until the
    # mask stopped it, which read the same from the empty plan alone.
    reached = np.zeros(n, dtype=np.int64)
    declined = np.zeros(n, dtype=bool)

    budget = int((cfg.max_micro_per_turn - starts.micro_count).max()) if n else 0
    for depth in range(budget if max_depth is None else min(budget, max_depth)):
        if state.batch_size == 0:
            break
        summary = summarize(cfg, state.table_sets)
        mask = legal_actions(state, summary)
        rows_logp = log_softmax(score(encode(state, summary), mask, group))
        placed = state.placed_rack.sum(-1).astype(np.int64)

        parents: list[int] = []
        chosen: list[int] = []
        totals: list[float] = []
        for row in range(state.batch_size):
            options = np.flatnonzero(mask[row])
            ranked = options[np.argsort(-rows_logp[row, options])][:beam]
            for action in ranked.tolist():
                total = logp[row] + float(rows_logp[row, action])
                if action == cfg.draw_action:
                    declined[int(group[row])] = True
                    continue  # declining is always on offer and always worth zero
                if action == cfg.end_turn_action:
                    which = int(group[row])
                    tiles = int(placed[row] - opening[which])
                    standing = best[which]
                    if (tiles, total) > (standing.tiles, standing.logp):
                        best[which] = Decoded((*trail[row], action), tiles, total)
                    continue
                parents.append(row)
                chosen.append(action)
                totals.append(total)

        if not parents:
            break
        keep = _best_per_group(group[parents], np.asarray(totals), beam)
        survivors = np.asarray(parents)[keep]
        actions = np.asarray(chosen)[keep]
        trail = [
            (*trail[row], action)
            for row, action in zip(survivors.tolist(), actions.tolist(), strict=True)
        ]
        logp = np.asarray(totals)[keep]
        group = group[survivors]
        reached[group] = depth + 1
        state = state.select(survivors)
        # The selected rows carry their parents' mask unchanged, so the engine's own
        # legality check costs nothing beyond a gather.
        engine_step(state, actions, mask[survivors])

    return [
        replace(row, depth=int(reached[i]), declined=bool(declined[i]))
        for i, row in enumerate(best)
    ]


class PrimitiveTurnAgent(PlanningAgent):
    """Every turn decoded out of primitives, with no macro vocabulary anywhere.

    The honest arm: no template list, no solver, no gate deciding when to reach for
    one. At each turn boundary the network constructs a whole turn or declines it,
    and declining is `DRAW` -- which is what the primitive action space costs when
    nothing in it is chosen well.
    """

    name = "primitive-turn"

    def __init__(self, cfg: RummiConfig, scorer: Score, beam: int = 1) -> None:
        super().__init__(cfg)
        self.scorer = scorer
        self.beam = beam
        self.turns = 0
        self.committed = 0
        self.tiles = 0

    def plan_batch(self, obs: Observation, envs: np.ndarray) -> list[list[int]]:
        found = decode_turns(self.cfg, self.scorer, TurnStart.from_obs(obs, envs), self.beam)
        self.turns += len(found)
        self.committed += sum(1 for row in found if row.plays)
        self.tiles += sum(row.tiles for row in found)
        return [list(row.actions) if row.plays else [] for row in found]


class PrimitiveRepartition(MacroAgent):
    """`by_value`, with the stuck-state solve answered by a primitive decode.

    The one-variable cell against `train_repartition.NeuralRepartition`: same gate,
    same heuristic around it, same fall-through when the answer plays nothing. Only
    the vocabulary the decoder emits differs, so a difference in score is a
    difference between the two action spaces and nothing else.
    """

    name = "primitive-repartition"

    def __init__(self, cfg: RummiConfig, scorer: Score, beam: int = 1) -> None:
        super().__init__(cfg, choose=by_value(cfg), repartition=True)
        self.scorer = scorer
        self.beam = beam
        self.asked = 0
        self.answered = 0

    def _repartition(self, obs: Observation, env: int) -> list[int]:
        self.asked += 1
        found = decode_turns(
            self.cfg, self.scorer, TurnStart.from_obs(obs, env), self.beam
        )[0]
        if not found.plays:
            return []
        self.answered += 1
        # Dropped for the reason `MacroAgent.expand` drops it from every macro: the
        # table is whole, so ending is legal, but whether to end is the next decision.
        return list(found.actions[:-1])
