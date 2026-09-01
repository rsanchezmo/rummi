"""The autoregressive repartition space, held to the engine it has to expand into.

The whole claim of `rummi/agents/learned/repartition_net.py` is that a whole-table
rearrangement can be *constructed* one template at a time under a mask, and that
what comes out is a legal turn. Neither half is checkable by inspection: the mask
is arithmetic over count vectors, and legality is the env's own kernel. So both are
checked against the things that own them -- the mask against what the multiset can
actually pay for, and every construction against `legal_actions` on a real state,
replayed micro-action by micro-action.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

pytest.importorskip("ortools")
pytest.importorskip("torch")

import torch

from rummi.agents.base import act_on_state, table
from rummi.agents.learned.repartition_net import (
    CANDIDATE_DYNAMIC,
    MASKED,
    RepartitionNet,
    build,
    candidate_features,
    feasible,
    initial_counts,
    label_sequence,
    laid_by,
    n_actions,
    padded_sets,
    present_counts,
    state_features,
    stop_action,
    template_table,
)
from rummi.agents.macro import MacroAgent, by_value
from rummi.env.numpy.deal import reset
from rummi.env.numpy.engine import step
from rummi.env.numpy.masks import legal_actions
from rummi.env.numpy.sets import evaluate_slots
from rummi.env.observation import encode
from rummi.rules.config import STANDARD
from rummi.rules.encoding import kind_of


@dataclass(frozen=True, slots=True)
class Stuck:
    """One state the repartition macro was offered in, and CP-SAT's answer to it."""

    rack: np.ndarray
    board: np.ndarray
    sets: tuple[tuple[int, ...], ...]
    played: np.ndarray
    probe: object
    """The single-env state it came from, so an expansion can be replayed on it."""


_STATES: list[Stuck] = []


def stuck_states(wanted: int = 90) -> list[Stuck]:
    """Real states where `by_value` can only draw and CP-SAT still plays.

    Driven once for the module, because reaching the gate needs a real game --
    `docs/EXPERIMENTS.md`' point that random play never reaches melding, let alone
    a table worth rearranging, applies with full force here.
    """
    if _STATES:
        return _STATES

    from rummi.solver.ilp import solve_turn

    cfg = STANDARD
    n = 16
    agent = MacroAgent(cfg, choose=by_value(cfg), repartition=True)
    agent.reset(n)
    state = reset(cfg, n, seed=17)
    for _ in range(900):
        if len(_STATES) >= wanted:
            break
        mask = legal_actions(state)
        obs = encode(state)
        for env in range(n):
            if not agent.legal_macros(obs, env)[agent.repartition_macro]:
                continue
            board = np.asarray(table(obs)[env]).copy()
            rack = np.asarray(obs["rack"][env]).astype(np.int64)
            solution = solve_turn(cfg, rack, board, True)
            assert solution.played is not None
            _STATES.append(
                Stuck(rack, board, tuple(solution.sets), np.asarray(solution.played), state.select(env))
            )
        step(state, act_on_state(agent, state, mask), mask)
    assert len(_STATES) > 40, f"only reached {len(_STATES)} stuck states"
    return _STATES


def test_every_solver_repartition_is_expressible_and_the_mask_admits_only_what_pays():
    """Coverage of the action space, and the feasibility test underneath it.

    A solve emits sets with the jokers already materialised, so which tile one stood
    in for is gone; `label_sequence` reads it back through `TemplateTable.by_content`.
    If that ever failed on a real solution the dataset would be silently truncated to
    the easy states and nothing downstream would say so.

    Walking that same sequence is also the honest place to test the mask: these are
    the states a decode actually visits. Feasibility is a dot product against "which
    kinds are exhausted", which is what keeps the ~330-way mask out of a search, and
    the equivalence it rests on -- template counts are 0 or 1 -- is checked against
    the per-kind maximum it replaces.
    """
    cfg = STANDARD
    tt = template_table(cfg)
    stop = stop_action(cfg)
    verdicts = {"exact": 0, "relaxed": 0, "none": 0}
    checked = 0

    for row in stuck_states():
        need, avail = initial_counts(cfg, row.rack, row.board)
        sequence, verdict = label_sequence(cfg, need, avail, row.sets)
        verdicts[verdict] += 1
        if verdict == "none":
            continue
        replayed = build(cfg, need, avail, list(sequence))
        assert replayed is not None
        # A different joker resting place is allowed; a different *turn* is not.
        assert int(replayed.played.sum()) == int(row.played.sum())

        present = present_counts(cfg, row.board)[None]
        last = 0
        for position in range(len(sequence) + 1):
            step_at, before = np.array([position]), np.array([last])
            dynamic, short = candidate_features(cfg, need[None], avail[None], present, before)
            legal = feasible(cfg, need[None], avail[None], step_at, before, short)
            assert dynamic.shape == (1, n_actions(cfg), CANDIDATE_DYNAMIC)
            assert bool(legal[0, stop]) == (need.sum() == 0)
            for template in np.flatnonzero(legal[0, :-1]).tolist():
                laid = laid_by(cfg, template, avail)
                assert (laid <= avail).all(), f"template {template} cannot be paid for"
                assert int(laid.sum()) == int(tt.length[template])
            gap = np.maximum(tt.counts.astype(np.int64) - avail[None], 0).sum(-1)
            payable = (gap <= avail[cfg.joker_kind]) & (np.arange(len(tt.counts)) >= last)
            assert (legal[0, :-1] == payable).all(), "the dot product and the maximum disagree"
            checked += 1
            if position == len(sequence):
                break
            last = int(sequence[position])
            laid = laid_by(cfg, last, avail)
            need, avail = np.maximum(need - laid, 0), avail - laid

    total = sum(verdicts.values())
    assert verdicts["none"] / total <= 0.02, verdicts
    assert verdicts["exact"] > 0, "nothing was reproduced tile for tile"
    assert checked > 200, f"only {checked} construction steps were checked"


def test_a_generated_repartition_replays_through_the_env_mask():
    """The construction is a legal turn, checked by playing it.

    `build` decides validity from count arithmetic alone -- that is the whole point,
    since a decode has to be checkable without an env. This is that arithmetic held
    against the thing it stands in for: expand through `to_actions.plan`, and every
    micro-action must pass `legal_actions` in turn, leave the workbench empty, leave
    every slot valid, and conserve every tile.
    """
    from rummi.solver.to_actions import plan

    cfg = STANDARD
    replayed = 0
    longest = 0
    for row in stuck_states():
        need, avail = initial_counts(cfg, row.rack, row.board)
        sequence, verdict = label_sequence(cfg, need, avail, row.sets)
        if verdict == "none":
            continue
        found = build(cfg, need, avail, list(sequence))
        assert found is not None and found.tiles_played >= 1
        assert evaluate_slots(cfg, padded_sets(cfg, found.sets)).is_valid.all()

        probe = row.probe
        actions = plan(cfg, row.board, list(found.sets), found.played)
        if len(actions) > cfg.max_micro_per_turn - int(probe.micro_count[0]):
            continue
        # Read before the replay: `plan` commits the turn, so by the end the acting
        # seat has moved on.
        seat = int(probe.current[0])
        before = int(probe.racks[0, seat].sum())
        for action in actions:
            probe_mask = legal_actions(probe)
            assert probe_mask[0, action], f"the expansion emitted {action}, illegal"
            step(probe, np.array([action]), probe_mask)
        probe.check_invariants()
        assert probe.workbench.sum() == 0, "left tiles held"
        after = evaluate_slots(cfg, probe.table_sets)
        assert (after.is_valid | after.is_empty).all(), "left a broken set"
        assert int(probe.racks[0, seat].sum()) < before
        replayed += 1
        longest = max(longest, len(actions))

    assert replayed > 40, f"only replayed {replayed} repartitions"
    assert longest > 10, f"the longest expansion was {longest}, so no real rearrangement"


def test_the_network_never_ranks_an_illegal_template_above_a_legal_one():
    """`MASKED` is -1e8, not -inf, for the reason in `learned/torch_net.py` -- and it
    still has to dominate, or a decode would propose a template the multiset cannot
    pay for and `build` would throw the whole construction away."""
    cfg = STANDARD
    board = np.full((cfg.max_sets, cfg.max_set_len), -1, dtype=np.int16)
    board[0, :3] = [kind_of(cfg, 0, n) for n in (1, 2, 3)]
    rack = np.zeros(cfg.n_kinds, dtype=np.int64)
    rack[kind_of(cfg, 0, 4)] = 1

    need, avail = initial_counts(cfg, rack, board)
    present = present_counts(cfg, board)[None]
    zero = np.array([0])
    dynamic, short = candidate_features(cfg, need[None], avail[None], present, zero)
    legal = feasible(cfg, need[None], avail[None], zero, zero, short)
    assert legal[0, : len(short[0])].sum() >= 2, "a one-run table admits more than one template"

    torch.manual_seed(0)
    net = RepartitionNet(cfg, hidden=32, key=16)
    with torch.no_grad():
        logits = net(
            torch.from_numpy(state_features(cfg, need[None], avail[None], zero, zero, present)),
            torch.from_numpy(dynamic),
            torch.from_numpy(legal),
        ).numpy()
    assert (logits[0, ~legal[0]] == MASKED).all()
    assert logits[0, legal[0]].min() > MASKED
    assert int(np.argmax(logits[0])) in np.flatnonzero(legal[0]).tolist()
