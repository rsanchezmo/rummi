"""The autoregressive repartition space, held to the engine it has to expand into.

The whole claim of `rummi/agents/learned/repartition_net.py` is that a whole-table
rearrangement can be *constructed* one template at a time under a mask, and that
what comes out is a legal turn. Neither half is checkable by inspection: the mask
is arithmetic over count vectors, and legality is the env's own kernel. So both are
checked against the things that own them -- the mask against what the multiset can
actually pay for, and every construction against `legal_actions` on a real state,
replayed micro-action by micro-action.

`learned/two_phase_net.py` splits the same construction into break-then-cover, and
is checked here rather than beside it because the states it needs are the same ones
and reaching them costs a real game. Both decoders are driven by a scorer that
returns the label -- an untrained net produces a valid repartition too rarely to
assert anything about the loop it came out of.
"""

from __future__ import annotations

from collections import Counter
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
    decode,
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
from rummi.agents.learned.two_phase_net import (
    BREAK_DYNAMIC,
    BreakNet,
    break_dynamic,
    break_feasible,
    break_state_features,
    break_static_dim,
    decode_two_phase,
    decompose,
    freed_counts,
    n_break_actions,
    slot_counts,
    slot_static,
    stop_break,
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

        # Cloned, because a replay commits the turn and the states are built once
        # for the module.
        probe = row.probe.clone()
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


# --- break, then cover ----------------------------------------------------


class _Teacher:
    """A scorer that answers with the label, so a decode is deterministic.

    Both loops are best-first over log-probabilities, so at `beam=1` naming one
    action per step drives them exactly along the sequence -- which is what makes
    the *loop* testable rather than whatever a network happens to prefer.
    """

    def __init__(self, breaks, covers, cfg) -> None:
        self.breaks = list(breaks)
        self.covers = list(covers)
        self.cfg = cfg

    @staticmethod
    def _pick(legal: np.ndarray, wanted: int) -> np.ndarray:
        out = np.where(legal, 0.0, MASKED).astype(np.float32)
        assert legal[0, wanted], f"the label names {wanted}, which the mask refuses"
        out[0, wanted] = 10.0
        return out

    def brk(self, state, static, dynamic, legal):
        return self._pick(legal, self.breaks.pop(0) if self.breaks else stop_break(self.cfg))

    def cover(self, state, dynamic, legal):
        return self._pick(legal, self.covers.pop(0) if self.covers else stop_action(self.cfg))


def _labelled(cfg, row) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """`(kept, broken, cover sequence)` for one stuck state, or `()` three times."""
    kept, broken, to_build = decompose(cfg, row.board, row.sets)
    need, avail = freed_counts(cfg, row.board, broken, row.rack)
    sequence, verdict = label_sequence(cfg, need, avail, to_build)
    return (kept, broken, sequence) if verdict != "none" else ((), (), ())


def test_the_break_and_cover_split_accounts_for_every_tile_on_the_table():
    """The decomposition is a partition of the solve, not an approximation of it.

    Kept slots plus built sets have to be the solver's answer exactly, and the
    tiles have to balance both ways -- what the broken slots free plus what leaves
    the rack is what the built sets consume. If that ever drifted, phase B would be
    trained to cover a multiset the decode is not handed.
    """
    cfg = STANDARD
    split = 0
    breaks = covers = 0
    for row in stuck_states():
        kept, broken, to_build = decompose(cfg, row.board, row.sets)
        contents = [tuple(sorted(int(k) for k in r if k >= 0)) for r in row.board]
        occupied = {s for s, c in enumerate(contents) if c}
        assert set(kept) | set(broken) == occupied
        assert not set(kept) & set(broken)

        rebuilt = Counter(contents[s] for s in kept) + Counter(
            tuple(sorted(s)) for s in to_build
        )
        assert rebuilt == Counter(tuple(sorted(int(k) for k in s)) for s in row.sets)

        need, avail = freed_counts(cfg, row.board, broken, row.rack)
        consumed = np.zeros(cfg.n_kinds, dtype=np.int64)
        for content in to_build:
            for kind in content:
                consumed[int(kind)] += 1
        assert (need + row.played == consumed).all(), "the freed tiles do not balance"
        assert (row.played <= avail - need).all(), "played more of a kind than the rack holds"

        sequence, verdict = label_sequence(cfg, need, avail, to_build)
        assert verdict != "none", "the cover is not expressible in template space"
        replayed = build(cfg, need, avail, list(sequence), reserved=len(kept))
        assert replayed is not None
        assert int(replayed.played.sum()) == int(row.played.sum())
        split += 1
        breaks += len(broken)
        covers += len(sequence)

    assert split > 40, f"only split {split} solves"
    assert breaks < covers < 2 * breaks + split, "the split is not the shape it claims"


def test_the_break_mask_offers_only_an_occupied_slot_above_the_last_one():
    """Phase A's whole mask, walked along the label it has to admit.

    An empty slot has nothing to dissolve and a slot already broken cannot be
    broken twice; both fall out of one comparison because the subset is emitted in
    slot order, and this is where that shortcut is held to what it stands for.
    """
    cfg = STANDARD
    checked = 0
    for row in stuck_states():
        kept, broken, _ = _labelled(cfg, row)
        if not kept and not broken:
            continue
        counts = slot_counts(cfg, row.board[None]).astype(np.int64)
        occupied = np.flatnonzero(counts[0].sum(-1) > 0)
        static = slot_static(cfg, counts)
        assert static.shape == (1, cfg.max_sets + 1, break_static_dim(cfg))
        assert static[0, cfg.max_sets, -1] == 1.0, "STOP does not describe itself"
        assert not static[0, :-1, -1].any(), "a slot claims to be STOP"

        freed = np.zeros((1, cfg.n_kinds), dtype=np.int64)
        last = -1
        for position in range(len(broken) + 1):
            before = np.array([last])
            legal = break_feasible(cfg, counts, before)
            dynamic = break_dynamic(cfg, counts, row.rack[None], freed, before)
            assert dynamic.shape == (1, n_break_actions(cfg), BREAK_DYNAMIC)
            assert break_state_features(
                cfg, row.rack[None], counts.sum(1), freed, np.array([len(occupied)]),
                np.array([position]), before
            ).shape == (1, 3 * cfg.n_kinds + 8)

            assert legal[0, stop_break(cfg)], "STOP is never masked"
            offered = np.flatnonzero(legal[0, :-1])
            assert set(offered) <= set(occupied.tolist()), "an empty slot was offered"
            assert not set(offered) & set(broken[:position]), "a broken slot was offered again"
            assert (offered > last).all()
            checked += 1
            if position == len(broken):
                break
            slot = broken[position]
            assert legal[0, slot], "the label breaks a slot the mask refuses"
            freed = freed + counts[:, slot]
            last = slot

    assert checked > 150, f"only {checked} break steps were checked"


def test_a_two_phase_decode_replays_through_the_env_mask():
    """Break, cover, expand, play -- the whole path, on a state a real game reached.

    The scorer answers with the label, so what is under test is the decode: that it
    threads phase A's subset into phase B's `need`/`avail`, puts the kept slots back
    into the finished repartition, and comes out as a turn `legal_actions` accepts
    micro-action by micro-action.
    """
    from rummi.solver.to_actions import plan

    cfg = STANDARD
    replayed = 0
    for row in stuck_states():
        kept, broken, sequence = _labelled(cfg, row)
        if not sequence and not broken:
            continue
        teacher = _Teacher(broken, sequence, cfg)
        found = decode_two_phase(cfg, teacher, row.rack, row.board, beam=1, monotone=True)
        assert found is not None, "the labelled split did not decode"
        assert int(found.played.sum()) == int(row.played.sum())
        assert len(found.sets) == len(kept) + len(sequence)
        assert evaluate_slots(cfg, padded_sets(cfg, found.sets)).is_valid.all()

        probe = row.probe.clone()
        actions = plan(cfg, row.board, list(found.sets), found.played)
        if len(actions) > cfg.max_micro_per_turn - int(probe.micro_count[0]):
            continue
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

    assert replayed > 40, f"only replayed {replayed} two-phase repartitions"


def test_the_one_phase_decode_still_follows_a_named_sequence():
    """The shared construction round, driven along a label the one-phase space owns.

    `expand` is what both decoders run, so a change to it that broke the whole-table
    path would otherwise only show up as a score.
    """
    cfg = STANDARD
    followed = 0
    for row in stuck_states():
        need, avail = initial_counts(cfg, row.rack, row.board)
        sequence, verdict = label_sequence(cfg, need, avail, row.sets)
        if verdict == "none":
            continue
        teacher = _Teacher((), sequence, cfg)
        found = decode(
            cfg, teacher.cover, need, avail, present_counts(cfg, row.board), beam=1
        )
        assert found is not None and found.templates == tuple(sequence)
        assert int(found.played.sum()) == int(row.played.sum())
        followed += 1
    assert followed > 40, f"only followed {followed} sequences"


def test_the_break_head_never_ranks_a_masked_slot_above_a_legal_one():
    """The same `MASKED` dominance the template head needs, for the slot pointer:
    a decode that dissolved an empty slot would hand `to_actions.plan` a target the
    table cannot reach."""
    cfg = STANDARD
    board = np.full((cfg.max_sets, cfg.max_set_len), -1, dtype=np.int16)
    board[0, :3] = [kind_of(cfg, 0, n) for n in (1, 2, 3)]
    board[2, :3] = [kind_of(cfg, c, 7) for c in (0, 1, 2)]
    rack = np.zeros(cfg.n_kinds, dtype=np.int64)
    rack[kind_of(cfg, 0, 4)] = 1

    counts = slot_counts(cfg, board[None]).astype(np.int64)
    freed = np.zeros((1, cfg.n_kinds), dtype=np.int64)
    start = np.array([-1])
    legal = break_feasible(cfg, counts, start)
    assert np.flatnonzero(legal[0]).tolist() == [0, 2, stop_break(cfg)]

    torch.manual_seed(0)
    net = BreakNet(cfg, hidden=32, key=16)
    with torch.no_grad():
        logits = net(
            torch.from_numpy(
                break_state_features(
                    cfg, rack[None], counts.sum(1), freed, np.array([2]), np.array([0]), start
                )
            ),
            torch.from_numpy(slot_static(cfg, counts)),
            torch.from_numpy(break_dynamic(cfg, counts, rack[None], freed, start)),
            torch.from_numpy(legal),
        ).numpy()
    assert (logits[0, ~legal[0]] == MASKED).all()
    assert logits[0, legal[0]].min() > MASKED
    assert int(np.argmax(logits[0])) in np.flatnonzero(legal[0]).tolist()
