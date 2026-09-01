"""The two differentiable phases, held to the decode they are fine-tuning.

`tools/finetune_two_phase.py` cannot reuse `decode_two_phase`: that loop scores
through `TwoPhaseScorer`, which drops the tape, and policy gradient needs the tape.
So there are two constructions of the same break-then-cover rollout, and the only
thing keeping them one space is that the sampling loops reproduce the decode exactly
when their draws are forced to the argmax -- in *both* phases, and in the derivation
that hands one to the other. An advantage measured against a greedy arm the sampler
does not share a space with would be measuring the difference between two
implementations.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

import torch

from rummi.agents.learned.repartition_net import present_counts
from rummi.agents.learned.two_phase_net import (
    TwoPhaseNet,
    TwoPhaseScorer,
    decode_two_phase,
    freed_counts,
    slot_present,
    stop_break,
)
from rummi.rules.config import STANDARD

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import finetune_two_phase as tool
from finetune_repartition import score_sequences, tile_price

from tests.test_finetune_repartition import synthetic_states


class Recording(TwoPhaseScorer):
    """`TwoPhaseScorer`, keeping what a beam-1 `decode_two_phase` picks in each phase.

    Illegal actions carry `MASKED`, so the argmax of the returned logits *is* the
    pick. The cover's mask can close on a row, which is a round the batched loop
    never makes -- it drops the row before the forward pass rather than soft-maxing
    over nothing -- so whether each round was open is recorded beside the pick.
    """

    def __init__(self, net: TwoPhaseNet) -> None:
        super().__init__(net)
        self.breaks: list[int] = []
        self.covers: list[int] = []
        self.open: list[bool] = []

    def brk(self, state, static, dynamic, legal):
        out = super().brk(state, static, dynamic, legal)
        self.breaks.append(int(out.argmax(-1)[0]))
        return out

    def cover(self, state, dynamic, legal):
        out = super().cover(state, dynamic, legal)
        self.covers.append(int(out.argmax(-1)[0]))
        self.open.append(bool(legal[0].any()))
        return out


def walked(recorder: Recording) -> tuple[int, ...]:
    """The cover picks a batched decode would have made, closed rounds dropped."""
    if False in recorder.open:
        return tuple(recorder.covers[: recorder.open.index(False)])
    return tuple(recorder.covers)


def test_both_phases_walk_the_same_rollout_as_a_beam_1_decode():
    cfg = STANDARD
    torch.manual_seed(0)
    net = TwoPhaseNet(cfg, hidden=32, key=16).eval()
    racks, boards, _ = synthetic_states(cfg, 24, seed=3)
    slots = tool.break_starts(cfg, racks, boards)

    with torch.no_grad():
        roll = tool.run_two_phase(cfg, net, slots, monotone=False, sample=False)

    price = tile_price(cfg)
    for i in range(len(racks)):
        recorder = Recording(net)
        found = decode_two_phase(cfg, recorder, racks[i], boards[i], 1, False)
        assert tuple(recorder.breaks) == roll.brk.actions[i]
        assert walked(recorder) == roll.cover.actions[i]
        # The reward is the thing the two loops have to agree on, so it is asserted
        # rather than only the sequences: a divergence in the phase-B start would
        # leave the actions matching and the construction worth something else.
        reward, valid, played = score_sequences(
            cfg, roll.starts.take(np.array([i])), [roll.cover.sequences[i]],
            mode="value", valid_bonus=0.0, price=price,
        )
        assert bool(valid[0]) == (found is not None)
        assert int(played[0]) == (0 if found is None else found.tiles_played)
        assert (reward[0] > 0.0) == (found is not None and found.tiles_played > 0)


def test_the_phase_b_start_is_what_the_decode_derives_for_that_subset():
    """`cover_starts` is a masked sum where `decode_two_phase` loops per subset."""
    cfg = STANDARD
    racks, boards, _ = synthetic_states(cfg, 8, seed=17)
    slots = tool.break_starts(cfg, racks, boards)
    rng = np.random.default_rng(0)

    subsets = []
    for board in boards:
        occupied = np.flatnonzero((board >= 0).any(-1))
        take = rng.integers(0, len(occupied) + 1)
        subsets.append(tuple(sorted(rng.permutation(occupied)[:take].tolist())))
    starts = tool.cover_starts(slots, subsets)

    for i, broken in enumerate(subsets):
        need, avail = freed_counts(cfg, boards[i], broken, racks[i])
        assert (starts.need[i] == need).all()
        assert (starts.avail[i] == avail).all()
        assert (starts.present[i] == present_counts(cfg, boards[i][list(broken)])).all()
        occupied = int((boards[i] >= 0).any(-1).sum())
        assert int(starts.base[i]) == occupied - len(broken)


def test_slot_present_sums_to_present_counts_over_any_subset():
    """The lookup phase B's `present` needs, paid once per slot instead of per subset."""
    cfg = STANDARD
    _, boards, _ = synthetic_states(cfg, 6, seed=23)
    per_slot = slot_present(cfg, boards)
    rng = np.random.default_rng(1)
    for i, board in enumerate(boards):
        occupied = np.flatnonzero((board >= 0).any(-1))
        broken = sorted(rng.permutation(occupied)[: len(occupied) // 2].tolist())
        assert (per_slot[i, broken].sum(0) == present_counts(cfg, board[broken])).all()


def test_a_break_that_matches_the_greedy_arm_carries_no_gradient_into_either_head():
    """The credit rule across the two phases, which is the composition's whole join.

    Forced onto the greedy arm's own decisions nothing is credited; and a rollout
    that leaves the arm in phase A credits its whole cover, because every template
    it then picks is conditioned on tiles the greedy arm never freed.
    """
    cfg = STANDARD
    torch.manual_seed(1)
    net = TwoPhaseNet(cfg, hidden=32, key=16)
    racks, boards, _ = synthetic_states(cfg, 16, seed=5)
    slots = tool.break_starts(cfg, racks, boards)

    with torch.no_grad():
        greedy = tool.run_two_phase(cfg, net, slots, monotone=False, sample=False)
    same = tool.run_two_phase(
        cfg, net, slots, monotone=False, sample=False, grad=True, baseline=greedy.decisions
    )
    assert same.brk.credited == 0
    assert same.cover.credited == 0
    assert torch.equal(same.logp, torch.zeros(len(racks)))

    # One extra break in phase A, above the last one so the mask still allows it,
    # and everything after it is a departure -- the cover included.
    stop = stop_break(cfg)
    forced_a = []
    for i, actions in enumerate(greedy.brk.actions):
        broken = greedy.brk.sequences[i]
        floor = broken[-1] if broken else -1
        spare = [
            int(s)
            for s in np.flatnonzero((boards[i] >= 0).any(-1))
            if s > floor and s not in broken
        ]
        forced_a.append((*actions[:-1], spare[-1], stop) if spare else actions)
    other = tool.run_two_phase(
        cfg, net, slots, monotone=False, sample=False, grad=True,
        baseline=greedy.decisions, forced=(forced_a, None),
    )
    moved = [i for i in range(len(racks)) if forced_a[i] != greedy.brk.actions[i]]
    assert moved
    assert (other.logp[moved] < 0).all()
    stayed = [i for i in range(len(racks)) if i not in moved]
    assert torch.equal(other.logp[stayed], torch.zeros(len(stayed)))


def test_replaying_a_rollout_scores_it_exactly_as_drawing_it_did():
    """What `--samples` rests on: the winner is replayed across both phases."""
    cfg = STANDARD
    torch.manual_seed(3)
    net = TwoPhaseNet(cfg, hidden=32, key=16)
    racks, boards, _ = synthetic_states(cfg, 16, seed=11)
    slots = tool.break_starts(cfg, racks, boards)

    drawn = tool.run_two_phase(
        cfg, net, slots, monotone=False, sample=True,
        generator=torch.Generator().manual_seed(4), grad=True,
    )
    replayed = tool.run_two_phase(
        cfg, net, slots, monotone=False, sample=False, grad=True, forced=drawn.decisions
    )
    assert replayed.decisions == drawn.decisions
    assert replayed.cover.sequences == drawn.cover.sequences
    assert (replayed.starts.need == drawn.starts.need).all()
    assert torch.allclose(replayed.logp, drawn.logp)


def test_breaking_nothing_is_valid_and_worth_zero():
    """The reward's nearest bad answer, which is why the validity bonus is 0 here.

    Phase A reaches it in one decision -- `STOP` before any slot -- and phase B can
    then `STOP` immediately, because nothing is uncovered. `build` accepts that: the
    table stands as it is, and no tile leaves the rack.
    """
    cfg = STANDARD
    racks, boards, _ = synthetic_states(cfg, 4, seed=29)
    slots = tool.break_starts(cfg, racks, boards)
    starts = tool.cover_starts(slots, [()] * len(racks))

    reward, valid, played = score_sequences(
        cfg, starts, [()] * len(racks), mode="value", valid_bonus=0.0, price=tile_price(cfg)
    )
    assert valid.all()
    assert not played.any()
    assert not reward.any()

    paid, _, _ = score_sequences(
        cfg, starts, [()] * len(racks), mode="value", valid_bonus=0.1, price=tile_price(cfg)
    )
    assert paid.tolist() == pytest.approx([0.1] * len(racks))
