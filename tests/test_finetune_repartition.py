"""The differentiable decode, held to the one it is fine-tuning.

`tools/finetune_repartition.py` cannot reuse `repartition_net.decode`: that loop
scores through `Scorer`, which drops the tape, and policy gradient needs the tape.
So there are two constructions of the same sequence, and the only thing keeping
them one space is that the sampling loop reproduces the decode exactly when its
draw is forced to the argmax. Everything else here rests on that: an advantage
measured against a greedy arm the sampler does not actually share a space with
would be measuring the difference between two implementations.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

import torch

from rummi.agents.learned.repartition_net import (
    RepartitionNet,
    Scorer,
    decode,
    initial_counts,
    present_counts,
    stop_action,
    template_table,
)
from rummi.rules.config import STANDARD
from rummi.rules.encoding import EMPTY

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import finetune_repartition as tool


def board_of(cfg, templates: list[int]) -> np.ndarray:
    """A table holding exactly these templates, `EMPTY` padded as the env pads it."""
    tt = template_table(cfg)
    board = np.full((cfg.max_sets, cfg.max_set_len), EMPTY, dtype=np.int16)
    for row, template in enumerate(templates):
        content = tt.contents[template]
        board[row, : len(content)] = content
    return board


def synthetic_states(cfg, count: int, seed: int) -> tuple[np.ndarray, np.ndarray, list[list[int]]]:
    """Tables built out of templates, with a rack drawn against the copies left.

    A real stuck state costs a real game -- `tests/test_repartition_net.py` pays
    that -- and nothing here needs the state to be stuck: the construction space
    reads a multiset, and where it came from is not part of its arithmetic.
    """
    tt = template_table(cfg)
    rng = np.random.default_rng(seed)
    racks, boards, laid = [], [], []
    for _ in range(count):
        held = np.zeros(cfg.n_kinds, dtype=np.int64)
        chosen: list[int] = []
        for template in rng.permutation(len(tt.counts))[:40]:
            counts = tt.counts[template].astype(np.int64)
            if len(chosen) >= 4 or (held + counts > cfg.n_copies).any():
                continue
            held = held + counts
            chosen.append(int(template))
        rack = np.zeros(cfg.n_kinds, dtype=np.int16)
        for kind in rng.choice(cfg.n_kinds - 1, size=6, replace=False):
            rack[kind] = min(cfg.n_copies - held[kind], 1)
        racks.append(rack)
        boards.append(board_of(cfg, chosen))
        laid.append(chosen)
    return np.stack(racks), np.stack(boards), laid


class Recording(Scorer):
    """`Scorer`, keeping the action a beam-1 `decode` takes from each of its rows.

    Illegal actions carry `MASKED`, so the argmax of the returned logits *is* the
    pick -- there is nothing to restate about how the decode chooses.
    """

    def __init__(self, net: RepartitionNet) -> None:
        super().__init__(net)
        self.picks: list[int] = []
        self.open: list[bool] = []

    def __call__(self, state, dynamic, legal):
        out = super().__call__(state, dynamic, legal)
        self.picks.append(int(out.argmax(-1)[0]))
        self.open.append(bool(legal[0].any()))
        return out


def test_the_sampling_loop_walks_the_same_construction_as_a_beam_1_decode():
    cfg = STANDARD
    torch.manual_seed(0)
    net = RepartitionNet(cfg, hidden=32, key=16).eval()
    racks, boards, _ = synthetic_states(cfg, 24, seed=3)
    starts = tool.starting_states(cfg, racks, boards)

    with torch.no_grad():
        walked = tool.run_decode(cfg, net, starts, monotone=False, sample=False)

    for i in range(len(racks)):
        need, avail = initial_counts(cfg, racks[i], boards[i])
        recorder = Recording(net)
        found = decode(cfg, recorder, need, avail, present_counts(cfg, boards[i]), 1, False)
        # A depth where the mask closed is a call the batched loop never makes: it
        # drops the row before the forward pass rather than soft-maxing over
        # nothing.
        taken = recorder.picks[: recorder.open.index(False)] if False in recorder.open else recorder.picks
        assert tuple(taken) == walked.actions[i]
        replayed = tool.score_sequences(
            cfg, starts.take(np.array([i])), [walked.sequences[i]],
            mode="value", valid_bonus=0.0, price=tool.tile_price(cfg),
        )
        assert replayed[1][0] == (found is not None)


def test_a_construction_is_worth_what_it_sheds_and_nothing_if_build_rejects_it():
    """The three rungs the reward has to keep apart, on a table built to order."""
    cfg = STANDARD
    tt = template_table(cfg)
    price = tool.tile_price(cfg)

    covered = next(t for t in range(len(tt.counts)) if len(tt.contents[t]) == 3)
    wider = next(
        t for t in range(len(tt.counts))
        if set(tt.contents[covered]) < set(tt.contents[t])
    )
    extra = sorted(set(tt.contents[wider]) - set(tt.contents[covered]))
    board = board_of(cfg, [covered])
    rack = np.zeros(cfg.n_kinds, dtype=np.int16)
    for kind in extra:
        rack[kind] = 1
    starts = tool.starting_states(cfg, rack[None], board[None])

    def paid(sequence: tuple[int, ...]) -> float:
        reward, _, _ = tool.score_sequences(
            cfg, starts, [sequence], mode="value", valid_bonus=0.1, price=price
        )
        return float(reward[0])

    assert paid(()) == 0.0  # the table is left uncovered
    assert paid((covered,)) == pytest.approx(0.1)  # covered, nothing shed
    assert paid((wider,)) == pytest.approx(
        0.1 + float(price[extra].sum()) / tool.VALUE_SCALE
    )


def test_the_advantage_credits_only_the_steps_after_the_greedy_arm_was_left():
    """Forced onto the greedy arm's own actions, a rollout carries no gradient.

    Which is the property the credit rule exists for: a shared prefix cannot have
    caused the difference in reward, and here the whole sequence is shared.
    """
    cfg = STANDARD
    torch.manual_seed(1)
    net = RepartitionNet(cfg, hidden=32, key=16)
    racks, boards, _ = synthetic_states(cfg, 16, seed=5)
    starts = tool.starting_states(cfg, racks, boards)

    with torch.no_grad():
        greedy = tool.run_decode(cfg, net, starts, monotone=False, sample=False)
    same = tool.run_decode(
        cfg, net, starts, monotone=False, sample=False, grad=True, baseline=greedy.actions
    )
    assert same.credited == 0
    assert same.steps == greedy.steps
    assert torch.equal(same.logp, torch.zeros(len(racks)))

    free = tool.run_decode(cfg, net, starts, monotone=False, sample=False, grad=True)
    assert free.credited == free.steps
    assert (free.logp < 0).all()


def test_replaying_a_construction_scores_it_exactly_as_walking_it_did():
    """What `--samples` rests on: the winner is replayed, not re-drawn.

    Several samples with a tape each would cost K times the memory to produce one
    gradient, so only the best is differentiated -- and that is sound only if a
    replay is the same trajectory at the same log-probability.
    """
    cfg = STANDARD
    torch.manual_seed(3)
    net = RepartitionNet(cfg, hidden=32, key=16)
    racks, boards, _ = synthetic_states(cfg, 16, seed=11)
    starts = tool.starting_states(cfg, racks, boards)

    drawn = tool.run_decode(
        cfg, net, starts, monotone=False, sample=True,
        generator=torch.Generator().manual_seed(4), grad=True,
    )
    replayed = tool.run_decode(
        cfg, net, starts, monotone=False, sample=False, grad=True, forced=drawn.actions
    )
    assert replayed.actions == drawn.actions
    assert replayed.sequences == drawn.sequences
    assert torch.allclose(replayed.logp, drawn.logp)


def test_every_construction_ends_at_stop_or_at_a_mask_with_nothing_left():
    """The loop's own termination, which is what bounds a rollout's cost."""
    cfg = STANDARD
    torch.manual_seed(2)
    net = RepartitionNet(cfg, hidden=32, key=16)
    racks, boards, _ = synthetic_states(cfg, 16, seed=7)
    starts = tool.starting_states(cfg, racks, boards)
    rolled = tool.run_decode(
        cfg, net, starts, monotone=False, sample=True,
        generator=torch.Generator().manual_seed(0),
    )
    stop = stop_action(cfg)
    for actions, sequence in zip(rolled.actions, rolled.sequences, strict=True):
        assert len(actions) <= cfg.max_sets + 1
        assert tuple(a for a in actions if a != stop) == sequence
        if stop in actions:
            assert actions.index(stop) == len(actions) - 1
