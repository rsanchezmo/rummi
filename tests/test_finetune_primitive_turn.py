"""The sampling decode, held to the one it is fine-tuning.

`tools/finetune_primitive_turn.py` cannot reuse `primitive_turn.decode_turns`: that
loop scores through `Scorer`, which drops the tape, and policy gradient needs the
tape. So there are two constructions of the same turn, and the only thing keeping
them one space is that the sampling loop reproduces the decode exactly when its draw
is forced to the argmax. Everything else rests on that: an advantage measured
against a greedy arm the sampler does not share a space with would be measuring the
difference between two implementations.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("ortools")
pytest.importorskip("torch")

import torch
from torch import nn

from rummi.agents.learned.primitive_turn import Scorer, decode_turns
from rummi.agents.learned.torch_net import MASKED, TorchPolicy
from rummi.rules.config import STANDARD
from tests.test_primitive_turn import _teacher_turns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import finetune_primitive_turn as tool


class _TeacherNet(nn.Module):
    """A net whose argmax is the recorded plan's next action, one turn at a time.

    Driven by the label for the reason `tests/test_repartition_net.py` gives: an
    untrained policy commits a turn far too rarely for an agreement test between
    two decodes to be about anything.
    """

    def __init__(self, plan: list[int]) -> None:
        super().__init__()
        self.plan = plan
        self.depth = 0

    def forward(self, obs, mask):
        logits = torch.full(mask.shape, MASKED, dtype=torch.float32)
        wanted = self.plan[self.depth] if self.depth < len(self.plan) else STANDARD.draw_action
        logits[:, wanted] = 10.0
        self.depth += 1
        return logits, torch.zeros(mask.shape[0])


def test_the_sampler_reproduces_the_decode_when_it_is_not_sampling() -> None:
    cfg = STANDARD
    records = _teacher_turns(cfg, envs=2, max_steps=400)[:12]
    assert len(records) == 12

    for _, start, plan in records:
        rolled = tool.roll(cfg, _TeacherNet(plan), start, sample=False)
        decoded = decode_turns(cfg, Scorer(_TeacherNet(plan)), start, beam=1)[0]
        assert rolled.actions[0] == plan
        assert list(decoded.actions) == plan
        assert bool(rolled.committed[0])
        assert int(rolled.tiles[0]) == decoded.tiles
        assert tool.reward_of(rolled)[0] == pytest.approx(decoded.tiles / tool.TILE_SCALE)


def test_a_forced_replay_returns_the_same_turn_with_a_tape() -> None:
    """Best-of-N scores one winner with a tape, so the replay has to be exact."""
    cfg = STANDARD
    records = _teacher_turns(cfg, envs=2, max_steps=400)[:8]
    net = TorchPolicy(cfg, seed=0)
    for _, start, plan in records:
        drawn = tool.roll(cfg, _TeacherNet(plan), start, sample=False)
        replayed = tool.roll(cfg, net, start, sample=False, grad=True, forced=drawn.actions)
        assert replayed.actions == drawn.actions
        assert replayed.logp.requires_grad
        assert torch.isfinite(replayed.logp).all()


def test_a_declined_turn_is_worth_nothing() -> None:
    cfg = STANDARD
    records = _teacher_turns(cfg, envs=2, max_steps=400)[:4]
    for _, start, _plan in records:
        # An empty plan makes the teacher emit DRAW at the first step, which is what
        # declining is; the reward has to be exactly zero and not merely small.
        rolled = tool.roll(cfg, _TeacherNet([]), start, sample=False)
        assert rolled.actions[0] == [cfg.draw_action]
        assert not rolled.committed[0]
        assert tool.reward_of(rolled)[0] == 0.0
        assert np.asarray(rolled.tiles)[0] == 0
