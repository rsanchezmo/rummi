"""The value head over afterstates, and the policy it makes: score, take the max.

`afterstate.py` builds the rows a position is described by; this scores them. It
sits in the package rather than beside `tools/train_afterstate.py`, which grew it,
for the reason `macro_net.py` does: an agent loads a saved state dict into it
(`solver_free.py`), a package cannot import from `tools/`, and one definition read
by the trainer, the search and the agent is what keeps a checkpoint loadable by
all three.

The chooser is the whole policy. `REPARTITION` short-circuits it rather than being
ranked, which is the rule `afterstate.py` refuses to build a row for: the macro is
offered only where nothing else plays, so taking it whenever it is legal is all
there is to decide.
"""

from __future__ import annotations

import pathlib
from collections.abc import Callable

import numpy as np
import torch
from torch import nn

from rummi.agents.base import Observation
from rummi.agents.learned.afterstate import afterstate_batch, afterstate_dim
from rummi.agents.macro import Choose, MacroAgent
from rummi.rules.config import RummiConfig

Value = Callable[[np.ndarray], np.ndarray]
"""``(n, afterstate_dim) -> (n,)``. Batched, because a decision offers tens of
macros and a call per row would be most of what scoring one costs."""


class ValueNet(nn.Module):
    """One scalar over an afterstate. Deterministic from the trainer's seed alone.

    Plain MLP on purpose: the question the training run asks is whether outcome
    regression over afterstates ranks macros at all, and any architecture that
    answered it would leave the same question about the plain one.
    """

    def __init__(self, dim: int, hidden: tuple[int, ...] = (256, 256)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = dim
        for size in hidden:
            layers += [nn.Linear(width, size), nn.ReLU()]
            width = size
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def load_value_net(path: pathlib.Path, cfg: RummiConfig) -> tuple[ValueNet, bool]:
    """The checkpoint's net and whether it was trained with `REPARTITION` on.

    Refused rather than reshaped where the width does not fit: seat count and
    preset widen the afterstate, so a net moved between them would load cleanly
    and score something meaningless.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    dim = afterstate_dim(cfg)
    if int(checkpoint["dim"]) != dim:
        raise ValueError(
            f"{path}: afterstate dim {checkpoint['dim']} != this config's {dim} "
            f"(trained on config {checkpoint['cfg']})"
        )
    net = ValueNet(dim, tuple(checkpoint["hidden"]))
    net.load_state_dict(checkpoint["state"])
    net.eval()
    return net, bool(checkpoint["repartition"])


def value_fn(net: ValueNet) -> Value:
    """`net` as the batched scorer the chooser and the search both take."""

    def value_of(rows: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return net(torch.as_tensor(rows)).numpy()

    return value_of


def argmax_chooser(cfg: RummiConfig, agent: MacroAgent, value_of: Value) -> Choose:
    """Argmax V over the afterstates the legal macros lead to.

    Deterministic: a published score has to be reproducible, and the protocol pins
    every seed precisely so two people get the same number.
    """

    def choose(obs: Observation, env: int, legal: np.ndarray) -> int:
        if agent.repartition_macro is not None and legal[agent.repartition_macro]:
            return int(agent.repartition_macro)
        options = np.flatnonzero(legal)
        rows = afterstate_batch(cfg, obs, env, options.tolist(), agent)
        return int(options[int(np.argmax(value_of(rows)))])

    return choose
