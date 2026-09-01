"""The network the macro-space agents run on: logits over macros, and a value.

In the package rather than beside the trainer that grew it, because a bundled
ladder rung loads weights into it -- `clone.py` cannot import from `tools/`, and
the trainer and the evaluator reading one definition is what keeps a saved
state dict loadable by both.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from rummi.agents.macro import action_features
from rummi.agents.learned.features import feature_dim
from rummi.rules.config import RummiConfig


class MacroNet(nn.Module):
    """Logits over the macro actions, and a value.

    `flat` gives every macro its own row, so nothing learned about one set carries
    to a similar one. `pointer` scores each macro against `macro.action_features`
    -- what the action *does* -- so the scoring function is shared and a per-action
    bias carries whatever is left over.

    `memory="lstm"` puts an LSTM cell between the trunk and the heads, stepped once
    per decision, so the policy can *learn* what to carry across steps where the
    engineered history block hands it a fixed summary. The heads read the trunk's
    output and the cell's side by side rather than the cell's alone: the snapshot
    path stays intact and memory is strictly additive, which is what makes an arm
    without it a control.
    """

    # Declared so the registered buffer types as a Tensor: `nn.Module.__getattr__`
    # otherwise widens every attribute to `Tensor | Module`.
    desc: torch.Tensor

    def __init__(
        self, cfg: RummiConfig, macros: int, hidden: int = 256,
        head: str = "flat", key_dim: int = 64, describe: np.ndarray | None = None,
        extra: int = 0, memory: str = "none", memory_dim: int = 128,
    ) -> None:
        super().__init__()
        self.head = head
        self.memory = memory
        self.memory_dim = memory_dim if memory == "lstm" else 0
        # `extra` widens the input alone -- the opponent-history block is appended
        # to the observation features, so at 0 the trunk is the one it always was.
        self.trunk = nn.Sequential(
            nn.Linear(feature_dim(cfg) + extra, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.cell = nn.LSTMCell(hidden, self.memory_dim) if memory == "lstm" else None
        width = hidden + self.memory_dim
        if head == "pointer":
            desc = torch.as_tensor(
                action_features(cfg) if describe is None else describe
            )
            self.register_buffer("desc", desc)
            self.key = nn.Linear(desc.shape[1], key_dim, bias=False)
            self.query = nn.Linear(width, key_dim)
            self.action_bias = nn.Parameter(torch.zeros(macros))
            # Small, for the same reason the flat head uses gain 0.01: a fresh
            # policy should be near-uniform over the legal macros.
            nn.init.orthogonal_(self.query.weight, 0.01)
            nn.init.zeros_(self.query.bias)
        else:
            self.pi = nn.Linear(width, macros)
            nn.init.orthogonal_(self.pi.weight, 0.01)
            nn.init.zeros_(self.pi.bias)
        self.v = nn.Linear(width, 1)

    def heads(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Logits and value from the representation the heads read -- the trunk's
        output, with the cell's concatenated when there is one."""
        if self.head == "pointer":
            logits = self.query(h) @ self.key(self.desc).T + self.action_bias
        else:
            logits = self.pi(h)
        return logits, self.v(h).squeeze(-1)

    def forward(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ):
        h = self.trunk(x)
        if self.cell is None:
            return self.heads(h)
        hx, cx = self.cell(h, state)
        logits, v = self.heads(torch.cat([h, hx], -1))
        return logits, v, (hx, cx)
