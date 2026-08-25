"""Torch implementation of the reference policy.

Consumes the observation in whatever array type it arrives in -- the whole point
of the torch backend is that it never left the device -- so `forward` takes the
dict as-is and only converts dtype.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from rummi.rules.config import RummiConfig
from rummi.agents.learned.architecture import Architecture, init_params
from rummi.agents.learned.features import FEATURE_FIELDS, feature_dim, feature_scale

MASKED = -1e8
"""Logit for an illegal action.

Finite on purpose. With `-inf`, an illegal action's probability is exactly 0 and
the entropy term computes `0 * -inf`, which is NaN. At -1e8 the probability
underflows to 0 while the product stays 0, so entropy needs no special case.
"""


class TorchPolicy(nn.Module):
    # Declared so the registered buffer types as a Tensor: `nn.Module.__getattr__`
    # otherwise widens every attribute to `Tensor | Module`.
    scale: torch.Tensor

    def __init__(
        self,
        cfg: RummiConfig,
        arch: Architecture | None = None,
        params: dict[str, np.ndarray] | None = None,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.arch = arch or Architecture()
        params = params if params is not None else init_params(cfg, self.arch, seed)

        self.register_buffer("scale", torch.as_tensor(feature_scale(cfg)))
        self.trunk = nn.ModuleList()
        for i, (fan_in, fan_out) in enumerate(self.arch.layer_sizes(cfg)):
            layer = nn.Linear(fan_in, fan_out)
            _load(layer, params[f"w{i}"], params[f"b{i}"])
            self.trunk.append(layer)

        last = self.arch.hidden[-1]
        if self.arch.head == "bilinear":
            self.kind = nn.Linear(last, cfg.n_kinds * self.arch.head_dim)
            _load(self.kind, params["w_kind"], params["b_kind"])
            self.slot = nn.Linear(last, cfg.max_sets * self.arch.head_dim)
            _load(self.slot, params["w_slot"], params["b_slot"])
            self.pos = nn.Linear(last, cfg.max_set_len * self.arch.head_dim)
            _load(self.pos, params["w_pos"], params["b_pos"])
            self.flat = nn.Linear(last, cfg.n_kinds + cfg.max_sets + 2)
            _load(self.flat, params["w_flat"], params["b_flat"])
        else:
            self.pi = nn.Linear(last, cfg.n_actions)
            _load(self.pi, params["w_pi"], params["b_pi"])
        self.v = nn.Linear(self.arch.hidden[-1], 1)
        _load(self.v, params["w_v"], params["b_v"])

    def slot_counts(self, table_sets: torch.Tensor) -> torch.Tensor:
        """`(B, S, K)` count of each kind per slot. See `features.slot_counts_numpy`.

        `EMPTY` is `-1`, so it is masked out rather than clamped into kind 0 --
        clamping would put a phantom tile of the lowest kind in every empty
        position, which is `S*L` of them on a fresh deal.
        """
        cfg = self.cfg
        b, s, _ = table_sets.shape
        valid = (table_sets >= 0).to(self.scale.dtype)
        idx = table_sets.clamp(min=0).long()
        counts = torch.zeros((b, s, cfg.n_kinds), dtype=self.scale.dtype, device=idx.device)
        return counts.scatter_add_(-1, idx, valid)

    def features(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        batch = obs["rack"].shape[0]
        flat = torch.cat(
            [torch.as_tensor(obs[f]).reshape(batch, -1) for f in FEATURE_FIELDS], dim=-1
        ).to(self.scale.dtype)
        assert flat.shape[-1] == feature_dim(self.cfg), flat.shape
        return flat / self.scale

    def _act(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x) if self.arch.activation == "relu" else torch.tanh(x)

    def _logits(self, x: torch.Tensor) -> torch.Tensor:
        """Action logits, assembled in the block order of SPEC.md section 4.

        `ASSIGN(kind, slot)` is `assign_offset + kind*S + slot` and
        `PICK(slot, pos)` is `pick_offset + slot*L + pos` -- both row-major, so a
        `(K, S)` or `(S, L)` matrix flattens straight onto its block.
        """
        if self.arch.head != "bilinear":
            return self.pi(x)

        cfg, d = self.cfg, self.arch.head_dim
        b = x.shape[0]
        kind = self.kind(x).view(b, cfg.n_kinds, d)
        slot = self.slot(x).view(b, cfg.max_sets, d)
        pos = self.pos(x).view(b, cfg.max_set_len, d)
        flat = self.flat(x)

        place, dissolve, tail = flat.split([cfg.n_kinds, cfg.max_sets, 2], dim=-1)
        pick = torch.einsum("bsd,bpd->bsp", slot, pos).reshape(b, -1)
        assign = torch.einsum("bkd,bsd->bks", kind, slot).reshape(b, -1)
        return torch.cat([place, pick, dissolve, assign, tail], dim=-1)

    def head(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """`(masked_logits, value)` from already-scaled features.

        Split out so a caller can cache features instead of observations. That
        matters for expert cloning: `slot_features` and the count vectors are ~570
        floats where the raw observation is ~1025, and an offline dataset large
        enough to clone CP-SAT is the difference between fitting in memory and not.
        """
        for layer in self.trunk:
            x = self._act(layer(x))
        logits = self._logits(x)
        legal = torch.as_tensor(mask, dtype=torch.bool, device=logits.device)
        return torch.where(legal, logits, torch.full_like(logits, MASKED)), self.v(x).squeeze(-1)

    def forward(
        self, obs: dict[str, torch.Tensor], mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`(masked_logits, value)`."""
        return self.head(self.features(obs), mask)


def _load(layer: nn.Linear, w: np.ndarray, b: np.ndarray) -> None:
    # NumPy holds (fan_in, fan_out); nn.Linear wants (fan_out, fan_in).
    with torch.no_grad():
        layer.weight.copy_(torch.as_tensor(w.T.copy()))
        layer.bias.copy_(torch.as_tensor(b))


def sample(logits: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    """Gumbel-max rather than `multinomial`, so it needs no normalisation and
    matches what the JAX side does with `categorical`."""
    u = torch.rand(logits.shape, generator=generator, device=logits.device)
    return torch.argmax(logits - torch.log(-torch.log(u.clamp_min(1e-20))), dim=-1)


def log_prob(logits: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    logp = torch.log_softmax(logits, dim=-1)
    return logp.gather(-1, actions.unsqueeze(-1).long()).squeeze(-1)


def entropy(logits: torch.Tensor) -> torch.Tensor:
    logp = torch.log_softmax(logits, dim=-1)
    return -(logp.exp() * logp).sum(-1)
