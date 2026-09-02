"""The repartition state read as one token per tile kind, related by attention.

`RepartitionNet`'s trunk is a flat MLP over `[need, avail, avail - need, scalars]`,
so kind 7's count and kind 8's count reach the first weight matrix as two unrelated
inputs -- while what a template asks for is exactly a *relation* between kinds: a
run is consecutive numbers of one colour, a group one number across colours. This
reads the same vector back as the tokens it was concatenated from and lets the
kinds attend to one another before the pointer head scores templates against the
summary.

It is a drop-in for the trunk -- `(B, state_dim)` in, `(B, hidden)` out -- so the
bilinear head, the folding trick `q . (W d) == (q W) . d`, the features, the mask,
the decode and the RL fine-tune are all untouched, and an arm differs from its
control in the encoder alone.

`attention` is an ablation switch rather than a second class. `uniform` replaces
every attention weight with `1/N`, which keeps the token-wise pathway and the
permutation-invariant pooling and removes only the content-based routing -- so a
flip rate measured against it is a statement about *relations between kinds*, not
about tokenisation. `self` removes the mixing entirely, which leaves the summary
token reading its own nine scalars and is the floor rather than a control.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

from rummi.rules.config import RummiConfig

Attention = Literal["attend", "uniform", "self"]


@dataclass(frozen=True, slots=True)
class EncoderSpec:
    """How to build a trunk. `mlp` is exactly the two layers it replaces.

    Stored in a checkpoint as a plain dict, because `torch.load(weights_only=True)`
    reconstructs containers and primitives and nothing else.
    """

    kind: Literal["mlp", "attention"] = "mlp"
    dim: int = 128
    layers: int = 2
    heads: int = 4
    ffn: int = 256

    def as_dict(self) -> dict[str, int | str]:
        return asdict(self)


def build_trunk(
    cfg: RummiConfig, state_width: int, hidden: int, spec: EncoderSpec | None
) -> nn.Module:
    """The MLP by default, so an untouched caller keeps its exact state-dict keys."""
    if spec is None or spec.kind == "mlp":
        return nn.Sequential(
            nn.Linear(state_width, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
    return KindAttentionTrunk(cfg, state_width, hidden, spec)


class _Block(nn.Module):
    """Pre-norm self-attention then a feed-forward, the usual pair.

    Written out rather than taken from `nn.MultiheadAttention` because the ablation
    has to replace the attention weights themselves, which that module does not
    expose.
    """

    def __init__(self, dim: int, heads: int, ffn: int) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError(f"{dim} channels do not split into {heads} heads")
        self.heads = heads
        self.norm_attn = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn), nn.GELU(), nn.Linear(ffn, dim))

    def forward(self, x: torch.Tensor, attention: Attention) -> torch.Tensor:
        rows, tokens, dim = x.shape
        q, k, v = self.qkv(self.norm_attn(x)).chunk(3, dim=-1)
        if attention == "attend":
            shape = (rows, tokens, self.heads, dim // self.heads)
            mixed = (
                F.scaled_dot_product_attention(
                    q.view(shape).transpose(1, 2),
                    k.view(shape).transpose(1, 2),
                    v.view(shape).transpose(1, 2),
                )
                .transpose(1, 2)
                .reshape(rows, tokens, dim)
            )
        elif attention == "uniform":
            mixed = v.mean(1, keepdim=True).expand(rows, tokens, dim)
        else:
            mixed = v
        x = x + self.proj(mixed)
        return x + self.ffn(self.norm_ffn(x))


class KindAttentionTrunk(nn.Module):
    """`(B, state_dim)` in, `(B, hidden)` out, by way of `n_kinds + 1` tokens.

    One token per kind carries that kind's three counts plus a learned identity --
    which is what a colour relabelling permutes, so the augmentation acts on the
    tokens rather than on an opaque input vector. The state's scalars become a
    summary token, and the readout is that token alone: everything the head is
    allowed to know about the multiset has to have been attended into it.
    """

    def __init__(
        self, cfg: RummiConfig, state_width: int, hidden: int, spec: EncoderSpec
    ) -> None:
        super().__init__()
        self.n_kinds = cfg.n_kinds
        self.n_scalars = state_width - 3 * cfg.n_kinds
        if self.n_scalars < 0:
            raise ValueError(f"{state_width} is narrower than three blocks of kinds")
        self.attention: Attention = "attend"
        self.identity = nn.Parameter(torch.empty(cfg.n_kinds, spec.dim))
        nn.init.normal_(self.identity, std=0.02)
        self.counts = nn.Linear(3, spec.dim, bias=False)
        self.summary = nn.Linear(self.n_scalars, spec.dim)
        self.blocks = nn.ModuleList(
            _Block(spec.dim, spec.heads, spec.ffn) for _ in range(spec.layers)
        )
        self.norm = nn.LayerNorm(spec.dim)
        self.out = nn.Linear(spec.dim, hidden)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        k = self.n_kinds
        counts = torch.stack((state[:, :k], state[:, k : 2 * k], state[:, 2 * k : 3 * k]), -1)
        tokens = self.counts(counts) + self.identity
        summary = self.summary(state[:, 3 * k :])[:, None, :]
        x = torch.cat((summary, tokens), dim=1)
        for block in self.blocks:
            x = block(x, self.attention)
        return F.relu(self.out(self.norm(x[:, 0])))


def set_attention(net: nn.Module, attention: Attention) -> int:
    """Switch every attention trunk under `net`, and say how many were found."""
    found = 0
    for module in net.modules():
        if isinstance(module, KindAttentionTrunk):
            module.attention = attention
            found += 1
    return found


__all__ = [
    "Attention",
    "EncoderSpec",
    "KindAttentionTrunk",
    "build_trunk",
    "set_attention",
]
