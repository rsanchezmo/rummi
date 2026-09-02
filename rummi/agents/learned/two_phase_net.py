"""The same repartition, cut in two: which sets to break, then what to build.

`repartition_net.py` constructs the whole table one template at a time, and what
separates its greedy decode from its beam is sequence length: a mean solve holds
12.7 sets but **8.7 of them are already on the table unchanged**, so thirteen
~330-way choices are spent re-deriving a partition the solver only edited. The
real decision is a mean of 2.9 slots to dissolve and 4.0 sets to build.

So phase A picks slots to break -- ~12 candidates, one per occupied slot, `STOP`
when done -- and phase B covers the freed tiles plus the rack with the existing
template machinery. Two things make that a decomposition rather than a heuristic.
The kept slots leave the problem entirely, so phase B's `need`/`avail` are a much
smaller multiset over a much shorter sequence; and a phase-A subset is coverable
whatever it is, because the freed tiles came off the table as valid sets and
putting them back is in the space -- so the commitment a cheap decode makes first
is one it can always retreat from, up to the freed-joker dead end
`repartition_net` already names.

Phase A is emitted in slot order, and unlike phase B's template order that costs
nothing: a subset has no order of its own, every subset is expressible in
increasing order, and the mask is then just "an occupied slot above the last one".

The break is a **set** choice, so its features are the slot's own contents rather
than a fixed table's -- there is no `TemplateTable` analogue, and `describe` runs
on a per-state `(S + 1, K + 6)` block instead of a shared one. At 36 rows that is
still cheaper than the template head it feeds.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import cache

import numpy as np
import torch
from torch import nn

from rummi.agents.learned.repartition_net import (
    MASKED,
    Repartition,
    RepartitionNet,
    _Partial,
    build,
    expand,
    present_counts,
    stop_action,
)
from rummi.agents.learned.set_encoder import EncoderSpec
from rummi.agents.macro import set_templates
from rummi.rules.config import RummiConfig
from rummi.rules.encoding import tables

BREAK_DYNAMIC = 6
"""Per `(state, slot)` features: how far above the last break it sits, twice; what
the rack and the already-freed pool duplicate of it; how much it could spare
without falling under `min_set`; and how much of the rack its tiles could combine
with at all.

The last is the one that is not bookkeeping. A slot is worth breaking when its
tiles let *rack* tiles into a set, and nothing in the slot's own description says
that -- the compatibility count is the cheapest statement of it that is still a
matmul rather than a search."""

BREAK_SCALARS = 8


@cache
def _compatible(cfg: RummiConfig) -> np.ndarray:
    """`(K, K)` -- which kinds ever share a template, so which tiles can combine."""
    counts = set_templates(cfg).astype(np.float32)
    return ((counts.T @ counts) > 0).astype(np.float32)


def break_static_dim(cfg: RummiConfig) -> int:
    return cfg.n_kinds + 6


def break_state_dim(cfg: RummiConfig) -> int:
    return 3 * cfg.n_kinds + BREAK_SCALARS


def n_break_actions(cfg: RummiConfig) -> int:
    """One per slot, then `STOP`."""
    return cfg.max_sets + 1


def stop_break(cfg: RummiConfig) -> int:
    """Phase A's `STOP`, one past the last slot."""
    return cfg.max_sets


# --- the table, as slots --------------------------------------------------


def slot_counts(cfg: RummiConfig, board: np.ndarray) -> np.ndarray:
    """`(B, S, K)` counts from the `(B, S, L)` table of kind ids.

    A `bincount` over the flattened grid rather than a Python loop per slot: the
    trainer rebuilds this for every batch, and the boards are what it stores.
    """
    board = np.asarray(board)
    b, s, length = board.shape
    width = cfg.n_kinds + 1
    # `EMPTY` is -1, so shifting by one lands every pad on column 0 and needs no mask.
    flat = (
        np.arange(b * s, dtype=np.int64)[:, None] * width
        + board.reshape(b * s, length).astype(np.int64)
        + 1
    ).ravel()
    out = np.bincount(flat, minlength=b * s * width).reshape(b, s, width)
    return out[..., 1:].astype(np.int16)


def slot_static(cfg: RummiConfig, counts: np.ndarray) -> np.ndarray:
    """`(B, S + 1, K + 6)` description of each slot, `STOP` last."""
    counts = np.asarray(counts)
    b, s, _ = counts.shape
    value = tables(cfg).value.astype(np.float32)
    colour = tables(cfg).color
    scale = float(cfg.n_numbers * cfg.max_set_len)

    real = counts.astype(np.float32).copy()
    real[..., cfg.joker_kind] = 0.0
    length = counts.sum(-1).astype(np.float32)
    colours = np.zeros((b, s, max(cfg.n_colors, 1)), dtype=np.float32)
    for c in range(cfg.n_colors):
        colours[..., c] = real[..., np.flatnonzero(colour == c)].sum(-1)
    is_run = (length > 0) & ((colours > 0).sum(-1) <= 1)

    out = np.zeros((b, s + 1, break_static_dim(cfg)), dtype=np.float32)
    out[:, :s, : cfg.n_kinds] = counts
    out[:, :s, cfg.n_kinds] = length / cfg.max_set_len
    out[:, :s, cfg.n_kinds + 1] = (counts.astype(np.float32) @ value) / scale
    out[:, :s, cfg.n_kinds + 2] = is_run
    out[:, :s, cfg.n_kinds + 3] = (length > 0) & ~is_run
    out[:, :s, cfg.n_kinds + 4] = counts[..., cfg.joker_kind] / max(cfg.n_jokers, 1)
    out[:, s, cfg.n_kinds + 5] = 1.0  # STOP describes itself and nothing else
    return out


def break_feasible(cfg: RummiConfig, counts: np.ndarray, last: np.ndarray) -> np.ndarray:
    """`(B, S + 1)` -- an occupied slot above the last break, or `STOP`.

    Emitting the subset in slot order costs nothing and buys the whole mask: a
    slot already broken sits at or below `last`, so one comparison excludes it.
    """
    counts = np.asarray(counts)
    b, s, _ = counts.shape
    out = np.zeros((b, s + 1), dtype=bool)
    order = np.arange(s)[None]
    out[:, :s] = (counts.sum(-1) > 0) & (order > np.asarray(last)[:, None])
    out[:, s] = True
    return out


def break_dynamic(
    cfg: RummiConfig,
    counts: np.ndarray,
    rack: np.ndarray,
    freed: np.ndarray,
    last: np.ndarray,
) -> np.ndarray:
    """`(B, S + 1, BREAK_DYNAMIC)` -- what each slot would mean for *this* state."""
    counts = np.asarray(counts).astype(np.float32)
    rack = np.asarray(rack).astype(np.float32)
    freed = np.asarray(freed).astype(np.float32)
    b, s, _ = counts.shape
    length = np.maximum(counts.sum(-1), 1.0)
    ahead = np.arange(s, dtype=np.float32)[None] - np.asarray(last, dtype=np.float32)[:, None]
    reach = (counts > 0).astype(np.float32) @ _compatible(cfg)

    out = np.zeros((b, s + 1, BREAK_DYNAMIC), dtype=np.float32)
    out[:, :s, 0] = ahead / cfg.max_sets
    out[:, :s, 1] = np.where(ahead >= 0.0, 1.0 / (1.0 + np.maximum(ahead, 0.0)), 0.0)
    out[:, :s, 2] = np.minimum(counts, rack[:, None, :]).sum(-1) / length
    out[:, :s, 3] = np.minimum(counts, freed[:, None, :]).sum(-1) / length
    out[:, :s, 4] = np.maximum(counts.sum(-1) - cfg.min_set, 0.0) / cfg.max_set_len
    out[:, :s, 5] = ((reach > 0) & (rack[:, None, :] > 0)).sum(-1) / cfg.n_kinds
    return out


def break_state_features(
    cfg: RummiConfig,
    rack: np.ndarray,
    table: np.ndarray,
    freed: np.ndarray,
    occupied: np.ndarray,
    broken: np.ndarray,
    last: np.ndarray,
) -> np.ndarray:
    """`(B, break_state_dim)`. `last` is the slot broken previously, `-1` at the start."""
    rack = np.asarray(rack).astype(np.float32)
    table = np.asarray(table).astype(np.float32)
    freed = np.asarray(freed).astype(np.float32)
    copies = float(cfg.n_copies)
    total = float(cfg.n_tiles)
    jokers = float(max(cfg.n_jokers, 1))

    scalars = np.stack(
        [
            np.asarray(occupied, dtype=np.float32) / cfg.max_sets,
            np.asarray(broken, dtype=np.float32) / cfg.max_sets,
            freed.sum(-1) / total,
            rack.sum(-1) / total,
            table.sum(-1) / total,
            (np.asarray(last, dtype=np.float32) + 1.0) / cfg.max_sets,
            freed[:, cfg.joker_kind] / jokers,
            rack[:, cfg.joker_kind] / jokers,
        ],
        axis=-1,
    )
    return np.concatenate(
        [rack / copies, freed / copies, (table - freed) / copies, scalars], axis=-1
    ).astype(np.float32)


# --- reading a solve back into the two phases -----------------------------


def decompose(
    cfg: RummiConfig, board: np.ndarray, solution_sets
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[tuple[int, ...], ...]]:
    """`(kept, broken, to_build)` -- the solve, split at the slots it left alone.

    Only an exact content match counts as kept, which is the same rule
    `to_actions.plan` applies when it decides what to dissolve, so the two cannot
    disagree about what a decode has to expand into. Identical slots are
    interchangeable, so the lowest indices are kept and the assignment is canonical.
    """
    contents = [tuple(sorted(int(k) for k in row if k >= 0)) for row in np.asarray(board)]
    wanted = Counter(tuple(sorted(int(k) for k in s)) for s in solution_sets)
    kept, broken = [], []
    for slot, content in enumerate(contents):
        if not content:
            continue
        if wanted[content] > 0:
            wanted[content] -= 1
            kept.append(slot)
        else:
            broken.append(slot)
    return tuple(kept), tuple(broken), tuple(wanted.elements())


def freed_counts(
    cfg: RummiConfig, board: np.ndarray, broken, rack: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """`(need, avail)` for phase B: the dissolved tiles, and those plus the rack."""
    board = np.asarray(board)
    need = np.zeros(cfg.n_kinds, dtype=np.int64)
    for slot in broken:
        for kind in board[slot][board[slot] >= 0]:
            need[int(kind)] += 1
    return need, need + np.asarray(rack).astype(np.int64)


def slot_present(cfg: RummiConfig, boards: np.ndarray) -> np.ndarray:
    """`(B, S, T)` -- `present_counts` of each slot on its own.

    A subset's `present` is the sum of its slots' rows, so a caller that scores many
    subsets of the same table pays the template lookup once per slot instead of once
    per subset. Empty slots contribute nothing and are skipped.
    """
    boards = np.asarray(boards)
    b, s, _ = boards.shape
    out = np.zeros((b, s, stop_action(cfg)), dtype=np.float32)
    for i in range(b):
        for slot in range(s):
            row = boards[i, slot]
            if not (row >= 0).any():
                continue
            out[i, slot] = present_counts(cfg, row[None])
    return out


# --- the network ----------------------------------------------------------


class BreakNet(nn.Module):
    """Logits over the occupied slots plus `STOP`, scored against what each holds.

    The same bilinear shape as `RepartitionNet`, for the same reason -- a flat head
    would have to learn slot 7 from its index, and a slot index means nothing. What
    differs is that the candidate block is per state rather than shared, so
    `describe` cannot be folded into the query and runs on `(B, S + 1, ...)`.
    """

    stop_at: torch.Tensor

    def __init__(self, cfg: RummiConfig, hidden: int = 256, key: int = 64) -> None:
        super().__init__()
        stop = np.zeros(n_break_actions(cfg), dtype=np.float32)
        stop[-1] = 1.0
        self.register_buffer("stop_at", torch.as_tensor(stop))
        self.trunk = nn.Sequential(
            nn.Linear(break_state_dim(cfg), hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.query = nn.Linear(hidden, key)
        self.describe = nn.Sequential(
            nn.Linear(break_static_dim(cfg), key), nn.ReLU(), nn.Linear(key, key)
        )
        self.interact = nn.Linear(BREAK_DYNAMIC, key, bias=False)
        self.prior = nn.Linear(BREAK_DYNAMIC, 1, bias=False)
        self.stop_bias = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        state: torch.Tensor,
        static: torch.Tensor,
        dynamic: torch.Tensor,
        legal: torch.Tensor,
    ) -> torch.Tensor:
        q = self.query(self.trunk(state))
        logits = (self.describe(static) * q[:, None, :]).sum(-1)
        logits = logits + (dynamic * (q @ self.interact.weight)[:, None, :]).sum(-1)
        logits = logits + self.prior(dynamic).squeeze(-1) + self.stop_bias * self.stop_at
        return torch.where(legal, logits, torch.full_like(logits, MASKED))


class TwoPhaseNet(nn.Module):
    """The break head and the cover head, side by side.

    `cover` is a plain `RepartitionNet`, unchanged and with the same state dict, so
    a one-phase checkpoint initialises it and the two spaces stay comparable at the
    level of weights rather than only of scores.

    The heads share no parameters and no batch, so `cover_hidden` and
    `cover_encoder` change the cover alone: an arm that varies the cover's encoder
    trains a bit-identical break head beside it from the same seed.
    """

    def __init__(
        self,
        cfg: RummiConfig,
        hidden: int = 256,
        key: int = 64,
        cover_hidden: int | None = None,
        cover_encoder: EncoderSpec | None = None,
    ) -> None:
        super().__init__()
        self.breaker = BreakNet(cfg, hidden=hidden, key=key)
        self.cover = RepartitionNet(
            cfg, hidden=cover_hidden or hidden, key=key, encoder=cover_encoder
        )


def two_phase_from_checkpoint(cfg: RummiConfig, checkpoint: dict) -> TwoPhaseNet:
    """Rebuild the net a checkpoint holds -- widths, cover encoder and weights.

    Written once and read by both trainers, so a checkpoint carrying an encoder the
    reader does not reconstruct cannot silently load into the default one.
    """
    spec = checkpoint.get("cover_encoder")
    net = TwoPhaseNet(
        cfg,
        hidden=checkpoint["hidden"],
        key=checkpoint["key"],
        cover_hidden=checkpoint.get("cover_hidden"),
        cover_encoder=EncoderSpec(**spec) if spec else None,
    )
    net.load_state_dict(checkpoint["state"])
    return net


class TwoPhaseScorer:
    """The two forward passes `decode_two_phase` makes, with no autograd tape."""

    def __init__(self, net: TwoPhaseNet) -> None:
        self.net = net.eval()

    def brk(
        self, state: np.ndarray, static: np.ndarray, dynamic: np.ndarray, legal: np.ndarray
    ) -> np.ndarray:
        with torch.no_grad():
            out = self.net.breaker(
                torch.from_numpy(state),
                torch.from_numpy(static),
                torch.from_numpy(dynamic),
                torch.from_numpy(legal),
            )
        return out.numpy()

    def cover(
        self, state: np.ndarray, dynamic: np.ndarray, legal: np.ndarray
    ) -> np.ndarray:
        with torch.no_grad():
            out = self.net.cover(
                torch.from_numpy(state), torch.from_numpy(dynamic), torch.from_numpy(legal)
            )
        return out.numpy()


# --- decoding -------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Break:
    """One live break: the slots dissolved so far and the pool they left."""

    logp: float
    broken: tuple[int, ...]
    freed: np.ndarray
    last: int


def _break_round(
    cfg: RummiConfig,
    score,
    counts: np.ndarray,
    static: np.ndarray,
    rack: np.ndarray,
    table: np.ndarray,
    occupied: int,
    live: list[_Break],
    beam: int,
    finished: list[tuple[float, tuple[int, ...]]],
) -> list[_Break]:
    """One round of phase A, the same best-first shape as the cover loop.

    The table is the same for every live break, so its block is broadcast rather
    than rebuilt -- copied only where it crosses into torch, which will not read a
    zero stride.
    """
    rows = len(live)
    freed = np.stack([row.freed for row in live])
    last = np.array([row.last for row in live])
    counts = np.broadcast_to(counts, (rows, *counts.shape))
    racks = np.broadcast_to(rack, (rows, rack.shape[0]))
    legal = break_feasible(cfg, counts, last)
    logits = score(
        break_state_features(
            cfg,
            racks,
            np.broadcast_to(table, (rows, table.shape[0])),
            freed,
            np.full(rows, occupied),
            np.array([len(row.broken) for row in live]),
            last,
        ),
        np.broadcast_to(static, (rows, *static.shape)).copy(),
        break_dynamic(cfg, counts, racks, freed, last),
        legal,
    )
    logits = logits - logits.max(-1, keepdims=True)
    logp = logits - np.log(np.exp(logits).sum(-1, keepdims=True))

    stop = stop_break(cfg)
    proposals: list[_Break] = []
    for row, partial in enumerate(live):
        options = np.flatnonzero(legal[row])
        for action in options[np.argsort(-logp[row, options])][:beam].tolist():
            total = partial.logp + float(logp[row, action])
            if action == stop:
                finished.append((total, partial.broken))
                continue
            proposals.append(
                _Break(
                    total,
                    (*partial.broken, action),
                    partial.freed + counts[row, action].astype(np.int64),
                    action,
                )
            )
    proposals.sort(key=lambda row: -row.logp)
    return proposals[:beam]


def decode_two_phase(
    cfg: RummiConfig,
    scorer: TwoPhaseScorer,
    rack: np.ndarray,
    board: np.ndarray,
    beam: int = 1,
    monotone: bool = True,
    breaks: int | None = None,
) -> Repartition | None:
    """Break, then cover: a repartition of the whole table, or `None`.

    The two phases share one beam rather than nesting: phase A's best `breaks`
    subsets seed a single cover search of width `beam`, so the cost stays linear
    where a cover per subset would be quadratic, and a subset that covers badly is
    simply outbid by one that covers well.

    `breaks` defaults to `beam`, which is what makes the widths comparable to the
    one-phase decode's. Widening it alone is nearly free -- phase A scores ~12
    candidates where the cover scores ~330 -- so it is a separate knob rather than
    a fixed ratio.
    """
    if breaks is None:
        breaks = beam
    board = np.asarray(board)
    rack = np.asarray(rack).astype(np.int64)
    counts = slot_counts(cfg, board[None])[0].astype(np.int64)
    static = slot_static(cfg, counts[None])[0]
    occupied = int((counts.sum(-1) > 0).sum())
    table = counts.sum(0)

    subsets: list[tuple[float, tuple[int, ...]]] = []
    live_a = [_Break(0.0, (), np.zeros(cfg.n_kinds, dtype=np.int64), -1)]
    for _ in range(occupied + 1):
        if not live_a:
            break
        live_a = _break_round(
            cfg, scorer.brk, counts, static, rack, table, occupied, live_a, breaks, subsets
        )
    if not subsets:
        return None
    subsets.sort(key=lambda row: -row[0])

    start: dict[tuple[int, ...], tuple[np.ndarray, np.ndarray, tuple[tuple[int, ...], ...]]] = {}
    live_b: list[_Partial] = []
    for logp, broken in subsets[:breaks]:
        need, avail = freed_counts(cfg, board, broken, rack)
        held = set(broken)
        kept = tuple(
            tuple(sorted(int(k) for k in board[slot] if k >= 0))
            for slot in range(board.shape[0])
            if slot not in held and (board[slot] >= 0).any()
        )
        start[broken] = (need, avail, kept)
        live_b.append(
            _Partial(
                logp,
                (),
                need,
                avail,
                present_counts(cfg, board[list(broken)]),
                0 if monotone else -1,
                len(kept),
                broken,
            )
        )

    finished: list[tuple[float, tuple[int, ...], tuple[int, ...]]] = []
    for _ in range(cfg.max_sets + 1):
        if not live_b:
            break
        live_b = expand(cfg, scorer.cover, live_b, beam, monotone, finished)

    finished.sort(key=lambda row: -row[0])
    best: Repartition | None = None
    for _, sequence, broken in finished[: 4 * beam]:
        need, avail, kept = start[broken]
        found = build(cfg, need, avail, list(sequence), reserved=len(kept))
        if found is None:
            continue
        whole = Repartition(found.templates, (*kept, *found.sets), found.played)
        if best is None or whole.tiles_played > best.tiles_played:
            best = whole
    return best


__all__ = [
    "BREAK_DYNAMIC",
    "BREAK_SCALARS",
    "BreakNet",
    "TwoPhaseNet",
    "TwoPhaseScorer",
    "break_dynamic",
    "break_feasible",
    "break_state_dim",
    "break_state_features",
    "break_static_dim",
    "decode_two_phase",
    "decompose",
    "freed_counts",
    "n_break_actions",
    "slot_counts",
    "slot_present",
    "slot_static",
    "stop_break",
    "two_phase_from_checkpoint",
]
