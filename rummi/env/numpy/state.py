"""Batched game state: a struct of fixed-shape arrays with leading batch dim ``B``.

Two deliberate choices shape this module.

*No randomness in the step loop.* Each env's deck is permuted once at reset into
``deck_order``; drawing just advances ``draw_ptr``. So a step is a pure function
of state and action, per-env streams are independent by construction, and the
torch/JAX ports need no PRNG plumbing in their hot loop.

*Mutation in place.* The NumPy reference mutates buffers to avoid per-step
allocation. The arithmetic is still index-and-mask only, so the JAX port is the
same expressions written functionally with ``.at[]``.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

from rummi.rules.config import RummiConfig
from rummi.rules.encoding import EMPTY, tables

NO_WINNER = -1
HISTORY_LEN = 8
"""Recent actions kept per env. Lives in the state, not the renderer, so the log
is complete even when rendering is throttled or switched off."""


@dataclass(slots=True)
class BatchState:
    cfg: RummiConfig
    racks: np.ndarray
    """``(B, P, K)`` per-player kind counts."""
    table_sets: np.ndarray
    """``(B, S, L)`` kind ids, ``EMPTY`` padded, tiles sorted within each slot."""
    workbench: np.ndarray
    """``(B, K)`` loose tiles held mid-turn; must be empty to end a turn."""
    placed_rack: np.ndarray
    """``(B, K)`` tiles that left the acting player's rack this turn."""
    slot_new: np.ndarray
    """``(B, S)`` slot was created during the current turn."""
    table_snapshot: np.ndarray
    """``(B, S, L)`` turn-start copy of ``table_sets``, restored by DRAW."""
    pool: np.ndarray
    """``(B, K)`` counts still in the bag; derived from ``deck_order``/``draw_ptr``
    but materialised because observations need it every step."""
    deck_order: np.ndarray
    """``(B, n_tiles)`` the env's shuffled deck, fixed at reset."""
    draw_ptr: np.ndarray
    """``(B,)`` index of the next tile to be drawn from ``deck_order``."""
    melded: np.ndarray
    current: np.ndarray
    micro_count: np.ndarray
    turn_count: np.ndarray
    consecutive_draws: np.ndarray
    last_action: np.ndarray
    action_history: np.ndarray
    """``(B, HISTORY_LEN)`` most recent actions, oldest first; ``-1`` for unused."""
    winner: np.ndarray
    done: np.ndarray
    truncated: np.ndarray

    # --- shapes --------------------------------------------------------------
    @property
    def batch_size(self) -> int:
        return self.racks.shape[0]

    @property
    def pool_size(self) -> np.ndarray:
        return (self.cfg.n_tiles - self.draw_ptr).astype(np.int16)

    def clone(self) -> BatchState:
        return BatchState(
            cfg=self.cfg,
            **{
                f.name: getattr(self, f.name).copy()
                for f in fields(self)
                if f.name != "cfg"
            },
        )

    def digest(self) -> str:
        """Stable hash of the whole batch state.

        The contract every backend must reproduce: two implementations that agree
        on this after the same seeded action sequence agree on the rules. Only
        semantic fields are included -- ``last_action`` and ``action_history`` are
        presentation, and dtypes are pinned so a backend cannot pass by accident
        of width.
        """
        import hashlib

        h = hashlib.sha256()
        for name in (
            "racks", "table_sets", "workbench", "placed_rack", "slot_new",
            "table_snapshot", "pool", "deck_order", "draw_ptr", "melded",
            "current", "micro_count", "turn_count", "consecutive_draws",
            "winner", "done", "truncated",
        ):
            arr = np.ascontiguousarray(getattr(self, name))
            h.update(name.encode())
            h.update(str(arr.shape).encode())
            h.update(arr.astype(np.int64).tobytes())
        return h.hexdigest()

    def select(self, indices) -> BatchState:
        """A new state holding just the given envs, in the given order.

        Used to check that a batched rollout equals the same envs run
        individually, and to hand a single env to a renderer without copying the
        whole batch.
        """
        idx = np.atleast_1d(np.asarray(indices))
        return BatchState(
            cfg=self.cfg,
            **{
                f.name: getattr(self, f.name)[idx].copy()
                for f in fields(self)
                if f.name != "cfg"
            },
        )

    # --- derived quantities --------------------------------------------------
    def table_counts(self) -> np.ndarray:
        """``(B, K)`` counts of every kind currently on the table."""
        return counts_of(self.cfg, self.table_sets)

    def slot_lengths(self) -> np.ndarray:
        """``(B, S)`` occupied positions per slot."""
        return (self.table_sets >= 0).sum(-1).astype(np.int16)

    def rack_values(self) -> np.ndarray:
        """``(B, P)`` penalty value of every rack, jokers at ``joker_penalty``."""
        t = tables(self.cfg)
        value = t.value.astype(np.int32).copy()
        value[self.cfg.joker_kind] = self.cfg.joker_penalty
        return self.racks.astype(np.int32) @ value

    def check_invariants(self) -> None:
        """Assert tile conservation. Cheap enough for fuzzing, not for benchmarks."""
        cfg = self.cfg
        total = (
            self.racks.sum(1).astype(np.int32)
            + self.table_counts().astype(np.int32)
            + self.workbench.astype(np.int32)
            + self.pool.astype(np.int32)
        )
        expected = tables(cfg).copies.astype(np.int32)
        bad = np.argwhere(total != expected[None, :])
        if bad.size:
            b, k = bad[0]
            raise AssertionError(
                f"tile conservation violated in env {b} for kind {k}: "
                f"{total[b, k]} != {expected[k]}"
            )
        if (self.pool.sum(-1) != self.pool_size).any():
            raise AssertionError("pool counts disagree with the draw pointer")


def counts_of(cfg: RummiConfig, kinds: np.ndarray) -> np.ndarray:
    """``(B, K)`` counts from any ``(B, ...)`` array of kind ids with ``EMPTY`` padding.

    Uses the offset-bincount trick so the scatter is one vectorised pass rather
    than a Python loop over the batch.
    """
    kinds = np.asarray(kinds)
    b = kinds.shape[0]
    flat = kinds.reshape(b, -1)
    occupied = flat >= 0
    offset = np.arange(b, dtype=np.int64)[:, None] * cfg.n_kinds
    # EMPTY entries are folded onto a scratch row that is then discarded.
    idx = np.where(occupied, flat.astype(np.int64) + offset, b * cfg.n_kinds)
    counts = np.bincount(idx.ravel(), minlength=b * cfg.n_kinds + 1)[: b * cfg.n_kinds]
    return counts.reshape(b, cfg.n_kinds).astype(np.int16)


def allocate(cfg: RummiConfig, batch_size: int) -> BatchState:
    """Allocate a zeroed state; :func:`rummi.env.numpy.deal.reset` fills it in."""
    b, p, k = batch_size, cfg.n_players, cfg.n_kinds
    s, ell = cfg.max_sets, cfg.max_set_len
    return BatchState(
        cfg=cfg,
        racks=np.zeros((b, p, k), dtype=np.int16),
        table_sets=np.full((b, s, ell), EMPTY, dtype=np.int16),
        workbench=np.zeros((b, k), dtype=np.int16),
        placed_rack=np.zeros((b, k), dtype=np.int16),
        slot_new=np.zeros((b, s), dtype=bool),
        table_snapshot=np.full((b, s, ell), EMPTY, dtype=np.int16),
        pool=np.zeros((b, k), dtype=np.int16),
        deck_order=np.zeros((b, cfg.n_tiles), dtype=np.int16),
        draw_ptr=np.zeros(b, dtype=np.int32),
        melded=np.zeros((b, p), dtype=bool),
        current=np.zeros(b, dtype=np.int16),
        micro_count=np.zeros(b, dtype=np.int32),
        turn_count=np.zeros(b, dtype=np.int32),
        consecutive_draws=np.zeros(b, dtype=np.int16),
        last_action=np.full(b, -1, dtype=np.int32),
        action_history=np.full((b, HISTORY_LEN), -1, dtype=np.int32),
        winner=np.full(b, NO_WINNER, dtype=np.int16),
        done=np.zeros(b, dtype=bool),
        truncated=np.zeros(b, dtype=bool),
    )
