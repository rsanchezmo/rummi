"""One's own turn, simulated from the observation alone.

A decoder that searches over primitive actions needs two things at every step of a
turn it has not played yet: the legal-action mask, and the observation the network
reads. Both come from a `BatchState`, and an agent is never handed one -- so this
builds a *synthetic* state that the env's own :func:`~rummi.env.numpy.masks.legal_actions`
and :func:`~rummi.env.observation.encode` cannot tell from the real thing, and the
engine steps it exactly as it would step the game.

That is sound because `PLACE`, `PICK`, `DISSOLVE`, `ASSIGN` and `END_TURN` read
only the acting seat's rack, the table, the workbench, what has left the rack this
turn, which slots are new, and the micro counter -- every one of which the
observation carries. `DRAW` is the exception: it draws from the pool, whose order
is exactly what a player may not know, and it is never simulated here. A decode
that reaches for it declines the turn instead.

**What is carried rather than reconstructed.** The observation merges the pool and
the opponents' racks into one `unseen` vector on purpose, so the split between them
is unknowable. This invents one: the other seats are dealt out of `unseen` until
their sizes match `rack_sizes`, and whatever is left is the pool, sized to
`scalars[POOL_SIZE]`. Nothing the mask or the encoder reads depends on the split
beyond those two -- `unseen` is a sum over it -- so the reconstruction is exact
rather than approximate, and `tests/test_turn_sim.py` holds it to real games step
by step. It also does not go stale: no opponent acts during one's own turn, so the
sizes, the meld flags and the pool count are as true at the end of the turn as at
its start.

`rummi/agents/learned/afterstate.py` answers the same question analytically for one
macro at a time -- it mirrors where the tiles go and rebuilds the observation field
by field. That is far cheaper per position and only says what a *macro* leaves
behind. A primitive decode has to see the table mid-turn, in pieces, which is
precisely where an analytic mirror has nothing to mirror.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rummi.agents.base import Observation
from rummi.env.numpy.state import BatchState, allocate
from rummi.rules.config import RummiConfig
from rummi.rules.observation import F_IS_NEW, MICRO_COUNT, POOL_SIZE


@dataclass(frozen=True, slots=True)
class TurnStart:
    """A batch of positions, in the fields an own-turn simulation needs.

    Batched rather than single because everything that consumes it -- a dataset, a
    beam, a whole vector env's turn boundaries -- has many at once, and one
    position is the batch of one.
    """

    rack: np.ndarray
    """`(B, K)` the acting seat's tiles."""
    board: np.ndarray
    """`(B, S, L)` the table as kind ids, `EMPTY` padded."""
    unseen: np.ndarray
    """`(B, K)` the pool and the other racks, indistinguishable."""
    rack_sizes: np.ndarray
    """`(B, P)` seat-rotated, so column 0 is the actor."""
    melded: np.ndarray
    """`(B, P)` seat-rotated."""
    workbench: np.ndarray
    """`(B, K)` tiles already lifted, non-empty only mid-turn."""
    placed: np.ndarray
    """`(B, K)` tiles that have left the rack this turn."""
    slot_new: np.ndarray
    """`(B, S)` slots created this turn -- what the opening meld is credited from."""
    pool_size: np.ndarray
    """`(B,)` how many tiles are left in the bag, which `unseen` does not say."""
    micro_count: np.ndarray
    """`(B,)` micro-actions already spent this turn."""

    @classmethod
    def from_obs(cls, obs: Observation, envs: np.ndarray | list[int] | int) -> TurnStart:
        index = np.atleast_1d(np.asarray(envs, dtype=np.int64))
        scalars = np.asarray(obs["scalars"])[index]
        return cls(
            rack=np.asarray(obs["rack"])[index].astype(np.int16),
            board=np.asarray(obs["table_sets"])[index].astype(np.int16),
            unseen=np.asarray(obs["unseen"])[index].astype(np.int16),
            rack_sizes=np.asarray(obs["rack_sizes"])[index].astype(np.int16),
            melded=np.asarray(obs["melded"])[index].astype(bool),
            workbench=np.asarray(obs["workbench"])[index].astype(np.int16),
            placed=np.asarray(obs["placed_this_turn"])[index].astype(np.int16),
            slot_new=np.asarray(obs["slot_features"])[index, :, F_IS_NEW].astype(bool),
            pool_size=scalars[:, POOL_SIZE].astype(np.int32),
            micro_count=scalars[:, MICRO_COUNT].astype(np.int32),
        )

    def __len__(self) -> int:
        return int(self.rack.shape[0])

    @staticmethod
    def stack(rows: list[TurnStart]) -> TurnStart:
        """One batch out of several, in order."""
        return TurnStart(
            **{
                name: np.concatenate([getattr(row, name) for row in rows])
                for name in TurnStart.__slots__
            }
        )

    def take(self, index: np.ndarray) -> TurnStart:
        """The rows at `index`, in that order. Repeats are allowed."""
        idx = np.asarray(index, dtype=np.int64)
        return TurnStart(
            rack=self.rack[idx],
            board=self.board[idx],
            unseen=self.unseen[idx],
            rack_sizes=self.rack_sizes[idx],
            melded=self.melded[idx],
            workbench=self.workbench[idx],
            placed=self.placed[idx],
            slot_new=self.slot_new[idx],
            pool_size=self.pool_size[idx],
            micro_count=self.micro_count[idx],
        )


def _dummy_racks(unseen: np.ndarray, rack_sizes: np.ndarray) -> np.ndarray:
    """`(B, P-1, K)` a deal of `unseen` that reproduces the other seats' sizes.

    Which tile goes to which opponent is arbitrary and unobservable, so the tiles
    are handed out in kind order and the split is read off two cumulative sums --
    seat *s* takes the tiles occupying positions `[lo_s, hi_s)` of the sorted
    multiset. Exact by construction: the sizes come out right and every tile that
    is not dealt is in the pool.
    """
    counts = unseen.astype(np.int64)
    upto = np.cumsum(counts, axis=-1)
    since = upto - counts
    bounds = np.cumsum(rack_sizes[:, 1:].astype(np.int64), axis=-1)
    hi = bounds[:, :, None]
    lo = (bounds - rack_sizes[:, 1:].astype(np.int64))[:, :, None]
    return np.maximum(
        np.minimum(upto[:, None, :], hi) - np.maximum(since[:, None, :], lo), 0
    )


def to_state(cfg: RummiConfig, starts: TurnStart, repeat: int = 1) -> BatchState:
    """A state per row of `starts`, each repeated `repeat` times, row-major.

    `repeat` is how a beam is seeded: `repeat` identical copies of a position that
    the search then drives apart, laid out so row `i * repeat + j` is hypothesis
    `j` of position `i`.
    """
    index = np.repeat(np.arange(len(starts)), repeat)
    rows = starts.take(index)
    b = len(rows)
    state = allocate(cfg, b)

    # Seat 0 acts, so the observation's already-rotated fields go straight in and
    # the encoder's own rotation is the identity.
    state.current[:] = 0
    state.racks[:, 0] = rows.rack
    state.racks[:, 1:] = _dummy_racks(rows.unseen, rows.rack_sizes)
    state.pool[:] = rows.unseen - state.racks[:, 1:].sum(1)
    state.draw_ptr[:] = cfg.n_tiles - rows.pool_size
    state.table_sets[:] = rows.board
    # Only DRAW reads the snapshot, and DRAW is never simulated -- but a turn that
    # started here would revert to exactly this table.
    state.table_snapshot[:] = rows.board
    state.workbench[:] = rows.workbench
    state.placed_rack[:] = rows.placed
    state.slot_new[:] = rows.slot_new
    state.melded[:] = rows.melded
    state.micro_count[:] = rows.micro_count
    return state


def state_from_obs(
    cfg: RummiConfig, obs: Observation, env: int, repeat: int = 1
) -> BatchState:
    """:func:`to_state` for one env of a batched observation."""
    return to_state(cfg, TurnStart.from_obs(obs, env), repeat)


def snapshot(cfg: RummiConfig, state: BatchState) -> TurnStart:
    """`state` back in :class:`TurnStart` form, through the env's own encoder.

    Read out rather than copied field by field, so a position walked forward on a
    simulated state re-enters the dataset as exactly what an agent would have been
    handed at that step.
    """
    from rummi.env.observation import encode

    return TurnStart.from_obs(encode(state), np.arange(state.batch_size))
