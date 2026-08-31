"""What the opponent did on its turns, carried across steps.

The observation is one state and the nets are feedforward, so opponent inference
is structurally impossible from a single row: the signal lives *between* rows.
Against a deterministic opponent that is a hard constraint rather than a hint --
`greedy` lays off every tile the table would take, so a lay-off it **declined**
proves it holds none of that kind.

Two seats only. Attribution rests on "everything the table gained while the
learner was not acting is the opponent's", which is exact with one opponent and a
sum over several otherwise, so :class:`OpponentHistory` refuses a wider config.

Owned by the agent, not by a trainer. A tracker wired into the training loop
alone would leave evaluation feeding the network zeros where it was trained on
history, and a train/eval mismatch of that kind invalidates the score rather than
degrading it.

Three traps, each paid for once:

* Aggregate per-kind counts, never per-slot deltas. Slots are re-sorted into
  canonical order at ``END_TURN``, so a slot-indexed diff across a turn boundary
  compares two unrelated sets.
* ``DRAW`` reverts the turn (SPEC.md section 4). The table the opponent inherited
  is then the one at the learner's *own* turn start, not the one it had built by
  the time it drew.
* An opponent's draw is invisible in ``unseen``: the pool and its rack are merged
  there precisely so neither can be read, and a draw moves tiles within the merged
  vector. It shows up instead as a turn whose table delta is empty.
"""

from __future__ import annotations

import numpy as np

from rummi.agents.base import Observation, turn_starting
from rummi.agents.greedy_agent import appendable
from rummi.agents.macro import Choose, MacroAgent
from rummi.rules.config import RummiConfig

H_DRAWS = 0
H_SINCE_PLAY = 1
H_RACK = 2
H_RACK_TREND = 3
H_TURNS = 4
HISTORY_SCALARS = 5
"""Index into the scalar tail of a history row. Named so nothing counts positions."""

SINCE_PLAY_SCALE = 10.0
"""An opponent that has drawn ten turns running is already at the tail of what
this distinguishes; past that the column saturating costs nothing."""

TURNS_SCALE = 40.0
"""Opponent turns seen, on the scale a standard game actually runs. It says how
much evidence stands behind the two histograms, which differ in age: `played` is
cumulative and `declined` decays."""

RACK_TREND_ALPHA = 0.5
"""EMA weight on the newest per-turn rack delta. High on purpose -- the useful
question is whether the opponent is shedding *now*, and a long memory of that is
what the cumulative histogram already is."""


def history_dim(cfg: RummiConfig) -> int:
    return 2 * cfg.n_kinds + HISTORY_SCALARS


def history_scale(cfg: RummiConfig, decay: float) -> np.ndarray:
    """`(history_dim,)` divisor per position, the same contract as `feature_scale`.

    Order and divisor as data, in one place, because the trainer and the evaluator
    both build this block and a silent disagreement between them would read as a
    weaker policy rather than as a bug.
    """
    k = cfg.n_kinds
    copies = float(max(cfg.n_copies, cfg.n_jokers, 1))

    parts = [
        np.full(k, copies, dtype=np.float32),                    # opp_played
        # The geometric ceiling of the decayed count: declining every turn forever.
        np.full(k, 1.0 / max(1.0 - decay, 1e-3), dtype=np.float32),   # declined
        np.zeros(HISTORY_SCALARS, dtype=np.float32),
    ]
    scalars = parts[-1]
    scalars[H_DRAWS] = max(1, cfg.n_tiles)
    scalars[H_SINCE_PLAY] = SINCE_PLAY_SCALE
    scalars[H_RACK] = max(1, cfg.n_tiles)
    scalars[H_RACK_TREND] = max(1, cfg.max_set_len)
    scalars[H_TURNS] = TURNS_SCALE

    scale = np.concatenate(parts).astype(np.float32)
    assert scale.shape == (history_dim(cfg),), (scale.shape, history_dim(cfg))
    assert (scale > 0).all(), "a zero divisor would produce inf features"
    return scale


def _table_counts(cfg: RummiConfig, boards: np.ndarray) -> np.ndarray:
    """`(M, K)` count of each kind on each `(S, L)` board, `EMPTY` padding dropped.

    One `bincount` over row-offset kind ids rather than `np.add.at`, which is the
    scattered-write path and an order of magnitude slower on batches this shape.
    """
    m = boards.shape[0]
    flat = np.asarray(boards).reshape(m, -1)
    held = flat >= 0
    rows = np.repeat(np.arange(m), flat.shape[1])[held.ravel()]
    kinds = flat.ravel()[held.ravel()].astype(np.int64)
    return np.bincount(rows * cfg.n_kinds + kinds, minlength=m * cfg.n_kinds).reshape(
        m, cfg.n_kinds
    )


class OpponentHistory:
    """Per-env memory of the opponent's turns, and the feature row it comes to.

    Driven from an agent's act path: :meth:`observe` before the decision,
    :meth:`record` after it. Both take the caller's ``active`` and touch nothing
    else, because a row where another seat is acting is not this opponent's turn.
    """

    def __init__(self, cfg: RummiConfig, decay: float = 0.8) -> None:
        if cfg.n_players != 2:
            raise ValueError(
                f"opponent history is exact at two seats and a sum over "
                f"{cfg.n_players}; the table delta cannot say which of several "
                "opponents played what"
            )
        self.cfg = cfg
        self.decay = float(decay)
        self.scale = history_scale(cfg, self.decay)
        self.reset(0)

    def reset(self, n_envs: int) -> None:
        cfg = self.cfg
        self.n_envs = n_envs
        self._has_turn = np.zeros(n_envs, dtype=bool)
        self._has_commit = np.zeros(n_envs, dtype=bool)
        # The env re-deals on the step *after* it reports done, so exactly one
        # observation -- the terminal one -- reaches the agent between a clear and
        # the fresh deal. Ignoring it is what keeps a turn boundary from being read
        # across two episodes.
        self._skip = np.zeros(n_envs, dtype=bool)

        shape = (n_envs, cfg.max_sets, cfg.max_set_len)
        self._turn_board = np.full(shape, -1, dtype=np.int16)
        self._turn_counts = np.zeros((n_envs, cfg.n_kinds), dtype=np.int64)
        self._turn_melded = np.zeros(n_envs, dtype=bool)
        self._commit_board = np.full(shape, -1, dtype=np.int16)
        self._commit_counts = np.zeros((n_envs, cfg.n_kinds), dtype=np.int64)
        self._commit_melded = np.zeros(n_envs, dtype=bool)

        self._played = np.zeros((n_envs, cfg.n_kinds), dtype=np.float32)
        self._declined = np.zeros((n_envs, cfg.n_kinds), dtype=np.float32)
        self._scalars = np.zeros((n_envs, HISTORY_SCALARS), dtype=np.float32)
        self._rack_prev = np.zeros(n_envs, dtype=np.float32)

    def clear(self, done: np.ndarray) -> None:
        """Forget everything about the envs that just finished.

        Envs are recycled, so nothing may survive a re-deal; and the terminal
        observation still to come is not a turn boundary of the next episode.
        """
        finished = np.asarray(done, dtype=bool)
        if not finished.any():
            return
        self._has_turn[finished] = False
        self._has_commit[finished] = False
        self._skip[finished] = True
        self._turn_board[finished] = -1
        self._turn_counts[finished] = 0
        self._turn_melded[finished] = False
        self._commit_board[finished] = -1
        self._commit_counts[finished] = 0
        self._commit_melded[finished] = False
        self._played[finished] = 0.0
        self._declined[finished] = 0.0
        self._scalars[finished] = 0.0
        self._rack_prev[finished] = 0.0

    def row(self, env: int) -> np.ndarray:
        """`(history_dim,)` scaled block for one env, to concatenate onto the
        observation features."""
        raw = np.concatenate(
            [self._played[env], self._declined[env], self._scalars[env]]
        )
        return (raw / self.scale).astype(np.float32)

    # --- the two hooks -------------------------------------------------------
    def observe(self, obs: Observation, active: np.ndarray | None = None) -> None:
        """Fold in the opponent's last turn, then snapshot this turn's start.

        Called before the decision, so :meth:`row` reads a block that already
        accounts for everything the opponent did since the learner last acted.
        """
        seen = self._seen(obs, active)
        # One observation each is dropped after a clear, not a whole flag's worth:
        # the next one is the fresh deal and starts a real turn.
        skipping = self._skip & seen
        self._skip &= ~seen
        fresh = turn_starting(obs) & seen & ~skipping
        envs = np.flatnonzero(fresh)
        if envs.size == 0:
            return

        boards = np.asarray(obs["table_sets"])[envs]
        counts = _table_counts(self.cfg, boards)
        opponent_rack = np.asarray(obs["rack_sizes"])[envs, 1].astype(np.float32)

        closing = envs[self._has_commit[envs]]
        if closing.size:
            self._attribute(closing, counts[self._has_commit[envs]],
                            opponent_rack[self._has_commit[envs]])

        self._turn_board[envs] = boards
        self._turn_counts[envs] = counts
        # The opponent cannot act between here and the learner's committing move,
        # so this one read answers "was it allowed to lay off" for both branches.
        self._turn_melded[envs] = np.asarray(obs["melded"])[envs, 1].astype(bool)
        self._scalars[envs, H_RACK] = opponent_rack
        self._rack_prev[envs] = opponent_rack
        self._has_turn[envs] = True

    def record(
        self,
        obs: Observation,
        actions: np.ndarray,
        active: np.ndarray | None = None,
    ) -> None:
        """Remember the table the opponent is about to inherit.

        ``END_TURN`` hands over what is on the table now; ``DRAW`` reverts the turn
        and hands over what was there when the turn began, which is why the
        turn-start snapshot is kept at all.
        """
        cfg = self.cfg
        seen = self._seen(obs, active) & self._has_turn
        chosen = np.asarray(actions)
        ending = np.flatnonzero(seen & (chosen == cfg.end_turn_action))
        drawing = np.flatnonzero(seen & (chosen == cfg.draw_action))
        if ending.size:
            boards = np.asarray(obs["table_sets"])[ending]
            self._commit_board[ending] = boards
            self._commit_counts[ending] = _table_counts(cfg, boards)
        if drawing.size:
            self._commit_board[drawing] = self._turn_board[drawing]
            self._commit_counts[drawing] = self._turn_counts[drawing]
        committed = np.concatenate([ending, drawing])
        if committed.size:
            self._commit_melded[committed] = self._turn_melded[committed]
            self._has_commit[committed] = True

    # --- internals -----------------------------------------------------------
    def _seen(self, obs: Observation, active: np.ndarray | None) -> np.ndarray:
        n = np.asarray(obs["rack"]).shape[0]
        if n != self.n_envs:
            self.reset(n)
        if active is None:
            return np.ones(n, dtype=bool)
        return np.asarray(active, dtype=bool)

    def _attribute(
        self, envs: np.ndarray, counts: np.ndarray, opponent_rack: np.ndarray
    ) -> None:
        """One opponent turn per env, read off the table it left behind."""
        cfg = self.cfg
        # Clipped because a rearranging opponent moves a tile between slots; the
        # count of a kind on the table never falls, so a negative is a rounding of
        # that rather than a play to attribute.
        delta = np.maximum(counts - self._commit_counts[envs], 0)
        played = delta > 0
        acted = delta.sum(-1) > 0

        # What the table the opponent inherited would have taken from *any* hand:
        # every kind it then failed to play is one it does not hold, given an
        # opponent that lays off whatever it can.
        hand = np.ones((envs.size, cfg.n_kinds), dtype=np.int16)
        takes = appendable(cfg, self._commit_board[envs], hand).any(-2)
        declined = takes & ~played & self._commit_melded[envs][:, None]

        self._played[envs] += delta
        block = self._declined[envs] * self.decay + declined
        # A kind it did play is evidence spent: it held one after all.
        self._declined[envs] = np.where(played, 0.0, block)

        scalars = self._scalars[envs]
        scalars[:, H_DRAWS] += ~acted
        scalars[:, H_SINCE_PLAY] = np.where(acted, 0.0, scalars[:, H_SINCE_PLAY] + 1.0)
        scalars[:, H_RACK_TREND] += RACK_TREND_ALPHA * (
            (opponent_rack - self._rack_prev[envs]) - scalars[:, H_RACK_TREND]
        )
        scalars[:, H_TURNS] += 1.0
        self._scalars[envs] = scalars


class HistoryMacroAgent(MacroAgent):
    """`MacroAgent` that keeps an :class:`OpponentHistory` in step with its own play.

    A subclass rather than a wrapper for the same reason `FixedOpponentEnv` is one:
    the tracker has to see the observation the decision was made on *and* the action
    that decision came to, and only :meth:`act` holds both at once.
    """

    name = "macro-history"

    def __init__(
        self,
        cfg: RummiConfig,
        history: OpponentHistory,
        choose: Choose | None = None,
    ) -> None:
        super().__init__(cfg, choose=choose)
        self.history = history

    def reset(self, n_envs: int) -> None:
        super().reset(n_envs)
        self.history.reset(n_envs)

    def act(
        self, obs: Observation, mask: np.ndarray, active: np.ndarray | None = None
    ) -> np.ndarray:
        self.history.observe(obs, active)
        actions = super().act(obs, mask, active)
        self.history.record(obs, actions, active)
        return actions
