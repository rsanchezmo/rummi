"""Oracle one-step regret: the ceiling on what choosing a turn can be worth.

Every agent in this repo maximises the current turn, and every strategy proposed
above that -- board shaping, rack potential, history, self-play -- has come back
null at the resolution the suites afford. Those arms each measured one candidate.
This measures the *bound*: how much is left for any of them to find.

The construction is hindsight, and exact rather than sampled. A step contains no
randomness -- the deck is permuted once at reset and drawing only advances a
pointer -- so a state cloned at a turn boundary and continued with deterministic
agents replays the rest of the game tile for tile. So the game that was played
*is* the baseline rollout for every decision in it, and a deviation is priced by
cloning that boundary, executing some other whole turn, and continuing. The
alternatives come from CP-SAT under added constraints: the same number of tiles
shed onto a different table, fewer tiles shed, no rearrangement at all, or
nothing played.

A mean over a deviation type cannot see a *targeted* effect inside it, so every
turn -- the alternatives and the one actually played -- also records what it leaves
behind. The oracle half is the opponent's best reply under CP-SAT against its true
rack: the most tiles it could shed, which answers both questions asked of it at
once, since zero means the table offers it nothing and its own rack size means the
door to going out is open. Beside it sits what the opponent actually shed on its
next turn in the continuation, and the unseen-weighted lay-off doors
`tools/denial_ab.py` scores a table by -- the same quantity, computed from nothing
hidden, so a targeted oracle effect can be checked against the signal a real agent
would have to find it with.

The hypothesis those exist for is endgame denial: the opponent is a tile or two
from finishing and, among the tables that shed the same tiles, one closes the door
it needs.

Two things this cannot bound. It is perfect hindsight -- the future deck and the
opponent's rack are both visible to the enumeration, so no real policy can reach
it -- and it deviates on one turn only, so a strategy that gives something up now
to collect later is outside it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field, fields
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from rummi.agents import build
from rummi.agents.base import act_by_seat
from rummi.agents.greedy_agent import appendable
from rummi.rules.config import CONFIG_BY_NAME, RummiConfig
from rummi.env.numpy.deal import reset as deal_reset
from rummi.env.numpy.deal import reset_envs
from rummi.env.numpy.engine import step as engine_step
from rummi.env.numpy.masks import legal_actions
from rummi.env.numpy.sets import summarize
from rummi.env.numpy.state import BatchState
from rummi.env.observation import encode
from rummi.solver.ilp import Objective, solve_turn
from rummi.solver.to_actions import plan, slot_contents

# Permeability is imported rather than recomposed. It is the metric the denial arm
# was scored on, and a second copy of it could disagree with that one about what a
# door is -- which is the whole point of recording it here.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from denial_ab import permeability

AGENT = "frugal"
KINDS = ("cpsat_max", "same_tiles_other_table", "fewer_tiles", "frozen_table", "draw")
"""Reported alternative types, in the order the enumeration assigns them: a
deduplicated turn keeps the tag of the first constraint that reached it."""

CONTINUE = "__continue__"
"""The zero-deviation candidate. Rolling it out must reproduce the game exactly,
which is the check that cloning a boundary is exact at all."""
REPLAY = "__base_replay__"
"""A CP-SAT alternative whose resulting table and played tiles match what the
game actually did. Its outcome must equal the baseline's, which is the check that
executing a turn through `plan` is the same turn."""

Signature = tuple[tuple[tuple[int, ...], ...], tuple[int, ...]]


class HarnessError(RuntimeError):
    """An illegal step or an unreachable target: a bug here, not a measurement."""


# --- what a turn leaves behind ----------------------------------------------


@dataclass(slots=True)
class Left:
    """What one finished turn leaves: the table's own properties, then what the
    continuation did with them.

    The oracle half reads the opponent's true rack, which is legitimate here for the
    same reason the rest of the file is -- this prices a bound, not a policy -- and
    it is one solve per opponent, not two: `opp_shed` is the *most* tiles that
    opponent could shed against this table, so zero means the table offers it
    nothing and its own rack size means its exit is open.
    """

    perm: float = 0.0
    """Unseen-weighted lay-off doors: `denial_ab.permeability`, the observable proxy."""
    doors: int = 0
    """How many distinct kinds any slot would accept, unweighted."""
    opp_shed: list[int] = field(default_factory=list)
    """Per opponent. -1 where the turn ended the game, which leaves no reply to price."""
    ended: bool = False
    next_shed: list[int | None] = field(default_factory=list)
    """Per opponent, what it actually shed on its own next turn: negative where it
    drew instead, None where it never got one."""
    next_win: list[bool] = field(default_factory=list)
    """Per opponent, whether that turn is the one it wins on."""
    own_shed: int | None = None
    """The deviator's own next turn, the same way."""

    def as_json(self) -> dict:
        return {
            "perm": round(self.perm, 3),
            "doors": self.doors,
            "opp_shed": self.opp_shed,
            "ended": self.ended,
            "next_shed": self.next_shed,
            "next_win": self.next_win,
            "own_shed": self.own_shed,
        }


def unseen_for(state: BatchState, env: int, seat: int) -> np.ndarray:
    """The pool plus every rack but ``seat``'s -- what the observation merges into
    `unseen`, read off the state rather than encoded, because by the time a turn has
    an afterstate `current` has already moved on to the next seat."""
    racks = state.racks[env].astype(np.int64)
    return racks.sum(0) - racks[seat] + state.pool[env].astype(np.int64)


def sparse_rack(rack: np.ndarray) -> list[list[int]]:
    return [[int(k), int(n)] for k, n in enumerate(rack) if n]


def board_features(
    cfg: RummiConfig,
    state: BatchState,
    env: int,
    seat: int,
    opps: list[int],
    time_limit: float,
) -> Left:
    """Read the afterstate the env actually reached -- not the target handed to
    `plan` -- so a turn is scored on the table it landed on."""
    rows = state.table_sets[env]
    unseen = unseen_for(state, env, seat)
    left = Left(
        perm=float(permeability(cfg, rows, unseen)),
        doors=int(appendable(cfg, rows, unseen).any(0).sum()),
        ended=bool(state.done[env]),
    )
    for p in opps:
        if left.ended:
            left.opp_shed.append(-1)
            continue
        reply = solve_turn(
            cfg,
            state.racks[env, p].astype(np.int64),
            rows,
            bool(state.melded[env, p]),
            time_limit=time_limit,
        )
        left.opp_shed.append(int(reply.tiles_played) if reply.feasible else 0)
    return left


@dataclass(slots=True)
class Candidate:
    kind: str
    tiles: int
    actions: list[int]
    decision: int = -1
    """Index into the game's decision list."""


@dataclass(slots=True)
class Decision:
    boundary: int
    turn: int
    seat: int
    pre_meld: bool
    own_rack: int
    opp_rack: int
    pool: int
    base_tiles: int
    base_won: bool
    cpsat_differs: bool
    n_tables: int
    """Distinct max-tiles target tables found, the played one included."""
    exhausted: bool
    """The k-best enumeration proved there are no further max-tiles tables."""
    freeze: str
    """Why the no-rearrangement turn is or is not a distinct alternative."""
    opps: list[int] = field(default_factory=list)
    """The other seats, ascending. Every per-opponent list below is aligned with it."""
    opp_sizes: list[int] = field(default_factory=list)
    opp_racks: list[list[list[int]]] = field(default_factory=list)
    """Each opponent's true rack as ``[kind, count]`` pairs -- the oracle the
    feasibility check reads, kept so a later question can be asked of the same run
    without replaying it."""
    base: Left = field(default_factory=Left)
    """The played turn, measured exactly as an alternative is, so a delta is paired."""
    alts: list[dict] = field(default_factory=list)


# --- playing the baseline ---------------------------------------------------


@dataclass(slots=True)
class Boundary:
    state: BatchState
    seat: int
    turn: int
    racks: np.ndarray
    melded: np.ndarray
    pool: int
    actions: list[int] = field(default_factory=list)


def play_base(cfg: RummiConfig, seed: np.random.SeedSequence, max_steps: int) -> tuple[
    list[Boundary], BatchState
]:
    """One `frugal`-vs-`frugal` deal, with the state cloned at every turn boundary."""
    state = deal_reset(cfg, 1, seed=0)
    reset_envs(state, np.arange(1), [seed])
    seats = [build(AGENT, cfg) for _ in range(cfg.n_players)]
    for agent in seats:
        agent.reset(1)

    boundaries: list[Boundary] = []
    for _ in range(max_steps):
        if state.done[0]:
            break
        if state.micro_count[0] == 0:
            boundaries.append(
                Boundary(
                    state=state.clone(),
                    seat=int(state.current[0]),
                    turn=int(state.turn_count[0]),
                    racks=state.racks[0].copy(),
                    melded=state.melded[0].copy(),
                    pool=int(state.pool_size[0]),
                )
            )
        summary = summarize(cfg, state.table_sets)
        mask = legal_actions(state, summary)
        actions, illegal = act_by_seat(
            seats, cfg, state.current, state.done, mask, encode(state, summary)
        )
        if illegal:
            raise HarnessError("the baseline agents proposed a masked action")
        boundaries[-1].actions.append(int(actions[0]))
        engine_step(state, actions, mask)
    return boundaries, state


def signature(table: np.ndarray, played: np.ndarray) -> Signature:
    """What identifies a turn: the table it leaves and the tiles it sheds."""
    sets = tuple(sorted(c for c in slot_contents(table) if c))
    return sets, tuple(int(v) for v in played)


# --- enumerating the alternatives -------------------------------------------


def _target(board: np.ndarray, solution, melded: bool) -> list[tuple[int, ...]]:
    """The whole resulting table. Pre-meld the solver never sees the existing sets,
    which it may not touch, so they are added back exactly as `optimal` does."""
    target = list(solution.sets)
    if not melded:
        target += [c for c in slot_contents(board) if c]
    return target


def enumerate_turns(
    cfg: RummiConfig,
    boundary: Boundary,
    base_sig: Signature,
    kbest: int,
    time_limit: float,
    replay_base: bool,
) -> tuple[list[Candidate], dict]:
    """Every other whole turn reachable from this boundary, deduplicated.

    Turns are deduplicated by what they leave behind -- the resulting table and
    the tiles shed -- so the same turn found under two different constraints is
    rolled out once, tagged by whichever constraint reached it first.
    """
    board = boundary.state.table_sets[0]
    rack = boundary.state.racks[0, boundary.seat].astype(np.int64)
    melded = bool(boundary.melded[boundary.seat])

    candidates: list[Candidate] = []
    seen: set[Signature] = set()
    diag = {"unproven": 0, "freeze": "no_play"}

    def offer(kind: str, solution) -> bool:
        """True where the turn is new. Duplicates are the common case: the
        constraints overlap heavily, which is itself part of the answer."""
        if solution is None or not solution.feasible:
            return False
        if solution.status != "OPTIMAL":
            diag["unproven"] += 1
        if not solution.plays_anything or solution.played is None:
            return False
        target = _target(board, solution, melded)
        sig = signature_of_target(target, solution.played)
        if sig in seen:
            return False
        seen.add(sig)
        actions = plan(cfg, board, target, solution.played)
        if len(actions) > cfg.max_micro_per_turn:
            raise HarnessError(f"a {len(actions)}-action turn overruns the micro budget")
        tag = REPLAY if sig == base_sig else kind
        if tag == REPLAY and not replay_base:
            return True
        candidates.append(Candidate(kind=tag, tiles=int(solution.tiles_played), actions=actions))
        return True

    best = solve_turn(cfg, rack, board, melded, time_limit=time_limit)
    if not best.plays_anything:
        raise HarnessError("CP-SAT finds no play at a boundary where the baseline played")
    max_tiles = int(best.tiles_played)
    diag["cpsat_differs"] = signature_of_target(
        _target(board, best, melded), best.played
    ) != base_sig
    offer("cpsat_max", best)

    # The free parameter this section exists to price: which table to leave behind
    # while shedding the same number of tiles. The two objective variants reach
    # tables the lexicographic order hides, then no-good cuts walk the rest. The
    # cuts are on the set-count vector, so they must be deduplicated on it too --
    # a variant that returns the same table again would otherwise be counted as a
    # second one.
    excluded = {best.set_counts.tobytes(): best.set_counts}
    for solution in (
        solve_turn(cfg, rack, board, melded, time_limit=time_limit, keep_weight=0),
        solve_turn(cfg, rack, board, melded, objective=Objective.MAX_VALUE, time_limit=time_limit),
    ):
        same = solution.feasible and solution.tiles_played == max_tiles
        offer("same_tiles_other_table" if same else "fewer_tiles", solution)
        if same and solution.set_counts is not None:
            excluded.setdefault(solution.set_counts.tobytes(), solution.set_counts)

    diag["exhausted"] = False
    for _ in range(kbest):
        solution = solve_turn(
            cfg,
            rack,
            board,
            melded,
            time_limit=time_limit,
            tiles_min=max_tiles,
            tiles_cap=max_tiles,
            exclude=list(excluded.values()),
        )
        if not solution.feasible:
            diag["exhausted"] = True
            break
        offer("same_tiles_other_table", solution)
        excluded[solution.set_counts.tobytes()] = solution.set_counts
    diag["n_tables"] = len(excluded)

    for cap in dict.fromkeys(c for c in (max_tiles - 1, max_tiles - 2, 1) if c >= 1):
        offer("fewer_tiles", solve_turn(cfg, rack, board, melded, time_limit=time_limit, tiles_cap=cap))

    frozen = solve_turn(cfg, rack, board, melded, time_limit=time_limit, freeze_table=True)
    if not frozen.plays_anything:
        diag["freeze"] = "nothing_plays"
    else:
        diag["freeze"] = "distinct" if offer("frozen_table", frozen) else "duplicate"

    candidates.append(Candidate(kind="draw", tiles=0, actions=[cfg.draw_action]))
    return candidates, diag


def signature_of_target(target: list[tuple[int, ...]], played: np.ndarray) -> Signature:
    sets = tuple(sorted(tuple(sorted(s)) for s in target))
    return sets, tuple(int(v) for v in played)


# --- executing and rolling out ----------------------------------------------


def execute(cfg: RummiConfig, state: BatchState, actions: list[int]) -> None:
    """Apply one whole turn, one primitive action at a time, against the mask.

    An illegal step means the target was unreachable or the plan wrong, which is a
    bug in this harness -- never a measurement -- so it raises.
    """
    buffer = np.zeros(1, dtype=np.int64)
    for action in actions:
        summary = summarize(cfg, state.table_sets)
        mask = legal_actions(state, summary)
        if not mask[0, action]:
            raise HarnessError(f"action {action} is masked out mid-turn")
        buffer[0] = action
        engine_step(state, buffer, mask)


def stack(states: list[BatchState]) -> BatchState:
    return BatchState(
        cfg=states[0].cfg,
        **{
            f.name: np.concatenate([getattr(s, f.name) for s in states])
            for f in fields(BatchState)
            if f.name != "cfg"
        },
    )


@dataclass(slots=True)
class Trace:
    """The turn boundaries at the head of a rollout, and nothing after them.

    Depth is `n_players + 2`: every seat's next turn is inside `n_players`
    boundaries, its *end* one further, and the zero-deviation continuation still has
    the played turn in front of it so its own boundaries are one later than
    everything else's.
    """

    seat: np.ndarray
    ended: np.ndarray
    racks: np.ndarray
    """``(B, depth, P)`` rack totals, which is all a shed count needs."""
    n: np.ndarray

    @classmethod
    def blank(cls, batch: int, depth: int, players: int) -> Trace:
        return cls(
            seat=np.full((batch, depth), -1, np.int64),
            ended=np.zeros((batch, depth), bool),
            racks=np.zeros((batch, depth, players), np.int64),
            n=np.zeros(batch, np.int64),
        )

    def record(self, state: BatchState) -> None:
        at = np.flatnonzero((state.micro_count == 0) & (self.n < self.seat.shape[1]))
        if not at.size:
            return
        k = self.n[at]
        self.seat[at, k] = state.current[at]
        self.ended[at, k] = state.done[at]
        self.racks[at, k] = state.racks[at].sum(-1)
        self.n[at] = k + 1

    def first_turn(
        self, env: int, seat: int, winner: int, skip: int
    ) -> tuple[int | None, bool]:
        """Tiles ``seat`` sheds on its first turn from ``skip`` boundaries in, and
        whether that is the turn it wins on. None where it never gets one."""
        for i in range(skip, int(self.n[env]) - 1):
            if self.ended[env, i] or int(self.seat[env, i]) != seat:
                continue
            shed = int(self.racks[env, i, seat] - self.racks[env, i + 1, seat])
            return shed, bool(self.ended[env, i + 1]) and winner == seat
        return None, False


def first_turn_in_game(
    boundaries: list[Boundary], final: BatchState, skip: int, seat: int, winner: int
) -> tuple[int | None, bool]:
    """:meth:`Trace.first_turn` read off the baseline's own boundary list instead.

    The played turn's continuation is the game itself, so rolling it out again would
    only re-derive it. The two derivations share nothing, which is what makes
    ``--check`` comparing them worth running.
    """
    for j in range(skip, len(boundaries)):
        if boundaries[j].seat != seat:
            continue
        before = int(boundaries[j].racks[seat].sum())
        if j + 1 < len(boundaries):
            return before - int(boundaries[j + 1].racks[seat].sum()), False
        return before - int(final.racks[0, seat].sum()), winner == seat
    return None, False


def rollout(
    cfg: RummiConfig, state: BatchState, max_steps: int
) -> tuple[BatchState, Trace]:
    """Continue every env to the end with fresh agents on every seat."""
    seats = [build(AGENT, cfg) for _ in range(cfg.n_players)]
    for agent in seats:
        agent.reset(state.batch_size)
    trace = Trace.blank(state.batch_size, cfg.n_players + 2, cfg.n_players)
    for _ in range(max_steps):
        if state.done.all():
            break
        trace.record(state)
        summary = summarize(cfg, state.table_sets)
        mask = legal_actions(state, summary)
        actions, illegal = act_by_seat(
            seats, cfg, state.current, state.done, mask, encode(state, summary)
        )
        if illegal:
            raise HarnessError("a rollout agent proposed a masked action")
        engine_step(state, actions, mask)
    return state, trace


# --- one game ----------------------------------------------------------------


def run_game(job: tuple) -> dict:
    """One deal. A harness failure is returned rather than raised so that a long
    run reports it beside the measurement instead of losing both."""
    import traceback

    try:
        return _run_game(job)
    except Exception:  # reported beside the measurement, never counted as one
        return {"game": job[0], "error": traceback.format_exc()}


def _run_game(job: tuple) -> dict:
    (
        index,
        cfg,
        seed_base,
        kbest,
        time_limit,
        chunk,
        max_steps,
        check,
        replay_base,
        features,
    ) = job
    started = time.perf_counter()
    seed = np.random.SeedSequence([seed_base, index])
    boundaries, final = play_base(cfg, seed, max_steps)
    if not final.done[0]:
        return {"game": index, "skipped": "baseline did not finish"}

    winner = int(final.winner[0])
    base_turns = int(final.turn_count[0])
    decisions: list[Decision] = []
    candidates: list[Candidate] = []
    unproven = 0

    for i, boundary in enumerate(boundaries):
        if not boundary.actions or boundary.actions[-1] != cfg.end_turn_action:
            continue  # a drawn turn: nothing was chosen
        after = boundaries[i + 1].state if i + 1 < len(boundaries) else final
        after_racks = after.racks[0]
        played = boundary.racks[boundary.seat].astype(np.int64) - after_racks[
            boundary.seat
        ].astype(np.int64)
        base_sig = signature(after.table_sets[0], played)

        found, diag = enumerate_turns(
            cfg, boundary, base_sig, kbest, time_limit, replay_base
        )
        unproven += diag["unproven"]
        opponents = [p for p in range(cfg.n_players) if p != boundary.seat]
        base_left = (
            board_features(cfg, after, 0, boundary.seat, opponents, time_limit)
            if features
            else Left()
        )
        for p in opponents:
            shed, win = first_turn_in_game(boundaries, final, i + 1, p, winner)
            base_left.next_shed.append(shed)
            base_left.next_win.append(win)
        base_left.own_shed = first_turn_in_game(
            boundaries, final, i + 1, boundary.seat, winner
        )[0]
        decision = Decision(
            boundary=i,
            turn=boundary.turn,
            seat=boundary.seat,
            pre_meld=not bool(boundary.melded[boundary.seat]),
            own_rack=int(boundary.racks[boundary.seat].sum()),
            opp_rack=int(min(boundary.racks[p].sum() for p in opponents)),
            pool=boundary.pool,
            base_tiles=int(played.sum()),
            base_won=winner == boundary.seat,
            cpsat_differs=bool(diag["cpsat_differs"]),
            n_tables=int(diag["n_tables"]),
            exhausted=bool(diag["exhausted"]),
            freeze=str(diag["freeze"]),
            opps=opponents,
            opp_sizes=[int(boundary.racks[p].sum()) for p in opponents],
            opp_racks=[sparse_rack(boundary.racks[p]) for p in opponents],
            base=base_left,
        )
        if check:
            found.append(Candidate(kind=CONTINUE, tiles=0, actions=[]))
        for candidate in found:
            candidate.decision = len(decisions)
            candidates.append(candidate)
        decisions.append(decision)

    # Longest rollouts first so a batch is homogeneous in how far it still has to
    # run: an env that finished early still pays for the batch's mask.
    candidates.sort(key=lambda c: decisions[c.decision].turn)
    failures: list[str] = []
    for start in range(0, len(candidates), chunk):
        batch = candidates[start : start + chunk]
        states = []
        lefts = []
        for candidate in batch:
            decision = decisions[candidate.decision]
            state = boundaries[decision.boundary].state.clone()
            execute(cfg, state, candidate.actions)
            # The zero-deviation candidate has not played the turn yet, so its
            # afterstate is not one: only the outcome and the sheds are read from it.
            lefts.append(
                board_features(cfg, state, 0, decision.seat, decision.opps, time_limit)
                if features and candidate.kind != CONTINUE
                else Left()
            )
            states.append(state)
        rolled, trace = rollout(cfg, stack(states), max_steps)
        for offset, candidate in enumerate(batch):
            decision = decisions[candidate.decision]
            outcome = _outcome(rolled, offset, decision.seat)
            left = lefts[offset]
            skip = 1 if candidate.kind == CONTINUE else 0
            rolled_winner = int(rolled.winner[offset])
            for p in decision.opps:
                shed, win = trace.first_turn(offset, p, rolled_winner, skip)
                left.next_shed.append(shed)
                left.next_win.append(win)
            left.own_shed = trace.first_turn(offset, decision.seat, rolled_winner, skip)[0]
            if candidate.kind == CONTINUE:
                turns = int(rolled.turn_count[offset])
                won = outcome == 1
                if won != decision.base_won or turns != base_turns:
                    failures.append(
                        f"continue at turn {decision.turn} gave won={won} turns={turns}, "
                        f"baseline won={decision.base_won} turns={base_turns}"
                    )
                if (left.next_shed, left.own_shed) != (
                    decision.base.next_shed,
                    decision.base.own_shed,
                ):
                    failures.append(
                        f"continue at turn {decision.turn} sheds "
                        f"{left.next_shed}/{left.own_shed}, the baseline's own "
                        f"boundaries say {decision.base.next_shed}/{decision.base.own_shed}"
                    )
                continue
            if candidate.kind == REPLAY and outcome != (1 if decision.base_won else 0):
                failures.append(
                    f"replaying the played turn at turn {decision.turn} gave {outcome}, "
                    f"baseline won={decision.base_won}"
                )
            decision.alts.append(
                {
                    "kind": candidate.kind,
                    "tiles": candidate.tiles,
                    "won": outcome,
                    **left.as_json(),
                }
            )

    return {
        "game": index,
        "winner": winner,
        "turns": base_turns,
        "boundaries": len(boundaries),
        "continuations": sum(c.kind == CONTINUE for c in candidates),
        "unproven_solves": unproven,
        "seconds": time.perf_counter() - started,
        "failures": failures,
        "decisions": [
            {
                "turn": d.turn,
                "seat": d.seat,
                "pre_meld": d.pre_meld,
                "own": d.own_rack,
                "opp": d.opp_rack,
                "pool": d.pool,
                "base_tiles": d.base_tiles,
                "base_won": d.base_won,
                "cpsat_differs": d.cpsat_differs,
                "n_tables": d.n_tables,
                "exhausted": d.exhausted,
                "freeze": d.freeze,
                "opps": d.opps,
                "opp_sizes": d.opp_sizes,
                "opp_racks": d.opp_racks,
                "base": d.base.as_json(),
                "alts": d.alts,
            }
            for d in decisions
        ],
    }


def _outcome(state: BatchState, env: int, seat: int) -> int:
    """1 win, 0 loss, -1 neither -- a stalemate or a truncation."""
    if not state.done[env] or state.truncated[env]:
        return -1
    winner = int(state.winner[env])
    if winner < 0:
        return -1
    return 1 if winner == seat else 0


# --- aggregation -------------------------------------------------------------


PHASES = ("pre-meld", "midgame", "endgame")


def phase(d: dict) -> str:
    """Pre-meld, then whether either rack is short enough to end the game."""
    if d["pre_meld"]:
        return "pre-meld"
    if min(d["own"], d["opp"]) <= 4:
        return "endgame"
    return "midgame"


def _rate(hits: int, total: int) -> str:
    return f"{hits / total:6.1%}" if total else "     --"


def _cluster(values: list[tuple[int, float]]) -> tuple[int, float, float]:
    """n, mean and a 95% half-width clustered by deal, over ``(deal, value)`` pairs.

    Alternatives inside one deal share its deck and its opponent, so the deal is the
    unit of independence: counting them separately would divide by a sample size the
    measurement does not have.
    """
    by_deal: dict[int, list[float]] = defaultdict(list)
    for deal, value in values:
        by_deal[deal].append(value)
    means = [float(np.mean(v)) for v in by_deal.values()]
    if not means:
        return 0, float("nan"), float("nan")
    if len(means) == 1:
        return len(values), means[0], float("nan")
    half = 1.96 * float(np.std(means, ddof=1)) / len(means) ** 0.5
    return len(values), float(np.mean(means)), half


def _fmt(n: int, mean: float, half: float) -> str:
    """One cell: the delta, its interval and the alternatives behind it. A cell drawn
    from a single deal has a mean but no interval, and says so rather than quoting a
    zero."""
    if not n:
        return f"{'--':>22}"
    interval = "  --  " if np.isnan(half) else f"{half:<6.1%}"
    return f"{mean:>+7.1%} +-{interval} n={n:<5}"


def _independent_prediction(played: list[dict], seats: int) -> tuple[float, float]:
    """What the headline would read if a deviation's outcome carried no strategy.

    The pooled per-alternative rate of landing on the other side of the result,
    compounded over however many alternatives that seat was offered.
    """
    out = []
    for want in (1, 0):
        hits = decided = 0
        for game in played:
            for d in game["decisions"]:
                if d["base_won"] == bool(want):
                    continue
                for a in d["alts"]:
                    if a["won"] < 0 or a["kind"] == REPLAY:
                        continue
                    decided += 1
                    hits += a["won"] == want
        rate = hits / decided if decided else 0.0
        per_seat = []
        for game in played:
            if game["winner"] < 0:
                continue
            for seat in range(seats):
                if (game["winner"] == seat) == bool(want):
                    continue
                n = sum(
                    1
                    for d in game["decisions"]
                    if d["seat"] == seat
                    for a in d["alts"]
                    if a["kind"] != REPLAY
                )
                per_seat.append(1.0 - (1.0 - rate) ** n)
        out.append(float(np.mean(per_seat)) if per_seat else float("nan"))
    return out[0], out[1]


# --- the targeted analysis ---------------------------------------------------

SHAPED = "same_tiles_other_table"
"""The one free parameter in a per-turn-maximising turn, and so the type the tables
below pair against the played turn one alternative at a time."""


@dataclass(slots=True)
class Delta:
    """One alternative against the turn it replaced.

    ``deal`` is what the interval is clustered on and ``at`` identifies the decision
    inside it, so a table can report how many *decisions* a cell covers as well as
    how many alternatives. Everything else is a split.
    """

    deal: int
    at: int
    delta: float
    phase: str
    kind: str
    shed_vs_base: int
    """Tiles this alternative shed, minus what the played turn shed."""
    opp_size: int = 0
    leader: bool = False
    """This opponent holds the fewest tiles -- at two seats, always true."""
    next_up: bool = False
    """This opponent is the seat that plays next, and so the only one that meets this
    table as the deviation left it. Past two seats the others meet whatever the seats
    before them leave, so their reply is a door as it stands and not as they find it."""
    d_shed: int | None = None
    """Change in what the opponent actually shed on its next turn. None where one
    side of the pair never gave it a turn at all."""
    d_play: int = 0
    """Change in whether the opponent can play anything at all: -1 closes it."""
    d_out: int = 0
    """Change in whether the opponent can shed its whole rack: -1 closes the exit."""
    base_out: bool = False
    """The opponent could shed its whole rack against the table actually left, which
    is what makes a decision one where there is a door to close at all."""
    closed_out: bool = False
    d_perm: float = 0.0
    d_doors: int = 0


def tile_deltas(played: list[dict]) -> list[Delta]:
    """Every alternative against the played turn, split by how many tiles it shed
    rather than by which constraint reached it."""
    out: list[Delta] = []
    for game in played:
        for at, d in enumerate(game["decisions"]):
            for a in d["alts"]:
                if a["won"] < 0 or a["kind"] == REPLAY:
                    continue
                out.append(
                    Delta(
                        deal=game["game"],
                        at=at,
                        delta=float(a["won"] == 1) - float(d["base_won"]),
                        phase=phase(d),
                        kind=a["kind"],
                        shed_vs_base=a["tiles"] - d["base_tiles"],
                    )
                )
    return out


def denial_pairs(played: list[dict]) -> tuple[list[Delta], int]:
    """Every same-tiles alternative against the played turn, once per opponent.

    A turn that ends the game leaves no door to close, on either side of the pair, so
    those are dropped and counted rather than scored: they would read as a large
    positive delta with nothing to do with the table.
    """
    out: list[Delta] = []
    dropped = 0
    for game in played:
        for at, d in enumerate(game["decisions"]):
            base = d.get("base")
            # A run with the features turned off records the outcome and nothing to
            # pair it against, so there is nothing here to answer.
            if not base or len(base["opp_shed"]) != len(d.get("opp_sizes", ())):
                continue
            smallest = min(d["opp_sizes"])
            next_seat = (d["seat"] + 1) % (len(d["opps"]) + 1)
            for a in d["alts"]:
                if a["kind"] != SHAPED or a["won"] < 0:
                    continue
                if base["ended"] or a["ended"]:
                    dropped += 1
                    continue
                for o, size in enumerate(d["opp_sizes"]):
                    was, now = base["next_shed"][o], a["next_shed"][o]
                    base_out = base["opp_shed"][o] >= size
                    out.append(
                        Delta(
                            deal=game["game"],
                            at=at,
                            delta=float(a["won"] == 1) - float(d["base_won"]),
                            phase=phase(d),
                            kind=a["kind"],
                            shed_vs_base=a["tiles"] - d["base_tiles"],
                            opp_size=size,
                            leader=size == smallest,
                            next_up=d["opps"][o] == next_seat,
                            d_shed=None if was is None or now is None else now - was,
                            d_play=int(a["opp_shed"][o] > 0) - int(base["opp_shed"][o] > 0),
                            d_out=int(a["opp_shed"][o] >= size) - int(base_out),
                            base_out=base_out,
                            closed_out=base_out and a["opp_shed"][o] < size,
                            d_perm=a["perm"] - base["perm"],
                            d_doors=a["doors"] - base["doors"],
                        )
                    )
    return out, dropped


SPLITS = (
    ("all", lambda p: True),
    ("pre-meld", lambda p: p.phase == "pre-meld"),
    ("midgame", lambda p: p.phase == "midgame"),
    ("endgame", lambda p: p.phase == "endgame"),
    ("opp rack <=2", lambda p: p.opp_size <= 2),
    ("opp rack 3-4", lambda p: 3 <= p.opp_size <= 4),
    ("opp rack >4", lambda p: p.opp_size > 4),
)
"""The rows every targeted table carries. The phase rows and the rack rows overlap
on purpose: `phase` calls a decision endgame on *either* rack being short, and the
hypothesis is about the opponent's."""


def grid(
    lines: list[str],
    title: str,
    records: list[Delta],
    columns: tuple,
    splits: tuple = SPLITS,
) -> None:
    """One table: `splits` are its rows and `columns` the bucketing under test, both
    as (label, predicate) over a :class:`Delta`. Every cell is the win delta of the
    alternatives it holds against the turns they replaced."""
    lines.append(title)
    lines.append(f"  {'':<14}" + "".join(f"{name:>23}" for name, _ in columns))
    for row, keep in splits:
        kept = [r for r in records if keep(r)]
        cells = [
            _fmt(*_cluster([(r.deal, r.delta) for r in kept if inside(r)]))
            for _, inside in columns
        ]
        lines.append(f"  {row:<14}" + "".join(cells))
    lines.append("")


TILE_COLUMNS = (
    ("fewer tiles", lambda p: p.shed_vs_base < 0 and p.kind != "draw"),
    ("the same", lambda p: p.shed_vs_base == 0),
    ("more tiles", lambda p: p.shed_vs_base > 0),
    ("nothing (draw)", lambda p: p.kind == "draw"),
)
SHED_COLUMNS = (
    ("opp sheds fewer", lambda p: p.d_shed is not None and p.d_shed < 0),
    ("unchanged", lambda p: p.d_shed == 0),
    ("opp sheds more", lambda p: p.d_shed is not None and p.d_shed > 0),
)
PLAY_COLUMNS = (
    ("can no longer play", lambda p: p.d_play < 0),
    ("unchanged", lambda p: p.d_play == 0),
    ("can play, could not", lambda p: p.d_play > 0),
)
OUT_COLUMNS = (
    ("exit closed", lambda p: p.d_out < 0),
    ("unchanged", lambda p: p.d_out == 0),
    ("exit opened", lambda p: p.d_out > 0),
)
PERM_COLUMNS = (
    ("less permeable", lambda p: p.d_perm < -0.001),
    ("equal", lambda p: abs(p.d_perm) <= 0.001),
    ("more permeable", lambda p: p.d_perm > 0.001),
)


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 3:
        return float("nan")
    x, y = np.asarray(xs, float), np.asarray(ys, float)
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def targeted(played: list[dict]) -> list[str]:
    """The tables a mean by type cannot answer: is the effect targeted inside it?"""
    lines: list[str] = []
    grid(
        lines,
        "DELTA BY TILES SHED RELATIVE TO THE PLAYED TURN (every type, phase rows only)",
        tile_deltas(played),
        TILE_COLUMNS,
        splits=SPLITS[:4],
    )

    pairs, dropped = denial_pairs(played)
    if not pairs:
        lines.append("TARGETED DENIAL: no afterstate features in this run")
        lines.append("")
        return lines

    deals = max(1, len({r.deal for r in pairs}))
    lines.append(
        f"TARGETED DENIAL: {len(pairs)} (alternative, opponent) pairs over {deals} deals, "
        f"{dropped} alternatives dropped for ending the game"
    )
    # Past two seats an alternative is paired once per opponent and the outcome delta
    # is shared between them, so the extra rows are the ones worth reading: which
    # opponent's door it is.
    splits = SPLITS
    if any(not r.next_up for r in pairs):
        splits = (
            *SPLITS,
            ("the next seat", lambda p: p.next_up),
            ("a later seat", lambda p: not p.next_up),
            ("the leader", lambda p: p.leader),
            ("not the leader", lambda p: not p.leader),
        )
    lines.append("")
    for title, columns in (
        ("  vs WHAT THE OPPONENT ACTUALLY SHED NEXT TURN (oracle)", SHED_COLUMNS),
        ("  vs WHETHER THE OPPONENT CAN PLAY AT ALL (oracle)", PLAY_COLUMNS),
        ("  vs WHETHER THE OPPONENT CAN GO OUT (oracle)", OUT_COLUMNS),
    ):
        grid(lines, title, pairs, columns, splits=splits)
    # Permeability is a property of the table and not of an opponent, so it is scored
    # once per alternative rather than once per pair -- otherwise past two seats the
    # same delta is counted P-1 times and n reads larger than the sample is. The seat
    # that plays next is the one kept, since it is the seat that meets this table.
    grid(
        lines,
        "  vs PERMEABILITY OF THE TABLE LEFT (observable, per alternative)",
        [r for r in pairs if r.next_up],
        PERM_COLUMNS,
        splits=splits,
    )

    lines.append("  THE CELL THE HYPOTHESIS NAMES")
    for label, keep in (
        ("exit closed, opp <=2", lambda p: p.closed_out and p.opp_size <= 2),
        ("exit closed, opp 3-4", lambda p: p.closed_out and 3 <= p.opp_size <= 4),
        ("exit closed, any rack", lambda p: p.closed_out),
        ("  the same, leader", lambda p: p.closed_out and p.leader),
        ("  the same, not leader", lambda p: p.closed_out and not p.leader),
        ("play closed, any rack", lambda p: p.d_play < 0),
        ("opp shed fewer, opp <=2", lambda p: p.opp_size <= 2 and (p.d_shed or 0) < 0),
    ):
        hit = [r for r in pairs if keep(r)]
        seen = len({(r.deal, r.at) for r in hit})
        lines.append(
            f"    {label:<24} {_fmt(*_cluster([(r.deal, r.delta) for r in hit]))}"
            f"  decisions {seen:<5} {seen / len(played) if played else 0:.2f}/game"
        )
    # Whether the mechanism is even available, which a delta cannot say: a door can
    # only be closed at a decision where the opponent could walk through it.
    risk = [r for r in pairs if r.base_out]
    decisions = len({(r.deal, r.at) for r in risk})
    lines.append(
        f"    the opponent could go out against the played table in {len(risk)} pairs "
        f"({decisions} decisions, {decisions / max(1, len(played)):.2f} per game); an "
        f"equal-tile table takes that away in {sum(r.closed_out for r in risk)}"
    )
    lines.append("")

    # Whether the signal a real agent could compute points where the oracle does.
    # A correlation of zero here is the stronger null: it says the proxy cannot even
    # find the cells, let alone that the cells are worth nothing.
    lines.append("  DOES THE OBSERVABLE PROXY TRACK THE ORACLE?")
    with_shed = [r for r in pairs if r.d_shed is not None]
    shed = [float(r.d_shed or 0) for r in with_shed]
    # The outcome and the proxy are both per alternative, so those two read the
    # next-seat pairs; the opponent's shed is per opponent and reads all of them.
    one = [r for r in pairs if r.next_up]
    for label, xs, ys in (
        ("corr(d perm, d opponent shed)", [r.d_perm for r in with_shed], shed),
        ("corr(d doors, d opponent shed)", [float(r.d_doors) for r in with_shed], shed),
        ("corr(d perm, d win)", [r.d_perm for r in one], [r.delta for r in one]),
        ("corr(d doors, d win)", [float(r.d_doors) for r in one], [r.delta for r in one]),
    ):
        lines.append(f"    {label:<32} {_pearson(xs, ys):>+6.3f}   n={len(xs)}")
    for label, keep in (
        ("exit closed", lambda p: p.closed_out),
        ("exit not closed", lambda p: not p.closed_out),
        ("play closed", lambda p: p.d_play < 0),
        ("opp shed fewer", lambda p: (p.d_shed or 0) < 0),
        ("opp shed more", lambda p: (p.d_shed or 0) > 0),
    ):
        hit = [r for r in pairs if keep(r)]
        if not hit:
            continue
        lines.append(
            f"    mean d perm where {label:<16} {float(np.mean([r.d_perm for r in hit])):>+7.3f}"
            f"  d doors {float(np.mean([r.d_doors for r in hit])):>+6.3f}  n={len(hit)}"
        )
    lines.append("")
    lines.extend(reply_check(played))
    return lines


def reply_check(played: list[dict]) -> list[str]:
    """The oracle reply against what the seat that plays next actually did.

    It is a bound, not a model: nothing intervenes between the turn and that seat's
    reply, so a shed larger than CP-SAT's maximum is a bug in the harness and a
    table the reply calls dead must leave that seat drawing. Both are asserted to be
    zero. How often the maximum is *reached* is the other half -- it is the distance
    between `frugal` and the oracle in one number, and it is why an oracle feature
    is not the same as a feature the opponent will act on.
    """
    compared = exceeded = played_anyway = reached = offered = 0
    for game in played:
        for d in game["decisions"]:
            if not d.get("base") or len(d["base"]["opp_shed"]) != len(d.get("opps", ())):
                continue
            o = d["opps"].index((d["seat"] + 1) % (len(d["opps"]) + 1))
            for turn in (d["base"], *d["alts"]):
                shed = turn["next_shed"][o]
                if turn["ended"] or shed is None:
                    continue
                compared += 1
                exceeded += shed > turn["opp_shed"][o]
                played_anyway += turn["opp_shed"][o] == 0 and shed > 0
                if turn["opp_shed"][o] > 0:
                    offered += 1
                    reached += shed == turn["opp_shed"][o]
    if not compared:
        return []
    return [
        "  THE ORACLE REPLY AGAINST WHAT THE NEXT SEAT DID",
        f"    turns compared {compared}   the reply exceeded {exceeded}   "
        f"played where the reply was nothing {played_anyway}",
        f"    the next seat took the whole reply in {_rate(reached, offered).strip()} "
        f"of the {offered} turns it was offered one",
        "",
    ]


def summarise(games: list[dict]) -> list[str]:
    played = [g for g in games if "decisions" in g]
    lines: list[str] = []

    # Headline: one loser and one winner per game. Every seat counts, including
    # one that never played a turn -- it cannot be rescued, and dropping it would
    # be dividing by the games where a rescue was possible.
    losses = wins = flipped = fragile = 0
    seats = 1 + max(d["seat"] for g in played for d in g["decisions"])
    for game in played:
        if game["winner"] < 0:
            continue  # a stalemate has no side to rescue
        for seat in range(seats):
            own = [d for d in game["decisions"] if d["seat"] == seat]
            any_win = any(a["won"] == 1 for d in own for a in d["alts"])
            any_loss = any(a["won"] == 0 for d in own for a in d["alts"])
            if game["winner"] == seat:
                wins += 1
                fragile += any_loss
            else:
                losses += 1
                flipped += any_win

    # If a deviation only reshuffles the shared deck, each one is an independent
    # coin flip and the headline is just a maximum over however many were tried.
    # This is what that would predict, from the pooled per-alternative rate.
    predicted = _independent_prediction(played, seats)

    n_dec = sum(len(g["decisions"]) for g in played)
    n_alt = sum(len(d["alts"]) for g in played for d in g["decisions"])
    lines.append(
        f"games {len(played)}  decisions {n_dec}  alternatives {n_alt} "
        f"({n_alt / max(1, n_dec):.1f} per decision)"
    )
    lines.append("")
    lines.append("HEADLINE (per game, per seat)")
    lines.append(f"  lost games rescued by one deviation   {_rate(flipped, losses)}  n={losses}")
    lines.append(f"  won games thrown by one deviation     {_rate(fragile, wins)}  n={wins}")
    lines.append(
        f"  the same, if every deviation were an independent coin flip: "
        f"{predicted[0]:6.1%} / {predicted[1]:6.1%}"
    )
    lines.append("")

    def table(title: str, key, order=None) -> None:
        buckets: dict[object, list[int]] = defaultdict(lambda: [0, 0, 0, 0])
        for game in played:
            for d in game["decisions"]:
                cell = buckets[key(d)]
                cell[0] += 1
                if d["base_won"]:
                    cell[2] += 1
                    cell[3] += any(a["won"] == 0 for a in d["alts"])
                else:
                    cell[1] += any(a["won"] == 1 for a in d["alts"])
        lines.append(title)
        lines.append(f"  {'':<16} {'decisions':>9} {'loss->win':>10} {'win->loss':>10}")
        keys = order if order is not None else sorted(buckets, key=str)
        for name in keys:
            if name not in buckets:
                continue
            total, flips, won, throws = buckets[name]
            lines.append(
                f"  {name!s:<16} {total:>9} {_rate(flips, total - won):>10} "
                f"{_rate(throws, won):>10}"
            )
        lines.append("")

    def bucket(n: int) -> str:
        for hi, name in ((2, "1-2"), (4, "3-4"), (8, "5-8"), (13, "9-13")):
            if n <= hi:
                return name
        return "14+"

    table("BY PHASE", phase, list(PHASES))
    table("BY OWN RACK", lambda d: bucket(d["own"]), ["1-2", "3-4", "5-8", "9-13", "14+"])
    table("BY OPPONENT RACK", lambda d: bucket(d["opp"]), ["1-2", "3-4", "5-8", "9-13", "14+"])
    table("BY TURN INDEX", lambda d: f"{10 * (d['turn'] // 10):>3}-{10 * (d['turn'] // 10) + 9}")

    # Per alternative type: how often an alternative of that type is the one that
    # flips, over the decisions where the type had a candidate at all.
    lines.append("BY ALTERNATIVE TYPE")
    lines.append(
        f"  {'':<24} {'offered':>8} {'loss->win':>10} {'win->loss':>10} {'per alt':>8}"
    )
    for kind in KINDS:
        offered = lost = won = flips = throws = alts = 0
        for game in played:
            for d in game["decisions"]:
                mine = [a for a in d["alts"] if a["kind"] == kind]
                if not mine:
                    continue
                offered += 1
                alts += len(mine)
                if d["base_won"]:
                    won += 1
                    throws += any(a["won"] == 0 for a in mine)
                else:
                    lost += 1
                    flips += any(a["won"] == 1 for a in mine)
        moved = sum(
            1
            for game in played
            for d in game["decisions"]
            for a in d["alts"]
            if a["kind"] == kind and a["won"] == (0 if d["base_won"] else 1)
        )
        lines.append(
            f"  {kind:<24} {offered:>8} {_rate(flips, lost):>10} {_rate(throws, won):>10} "
            f"{_rate(moved, alts):>8}"
        )
    lines.append("")

    # The control the headline needs. The oracle takes a maximum over
    # alternatives, so a deviation set that only reshuffles the shared deck
    # rescues losses by chance alone -- and would throw wins at the same rate and
    # win no more often than the turn it replaced. A type that carries strategy
    # has to beat its own matched baseline on all three.
    lines.append("MEAN OUTCOME PER TYPE (the deviation against the turn it replaced)")
    lines.append(
        f"  {'':<24} {'alts':>7} {'base win':>9} {'alt win':>9} "
        f"{'delta (95% CI)':>20} {'changed':>8}"
    )
    for kind in (*KINDS, REPLAY):
        alts = wins_alt = decided = base_wins = base_decided = changed = 0
        paired: list[tuple[int, float]] = []
        for game in played:
            for d in game["decisions"]:
                mine = [a for a in d["alts"] if a["kind"] == kind]
                if not mine:
                    continue
                base_decided += 1
                base_wins += d["base_won"]
                for a in mine:
                    alts += 1
                    if a["won"] < 0:
                        continue
                    decided += 1
                    wins_alt += a["won"] == 1
                    paired.append((game["game"], float(a["won"] == 1) - float(d["base_won"])))
                    changed += (a["won"] == 1) != d["base_won"]
        if not alts:
            continue
        _, mean, half = _cluster(paired)
        lines.append(
            f"  {kind:<24} {alts:>7} {base_wins / base_decided:>8.1%} "
            f"{wins_alt / max(1, decided):>9.1%} "
            f"{mean:>+9.1%} +-{half:<7.1%} {changed / max(1, decided):>8.1%}"
        )
    lines.append("")

    # Where the deviation actually costs or gains, crossed with when it is taken.
    lines.append("DELTA (alt win - base win) BY TYPE AND PHASE")
    lines.append(f"  {'':<24} " + " ".join(f"{name:>18}" for name in PHASES))
    for kind in KINDS:
        cells = []
        for name in PHASES:
            alts = wins = base = decided = 0
            for game in played:
                for d in game["decisions"]:
                    if phase(d) != name:
                        continue
                    for a in d["alts"]:
                        if a["kind"] != kind:
                            continue
                        alts += 1
                        if a["won"] < 0:
                            continue
                        decided += 1
                        wins += a["won"] == 1
                        base += d["base_won"]
            if not decided:
                cells.append(f"{'--':>18}")
                continue
            cells.append(f"{(wins - base) / decided:>+11.1%} n={alts:<5}")
        lines.append(f"  {kind:<24} " + " ".join(cells))
    lines.append("")

    lines.extend(targeted(played))

    tables = Counter(d["n_tables"] for g in played for d in g["decisions"])
    total = sum(tables.values())
    exhausted = sum(d["exhausted"] for g in played for d in g["decisions"])
    differs = sum(d["cpsat_differs"] for g in played for d in g["decisions"])
    lines.append("MAX-TILES TABLES PER DECISION (CP-SAT's optimum, is it unique?)")
    for n in sorted(tables):
        lines.append(f"  {n:>3} table(s)  {_rate(tables[n], total)}  n={tables[n]}")
    lines.append(f"  enumeration exhaustive in {_rate(exhausted, total)} of decisions")
    freeze = Counter(d.get("freeze", "?") for g in played for d in g["decisions"])
    lines.append(
        "  no-rearrangement turn: "
        + ", ".join(f"{name} {_rate(freeze[name], total).strip()}" for name in sorted(freeze))
    )
    lines.append(f"  cpsat_max differs from the turn frugal played: {_rate(differs, total)}")
    lines.append("")

    flip_when_differs = sum(
        1
        for g in played
        for d in g["decisions"]
        if d["cpsat_differs"]
        and not d["base_won"]
        and any(a["won"] == 1 for a in d["alts"] if a["kind"] == "cpsat_max")
    )
    differ_losses = sum(
        1 for g in played for d in g["decisions"] if d["cpsat_differs"] and not d["base_won"]
    )
    lines.append(
        f"  and where it differs it alone flips a loss in {_rate(flip_when_differs, differ_losses)} "
        f"(n={differ_losses})"
    )
    lines.append("")

    unproven = sum(g.get("unproven_solves", 0) for g in played)
    failed = [f for g in games for f in g.get("failures", [])]
    errored = [g for g in games if "error" in g]
    lines.append(f"solves short of proven optimality: {unproven}")
    replays = sum(
        1 for g in played for d in g["decisions"] for a in d["alts"] if a["kind"] == REPLAY
    )
    lines.append(
        f"exactness checks: {replays} re-derived turns, "
        f"{sum(g.get('continuations', 0) for g in played)} zero-deviation continuations"
    )
    lines.append(f"exactness-check failures: {len(failed)}")
    lines.extend(f"  {f}" for f in failed[:10])
    lines.append(f"games lost to a harness error: {len(errored)}")
    lines.extend(f"  game {g['game']}: {g['error'].splitlines()[-1]}" for g in errored[:5])
    return lines


# --- entry point -------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--config", default="standard", choices=sorted(CONFIG_BY_NAME))
    parser.add_argument("--seed-base", type=int, default=91_000)
    parser.add_argument("--kbest", type=int, default=4)
    parser.add_argument("--time-limit", type=float, default=2.0)
    parser.add_argument("--chunk", type=int, default=48, help="alternatives rolled out per batch")
    parser.add_argument("--max-steps", type=int, default=20_000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--check",
        action="store_true",
        help="roll out the zero-deviation continuation at every decision too",
    )
    parser.add_argument("--no-replay-base", action="store_true")
    parser.add_argument(
        "--no-features",
        action="store_true",
        help="skip the per-turn afterstate records (one CP-SAT solve per opponent per turn)",
    )
    parser.add_argument("--out", type=Path, default=Path("runs/oracle-regret"))
    parser.add_argument(
        "--from-json",
        type=Path,
        nargs="+",
        help="re-print the tables from finished runs and exit. Several runs of the same "
        "config are pooled, their deal numbers renumbered so the clustered intervals "
        "still treat one deal as one observation",
    )
    args = parser.parse_args()

    if args.from_json is not None:
        pooled: list[dict] = []
        for path in args.from_json:
            offset = 1 + max((g["game"] for g in pooled), default=-1)
            pooled += [
                {**g, "game": g["game"] + offset}
                for g in json.loads(path.read_text())["per_game"]
            ]
        print("\n".join(summarise(pooled)))
        return

    cfg = CONFIG_BY_NAME[args.config]
    jobs = [
        (
            i,
            cfg,
            args.seed_base,
            args.kbest,
            args.time_limit,
            args.chunk,
            args.max_steps,
            args.check,
            not args.no_replay_base,
            not args.no_features,
        )
        for i in range(args.games)
    ]

    started = time.perf_counter()
    games: list[dict] = []
    if args.workers > 1:
        with get_context("spawn").Pool(args.workers) as pool:
            for game in pool.imap_unordered(run_game, jobs):
                games.append(game)
                print(
                    f"[{len(games):>4}/{args.games}] game {game['game']} "
                    f"{game.get('seconds', 0):.1f}s",
                    flush=True,
                )
    else:
        for job in jobs:
            games.append(run_game(job))
            print(f"[{len(games):>4}/{args.games}] {games[-1].get('seconds', 0):.1f}s", flush=True)
    wall = time.perf_counter() - started
    games.sort(key=lambda g: g["game"])

    lines = summarise(games)
    print()
    print("\n".join(lines))
    print(f"\nwall clock {wall:.1f}s over {args.workers} workers")

    args.out.mkdir(parents=True, exist_ok=True)
    path = (
        args.out
        / f"regret-{args.config}-{args.games}g-seed{args.seed_base}-k{args.kbest}.json"
    )
    path.write_text(
        json.dumps(
            {
                "agent": AGENT,
                "config": args.config,
                "games": args.games,
                "seed_base": args.seed_base,
                "kbest": args.kbest,
                "check": args.check,
                "wall_seconds": wall,
                "summary": lines,
                "per_game": games,
            },
            indent=1,
        )
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
