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

Two things this cannot bound. It is perfect hindsight -- the future deck and the
opponent's rack are both visible to the enumeration, so no real policy can reach
it -- and it deviates on one turn only, so a strategy that gives something up now
to collect later is outside it.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field, fields
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from rummi.agents import build
from rummi.agents.base import act_by_seat
from rummi.rules.config import STANDARD, RummiConfig
from rummi.env.numpy.deal import reset as deal_reset
from rummi.env.numpy.deal import reset_envs
from rummi.env.numpy.engine import step as engine_step
from rummi.env.numpy.masks import legal_actions
from rummi.env.numpy.sets import summarize
from rummi.env.numpy.state import BatchState
from rummi.env.observation import encode
from rummi.solver.ilp import Objective, solve_turn
from rummi.solver.to_actions import plan, slot_contents

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


def rollout(cfg: RummiConfig, state: BatchState, max_steps: int) -> BatchState:
    """Continue every env to the end with fresh agents on both seats."""
    seats = [build(AGENT, cfg) for _ in range(cfg.n_players)]
    for agent in seats:
        agent.reset(state.batch_size)
    for _ in range(max_steps):
        if state.done.all():
            break
        summary = summarize(cfg, state.table_sets)
        mask = legal_actions(state, summary)
        actions, illegal = act_by_seat(
            seats, cfg, state.current, state.done, mask, encode(state, summary)
        )
        if illegal:
            raise HarnessError("a rollout agent proposed a masked action")
        engine_step(state, actions, mask)
    return state


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
    index, cfg, seed_base, kbest, time_limit, chunk, max_steps, check, replay_base = job
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
        for candidate in batch:
            state = boundaries[decisions[candidate.decision].boundary].state.clone()
            execute(cfg, state, candidate.actions)
            states.append(state)
        rolled = rollout(cfg, stack(states), max_steps)
        for offset, candidate in enumerate(batch):
            decision = decisions[candidate.decision]
            outcome = _outcome(rolled, offset, decision.seat)
            if candidate.kind == CONTINUE:
                turns = int(rolled.turn_count[offset])
                won = outcome == 1
                if won != decision.base_won or turns != base_turns:
                    failures.append(
                        f"continue at turn {decision.turn} gave won={won} turns={turns}, "
                        f"baseline won={decision.base_won} turns={base_turns}"
                    )
                continue
            if candidate.kind == REPLAY and outcome != (1 if decision.base_won else 0):
                failures.append(
                    f"replaying the played turn at turn {decision.turn} gave {outcome}, "
                    f"baseline won={decision.base_won}"
                )
            decision.alts.append(
                {"kind": candidate.kind, "tiles": candidate.tiles, "won": outcome}
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
        # The interval is clustered by game: alternatives inside one deal share its
        # deck and its opponent, so counting them as independent would divide by a
        # sample size the measurement does not have.
        per_game = []
        for game in played:
            game_alt, game_base, game_n = 0, 0, 0
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
                    game_n += 1
                    wins_alt += a["won"] == 1
                    game_alt += a["won"] == 1
                    game_base += d["base_won"]
                    changed += (a["won"] == 1) != d["base_won"]
            if game_n:
                per_game.append((game_alt - game_base) / game_n)
        if not alts:
            continue
        half = 1.96 * float(np.std(per_game, ddof=1)) / len(per_game) ** 0.5
        lines.append(
            f"  {kind:<24} {alts:>7} {base_wins / base_decided:>8.1%} "
            f"{wins_alt / max(1, decided):>9.1%} "
            f"{float(np.mean(per_game)):>+9.1%} +-{half:<7.1%} {changed / max(1, decided):>8.1%}"
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
    parser.add_argument("--out", type=Path, default=Path("runs/oracle-regret"))
    parser.add_argument(
        "--from-json", type=Path, help="re-print the tables from a finished run and exit"
    )
    args = parser.parse_args()

    if args.from_json is not None:
        print("\n".join(summarise(json.loads(args.from_json.read_text())["per_game"])))
        return

    cfg = STANDARD
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
    path = args.out / f"regret-{args.games}g-seed{args.seed_base}-k{args.kbest}.json"
    path.write_text(
        json.dumps(
            {
                "agent": AGENT,
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
