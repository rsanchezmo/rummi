"""How much room is there between two agents at the top of the ladder?

Every strategy proposed above per-turn maximisation has come back null, and
`tools/oracle_regret.py` priced the ceiling on choosing a turn better. Neither
says what the games themselves are made of -- and that is what decides whether a
null is a measurement problem or the game's. This reports the shape: how often the
seat rather than the agent decides a deal, how close the loser was, how much of a
turn is a draw, and whether the lead at half-time survives.

Every deal is played twice, with A in seat 0 / B in seat 1 and then swapped, so a
pairing's win rate is free of turn order and a mirrored pairing reads exactly
even. Two seats only: the seat-decides and first-mover splits are statements about
a two-seat rotation, and generalising them is a separate measurement rather than a
loop bound.

    python tools/game_structure.py --a optimal --b frugal --deals 200 --seed-base 77000
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from rummi.agents import build
from rummi.agents.base import act_by_seat
from rummi.env.numpy.deal import reset as deal_reset
from rummi.env.numpy.deal import reset_envs
from rummi.env.numpy.engine import step as engine_step
from rummi.env.numpy.masks import legal_actions
from rummi.env.numpy.sets import summarize
from rummi.env.observation import encode
from rummi.rules.config import CONFIG_BY_NAME, RummiConfig

SEATS = 2


def play(
    cfg: RummiConfig, names: tuple[str, str], seeds: list[np.random.SeedSequence]
) -> list[dict]:
    """One batch of deals, one deal per env, `names[i]` in seat `i`.

    The lead history is sampled at turn boundaries only -- the table may be in
    pieces mid-turn, and a rack difference read there is a rack halfway through
    being emptied.
    """
    n = len(seeds)
    state = deal_reset(cfg, n, seed=0)
    reset_envs(state, np.arange(n), seeds)
    seats = [build(name, cfg) for name in names]
    for agent in seats:
        agent.reset(n)

    draws = np.zeros((n, SEATS), dtype=int)
    plays = np.zeros((n, SEATS), dtype=int)
    lead: list[list[int]] = [[] for _ in range(n)]
    while not state.done.all():
        summary = summarize(cfg, state.table_sets)
        mask = legal_actions(state, summary)
        actions, illegal = act_by_seat(
            seats, cfg, state.current, state.done, mask, encode(state, summary)
        )
        if illegal:
            raise RuntimeError(f"{illegal} illegal actions from {names}")
        live = ~state.done
        drew = (actions == cfg.draw_action) & live
        ended = (actions == cfg.end_turn_action) & live
        for env in np.flatnonzero(drew):
            draws[env, state.current[env]] += 1
        for env in np.flatnonzero(ended):
            plays[env, state.current[env]] += 1
        for env in np.flatnonzero(drew | ended):
            lead[env].append(int(state.racks[env, 0].sum() - state.racks[env, 1].sum()))
        engine_step(state, actions, mask)

    values = state.rack_values()
    out = []
    for env in range(n):
        sizes = state.racks[env].sum(-1)
        winner = int(state.winner[env])
        decided = winner in (0, 1)
        out.append(
            {
                "winner": winner,
                "truncated": bool(state.truncated[env]),
                "stalemate": bool(sizes.min() != 0),
                "loser_tiles": int(sizes[1 - winner]) if decided else None,
                "loser_value": int(values[env, 1 - winner]) if decided else None,
                "turns": int(state.turn_count[env]),
                "pool": int(state.pool[env].sum()),
                "draws": draws[env].tolist(),
                "plays": plays[env].tolist(),
                "lead": lead[env],
            }
        )
    return out


def summarise(a: str, b: str, deals: int, forward: list[dict], reverse: list[dict]) -> dict:
    """The pairing as numbers, both arms pooled where the metric is seat-free.

    `seat_decides` is the split the whole tool exists for: a deal where the same
    *seat* wins both games is one the deal handed to a position, not to an agent.
    """
    a_wins = np.array([g["winner"] == 0 for g in forward])
    a_wins_swapped = np.array([g["winner"] == 1 for g in reverse])
    both = int((a_wins & a_wins_swapped).sum())
    neither = int((~a_wins & ~a_wins_swapped).sum())

    games = forward + reverse
    tiles = np.array([g["loser_tiles"] for g in games if g["loser_tiles"] is not None])
    value = np.array([g["loser_value"] for g in games if g["loser_value"] is not None])
    drawn = np.array([g["draws"] for g in games])
    played = np.array([g["plays"] for g in games])

    # Whether the lead at half-time survives: the sign of (rack0 - rack1) at the
    # middle turn boundary against who won. Deals whose lead is tied there carry no
    # prediction, and a handful of boundaries is not a half-time.
    agree = []
    for g in games:
        history = g["lead"]
        if len(history) < 4:
            continue
        middle = history[len(history) // 2]
        if middle == 0:
            continue
        agree.append((middle < 0) == (g["winner"] == 0))

    return {
        "a": a,
        "b": b,
        "deals": deals,
        "games": len(games),
        "a_win_rate": float((a_wins.sum() + a_wins_swapped.sum()) / len(games)),
        "a_wins_both_seats": both,
        "b_wins_both_seats": neither,
        "seat_decides": deals - both - neither,
        "seat_decides_rate": (deals - both - neither) / deals,
        "first_mover_win_rate": float(np.mean([g["winner"] == 0 for g in games])),
        "stalemate_rate": float(np.mean([g["stalemate"] for g in games])),
        "truncated_rate": float(np.mean([g["truncated"] for g in games])),
        "loser_tiles": {
            "mean": float(tiles.mean()),
            "median": float(np.median(tiles)),
            "p_le_1": float(np.mean(tiles <= 1)),
            "p_le_2": float(np.mean(tiles <= 2)),
            "p_le_3": float(np.mean(tiles <= 3)),
        },
        "loser_value_mean": float(value.mean()),
        "turns_mean": float(np.mean([g["turns"] for g in games])),
        "pool_mean": float(np.mean([g["pool"] for g in games])),
        "p_pool_empty": float(np.mean([g["pool"] == 0 for g in games])),
        "draws_per_seat": [float(x) for x in drawn.mean(0)],
        "plays_per_seat": [float(x) for x in played.mean(0)],
        "half_time_leader_wins": float(np.mean(agree)),
        "half_time_n": len(agree),
    }


def report(result: dict) -> list[str]:
    tiles, a, b = result["loser_tiles"], result["a"], result["b"]
    return [
        f"\n{a} vs {b}, {result['deals']} deals x {SEATS} seats",
        f"  {a} win rate: {result['a_win_rate']:.1%}",
        f"  deals {a} wins from BOTH seats: {result['a_wins_both_seats']}   "
        f"{b} wins both: {result['b_wins_both_seats']}   "
        f"seat decides: {result['seat_decides']} ({result['seat_decides_rate']:.0%})",
        f"  seat 0 (first mover) wins: {result['first_mover_win_rate']:.1%}",
        f"  stalemates: {result['stalemate_rate']:.1%}  truncated: {result['truncated_rate']:.1%}",
        f"  loser tiles left: mean {tiles['mean']:.2f} median {tiles['median']:.0f}  "
        f"P(<=1)={tiles['p_le_1']:.1%} P(<=2)={tiles['p_le_2']:.1%} P(<=3)={tiles['p_le_3']:.1%}",
        f"  loser value left: mean {result['loser_value_mean']:.1f}",
        f"  turns: mean {result['turns_mean']:.1f}   pool left: mean {result['pool_mean']:.1f}  "
        f"P(pool==0)={result['p_pool_empty']:.1%}",
        f"  draws per game per seat: {np.round(result['draws_per_seat'], 2)}   "
        f"playing turns per seat: {np.round(result['plays_per_seat'], 2)}",
        f"  leader at half-time wins: {result['half_time_leader_wins']:.1%} "
        f"(n={result['half_time_n']})",
    ]


def merge(path: Path, config: str, result: dict) -> dict:
    """The pairing into whatever `path` already holds, keyed by the two agents.

    One file holds every pairing the docs quote and a pairing is one invocation, so
    a second run has to land beside the first rather than replace it.
    """
    payload: dict = {"config": config, "pairings": {}}
    if path.exists():
        payload = json.loads(path.read_text())
        if payload.get("config") != config:
            raise ValueError(f"{path} holds {payload.get('config')!r}, not {config!r}")
    payload["pairings"][f"{result['a']}-vs-{result['b']}"] = result
    return payload


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--a", default="optimal", help="the agent the win rate is reported for")
    p.add_argument("--b", default="frugal")
    p.add_argument("--deals", type=int, default=200)
    p.add_argument("--seed-base", type=int, default=77_000)
    p.add_argument("--batch", type=int, default=25, help="deals played per arm at once")
    p.add_argument("--config", default="standard", choices=sorted(CONFIG_BY_NAME))
    p.add_argument("--json", type=Path, default=None)
    args = p.parse_args()

    cfg = CONFIG_BY_NAME[args.config]
    if cfg.n_players != SEATS:
        raise SystemExit(
            f"{args.config} has {cfg.n_players} seats; this tool is two-seat only "
            "(the seat-decides and first-mover splits are defined on a swap)"
        )

    started = time.perf_counter()
    forward: list[dict] = []
    reverse: list[dict] = []
    for start in range(0, args.deals, args.batch):
        # The seed depends on the deal index alone, so the batch size does not
        # change which deals are played -- only how many run at once.
        seeds = [
            np.random.SeedSequence([args.seed_base, start + i])
            for i in range(min(args.batch, args.deals - start))
        ]
        forward += play(cfg, (args.a, args.b), seeds)
        reverse += play(cfg, (args.b, args.a), seeds)
        print(f"  {len(forward)}/{args.deals} deals, {time.perf_counter() - started:.0f}s", flush=True)

    result = summarise(args.a, args.b, args.deals, forward, reverse)
    result["seed_base"] = args.seed_base
    result["wall_seconds"] = time.perf_counter() - started
    print("\n".join(report(result)))

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(merge(args.json, args.config, result), indent=1) + "\n")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
