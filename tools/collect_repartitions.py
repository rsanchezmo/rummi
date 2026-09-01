"""Every CP-SAT repartition `by_value` asks for, as a labelled dataset.

    python tools/collect_repartitions.py --target 20000 --out checkpoints/repartitions.npz

`macro/by_value` with `repartition=True` reaches for the solver only where nothing
else plays -- and that gate is the dataset's definition, not a filter applied
afterwards. Each row is the state the gate saw (the rack and the table) together
with the whole-table rearrangement CP-SAT answered with, which is the thing
`tools/train_repartition.py` learns to construct without it.

States where the gate fires and the solver *declines* are counted rather than
stored, and the counter is worth reading: with the macro **on**, CP-SAT plays in
only **20.9%** of the states it is asked about, against the 47.7% `docs/EXPERIMENTS.md`
measures with it off. The macro consumes its own opportunities -- a repartition
leaves a table the next solve can find nothing in -- so what is left is a harder
residual than the published figure describes. Those states carry no repartition to
learn, and an offline "how often does the net play" that counted them would be
measuring the position rather than the network.

Rows are split by **game**, never by state: two decisions of one deal share a rack
and most of a table, so a state-level split scores memorisation.
"""

from __future__ import annotations

import argparse
import dataclasses
import pathlib
import time
from collections import Counter

import numpy as np

from rummi.agents.base import Observation, table
from rummi.agents.learned.repartition_net import initial_counts, label_sequence
from rummi.agents.macro import MacroAgent, by_value
from rummi.env.fixed_opponent import FixedOpponentEnv
from rummi.rules.config import CONFIG_BY_NAME, RewardMode, RummiConfig
from rummi.rules.observation import MICRO_COUNT
from rummi.solver.ilp import Solution


class RecordingAgent(MacroAgent):
    """`by_value+repartition`, keeping the solve its own gate already ran.

    `MacroAgent._repartition` is re-stated here rather than wrapped because the
    `Solution` is what the dataset is made of and the base method returns only the
    micro-actions. Solving the state a second time would double the collection
    cost and is not guaranteed to answer identically under a wall-clock limit.
    """

    def __init__(self, cfg: RummiConfig) -> None:
        super().__init__(cfg, choose=by_value(cfg), repartition=True)
        self.solution: dict[int, Solution] = {}

    def _repartition(self, obs: Observation, env: int) -> list[int]:
        from rummi.solver.ilp import solve_turn
        from rummi.solver.to_actions import plan

        cfg = self.cfg
        board = table(obs)[env]
        self.solution.pop(env, None)
        solution = solve_turn(cfg, np.asarray(obs["rack"][env]).astype(np.int64), board, True)
        if not solution.plays_anything or solution.played is None:
            return []
        actions = plan(cfg, board, list(solution.sets), solution.played)
        spent = int(np.asarray(obs["scalars"])[env, MICRO_COUNT])
        if len(actions) > cfg.max_micro_per_turn - spent:
            return []
        actions.pop()
        self.solution[env] = solution
        return actions


def _eligible(cfg: RummiConfig, obs: Observation, env: int) -> bool:
    """The rest of `legal_macros`' repartition gate, beyond "nothing else plays".

    Restated so the declined states counted here are the same population the
    solver was actually asked about -- a pre-meld stall is not one it declined.
    """
    return bool(
        np.asarray(obs["melded"])[env, 0]
        and not np.asarray(obs["workbench"])[env].any()
        and np.asarray(obs["rack"])[env].any()
        and np.asarray(table(obs)[env]).max() >= 0
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="standard", choices=sorted(CONFIG_BY_NAME))
    p.add_argument("--envs", type=int, default=32)
    p.add_argument("--target", type=int, default=20_000, help="labelled states to collect")
    p.add_argument("--max-steps", type=int, default=1_000_000)
    p.add_argument("--holdout", type=float, default=0.15, help="fraction of *games* held out")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=pathlib.Path, default=pathlib.Path("checkpoints/repartitions.npz"))
    args = p.parse_args()

    # The reward is never read here; normalising it keeps the rollout identical to
    # the one every other tool in this directory drives.
    cfg = dataclasses.replace(
        CONFIG_BY_NAME[args.config], reward_mode=RewardMode.SCORE_NORMALIZED
    )

    env = FixedOpponentEnv(num_envs=args.envs, cfg=cfg, seed=args.seed, opponent="greedy")
    agent = RecordingAgent(cfg)
    agent.reset(args.envs)

    racks: list[np.ndarray] = []
    boards: list[np.ndarray] = []
    solved: list[np.ndarray] = []
    played: list[np.ndarray] = []
    micro: list[int] = []
    games: list[int] = []
    status = Counter[str]()
    stuck = declined = 0
    episode = np.zeros(args.envs, dtype=np.int64)

    inner = agent.choose

    def choose(obs: Observation, e: int, legal: np.ndarray) -> int:
        nonlocal stuck, declined
        repart = agent.repartition_macro
        assert repart is not None
        if legal[repart]:
            stuck += 1
            solution = agent.solution.get(e)
            assert solution is not None and solution.played is not None
            board = np.asarray(table(obs)[e])
            rack = np.asarray(obs["rack"][e]).astype(np.int64)
            sets = np.full((cfg.max_sets, cfg.max_set_len), -1, dtype=np.int16)
            for i, content in enumerate(solution.sets):
                sets[i, : len(content)] = content
            racks.append(rack.astype(np.int16))
            boards.append(board.astype(np.int16))
            solved.append(sets)
            played.append(np.asarray(solution.played).astype(np.int16))
            micro.append(int(np.asarray(obs["scalars"])[e, MICRO_COUNT]))
            games.append(int(e * 1_000_000 + episode[e]))
        elif not legal[:repart].any() and _eligible(cfg, obs, e):
            # The gate fired and the solver had nothing. Counted, not stored: it
            # is the rest of the denominator an offline "how often does the net
            # play" has to be read against, and no repartition exists to learn.
            stuck += 1
            declined += 1
        return inner(obs, e, legal)

    agent.choose = choose

    obs, info = env.reset()
    started = time.perf_counter()
    steps = 0
    while len(racks) < args.target and steps < args.max_steps:
        actions = agent.act(obs, np.asarray(info["action_mask"]))
        obs, _, term, trunc, info = env.step(actions)
        done = np.asarray(term) | np.asarray(trunc)
        episode[done] += 1
        steps += 1
        if steps % 2000 == 0:
            rate = len(racks) / max(time.perf_counter() - started, 1e-9)
            print(
                f"step {steps:>7,}  states {len(racks):>6,}  "
                f"stuck {stuck:>6,}  plays {1 - declined / max(stuck, 1):>5.1%}  "
                f"{rate:>5.1f} states/s",
                flush=True,
            )
    elapsed = time.perf_counter() - started
    env.close()

    if not racks:
        raise SystemExit("collected nothing")

    rack_arr = np.stack(racks)
    board_arr = np.stack(boards)
    game_arr = np.asarray(games, dtype=np.int64)

    # The label sequence is derived here, once, so the trainer and the tests read
    # one definition of what the action space can say.
    sequences: list[np.ndarray] = []
    keep: list[int] = []
    for i in range(len(racks)):
        need, avail = initial_counts(cfg, rack_arr[i], board_arr[i])
        content = [tuple(int(k) for k in row if k >= 0) for row in solved[i] if (row >= 0).any()]
        sequence, verdict = label_sequence(cfg, need, avail, content)
        status[verdict] += 1
        if verdict != "none":
            sequences.append(np.asarray(sequence, dtype=np.int16))
            keep.append(i)

    lengths = np.asarray([len(s) for s in sequences], dtype=np.int16)
    width = int(lengths.max())
    padded = np.full((len(sequences), width), -1, dtype=np.int16)
    for i, sequence in enumerate(sequences):
        padded[i, : len(sequence)] = sequence

    kept = np.asarray(keep)
    unique = np.unique(game_arr[kept])
    rng = np.random.default_rng(args.seed)
    held = set(rng.permutation(unique)[: max(1, round(args.holdout * len(unique)))].tolist())
    is_holdout = np.asarray([g in held for g in game_arr[kept]], dtype=bool)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        config=args.config,
        rack=rack_arr[kept],
        board=board_arr[kept],
        sets=np.stack(solved)[kept],
        played=np.stack(played)[kept],
        micro=np.asarray(micro, dtype=np.int16)[kept],
        game=game_arr[kept],
        sequence=padded,
        length=lengths,
        holdout=is_holdout,
        stuck=stuck,
        declined=declined,
        seconds=elapsed,
    )

    total = len(racks)
    tiles = np.stack(played)[kept].sum(-1)
    print(
        f"\ncollected {total:,} labelled states in {elapsed / 60:.1f} min "
        f"({total / max(elapsed, 1e-9):.1f}/s) over {len(unique):,} games\n"
        f"  gate fired {stuck:,} times, CP-SAT played in {1 - declined / max(stuck, 1):.1%}\n"
        f"  action-space coverage  exact {status['exact'] / total:.2%}  "
        f"relaxed {status['relaxed'] / total:.2%}  unrepresentable {status['none'] / total:.2%}\n"
        f"  sets per solution      mean {lengths.mean():.2f}  "
        f"median {np.median(lengths):.0f}  max {width}\n"
        f"  tiles played           mean {tiles.mean():.2f}  max {tiles.max()}\n"
        f"  holdout {is_holdout.mean():.1%} of {len(kept):,} rows\n"
        f"wrote {args.out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
