"""Head-to-head play between baselines.

The point is a sanity check on the simulator, not a leaderboard: if optimal does
not clearly beat greedy and greedy does not clearly beat random, something in the
rules, the masks or the reward is wrong.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field

import numpy as np

from rummi.rules.config import STANDARD, TINY, TINY_GROUPS, RummiConfig
from rummi.env.numpy.deal import env_seeds, reset, reset_envs
from rummi.env.numpy.engine import step
from rummi.env.numpy.masks import legal_actions
from rummi.env.numpy.state import BatchState

CONFIGS = {"standard": STANDARD, "tiny": TINY, "tiny_groups": TINY_GROUPS}


def build(cfg: RummiConfig, name: str, seed: int = 0):
    """Return ``act(state, mask) -> actions`` for a named baseline."""
    if name == "random":
        from rummi.policies.random_policy import RandomPolicy

        p = RandomPolicy(cfg, seed=seed)
        return lambda state, mask, envs=None: p.act(mask)
    if name == "greedy":
        from rummi.policies.greedy_policy import GreedyPolicy

        return GreedyPolicy(cfg).act
    if name == "optimal":
        from rummi.policies.optimal_policy import OptimalPolicy

        return OptimalPolicy(cfg).act
    raise ValueError(f"unknown policy {name!r}")


class SeatPolicies:
    """Dispatch each env to the policy of whichever seat is acting.

    Each policy is told which envs are its own. That is not an optimisation: the
    planning policies hold a per-env plan and pop one action per call, so a policy
    that peeked at an env it does not control would consume that env's plan.
    """

    def __init__(self, policies: list) -> None:
        self.policies = policies

    def act(self, state: BatchState, mask: np.ndarray) -> np.ndarray:
        out = np.full(state.batch_size, state.cfg.draw_action, dtype=np.int64)
        for seat, policy in enumerate(self.policies):
            mine = state.current == seat
            if not mine.any():
                continue
            out[mine] = policy(state, mask, mine)[mine]
        return out


@dataclass
class Match:
    names: tuple[str, ...]
    wins: np.ndarray
    stalemates: int = 0
    truncations: int = 0
    games: int = 0
    turns: list[int] = field(default_factory=list)
    seconds: float = 0.0

    def report(self) -> str:
        rates = " ".join(
            f"{n}={w}/{self.games} ({100 * w / max(1, self.games):.0f}%)"
            for n, w in zip(self.names, self.wins)
        )
        mean_turns = sum(self.turns) / len(self.turns) if self.turns else float("nan")
        return (
            f"{' vs '.join(self.names)}: {rates}  "
            f"stalemates={self.stalemates} truncations={self.truncations}  "
            f"mean turns={mean_turns:.1f}  {self.seconds:.1f}s"
        )


def play_match(
    cfg: RummiConfig, names: list[str], games: int, batch_size: int, seed: int = 0
) -> Match:
    if len(names) != cfg.n_players:
        raise ValueError(f"{cfg.n_players} seats need {cfg.n_players} policies")

    policy = SeatPolicies([build(cfg, n, seed + i) for i, n in enumerate(names)])
    state = reset(cfg, batch_size, seed=seed)
    match = Match(names=tuple(names), wins=np.zeros(cfg.n_players, dtype=int))
    next_seed = batch_size
    t0 = time.perf_counter()

    while match.games < games:
        mask = legal_actions(state)
        step(state, policy.act(state, mask), mask)
        finished = np.flatnonzero(state.done)
        if not finished.size:
            continue
        for env in finished:
            match.games += 1
            match.turns.append(int(state.turn_count[env]))
            emptied = state.racks[env].sum(-1).min() == 0
            if state.truncated[env]:
                match.truncations += 1
            elif not emptied:
                match.stalemates += 1
            if state.winner[env] >= 0:
                match.wins[int(state.winner[env])] += 1
        seeds = env_seeds(seed + 50_000, next_seed + finished.size)[next_seed:]
        next_seed += finished.size
        reset_envs(state, finished, seeds)

    match.seconds = time.perf_counter() - t0
    return match


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", choices=sorted(CONFIGS), default="standard")
    p.add_argument("--games", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--matches",
        nargs="+",
        default=["optimal,greedy", "greedy,random", "optimal,random"],
        help="comma-separated seat policies, one match per argument",
    )
    args = p.parse_args()
    cfg = CONFIGS[args.config]
    for spec in args.matches:
        names = spec.split(",")
        print(play_match(cfg, names, args.games, args.batch_size, args.seed).report())


if __name__ == "__main__":
    main()
