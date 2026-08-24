"""Random-rollout fuzzing with invariants checked on every step."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import numpy as np

from rummi.core.config import STANDARD, TINY, TINY_GROUPS, RummiConfig
from rummi.core.deal import env_seeds, reset, reset_envs
from rummi.core.engine import step
from rummi.core.masks import legal_actions
from rummi.core.sets import evaluate_slots
from rummi.core.state import BatchState

CONFIGS = {"standard": STANDARD, "tiny": TINY, "tiny_groups": TINY_GROUPS}


@dataclass
class Stats:
    games: int = 0
    steps: int = 0
    end_turns: int = 0
    melds: int = 0
    wins: int = 0
    stalemates: int = 0
    truncations: int = 0
    turn_lengths: list[int] = field(default_factory=list)
    game_turns: list[int] = field(default_factory=list)

    def report(self) -> str:
        def mean(xs):
            return sum(xs) / len(xs) if xs else float("nan")

        return (
            f"games={self.games} steps={self.steps} "
            f"end_turns={self.end_turns} melds={self.melds} "
            f"wins={self.wins} stalemates={self.stalemates} truncations={self.truncations}\n"
            f"mean micro-actions/turn={mean(self.turn_lengths):.2f} "
            f"mean turns/game={mean(self.game_turns):.1f}"
        )


def check_step_invariants(state: BatchState, mask: np.ndarray) -> None:
    state.check_invariants()

    if not mask.any(-1).all():
        raise AssertionError("an env has no legal action at all")

    ev = evaluate_slots(state.cfg, state.table_sets)
    # Mid-turn the table is allowed to be in pieces -- that is the point of the
    # workbench. It must be whole at every turn boundary, which is exactly where
    # micro_count is back to zero.
    at_boundary = (state.micro_count == 0) & ~state.done
    broken = at_boundary & ~(ev.is_valid | ev.is_empty).all(-1)
    if broken.any():
        env = int(np.argmax(broken))
        raise AssertionError(f"env {env} began a turn with an invalid table")
    if (state.workbench[at_boundary] != 0).any():
        raise AssertionError("workbench not empty at a turn boundary")

    if (state.micro_count > state.cfg.max_micro_per_turn).any():
        raise AssertionError("micro-action budget exceeded")


def make_policy(cfg: RummiConfig, name: str, seed: int):
    """Return ``act(state, mask) -> actions``.

    Random play exercises the mask and the mid-turn machinery but essentially
    never assembles a 30-point opening meld by chance, so it leaves END_TURN,
    melding and win detection untouched. Greedy reaches those paths.
    """
    if name == "random":
        from rummi.policies.random_policy import RandomPolicy

        policy = RandomPolicy(cfg, seed=seed)
        return lambda state, mask, envs=None: policy.act(mask)
    if name == "greedy":
        from rummi.policies.greedy_policy import GreedyPolicy

        return GreedyPolicy(cfg).act
    if name == "optimal":
        from rummi.policies.optimal_policy import OptimalPolicy

        return OptimalPolicy(cfg).act
    raise ValueError(f"unknown policy {name!r}")


def fuzz(
    cfg: RummiConfig,
    games: int = 200,
    batch_size: int = 32,
    seed: int = 0,
    check_every: int = 1,
    policy_name: str = "random",
) -> Stats:
    policy = make_policy(cfg, policy_name, seed)
    state = reset(cfg, batch_size, seed=seed)
    stats = Stats()
    next_seed = batch_size
    turn_start = state.turn_count.copy()
    micro_at_turn_start = np.zeros(batch_size, dtype=np.int64)
    step_index = 0

    while stats.games < games:
        mask = legal_actions(state)
        if step_index % check_every == 0:
            check_step_invariants(state, mask)

        actions = policy(state, mask)
        stats.end_turns += int((actions == cfg.end_turn_action).sum())
        melded_before = state.melded.sum()
        step(state, actions, mask)
        stats.melds += int(state.melded.sum() - melded_before)
        step_index += 1
        stats.steps += batch_size

        committed = state.turn_count > turn_start
        if committed.any():
            lengths = step_index - micro_at_turn_start[committed]
            stats.turn_lengths.extend(int(x) for x in lengths)
            micro_at_turn_start[committed] = step_index
            turn_start = state.turn_count.copy()

        finished = np.flatnonzero(state.done)
        if finished.size:
            for env in finished:
                stats.games += 1
                stats.game_turns.append(int(state.turn_count[env]))
                if state.truncated[env]:
                    stats.truncations += 1
                elif state.pool_size[env] == 0 and state.consecutive_draws[env] >= cfg.n_players:
                    stats.stalemates += 1
                else:
                    stats.wins += 1
            seeds = env_seeds(seed + 10_000, next_seed + finished.size)[next_seed:]
            next_seed += finished.size
            reset_envs(state, finished, seeds)
            turn_start[finished] = 0
            micro_at_turn_start[finished] = step_index

    return stats


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", choices=sorted(CONFIGS), default="standard")
    p.add_argument("--games", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--check-every", type=int, default=1)
    p.add_argument("--policy", choices=["random", "greedy", "optimal"], default="random")
    args = p.parse_args()

    stats = fuzz(
        CONFIGS[args.config],
        games=args.games,
        batch_size=args.batch_size,
        seed=args.seed,
        check_every=args.check_every,
        policy_name=args.policy,
    )
    print(f"[{args.config}/{args.policy}] {stats.report()}")


if __name__ == "__main__":
    main()
