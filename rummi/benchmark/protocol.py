"""The frozen evaluation protocol.

A benchmark is only useful if two people running it get comparable numbers, so
everything that could vary is pinned here and versioned: configs, opponents, game
counts, and the seed of every individual game. Change any of it and
:data:`PROTOCOL_VERSION` must change with it, because older scores stop being
comparable.

Two choices are worth explaining.

**Mirrored matches.** Every game is played twice from the same deal, with the
agents swapped between seats. That cancels both the first-player advantage and
the luck of the deal, so a 50% win rate against a mirror of yourself is exactly
50% rather than something that needs error bars.

**Illegal actions disqualify rather than penalise.** The mask is never all-zero
and always exactly describes what the rules permit, so proposing a masked-out
action is a bug in the agent, not a bad strategy. Scoring it would invite tuning
against the penalty.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from rummi.agents.base import Agent
from rummi.core.config import STANDARD, TINY_GROUPS, RummiConfig
from rummi.core.deal import reset as deal_reset
from rummi.core.deal import reset_envs
from rummi.core.engine import step as engine_step
from rummi.core.masks import legal_actions
from rummi.envs.observation import encode

PROTOCOL_VERSION = "1.0"


@dataclass(frozen=True, slots=True)
class Suite:
    name: str
    cfg: RummiConfig
    opponent: str
    games: int
    """Distinct deals. Each is played mirrored, so twice this many games run."""
    seed_base: int
    """Fixed so game *i* of a suite is always the same deal, for everyone."""
    batch_size: int = 32
    max_steps: int = 20_000

    @property
    def total_games(self) -> int:
        return self.games * 2


SUITES: tuple[Suite, ...] = (
    Suite("tiny", TINY_GROUPS, opponent="greedy", games=100, seed_base=1_000, batch_size=32),
    Suite("standard-greedy", STANDARD, opponent="greedy", games=200, seed_base=2_000, batch_size=32),
    Suite("standard-optimal", STANDARD, opponent="optimal", games=100, seed_base=3_000, batch_size=16),
)
SUITE_BY_NAME = {s.name: s for s in SUITES}


@dataclass
class Result:
    suite: str
    agent: str
    opponent: str
    games: int = 0
    wins: int = 0
    losses: int = 0
    stalemates: int = 0
    """Games that ended on an exhausted pool rather than an emptied rack."""
    truncations: int = 0
    illegal_attempts: int = 0
    turns: list[int] = field(default_factory=list)
    scores: list[int] = field(default_factory=list)
    final_racks: list[int] = field(default_factory=list)

    @property
    def disqualified(self) -> bool:
        return self.illegal_attempts > 0

    @property
    def win_rate(self) -> float:
        return self.wins / self.games if self.games else float("nan")

    @property
    def mean_score(self) -> float:
        """Official Rummikub scoring from the agent's side: losers pay their rack,
        the winner collects. Captures margin, which win rate alone hides."""
        return float(np.mean(self.scores)) if self.scores else float("nan")

    @property
    def mean_turns(self) -> float:
        return float(np.mean(self.turns)) if self.turns else float("nan")

    @property
    def mean_final_rack(self) -> float:
        return float(np.mean(self.final_racks)) if self.final_racks else float("nan")

    def report(self) -> str:
        if self.disqualified:
            return (
                f"{self.suite:<18} {self.agent} DISQUALIFIED "
                f"({self.illegal_attempts} illegal actions)"
            )
        return (
            f"{self.suite:<18} win {self.win_rate:>6.1%}  score {self.mean_score:>+7.1f}  "
            f"turns {self.mean_turns:>6.1f}  left {self.mean_final_rack:>5.1f}  "
            f"stale {self.stalemates / max(1, self.games):>5.1%}  n={self.games}"
        )


def _game_seeds(suite: Suite, start: int, count: int) -> list[np.random.SeedSequence]:
    """Seed of game *i*, derived from position alone so it never depends on how
    the games happened to be batched."""
    return [np.random.SeedSequence([suite.seed_base, start + i]) for i in range(count)]


def _play_batch(
    suite: Suite,
    seats: list[Agent],
    seeds: list[np.random.SeedSequence],
    result: Result,
) -> None:
    """Run one batch to completion. Env slot *i* is the game with seed *i*.

    Autoreset is deliberately not used: one env slot is one game with one known
    seed, which is what makes a score reproducible.
    """
    cfg = suite.cfg
    n = len(seeds)
    state = deal_reset(cfg, n, seed=0)
    reset_envs(state, np.arange(n), seeds)
    for agent in seats:
        agent.reset(n)

    for _ in range(suite.max_steps):
        if state.done.all():
            break
        mask = legal_actions(state)
        obs = encode(state)
        actions = np.full(n, cfg.draw_action, dtype=np.int64)
        for seat, agent in enumerate(seats):
            active = (state.current == seat) & ~state.done
            if not active.any():
                continue
            proposed = np.asarray(agent.act(obs, mask, active))
            illegal = active & ~mask[np.arange(n), proposed]
            result.illegal_attempts += int(illegal.sum())
            # Substitute DRAW so the suite still completes and reports, rather
            # than dying halfway with no diagnosis.
            actions[active] = np.where(illegal[active], cfg.draw_action, proposed[active])
        engine_step(state, actions, mask)

    _score_batch(suite, state, result)


def _score_batch(suite: Suite, state, result: Result) -> None:
    cfg = suite.cfg
    values = state.rack_values()
    for env in range(state.batch_size):
        if not state.done[env]:
            result.truncations += 1
            continue
        result.games += 1
        result.turns.append(int(state.turn_count[env]))
        winner = int(state.winner[env])
        # Seat 0 is always the agent under test; mirroring is handled by the
        # caller swapping who sits there.
        result.final_racks.append(int(values[env, 0]))
        if state.truncated[env]:
            result.truncations += 1
        if state.racks[env].sum(-1).min() != 0:
            result.stalemates += 1
        if winner == 0:
            result.wins += 1
            result.scores.append(int(values[env, 1:].sum()))
        else:
            result.losses += 1
            result.scores.append(-int(values[env, 0]))


def evaluate(
    agent_name: str,
    suite: Suite,
    build_agent=None,
    games: int | None = None,
) -> Result:
    """Run one suite and return its result.

    ``build_agent`` is a callable ``(cfg) -> Agent``; omit it to use a name from
    the reference registry. Passing your own is how you enter the benchmark
    without adding code to this package.
    """
    from rummi.agents.reference import build

    cfg = suite.cfg
    if cfg.n_players != 2:
        raise ValueError("the v1 protocol is two-seat only")
    make_agent = build_agent or (lambda c: build(agent_name, c))
    result = Result(suite=suite.name, agent=agent_name, opponent=suite.opponent)

    total = games if games is not None else suite.games
    done = 0
    while done < total:
        count = min(suite.batch_size, total - done)
        seeds = _game_seeds(suite, done, count)
        # Mirrored: the same deals played twice with the seats swapped, so neither
        # the deal nor the turn order can flatter either side.
        for mirrored in (False, True):
            under_test = make_agent(cfg)
            reference = build(suite.opponent, cfg)
            under_test.name = agent_name
            seats = [reference, under_test] if mirrored else [under_test, reference]
            if mirrored:
                # Scoring always reads seat 0, so evaluate the mirror from the
                # reference's side and invert it.
                mirror = Result(suite=suite.name, agent=agent_name, opponent=suite.opponent)
                _play_batch(suite, seats, seeds, mirror)
                _absorb_mirrored(result, mirror)
            else:
                _play_batch(suite, seats, seeds, result)
        done += count
    return result


def _absorb_mirrored(target: Result, mirror: Result) -> None:
    """Fold in a mirrored batch, flipping the perspective to the agent's side."""
    target.games += mirror.games
    target.wins += mirror.losses
    target.losses += mirror.wins
    target.stalemates += mirror.stalemates
    target.truncations += mirror.truncations
    target.illegal_attempts += mirror.illegal_attempts
    target.turns.extend(mirror.turns)
    target.scores.extend(-s for s in mirror.scores)
    # final_racks was recorded for seat 0, which in a mirrored game is the
    # reference agent, so those numbers are not the agent's and are dropped.


def run_all(agent_name: str, build_agent=None, suites=None, games=None) -> list[Result]:
    return [evaluate(agent_name, s, build_agent, games) for s in (suites or SUITES)]
