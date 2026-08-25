"""The frozen evaluation protocol.

A benchmark is only useful if two people running it get comparable numbers, so
everything that could vary is pinned here and versioned: configs, opponents, game
counts, and the seed of every individual game. Change any of it and
:data:`PROTOCOL_VERSION` must change with it, because older scores stop being
comparable.

Two choices are worth explaining.

**Seat rotation.** Every deal is played once per seat, with the agent under test
occupying a different one each time and reference agents filling the rest. That
cancels both the turn-order advantage and the luck of the deal: an agent mirrored
against itself scores exactly ``1 / n_players`` and exactly ``+0.0``, not
something that needs error bars. At two seats this is the swap it generalises.

**Illegal actions disqualify rather than penalise.** The mask is never all-zero
and always exactly describes what the rules permit, so proposing a masked-out
action is a bug in the agent, not a bad strategy. Scoring it would invite tuning
against the penalty.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from rummi.agents.base import Agent, act_by_seat
from rummi.rules.config import STANDARD, STANDARD_3P, STANDARD_4P, TINY_GROUPS, RummiConfig
from rummi.env.numpy.deal import reset as deal_reset
from rummi.env.numpy.deal import reset_envs
from rummi.env.numpy.engine import step as engine_step
from rummi.env.numpy.masks import legal_actions

PROTOCOL_VERSION = "2.0"


@dataclass(frozen=True, slots=True)
class Suite:
    name: str
    cfg: RummiConfig
    opponent: str
    games: int
    """Distinct deals. Each is played once per seat, so ``n_players`` times this
    many games run."""
    seed_base: int
    """Fixed so game *i* of a suite is always the same deal, for everyone."""
    batch_size: int = 32
    max_steps: int = 20_000

    @property
    def total_games(self) -> int:
        return self.games * self.cfg.n_players


SUITES: tuple[Suite, ...] = (
    Suite("tiny", TINY_GROUPS, opponent="greedy", games=100, seed_base=1_000, batch_size=32),
    Suite("standard-greedy", STANDARD, opponent="greedy", games=200, seed_base=2_000, batch_size=32),
    Suite("standard-optimal", STANDARD, opponent="optimal", games=100, seed_base=3_000, batch_size=16),
    # One per registered Gymnasium id, so every env you can train on has a score
    # to place a submission against. Fewer deals than the two-seat suites because
    # rotation multiplies each by the seat count: 70 deals at three seats is 210
    # games, 55 at four is 220, both comparable to standard-greedy's 400.
    Suite("standard-3p", STANDARD_3P, opponent="greedy", games=70, seed_base=4_000, batch_size=32),
    Suite("standard-4p", STANDARD_4P, opponent="greedy", games=55, seed_base=5_000, batch_size=32),
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
    under_test: int,
) -> None:
    """Run one batch to completion. Env slot *i* is the game with seed *i*.

    Autoreset is deliberately not used: one env slot is one game with one known
    seed, which is what makes a score reproducible.
    """
    n = len(seeds)
    state = deal_reset(suite.cfg, n, seed=0)
    reset_envs(state, np.arange(n), seeds)
    for agent in seats:
        agent.reset(n)

    for _ in range(suite.max_steps):
        if state.done.all():
            break
        mask = legal_actions(state)
        actions, illegal = act_by_seat(seats, state, mask)
        result.illegal_attempts += illegal
        engine_step(state, actions, mask)

    _score_batch(suite, state, result, under_test)


def _score_batch(suite: Suite, state, result: Result, under_test: int) -> None:
    """Score from ``under_test``'s side, whichever seat the rotation put it in.

    Reading the seat rather than always seat 0 is what generalises the two-seat
    swap: there is no perspective to invert afterwards, and every game contributes
    the agent's own final rack instead of half of them being discarded.
    """
    values = state.rack_values()
    for env in range(state.batch_size):
        if not state.done[env]:
            result.truncations += 1
            continue
        result.games += 1
        result.turns.append(int(state.turn_count[env]))
        winner = int(state.winner[env])
        result.final_racks.append(int(values[env, under_test]))
        if state.truncated[env]:
            result.truncations += 1
        if state.racks[env].sum(-1).min() != 0:
            result.stalemates += 1
        if winner == under_test:
            result.wins += 1
            # Official scoring: the winner collects every loser's rack.
            others = [p for p in range(suite.cfg.n_players) if p != under_test]
            result.scores.append(int(values[env, others].sum()))
        else:
            result.losses += 1
            result.scores.append(-int(values[env, under_test]))


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
    from rummi.agents import build

    cfg = suite.cfg
    make_agent = build_agent or (lambda c: build(agent_name, c))
    result = Result(suite=suite.name, agent=agent_name, opponent=suite.opponent)

    total = games if games is not None else suite.games
    done = 0
    while done < total:
        count = min(suite.batch_size, total - done)
        seeds = _game_seeds(suite, done, count)
        # The same deals played once per seat, so neither the deal nor the turn
        # order can flatter either side. A fresh agent per rotation: one carrying
        # plans from the previous seat would consume them in the wrong game.
        for seat in range(cfg.n_players):
            agent = make_agent(cfg)
            agent.name = agent_name
            seats: list[Agent] = [build(suite.opponent, cfg) for _ in range(cfg.n_players)]
            seats[seat] = agent
            _play_batch(suite, seats, seeds, result, seat)
        done += count
    return result


def run_all(agent_name: str, build_agent=None, suites=None, games=None) -> list[Result]:
    return [evaluate(agent_name, s, build_agent, games) for s in (suites or SUITES)]
