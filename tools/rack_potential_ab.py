"""Is a rack worth more than its size? The one axis no agent here has ever ranked.

    python tools/rack_potential_ab.py --arena diagnose --deals 400 --seed-base 94000
    python tools/rack_potential_ab.py --arena sweep --deals 400 --seed-base 93000 \
        --modes ready reach both stop
    python tools/rack_potential_ab.py --arena both --deals 1600 --games 800 --w 1.0 \
        --seed-base 94000 --modes ready ready+null both both+null joker joker+null

Every ranking in this repo -- `greedy`, `rearrange`, `by_value`, `frugal`, `optimal`,
the clone -- maximises how many tiles *leave* the rack. The win condition is emptying
the rack **first**, so what stays should matter: seven tiles that combine are closer
to done than five that do not, and a held joker guarantees a finish that spending it
forfeits. This measures that, and it is deliberately **not** a tie-break -- the denial
experiment already bounded the indifference set at ~1.6pp, and a potential tie-break
sits inside that bound. Here potential is a **weighted objective**:

    score(play) = by_value's own rank  +  w * potential(rack left behind)

so a positive `w` can overrule the ranking and shed *fewer* tiles when what remains is
worth more. `w = 0` is `by_value` exactly, down to `argmax`'s first-maximum order --
which is the control and the correctness check in one
(`test_a_zero_weight_is_by_value_exactly`).

**Potential**, from `macro.shortfall` -- which is exact as a matmul, because every
template row is binary, so `shortfall = templates @ (rack == 0)`:

- `ready` -- tiles in the largest set the remaining rack can still lay down, from
  `macro.playable`, the same predicate `legal_macros` gates a new set on. Joker-aware
  for free: with a joker held every one-away template is playable, so **spending the
  joker collapses `ready`**, which is the sharpest instance of the whole hypothesis.
  In tiles, so `w = 1` reads "a tile I can shed next turn is worth exactly a tile shed
  now" -- a natural unit and the centre of the sweep.
- `reach` -- the chance the next draw completes a set: the unseen copies of every kind
  that would finish some template, over all unseen copies. **Deduplicated by kind**, not
  counted per template, because one tile serving four near-runs is one draw, and
  **gated on availability**, because a one-away set whose missing kind is all on the
  table is not a chance. Both matter: undeduplicated, a rack of 5-6-7-8 scores its two
  real doors as six.

Two-away templates are deliberately *not* a third term, and that is measured rather
than assumed: a rack at a turn boundary holds 8.1 tiles, 5.7 templates one tile away
and **35.3 two tiles away**, and 63.3% of unseen copies participate in *some* two-away
set against 12.6% one-away. A quantity two thirds saturated cannot separate two
candidate plays, so it would swamp the term it is meant to refine.

The arms, and what each is for. `ready`, `reach` and `both` are the hypothesis at its
three weightings. `stop` adds `END_TURN` to the same objective at rank 0 -- the most
aggressive override available, keeping the whole rack rather than shedding anything.
`joker` is the hard rule: never spend the rack's joker unless the play empties the
rack. Each has a **`+null`** twin, and none of the numbers means anything without it:
the null keeps the arm's own potential values and **permutes them across the
candidates** on a state-seeded shuffle, so the perturbation has the same magnitude and
very nearly the same firing rate on a basis that cannot be right. `joker+null` blocks a
seeded same-sized subset instead. The denial arm's apparent +1.06pp evaporated against
exactly this control.

**What it measured.** The diagnostic came first and it is most of the answer. 400 deals
from both seats at `w = 1`, `--arena diagnose`:

| | ready | reach | both | stop | joker |
|---|---|---|---|---|---|
| candidates per decision | 1.76 | 1.75 | 1.76 | 1.92 | 1.81 |
| decisions offering more than one | 31.9% | 31.4% | 31.8% | 43.1% | 32.0% |
| decisions the arm moves | 1.62% | 2.28% | 3.79% | 5.63% | 0.98% |
| potential of the rack as it stands | 0.83 | 0.15 | 0.98 | 1.03 | -- |
| its spread over the candidates | 0.10 | 0.01 | 0.11 | 0.21 | 0.00 |
| potential the move buys / rank it gives up | +3.09 / 1.78 | +0.05 / 0.00 | +1.34 / 0.75 | +1.91 / 1.41 | 0 / 15.77 |
| **deals it changes the outcome of** | **9.8%** | 43.2% | 48.0% | 75.8% | 18.5% |

Two rows carry it. **There is usually nothing to choose between**: 1.76 candidates per
decision, only 31.9% of decisions offering a second one at all, and a potential spread
across them of 0.10 tiles. And **a moved decision is mostly not a changed game**:
`ready` moves 1.62% of decisions but changes the outcome of 9.8% of deals, which caps
the win rate it can move at 4.9pp before a game is scored. Replaying both picks from a
turn's *first* decision through to the end of that turn says why: the turn finishes
with the identical rack **and** the identical table **83.7%** of the time under `ready`
(n=98 moved decisions) and 69.0% under `both` (n=258). The override reorders the sets
of a turn far more often than it changes what the turn achieves.

The `w` sweep, 400 deals at `--seed-base 93000` (win rate against plain `by_value`):

| arm | w=0.5 | w=1 | w=2 | w=4 | w=8 |
|---|---|---|---|---|---|
| `ready` | 50.50% | 50.62% | 50.50% | 50.00% | 50.62% |
| `reach` | 51.62% | 51.62% | 51.62% | 51.62% | 51.12% |
| `both` | **51.75%** | 51.62% | 51.62% | 51.25% | 51.50% |
| `stop` | **51.75%** | 48.50% | 48.38% | 48.12% | 49.38% |

Every cell is inside its own interval -- those run from +-0.8 for the arms that barely
act to +-3.4 for `stop` at w=8 -- and `reach` is *constant* from w=0.5 to w=4 because
its term never exceeds a rank gap of one tile there: below w=8 it is a tie-break, not
an override, which is what its `give 0.00` says above.

The sweep's best cell -- `both` at w=0.5, 51.75% -- **does not survive disjoint deals**:
1600 fresh deals at `--seed-base 94000` put it at **49.78% +-1.15%**, a 2pp swing, which
is the whole reason the confirmation run exists and the lesson this repo has already
published once about a max over fifteen noisy evals.

Head to head against plain `by_value`, 1600 deals from both seats (3200 games per arm),
w=1 unless stated, with every arm's own null beside it:

| arm | head-to-head win | vs its null, paired | `standard-greedy`, n=1600 |
|---|---|---|---|
| `ready` | 50.41% +-0.53% | +0.28pp +-0.79 | 81.9% +-1.9 / +27.95 |
| `reach` | 49.84% +-1.13% | -0.06pp +-1.36 | 81.9% +-1.9 / +28.23 |
| `both` | 50.25% +-1.19% | +0.62pp +-1.48 | 81.8% +-1.9 / +28.14 |
| `stop` | 50.31% +-1.46% | +1.09pp +-1.75 | 82.0% +-1.9 / +28.16 |
| `joker` | 50.75% +-0.76% | +0.53pp +-1.02 | 82.6% +-1.9 / +28.08 |
| `both` at w=0.5 (the sweep's pick) | 49.78% +-1.15% | -0.06pp +-1.42 | -- |
| `ready+null` | 50.12% +-0.73% | -- | 81.9% +-1.9 / +27.93 |
| `reach+null` | 49.91% +-1.21% | -- | 83.2% +-1.8 / +29.15 |
| `both+null` | 49.62% +-1.29% | -- | 82.7% +-1.9 / +28.81 |
| `stop+null` | 49.22% +-1.36% | -- | 82.9% +-1.8 / +28.90 |
| `joker+null` | 50.22% +-1.00% | -- | 82.6% +-1.9 / +27.93 |
| `base` (`w = 0`) | exactly 50.00% +-0.00% | -- | 82.2% +-1.9 / +28.16 |

`base` mirrored against itself reads *exactly* 50.00% and +0.00, so the rotation is
exact and none of this is turn-order bias. Power was sized before running from the
preceding experiment's pilot -- per-deal sd 0.305, so 894 deals for a 2pp half-width --
and run at 1600; the arms that change fewer deals come in tighter than that, because the
pairing is per deal and a deal played identically contributes no variance. **No arm
beats its own control by more than 1.1pp, every paired interval covers zero, and on
`standard-greedy` three of the five nulls are ahead of their own arm.** So rack
potential is worth nothing measurable here, with the axis bounded at roughly 1.8pp --
and there was no case for arena (c), the `optimal` opponent, which only pays for its
compute if the head-to-head shows something.

Two rows read as nominally above even, and both are tight for the same reason they are
uninteresting -- they change few deals: `ready` at 50.41% +-0.53% and `joker` at
50.75% +-0.76%. Neither survives its own control (+0.28pp +-0.79 and +0.53pp +-1.02,
with the controls themselves at +0.12pp and +0.22pp), and on `standard-greedy` each arm
and its control are indistinguishable -- 81.9% / +27.95 against 81.9% / +27.93, and
82.6% / +28.08 against 82.6% / +27.93. That an arbitrary perturbation gains anything at
all is a fact about `by_value`'s fallback -- a tie goes to the lowest template index,
which is the cheapest, lowest-numbered set -- not about rack potential.

**Unlike board shaping, this null needs its own explanation.** A rack is private, so
improving one's own finishing potential costs the opponent nothing and there is no
symmetry to blame. Three measurements give it, and the third is the transferable one:

- **The residue is already the residue.** Mean `ready` of the rack at a decision is
  **0.83 tiles** against a rack of 8.1 -- usually there is no complete set left at all,
  because a decision recurs after every set and `by_value` has already played whatever
  it could. There is little to protect.
- **And the candidates barely differ on it.** `reach` sits at 0.15 with a spread of
  **0.01** across the candidate plays: the plays on offer leave near-identical
  finishing chances, so no weighting of the term can separate them. That is the same
  shape as denial's "58.9% of ties have exactly zero spread", reached from the private
  side of the game rather than the shared one.
- **The lookahead is inside the turn, so it is not a lookahead.** A macro decision
  recurs immediately after the play, so a set kept in the rack is played *in the same
  turn* -- 83.7% of the first-of-turn decisions `ready` moves end that turn in exactly
  the same place. What survives to the *next* turn is what no partition of the rack
  could place, and choosing among tiles that are all stuck is choosing nothing. `stop`
  is the proof by
  exaggeration: given `END_TURN` at rank 0 it takes it in 1.42% of decisions, changes
  75.8% of deals, and still scores even -- 155 micro-actions per turn and a decision
  after every set mean stopping early keeps nothing that playing on would have spent.

What that leaves open is narrow and specific: rack potential is an axis only where the
turn boundary *forces* you to live with the residue. On a config with a tight micro
budget, a cap of one set per turn, or a rack large enough that no turn can drain its
playable part, the third mechanism above disappears and the question is open again. On
the standard config it is closed.
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import time
from dataclasses import dataclass

import numpy as np

from rummi.agents.base import Agent, Observation, has_melded, table
from rummi.agents.macro import (
    MacroAgent,
    by_value,
    extend_offset,
    laid_tiles,
    repartition_offset,
    set_templates,
    steal_offset,
)
from rummi.rules.config import CONFIG_BY_NAME, RummiConfig
from rummi.evaluate.protocol import SUITE_BY_NAME, evaluate

# The preceding experiment's runner, imported rather than rewritten: it already pins
# the self-mirror to exactly 50.00% and the ranking it recomposes is pinned to
# `by_value`'s own argmax, which is the base this objective adds a term to.
from denial_ab import _deals_for, head_to_head, interval, play_deals, rank_tables


@dataclass(frozen=True)
class Terms:
    """Weights inside `potential`, fixed per mode so the sweep is over one scalar.

    A sweep over both at once is a maximum over a grid of noisy measurements, which
    is the trap this whole file is written around.
    """

    ready: float = 0.0
    reach: float = 0.0


@dataclass(frozen=True)
class Mode:
    terms: Terms = Terms()
    stop: bool = False
    """Offer `END_TURN` inside the same objective, at rank 0 -- shedding nothing at
    all is the strongest form of "keep what combines"."""
    joker_hold: bool = False
    """Refuse to spend the rack's joker unless the play empties the rack."""


MODES: dict[str, Mode] = {
    "ready": Mode(Terms(ready=1.0)),
    "reach": Mode(Terms(reach=1.0)),
    "both": Mode(Terms(ready=1.0, reach=1.0)),
    "stop": Mode(Terms(ready=1.0, reach=1.0), stop=True),
    "joker": Mode(joker_hold=True),
}
"""`ready` is in tiles and `reach` is a probability, so both are about "one tile" at
their own scale and `w` reads the same way in every arm: how many tiles shed now a
unit of potential is worth."""

ARMS = tuple(name + suffix for name in MODES for suffix in ("", "+null"))


def state_rng(board: np.ndarray, rack: np.ndarray) -> np.random.Generator:
    """A generator fixed by the state, so a perturbing arm is still deterministic.

    `blake2b` rather than `hash`, whose seed varies per process: a control that played
    different games on every run could not be compared with anything.
    """
    digest = hashlib.blake2b(
        np.asarray(board, dtype=np.int16).tobytes()
        + np.asarray(rack, dtype=np.int64).tobytes(),
        digest_size=8,
    )
    return np.random.default_rng(int.from_bytes(digest.digest(), "little"))


class RackPotential:
    """What a rack is worth beyond its size, per candidate afterstate.

    Both terms come off one `(n, T)` shortfall, and that shortfall is a matmul rather
    than `macro.shortfall`'s broadcast subtraction: every template row is binary -- a
    run holds distinct kinds and a group holds distinct colours of one number -- so
    `max(template - rack, 0)` collapses to `template & (rack == 0)`. Exact, not an
    approximation, and `test_the_matmul_shortfall_is_the_reference_one` holds it to the
    reference over random racks. It has to be cheap: the objective scores every legal
    candidate at every decision, where a tie-break scored two.
    """

    def __init__(self, cfg: RummiConfig) -> None:
        templates = set_templates(cfg)
        assert templates.max() == 1, "the matmul shortfall needs binary templates"
        self.cfg = cfg
        self.templates = templates.astype(np.int64)
        self.rows = templates.astype(np.float32)
        self.sizes = templates.sum(-1).astype(np.float32)
        self.ext = extend_offset(cfg)
        self.steal = steal_offset(cfg)

    def played(self, options: np.ndarray, hand: np.ndarray) -> np.ndarray:
        """`(n, K)` tiles each option takes out of the hand.

        The three template blocks differ in exactly what leaves the rack: a new set
        lays what the hand can cover and stands a joker in for the rest, a lay-off
        plays its one kind, and a steal takes its missing tile off the table instead of
        out of the hand. Only options `legal_macros` offered may be passed -- that is
        what bounds a new set's joker substitution to one.
        """
        out = np.zeros((options.size, self.cfg.n_kinds), dtype=np.int64)
        new = options < self.ext
        if new.any():
            out[new] = laid_tiles(self.cfg, self.templates[options[new]], hand)
        lay = ~new & (options < self.steal)
        if lay.any():
            out[lay, options[lay] - self.ext] = 1
        take = options >= self.steal
        if take.any():
            out[take] = np.minimum(self.templates[options[take] - self.steal], hand)
        return out

    def parts(self, racks: np.ndarray, unseen: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """`(n,)` ready tiles and `(n,)` reach probability, for a batch of racks."""
        racks = np.asarray(racks)
        lack = (racks == 0).astype(np.float32)
        short = lack @ self.rows.T
        jokers = racks[:, self.cfg.joker_kind][:, None]
        ready = ((short == 0) | ((short == 1) & (jokers >= 1))) * self.sizes
        # A kind completes a set iff it is missing and some one-away template holds
        # it: `hit` counts those templates per kind, and `lack` picks out the one kind
        # each of them is short of. Per kind rather than per template, because one tile
        # serving four near-runs is one draw.
        hit = (short == 1).astype(np.float32) @ self.rows
        completing = (hit > 0) & (lack > 0)
        unseen = np.asarray(unseen, dtype=np.float64)
        reach = (completing * unseen).sum(-1) / max(float(unseen.sum()), 1.0)
        return ready.max(-1).astype(np.float64), reach


@dataclass
class MoveStats:
    """How often a positive `w` overrules the ranking, and what it pays for it."""

    decisions: int = 0
    options: int = 0
    choices: int = 0
    """Decisions offering more than one candidate: the objective cannot act elsewhere,
    however it is weighted."""
    moved: int = 0
    spread: float = 0.0
    """Summed `max - min` potential over the candidates, so the mean says how much
    room the objective has to act in at all."""
    ready: float = 0.0
    reach: float = 0.0
    """Summed potential of the rack *as it stands*, both terms, whatever the mode
    weights them at -- the size the spread has to be read against."""
    gained: float = 0.0
    """Summed potential the move buys, over the decisions it moves."""
    shed: float = 0.0
    """Summed rank the move gives up -- tiles after the opening meld, points before."""
    stopped: int = 0
    blocked: int = 0
    """Decisions where the joker rule removed a candidate."""

    def report(self) -> str:
        moved = max(self.moved, 1)
        decisions = max(self.decisions, 1)
        return (
            f"decisions {self.decisions:>7,}  cands {self.options / decisions:>4.2f}  "
            f"open {self.choices / decisions:>5.1%}  moved {self.moved / decisions:>6.2%}  "
            f"held {self.ready / decisions:>4.2f}/{self.reach / decisions:<4.2f} "
            f"spread {self.spread / decisions:>4.2f}  "
            f"gain {self.gained / moved:>+5.2f}  give {self.shed / moved:>4.2f}  "
            f"stop {self.stopped / decisions:>5.2%}  block {self.blocked / decisions:>5.2%}"
        )


class WeightedRack:
    """`by_value`'s ranking plus `w` times the potential of the rack left behind.

    A weighted objective, not a refinement: a positive `w` may take a play the ranking
    puts *below* the best one, which is the only shape in which this hypothesis has
    teeth -- reordering `by_value`'s indifference set is already bounded at ~1.6pp by
    `tools/denial_ab.py`.

    At `w = 0` the potential term is exactly zero and `argmax` returns the first
    maximum of the rank alone, which is `by_value`'s own pick down to index order. So
    the control is the same code, not a second agent.
    """

    def __init__(
        self,
        cfg: RummiConfig,
        mode: str = "ready",
        w: float = 1.0,
        null: bool = False,
        stats: MoveStats | None = None,
    ) -> None:
        assert mode in MODES, mode
        self.cfg = cfg
        self.mode = MODES[mode]
        self.w = w
        self.null = null
        self.ranked = repartition_offset(cfg)
        self.points, self.tiles = rank_tables(cfg)
        self.potential = RackPotential(cfg)
        self.stats = stats if stats is not None else MoveStats()

    def __call__(self, obs: Observation, env: int, legal: np.ndarray) -> int:
        options = np.flatnonzero(legal[: self.ranked])
        if not options.size:
            # Nothing a template describes is legal, so there is nothing to score:
            # REPARTITION, END_TURN or DRAW, exactly as `by_value` leaves it.
            return int(np.flatnonzero(legal)[0])

        cfg = self.cfg
        rank = (self.points if not bool(has_melded(obs)[env]) else self.tiles)[options]
        rank = rank.astype(np.float64)
        rack = np.asarray(obs["rack"][env]).astype(np.int64)
        hand = rack + np.asarray(obs["workbench"][env]).astype(np.int64)
        unseen = np.asarray(obs["unseen"][env])

        played = self.potential.played(options, hand)
        left = np.vstack([hand[None] - played, hand[None]])
        # The hand's own row rides along, which is both what `stop` scores ending the
        # turn at and the level every spread is read against.
        ready, reach = self.potential.parts(left, unseen)
        terms = self.mode.terms
        weighted = terms.ready * ready + terms.reach * reach
        value, held = weighted[:-1], weighted[-1]

        end_macro = len(legal) - 2
        stopping = self.mode.stop and bool(legal[end_macro])
        if stopping:
            # Ending the turn sheds nothing, so its rank is 0 and its potential is the
            # rack as it stands. Appended last, so at `w = 0` the first maximum is
            # still a play.
            options = np.append(options, end_macro)
            rank = np.append(rank, 0.0)
            value = np.append(value, held)
            played = np.vstack([played, np.zeros_like(hand)[None]])
        else:
            left = left[:-1]

        stats = self.stats
        stats.decisions += 1
        stats.options += int(options.size)
        stats.choices += int(options.size > 1)
        stats.spread += float(value.max() - value.min())
        stats.ready += float(ready[-1])
        stats.reach += float(reach[-1])

        board = np.asarray(table(obs)[env])
        if self.null:
            # Same values, wrong owners: a perturbation of the arm's own magnitude on a
            # basis that cannot be right. A control firing at a different rate would
            # answer a different question.
            value = value[state_rng(board, rack).permutation(value.size)]

        allowed = np.ones(options.size, dtype=bool)
        if self.mode.joker_hold:
            spends = played[:, cfg.joker_kind] > 0
            # A joker spent to win is not a joker lost, and the rule has to say so or
            # it refuses the finish it exists to protect.
            finishes = left.sum(-1) == 0
            blocked = spends & ~finishes
            if self.null and blocked.any():
                blocked = np.zeros(options.size, dtype=bool)
                picked = state_rng(board, rack).choice(
                    options.size, size=int((spends & ~finishes).sum()), replace=False
                )
                blocked[picked] = True
            if blocked.any() and not blocked.all():
                allowed = ~blocked
                stats.blocked += 1

        # `by_value`'s own pick, unrestricted, so `moved` counts every way this arm
        # departs from it -- the weight overruling the rank and the joker rule
        # withdrawing a candidate alike.
        base = int(np.argmax(rank))
        score = np.where(allowed, rank + self.w * value, -np.inf)
        chosen = int(np.argmax(score))
        if chosen != base:
            stats.moved += 1
            stats.gained += float(value[chosen] - value[base])
            stats.shed += float(rank[base] - rank[chosen])
            stats.stopped += int(options[chosen] == end_macro and self.mode.stop)
        return int(options[chosen])


def build_arm(
    cfg: RummiConfig, arm: str, w: float, repartition: bool, stats: MoveStats
) -> Agent:
    """One arm as an agent. `base` is `by_value` itself, the thing being reweighted."""
    if arm == "base":
        return MacroAgent(cfg, choose=by_value(cfg), repartition=repartition)
    mode, _, suffix = arm.partition("+")
    choose = WeightedRack(cfg, mode=mode, w=w, null=suffix == "null", stats=stats)
    return MacroAgent(cfg, choose=choose, repartition=repartition)


def outcome_changes(
    cfg: RummiConfig, make_a, make_b, deals: int, seed_base: int, batch: int = 32
) -> np.ndarray:
    """`(deals,)` -- did seating the arm change how the deal ended, from any seat?

    The bound a win rate cannot give. Both agents are deterministic and the deal fixes
    the pool, so a deal whose winner and final racks are untouched is a deal the arm
    played identically: if it changes `f` of them, the win rate it can move is at most
    `f / 2`, whatever the ranking inside. That ceiling is what tells a 4%-of-decisions
    intervention apart from a 4%-of-*games* one, and the two are not the same thing
    here.
    """
    out = np.zeros(deals, dtype=bool)
    for start in range(0, deals, batch):
        count = min(batch, deals - start)
        seeds = [np.random.SeedSequence([seed_base, start + i]) for i in range(count)]
        control = [make_b(cfg) for _ in range(cfg.n_players)]
        winner, values = play_deals(cfg, control, seeds)
        for seat in range(cfg.n_players):
            seats: list[Agent] = [make_b(cfg) for _ in range(cfg.n_players)]
            seats[seat] = make_a(cfg)
            other, theirs = play_deals(cfg, seats, seeds)
            out[start : start + count] |= (other != winner) | (theirs != values).any(-1)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="standard", choices=sorted(CONFIG_BY_NAME))
    p.add_argument(
        "--arena", default="both", choices=("head2head", "suite", "sweep", "diagnose", "both"),
        help="'diagnose' plays the arm and reports only how often the weight acts; "
        "'sweep' scores every --sweep weight on one seed base, which is a maximum "
        "over noisy measurements and must be confirmed at another",
    )
    p.add_argument("--deals", type=int, default=1600, help="distinct deals per head-to-head arm")
    p.add_argument("--games", type=int, default=400, help="deals for the standard-greedy arm")
    p.add_argument("--modes", nargs="+", default=list(ARMS), choices=sorted({*ARMS, "base"}))
    p.add_argument("--w", type=float, default=1.0, help="weight on the potential term")
    p.add_argument(
        "--sweep", nargs="+", type=float, default=[0.5, 1.0, 2.0, 4.0, 8.0],
        help="weights for --arena sweep",
    )
    p.add_argument(
        "--opponent", default="base", choices=("base", *ARMS, "optimal", "frugal", "greedy"),
        help="who the head-to-head is against; 'base' is plain by_value",
    )
    p.add_argument(
        "--repartition", action="store_true",
        help="run every arm and the baseline at frugal tier, with the stuck-state solve on",
    )
    p.add_argument("--seed-base", type=int, default=94_000, help="ad-hoc suite, outside the frozen ones")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument(
        "--out", type=pathlib.Path, default=None,
        help="npz of the per-deal head-to-head outcomes, for pairing arms across runs",
    )
    args = p.parse_args()

    cfg = CONFIG_BY_NAME[args.config]
    stats: dict[str, MoveStats] = {}

    def make(arm: str, w: float | None = None):
        key = arm if w is None else f"{arm}@{w}"
        stats.setdefault(key, MoveStats())
        weight = args.w if w is None else w
        return lambda c: build_arm(c, arm, weight, args.repartition, stats[key])

    if args.opponent in ("base", *ARMS):
        make_b = make(args.opponent)
        opponent = f"{args.opponent}{'+repartition' if args.repartition else ''}"
    else:
        from rummi.agents import build as build_registry

        make_b = lambda c: build_registry(args.opponent, c)  # noqa: E731
        opponent = args.opponent

    print(
        f"config {args.config}  repartition {args.repartition}  w {args.w}  "
        f"baseline {'by_value' if args.opponent == 'base' else opponent}"
    )

    if args.arena == "diagnose":
        # The cheap answer, and it comes before any game is scored: if a positive
        # weight rarely moves the pick and barely changes the potential when it does,
        # that is the finding.
        print(f"\nhow often the weight acts, {args.deals} deals x {cfg.n_players} seats")
        for arm in args.modes:
            began = time.perf_counter()
            changed = outcome_changes(
                cfg, make(arm), make("base"), args.deals, args.seed_base, args.batch
            )
            print(f"  {arm:<12} {stats[arm].report()}  {time.perf_counter() - began:.0f}s")
            print(
                f"  {'':<12} deals changed {changed.mean():>6.1%}  "
                f"-> the win rate can move at most {changed.mean() / 2:>5.2%}",
                flush=True,
            )
        return

    if args.arena == "sweep":
        print(f"\nweight sweep against {opponent}, {args.deals} deals x {cfg.n_players} seats")
        print("  a maximum over noisy cells -- confirm the pick at another --seed-base")
        for arm in args.modes:
            for w in args.sweep:
                key = f"{arm}@{w}"
                began = time.perf_counter()
                wins, scores = head_to_head(
                    cfg, make(arm, w), make_b, args.deals, args.seed_base, args.batch
                )
                win, win_ci = interval(wins)
                score, score_ci = interval(scores)
                print(
                    f"  {arm:<12} w {w:<5} win {win:>6.2%} +-{win_ci:>5.2%}  "
                    f"score {score:>+7.2f} +-{score_ci:>5.2f}  "
                    f"moved {stats[key].moved / max(stats[key].decisions, 1):>6.2%}  "
                    f"{time.perf_counter() - began:.0f}s",
                    flush=True,
                )
        return

    if args.arena in ("head2head", "both"):
        print(f"\nhead-to-head against {opponent}, {args.deals} deals x {cfg.n_players} seats")
        played: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for arm in args.modes:
            stats[arm] = MoveStats()  # one count per measurement, not per process
            began = time.perf_counter()
            wins, scores = head_to_head(
                cfg, make(arm), make_b, args.deals, args.seed_base, args.batch
            )
            played[arm] = (wins, scores)
            win, win_ci = interval(wins)
            score, score_ci = interval(scores)
            even = 1.0 / cfg.n_players
            print(
                f"  {arm:<12} win {win:>6.2%} +-{win_ci:>5.2%} (even {even:.1%})  "
                f"score {score:>+7.2f} +-{score_ci:>5.2f}  "
                f"{time.perf_counter() - began:.0f}s",
                flush=True,
            )
            print(f"  {'':<12} {stats[arm].report()}")
            print(
                f"  {'':<12} 2pp needs {_deals_for(0.02, float(np.std(wins, ddof=1))):,} deals",
                flush=True,
            )

        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                args.out,
                deals=args.deals,
                seed_base=args.seed_base,
                w=args.w,
                **{f"{arm}_{field}": played[arm][i] for arm in played
                   for i, field in enumerate(("wins", "scores"))},
            )
            print(f"\n  wrote {args.out}")

        # Every arm met the same baseline on the same deals, so an arm pairs with its
        # own null on the deal -- which is the tighter comparison, and the only one
        # that says whether the arm beats the perturbation rather than the mean.
        pairs = [(a, f"{a}+null") for a in args.modes if f"{a}+null" in played]
        if pairs:
            print(f"\n  paired against each arm's own null, {args.deals} deals")
            for arm, null in pairs:
                win, win_ci = interval(played[arm][0] - played[null][0])
                score, score_ci = interval(played[arm][1] - played[null][1])
                print(
                    f"  {arm + ' - null':<14} win {win:>+6.2%} +-{win_ci:>5.2%}  "
                    f"score {score:>+7.2f} +-{score_ci:>5.2f}"
                )

    if args.arena in ("suite", "both"):
        suite = SUITE_BY_NAME["standard-greedy" if args.config == "standard" else "tiny"]
        print(f"\n{suite.name}, {args.games} deals x {cfg.n_players} seats")
        for arm in dict.fromkeys(("base", *args.modes)):
            stats[arm] = MoveStats()
            began = time.perf_counter()
            result = evaluate(arm, suite, build_agent=make(arm), games=args.games)
            assert not result.disqualified, f"{arm} was disqualified"
            # Error bars over games rather than deals, because `evaluate` reports the
            # suite in aggregate: the two seats of a deal are correlated, so these are
            # the *narrowest* the intervals could honestly be, not the true widths.
            games = max(result.games, 1)
            win_ci = 1.96 * np.sqrt(result.win_rate * (1 - result.win_rate) / games)
            score_ci = 1.96 * float(np.std(result.scores, ddof=1)) / np.sqrt(games)
            print(
                f"  {arm:<12} win {result.win_rate:>6.1%} +-{win_ci:>4.1%}  "
                f"score {result.mean_score:>+8.2f} +-{score_ci:>5.2f}  "
                f"stale {result.stalemates / games:>5.1%}  n={games}  "
                f"{time.perf_counter() - began:.0f}s",
                flush=True,
            )
            if stats[arm].decisions:
                print(f"  {'':<12} {stats[arm].report()}")


if __name__ == "__main__":
    main()
