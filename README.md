# rummi

[![ci](https://github.com/rsanchezmo/rummi/actions/workflows/ci.yml/badge.svg)](https://github.com/rsanchezmo/rummi/actions/workflows/ci.yml)

A Rummikub **environment** for reinforcement learning, with opponents to play
against. Batch-native from the ground up, wrapped as a Gymnasium vector env, and
implemented three times over — NumPy, torch and JAX — each verified against the
others bit for bit.

The bundled agents are the point as much as the env is. You get a floor
(`greedy`) and a ceiling (`optimal`, an OR-Tools CP-SAT solver that plays each
turn perfectly), so a new agent has something real to measure itself against on
day one.

**Why it exists.** The project set out to find a strategy — hand-written or
learned — that beats an optimal per-turn solver at Rummikub, and built the env,
the ladder and the training machinery in order to look for one. It did not find
one, and the search is now closed at the resolution these suites afford: **a
single round of standard Rummikub has no strategic layer above playing the best
available turn**, at two, three or four seats, within about a point of win rate,
even with perfect information. So `optimal` is the game's ceiling and not merely
the strongest thing in the box, and that is proven rather than asserted. The env,
the three backends, the ladder and that negative result are what this repo is.
[What the benchmark found](#what-the-benchmark-found) is the argument and the
numbers.

```bash
pip install -e '.[dev,env,render,solver]'

python -m rummi.render.record --play --render-mode ansi    # watch a game
python -m rummi.evaluate.run --agent greedy               # score against the agents
python -m rummi.bench.bench_backends --compile             # compare backends
```

```python
import gymnasium as gym
import rummi.env                                    # registers the ids below

env = gym.make_vec("Rummi-2p-v0", num_envs=256)     # or Rummi-3p-v0, Rummi-4p-v0
obs, info = env.reset()
obs, rewards, terminated, truncated, info = env.step(actions)
# info["action_mask"] is (num_envs, 2400) and never all-zero
# info["current_player"] says whose view obs is — one policy plays every seat
```

Seat count is in the id because it changes the observation and action spaces.
Only a vector entry point is registered, so `gym.make` fails by design: a batch
of one is not a single-agent env. `RummiVectorEnv` is importable directly from
`rummi.env.vector_env` when you want to pass a `cfg` of your own.

That env is **self-play**: one policy plays every seat. To hold one seat and let
the bundled agents play the rest:

```python
from rummi.env.fixed_opponent import FixedOpponentEnv
from rummi.rules.config import STANDARD_4P

env = FixedOpponentEnv(num_envs=256, cfg=STANDARD_4P, opponent="greedy")
obs, info = env.reset()      # obs is always a position your seat can act in
```

One `step` is your micro-action plus however many the other seats need to hand
control back, so reward covers their replies rather than arriving on a step you
could not act on. `opponent="optimal"` costs a CP-SAT solve per opponent turn per
env and cannot batch — that one is for evaluation, not training.

The env runs on any of the three backends — `backend="numpy" | "torch" |
"torch-mps" | "jax"` — and the observation comes back in that backend's own array
type, so a device rollout never round-trips through the host. That is worth about
**4x** over NumPy; the numbers are under [Backends](#backends). Rendering reads a
`BatchState`, so a device backend refuses a render mode at construction rather
than failing on the first frame.

Want tensors instead of arrays? Use Gymnasium's own wrappers rather than a mode of
this env — `gymnasium.wrappers.vector.NumpyToTorch` over the NumPy backend, or
`JaxToTorch` / `JaxToNumpy` over the JAX one. They convert through `from_dlpack`,
so a same-device hand-off is a view, not a copy.

One `step` is one primitive table operation, so a player's turn spans several
steps. Observations are always the acting seat's view, which is what lets a
single shared policy play self-play without learning two conventions.

| `render_mode="ansi"` | `render_mode="human"` |
|:--:|:--:|
| ![A full game in the terminal view](docs/render-terminal.gif) | ![The same game in the pygame window](docs/render-window.gif) |

The same game, frame for frame. Both are built from one pass over one `GameView`
sequence, so frame *N* of each is the same turn — the turn counter, the flagged
slots and the action log agree because they are the same data, not because two
renderers happen to concur. Every frame is a *committed turn*, which is what
`render_on="turn"` gives you live. Watch the table fill from three sets to
twenty-two and the winner's rack drain to nothing.

## Why Rummikub

Because unrestricted rearrangement makes it genuinely hard. A turn is not "play a
card": it is *leave the table partitioned into valid sets*, and you may take
every existing set apart to do it. Deciding whether a tile multiset admits such a
partition is the problem [Den Hertog & Hulshof
(2006)](https://academic.oup.com/comjnl/article-abstract/49/6/665/356316) solve
by integer programming. The interesting move — steal a tile out of a run to
complete a group, after extending the run so it survives losing it — is invisible
to anything that does not search.

## The opponents

Seven agents, all playable out of the box and all usable as opponents in your
own training loop. Scored here on the `standard-greedy` suite over 120
games; `score` is official Rummikub scoring from the agent's side.

![Agent win rate and stalemate rate, weakest to strongest](docs/charts/agents.svg)

| agent | win rate | score | turns | rack left | stalemates |
|---|---:|---:|---:|---:|---:|
| `random` | 0.0% | −442.3 | 90.6 | 442.3 | 100.0% |
| `weighted-random` | 0.0% | −442.3 | 90.6 | 442.3 | 100.0% |
| `greedy` | 50.0% | +0.0 | 124.7 | 48.2 | 98.3% |
| `rearrange` | 85.0% | +31.9 | 129.3 | 21.7 | 80.0% |
| `learned` (net + CP-SAT) | **100.0%** | +48.5 | 65.0 | 0.0 | 0.0% |
| `frugal` | **100.0%** | **+49.0** | 64.9 | 0.0 | 0.0% |
| `optimal` (CP-SAT) | **100.0%** | +44.6 | 65.7 | 0.0 | 0.0% |

**Read the random row as a warning, not a floor.** On the standard config random
play is *byte-identical to passing every turn* — same scores, same turn counts. It
can never assemble a legal 30-point opening meld, so it never places a tile and
its choices have no effect on the game whatsoever. Beating random says nothing.
**Greedy at 50% is the real floor**; it is also the suite's own opponent, which
is why it scores exactly even.

The ladder is one idea: **how much of the table are you willing to take apart?**
`greedy` never rearranges, so once it can neither append to a set nor lay one
from its rack it just draws — which is why it stalls out in 98% of games.
`rearrange` steals exactly **one** tile from a set that stays legal without it,
and that alone is worth 35 points of win rate. `frugal` plays template sets and
single-tile steals, and repartitions the whole table **only where nothing else
plays** — one CP-SAT solve at exactly the states that need one. `optimal`
repartitions every turn. That the two are statistically even (48.7%
head-to-head over 600 games) at an order of magnitude apart in compute is a
measured fact about the game: per-turn play above `frugal` buys nothing the
stuck-state solve does not already deliver.

`learned` is the same question answered by a network instead of a rule. It sees
one repartition among all the ordinary moves and picks — **the net decides *when*
to spend a solve, CP-SAT decides *how*** — and it lands even with the other two,
which is how a rung carrying weights earned a place here. It is a 233k-parameter
DAgger clone of `by_value` driving that same macro space with the stuck-state
solve on — an optimal-tier teacher at +48.01 — and it reproduces the teacher to
within noise from the observation alone, which is what keeps "the observation is
sufficient to play optimally" a tested claim rather than a hope. It needs the
`torch` extra (`pip install -e '.[torch]'`) and ships one file of weights per
seat count, since the observation widens with the table.

## What the benchmark found

The goal was a strategy that beats an optimal per-turn solver: something above
"play the best turn you can see". Two weeks of arms went at it — PPO in the
primitive, macro and hybrid action spaces, cloning and DAgger, self-play with a
snapshot league, best response against `optimal` itself, belief features and an
LSTM over decision history, a net handed the opponent's true rack, afterstate
value learning, turn-completion search, hand-written tie-breaks on the table and
on the rack, and last an opening rule. Then the ceiling itself was measured, by
hindsight instead of by learning.

**None of it found one.** In a single round of standard Rummikub there is no
strategic layer above playing the best available turn — at two, three or four
seats, at about one point of win-rate resolution, even with the opponent's rack
and the rest of the deck in view. The skill this game rewards is computational,
and that part is enormous: in the table above `greedy`, which never takes a set
apart, loses all 120 games to `optimal`, which repartitions every turn. Above
rearrangement it is the deck. [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) is the
lab notebook, a section per arm and the reason each stopped; this is the argument.

### The game, in numbers

`optimal` against `frugal`, 200 deals played from both seats:

- **The two top rungs are even at 47.2%**, and **the same seat wins both games of
  a deal in 68% of them.** The deal is a larger term than the agent, which is why
  every suite here rotates seats and pairs on deals.
- **58% of turns are a forced `DRAW`** — 19.1 draws per seat per game against 13.7
  playing turns. In most turns no legal play exists at all.
- **The loser is close**: 2 tiles left at the median, ≤1 in 30% of games, ≤3 in
  78%. Nothing runs out either — 40 of the 78 pool tiles are still unseen at the
  end, and none of the 400 games stalemated.
- **Turn order outweighs either agent's choices.** First mover wins 51.2%, 52.5%
  in the `optimal` mirror, and the leader at half-time wins 59%.

58% forced draws, decided by two tiles, with the seat worth 68% of the outcome:
that is the shape everything below is measured inside.

```bash
python tools/game_structure.py --a optimal --b frugal --deals 200
```

### The ceiling on choosing a turn

Each of those arms tested one candidate strategy, and a null reads the same
whether the axis is empty or merely out of reach. `tools/oracle_regret.py`
measures the bound instead: for every turn `frugal` played against itself, replay
the rest of the game from every *other* whole turn the rules admit. A step holds
no randomness — the deck is permuted once at `reset` and drawing only advances a
pointer — so a boundary cloned and continued with deterministic agents replays
the rest of the game tile for tile. **The game that was played is therefore the
baseline rollout for every decision in it**, and each deviation is priced against
that baseline on that deck, with no variance to average away. Every alternative
is a turn CP-SAT built under an added restriction and then run one primitive
action at a time against the env's own mask, so the engine accepts it as a turn.

300 deals, 8,185 playing decisions, **39,903 alternative turns each rolled out to
the end**:

![Win-rate change of one-turn deviations, by deviation type and seat count](docs/charts/regret.svg)

| deviation from the turn `frugal` played | alternatives | changes the winner | delta vs that turn |
|---|---:|---:|---:|
| a different CP-SAT optimum | 1,248 | 11.4% | +0.0 ±2.3 |
| **the same tiles onto a different table** | **15,801** | **3.6%** | **−0.2 ±0.5** |
| fewer tiles shed | 5,886 | 27.5% | −3.1 ±1.5 |
| no rearrangement at all | 768 | 27.3% | −5.9 ±3.9 |
| `DRAW` instead of playing | 8,185 | 26.3% | −0.4 ±1.1 |
| the played turn, re-derived | 8,015 | 0.0% | +0.0 ±0.0 |

- **The free parameter every table-shaping idea aimed at is worth nothing.**
  Shedding the same tiles onto a different table is **−0.2 ±0.5pp** over 15,801
  tries with the future deck and the opponent's rack both visible — the tightest
  interval in the repo. Nor was the choice empty: a single max-tiles table exists
  at only 24.3% of decisions.
- **The instrument detects a worse turn when handed one.** Shedding fewer tiles
  loses 3.1pp (−8.3 in the endgame) and freezing the table 5.9 (−12.4), at eight
  times the winner-change rate. The last row is the exactness check: 8,015 turns
  re-derived to the same table and tiles, every one reproducing its baseline.
- **`frugal` and `optimal` are not close, they are the same.** CP-SAT's optimum
  differs from the turn `frugal` played at 15.2% of decisions and is worth **+0.0
  ±2.3pp** there — their head-to-head tie re-derived per decision, not per match.
- **Never quote the headline.** *Some* deviation rescues 91.3% of lost games, and
  a coin flip over the same deviations predicts 99.5%; 93.0% of *won* games are
  thrown the same way. A maximum over ~120 rollouts per seat changes what the
  shared pool hands out from that turn on, so the game is re-decided. Deck chaos.

A mean over a type cannot see a targeted effect inside it, so every turn also
records the opponent's best reply, solved by CP-SAT against its true rack. The one
cell the hypothesis names — the opponent is a tile from finishing and an
equal-tile alternative shuts the door it needs — is worth **+11 ±22pp at two
seats, +20 ±21 at three, +18 ±19 at four**, and **fires 0.03, 0.07 and 0.11 times
per seat-game**: shutting every exit available moves a win rate by 0.2–0.5pp, 1pp
at the very top of those intervals. Why it is that rare outlives the measurement:
**a solved reply is a function of the tile multiset the table holds, not of its
arrangement**, pinned by a test on `solve_turn` itself. Against any opponent that
can rearrange, table *shape* carries nothing — only which tiles are shed can
matter, and 91.5% of equal-tile alternatives shed exactly the same tiles. Lay-off
permeability, the observable a real agent would have to find the door with,
correlates **+0.05** with what the opponent could do and moves two hundred times
more often than the door does.

```bash
python tools/oracle_regret.py --games 300 --seed-base 91000 --kbest 4 --workers 8
python tools/oracle_regret.py --config standard_3p --games 100 --seed-base 73000
python tools/oracle_regret.py --config standard_4p --games 150 --seed-base 76000
```

Three seats break the argument that closed two — the table is common property, so
at two seats a door closed on the opponent is closed on the closer, while with two
opponents it costs the closer once and the opposition twice — and four should be
weaker still for per-turn play, a turn of yours being three turns from mattering.
**Every deviation type is at or below zero at three seats** (same tiles, other
table: −1.2 ±1.1 over 7,280), and at four the only positive point estimate in the
three runs is +0.4 ±1.3, covering zero, while `fewer_tiles` loses 8.3pp there and a
frozen table 12.1 — more heavily than anywhere else.

### The one lead, and how it died

One cell of fourteen moved. Pre-meld, an alternative shedding *fewer* tiles than
the played turn was worth **+8.1 ±6.2pp** (n=384), against −1.1 midgame and −8.4
in the endgame, and the mechanism was specific: **the opening is the one decision
that does not recur.** Before melding the table is untouchable, so the sets you
open with are handed over as rigid sets, while a tile kept back is played later
with full rearrangement rights.

Tested as a policy, pre-sized and pre-registered: nine opening rules, each
`frugal` with the opening turn rebuilt and **every post-meld decision delegated to
`by_value` untouched**, over 1600 deals paired against plain `frugal`, then 800
against `optimal` and 800 each at three and four seats. **No arm met the bar and
no interval excludes zero.** `runs_first` is widest at +0.91 ±1.04 and its sign
flips at three seats; `min_sets` reads +0.56 ±1.02 and −0.67 at three;
`min_tiles`, the same class of deviation the oracle priced at +8.1, is **+0.12
±1.22** — about five times tighter than the cell, and excluding it. The only sign
that repeats in every arena is the negative control's: shedding *more* at the
opening is mildly bad, **−0.45 ±0.50pp** pooled over 2p, 3p and 4p.

The arms are not inert, which is what makes that a result rather than a
resolution failure: they move 6.0–12.5% of pre-meld decisions and change the
outcome of 17.1–25.9% of deals, so each had ~9pp available. The mechanism is
present at every step and **gone within two turns** — openings shrink 5.71 → 4.13
tiles and the opponent's next turn sheds 2.58 → 2.40, but a turn after that it is
1.141 against `base`'s 1.146 while the arm's own shedding rises 0.902 → 0.951. The
kept tiles are played a turn later by the hand that kept them. Two exactness
controls carry the reading: `frugal` mirrored against itself is *exactly* 50.00 /
33.33 / 25.00%, and `by_value`'s own opening rebuilt by the arms' planner is
exactly +0.00 over **24,490** pre-meld decisions with 0.00% moved.

```bash
python tools/opening_ab.py --arena head2head --deals 1600 --seed-base 93000
```

### Why no training run could have seen this

**An effect of 1pp on win rate is 0.02 in reward units** against a terminal reward
of ±1. The macro trainer normalises its advantage over ~30 closed episodes per
update, so the standard error of that mean is 1/√30 ≈ 0.18 — an order of magnitude
above the signal — before the one reward is credited across every decision of the
episode. From the other side, the ranking that separates two adjacent rungs spans
**0.012 normalised units against 0.073 of per-episode noise**, which is why the
afterstate value learner finds the heuristic band in 80 updates and never locks
the last three points, and why search over it is worth +0.71 ±2.07. Behaviour
agrees: everything that keeps a cloned policy competent moves it **at most 1.6% of
its decisions**, and the first setting that moves more destroys it. The hindsight
instrument resolves what none of that could for one reason — **it pairs each
deviation with its own baseline on the same deck**, so the variance a learner must
average over does not exist here at all.

### What is closed, and what is not

Closed at every seat count the repo scores: table shaping, rack potential, the
opening, and information — a net handed the opponent's true rack ties its own
control on the same deals, and zeroing that block flips **0.0%** of ~9,500 argmax
decisions at two seats and of 8,409 at three. **So the ladder's top rung is the
game's ceiling for one round of the standard rules**, and a submission's room is
the distance below it. Not closed, and named rather than waved at:

- **Multi-turn plans.** The construction deviates on *one* turn, so giving
  something up now to collect it later sits outside it. Three arms went at it
  directly — a delegating agent, self-play from the clone, best response against
  `optimal` — and all three are null, which is not the same as exhaustive.
- **Adaptation against a population.** A best response to one fixed deterministic
  opponent is itself a fixed policy, so "work out who you are playing" has no
  content in any arena measured here. Different benchmark; the precondition is
  whether the best responses to `greedy` and to `optimal` differ at all.
- **Configs that force a residue.** Rack potential is an axis only where a turn
  boundary *forces* tiles to be kept — a tight micro budget, a one-set-per-turn
  cap, a rack too large to drain in one turn. Standard forces none of them.
- **Margin, the cross-turn decision this scoring never poses.** Official Rummikub
  runs over rounds and the loser pays the face value left in the rack, so a round
  already lost plausibly holds a real decision: shed *value* rather than tiles,
  spend a held joker rather than keep it. The frozen suites cannot see it — they
  score one round by its winner — and it is the likeliest place a layer survives.
- **An edge below the noise floor**, about ±1pp at these game counts.

| measured | result | notebook | reproduce |
|---|---|---|---|
| the shape of a game between the top two rungs | 47.2%, and the seat decides 68% of deals | above | `python tools/game_structure.py --a optimal --b frugal --deals 200` |
| every single-turn deviation, 2p | same tiles / other table, −0.2 ±0.5pp over 15,801 | [Oracle one-step regret](docs/EXPERIMENTS.md#oracle-one-step-regret-the-bound-the-nulls-were-missing) | `python tools/oracle_regret.py --games 300 --seed-base 91000` |
| targeted endgame denial | +11 ±22pp, firing 0.03 times per seat-game | [Endgame denial](docs/EXPERIMENTS.md#endgame-denial-the-cell-that-survived-the-oracle-and-the-reason-it-is-empty) | the same run |
| the same at three and four seats | −1.2 ±1.1 and +0.4 ±1.3 | [Three seats](docs/EXPERIMENTS.md#three-seats-the-argument-that-killed-it-at-two-does-not-apply-and-the-answer-is-the-same), [four seats](docs/EXPERIMENTS.md#four-seats-and-the-number-the-three-runs-agree-on) | `python tools/oracle_regret.py --config standard_3p --games 100 --seed-base 73000` |
| the opening, played as a policy | best arm +0.91 ±1.04, control −0.45 ±0.50 pooled | [The opening](docs/EXPERIMENTS.md#the-opening-was-the-one-cell-that-moved-and-it-does-not-survive-being-played) | `python tools/opening_ab.py --arena head2head --deals 1600 --seed-base 93000` |
| table shaping, as a tie-break | +1.06 ±1.57pp against a coin flip over the same ties | [Board shaping](docs/EXPERIMENTS.md#board-shaping-the-axis-the-oracle-never-bounded-and-why-it-has-no-sign) | `python tools/denial_ab.py --arena both --deals 1600 --games 400` |
| rack potential, as a weighted objective | no arm beats its own permuted control; best +1.09 ±1.75 | [Rack potential](docs/EXPERIMENTS.md#rack-potential-the-asymmetric-half-and-why-a-recurring-decision-closes-it) | `python tools/rack_potential_ab.py --arena both --deals 1600 --games 800 --w 1.0` |
| every learning attempt, and what killed it | the strongest learned agent is a clone of a heuristic, +47.32 | [Training attempts](docs/EXPERIMENTS.md#training-attempts) | one seed each; see the file |

Summaries are committed as `docs/data/regret.json`, `docs/data/opening.json` and
`docs/data/game_structure.json`, and `tools/render_charts.py` draws the figure from
the first. The per-decision JSON runs to tens of megabytes, so it is regenerable
rather than committed; `oracle_regret.py --from-json` re-prints the tables from a
finished run, pooling several of one config with the deals renumbered so a
clustered interval still counts one deal once.

## Writing an agent

Agents plug into the same interface the bundled ones use — there is one way to
write an agent, and the bundled ones are not special. An agent sees an
observation and a legal-action mask, and nothing else. That is
the integrity property the whole benchmark rests on: the observation exposes only
what a player is entitled to know — your rack, the table, and `unseen` (the pool
and your opponents' racks combined, indistinguishable from one another). Every
reference agent obeys this, the CP-SAT one included, which is a proof the
observation is sufficient to play optimally.

```python
import numpy as np
from rummi.evaluate.protocol import SUITE_BY_NAME, evaluate

class MyAgent:
    name = "mine"

    def __init__(self, cfg):
        self.cfg = cfg

    def reset(self, n_envs):
        ...                      # clear per-env memory; envs are recycled

    def act(self, obs, mask, active=None):
        # obs["rack"], obs["table_sets"], obs["unseen"], ...
        # mask is (n_envs, n_actions) and never all-zero: DRAW is always legal.
        # `active` marks the envs you control — honour it if you cache plans.
        return np.argmax(mask, axis=-1)

print(evaluate("mine", SUITE_BY_NAME["tiny"], build_agent=MyAgent).report())
```

A turn spans several `act` calls, so an agent that plans a whole turn should
cache its plan and consume it; subclass `rummi.agents.base.PlanningAgent` to get
that bookkeeping for free. [`CONTRIBUTING.md`](CONTRIBUTING.md) has the rest. Proposing a masked-out action **disqualifies** the run
rather than costing reward: the mask exactly describes what the rules permit, so
an illegal action is a bug, not a strategy.

### Every registered env has a score

`Rummi-3p-v0` and `Rummi-4p-v0` are scored too, on the `standard-3p` (165 games)
and `standard-4p` (220 games) suites, so an agent trained at any seat count has a
baseline to beat.

| agent | win 2p | win 3p | win 4p | score 2p | score 3p | score 4p |
|---|---:|---:|---:|---:|---:|---:|
| `greedy` | 50.0% | 33.3% | 25.0% | +0.0 | +0.0 | +0.0 |
| `rearrange` | 85.0% | 64.8% | 54.1% | +31.9 | +49.9 | +60.1 |
| `learned` | 100.0% | 99.4% | 100.0% | +48.5 | +94.5 | +140.2 |
| `frugal` | 100.0% | 99.4% | 100.0% | +49.0 | +93.6 | +140.8 |
| `optimal` | 100.0% | 100.0% | 99.5% | +44.6 | +94.0 | +139.4 |

Two things the seat count does to the ladder. `greedy` lands on exactly
`1 / n_players` in every column — it is each suite's own opponent, and that the
number is *exact* rather than close is the rotation check working, not a
coincidence. And `optimal` stops being *perfect* at four seats — but only just:
99.5% is one stalemate in 220 games, which is not a gap to aim at. **What the top
rung now is, is the game's measured ceiling**: an agent that plays the best turn
available every turn, with nothing above that worth a point at any seat count
([What the benchmark found](#what-the-benchmark-found)). A submission places
itself on the rungs below it, and the interesting distance is `greedy` to
`rearrange` to the solver tier.

`learned` is one net per seat count, trained by the same recipe and nothing else
— the observation widens with the table, so a 2p net cannot be pointed at a 3p
suite and is refused rather than reshaped. That all three land on their teacher
is the point of the column: the recipe transfers, the weights do not.

## Scoring yourself

`rummi.evaluate` plays your agent against the bundled ones under a frozen,
versioned protocol (`PROTOCOL_VERSION`), so a number you quote today still means
the same thing later.

| suite | config | opponent | deals |
|---|---|---|---|
| `tiny` | 3 colours × 4 numbers | `greedy` | 100 |
| `standard-greedy` | full 106-tile game | `greedy` | 200 |
| `standard-optimal` | full 106-tile game | `optimal` | 100 |

`standard-optimal` runs CP-SAT once per turn per env, so it takes roughly a
minute per hundred games — fine for a score, too slow to iterate against. Use
`tiny` while developing.

Every deal is played **twice, with the seats swapped**. That cancels both the
turn-order advantage and the luck of the deal, so an agent mirrored against
itself scores exactly `1 / n_players` and +0.0 — not that within error bars. Each game's
seed is derived from its index alone, so batching cannot change which deals run.

[`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) records every attempt at an agent
strong enough to join the ladder — learned and hand-written — the reason each one
stopped, and the hindsight measurements that bound what was left to find. The
scores are the least interesting column: most failed for something that took a day
to find and a line to describe.
Deterministic entries are captured to `docs/data/experiments.json` with
`python tools/capture_experiments.py`; the training runs are one seed apiece and
cannot be.

## How a turn is expressed

One `step` is one primitive table operation; a turn is a sequence of them ending
in `END_TURN` or `DRAW`. Tiles picked up mid-turn sit in a *workbench*, and the
table only has to be whole again when the turn commits.

| action | count | effect |
|---|---:|---|
| `PLACE(kind)` | 53 | rack → workbench |
| `PICK(slot, pos)` | 455 | one tile of a set → workbench |
| `DISSOLVE(slot)` | 35 | whole set → workbench |
| `ASSIGN(kind, slot)` | 1855 | workbench → set slot |
| `END_TURN` | 1 | commit, only legal when the table is valid |
| `DRAW` | 1 | revert the turn, take a tile, pass |

This makes every legality check O(1) — no NP-hard partition test anywhere in the
hot loop — and it mirrors how the game is physically played. It is also
**complete**: every table CP-SAT proposes is reachable through these actions
within the per-turn budget, which the test suite checks constructively on all
three configs. `DRAW` is never masked, so the MDP cannot deadlock and no mask row
is ever all-zero.

Full rules-to-arrays contract: [`SPEC.md`](SPEC.md).

## Backends

Three implementations, written independently against `SPEC.md` rather than
against a shared abstraction, so a comparison measures implementations and not
the cost of a common layer. `rummi.env.api` reconciles them at the boundary,
so they are swappable by name.

There are **two** throughput figures below and they are not interchangeable. The
simulator one is the fair comparison *between implementations*; the env one is
what a training loop actually gets. Same units, different work.

### The simulator: transition and mask only

![Simulator throughput by backend and batch size](docs/charts/throughput.svg)

| backend | peak env-steps/s | vs NumPy | at batch |
|---|---:|---:|---:|
| NumPy (reference) | 195k | 1.0x | 16,384 |
| torch CPU | 227k | 1.2x | 16,384 |
| torch CPU + `compile` | 286k | 1.5x | 16,384 |
| torch MPS | 432k | 2.2x | 16,384 |
| torch MPS + `compile` | **809k** | **4.2x** | 16,384 |
| JAX (`lax.scan`) | 346k | 1.8x | 1,024 |

Standard config, best of three, peak over batch sizes up to 16,384, with the
action choice held to the cheapest possible so the figure measures the *simulator*
and not a policy. Numbers from `docs/data/backends.json`; regenerate with
`python -m rummi.bench.bench_backends --compile --json docs/data/backends.json`.
The sweep resets Dynamo before each compiled cell: its recompile limit is per code
object and shared across the whole sweep, and past it `compile` falls back to eager
silently — which publishes an eager number under a compiled name.

Read the shape of the curves rather than the top row:

- **NumPy saturates at 256 envs.** It reaches 157k there and is flat from a
  thousand on. The reference runs out of one core's memory bandwidth long before
  it runs out of batch, and that plateau is what everything else is measured
  against.
- **The GPU needs a batch to be worth having.** `torch-mps` is *six times slower
  than NumPy* at 64 envs, where the launch overhead per step dwarfs the step, and
  only overtakes it past a thousand.
- **Fusion multiplies the GPU, not the CPU.** `compile` is worth 1.8x on MPS at
  4,096 (720k against 399k) and 1.3x on torch CPU (312k against 233k at 16,384).
  What it removes is per-kernel overhead, and the CPU has less of it to remove.
- **Both frameworks agree on CPU** (1.5x and 1.8x). Two implementations written
  independently landing in the same band is decent evidence that the headroom over
  a vectorised NumPy reference is real rather than a tuning artefact.
- **The NumPy row is doing less work, not better work.** ASSIGN is legal only for a
  tile in hand, and a workbench holds ~6 kinds of 53, so the reference builds that
  block at the pairs held rather than over the whole grid. The pair count depends on
  the data, which is exactly what `jit` and `compile` cannot have, so the other two
  build it in full. See SPEC.md section 10.
- **JAX is fastest at small batch** — 1.4x at 64 envs, where torch is still
  paying fixed per-call costs — and flattest thereafter. `lax.scan` buys almost
  nothing over the fused version: the per-step work already dominates, so the
  Python loop was never the bottleneck.

JAX here is CPU-only; it has no production Metal backend, so it cannot be
compared against MPS on equal hardware. On a CUDA box, re-measure.

### The env: what a training loop gets

| backend | B=1024 | B=4096 | vs NumPy |
|---|---:|---:|---:|
| NumPy (reference) | 190k | 191k | 1.0x |
| torch CPU | 72k | 162k | 0.8x |
| torch CPU + `compile` | 206k | 334k | 1.8x |
| torch MPS | 146k | 300k | 1.6x |
| torch MPS + `compile` | **319k** | **694k** | **3.6x** |
| JAX CPU | 355k | 347k | 1.8x |

`RummiVectorEnv.step`, so the mask, the transition, the observation encoding *and*
next-step autoreset are all inside the figure. Every arm replays one script of
first-legal actions recorded on the reference backend, so no arm reads a mask to
choose — a read forces an async backend to finish, and the sampling is a policy's
cost anyway, not the env's. Data in
`docs/data/env_throughput.json`; regenerate with
`python -m rummi.bench.bench_env --compile --json docs/data/env_throughput.json`.

`+compile` is a backend name, not a mode: `RummiVectorEnv(backend="torch-mps+compile")`
puts the mask, the observation and the transition through `torch.compile`. Action
validation moves out of the compiled step and runs beside it, because branching on
a device boolean splits the graph in two — the same reason JAX validates outside
its jitted step.

`FixedOpponentEnv` runs on every backend too, converting the observation, the mask
and the seat vectors for its NumPy opponents — but which one is fastest there is
not this table: the hand-off happens once per opponent action, so at B=1024
against `greedy`, `jax` is 1.6x NumPy while `torch-mps+compile` is 0.71x. See
`--backend` in `tools/train_ppo.py`.

**Why the env numbers are lower than the simulator's**, since the tables invite
the comparison: the env encodes an observation every step, keeps next-step
autoreset, and reads a handful of `(N,)` telemetry vectors back to the host — and
it is measured at 4,096 rather than at each backend's best batch. Compiled MPS
puts a number on all of that: **694k against the simulator's 753k at the same
batch**, so everything the env adds over the simulator costs about 8%.

Conformance is not assumed. Every backend replays recorded trajectories and must
reproduce 42 state digests per config, and masks and rewards are compared against
the reference step for step.

## Layout

```
rummi/
  rules/        backend-free: the rules as data (config, tile encoding, actions)
  env/
    numpy/      the reference implementation: set kernel, masks, engine, dealing
    torch/      independent torch implementation
    jax/        independent JAX implementation
    api.py      uniform adapter, so backends are swappable by name
    vector_env.py     Gymnasium VectorEnv
    observation.py    the observation an agent is allowed to see
  agents/       the Agent protocol and the bundled opponents
  evaluate/     scoring an agent against those opponents
  solver/       brute-force oracles, candidate sets, CP-SAT, plan translator
  render/       shared board layout, terminal view, pygame window, play UI, record/replay
  bench/        throughput benchmarks and the invariant fuzzer
```

## What is verified

- **659 tests.** The set-validity kernel is checked *exhaustively* against a
  brute-force oracle on the reduced configs, including the closed-form
  "could I add this tile?" predicate.
- **10.5M fuzz steps, zero invariant violations.** Tile conservation, mask
  soundness, exact turn reversion, and termination, on every step.
- **Random play cannot test this environment.** In 10M steps it assembled a legal
  opening meld four times. Any test suite here needs greedy or better to reach
  melding, winning, or `END_TURN` at all — which is why the fuzzer takes a
  `--policy` flag.

```bash
pytest                                                  # everything
python -m rummi.bench.fuzz --policy greedy --games 500  # invariant fuzzing
python -m rummi.bench.tournament                        # baselines head to head
```

## Watching games

Two live views over one shared view model — a pygame window and an in-place
terminal view that redraws only the lines that changed.

```bash
python -m rummi.render.record --play --render-mode ansi --fps 6
python -m rummi.render.record --play --render-mode human --policy optimal
python -m rummi.render.record --play --out game.jsonl && \
python -m rummi.render.record --replay game.jsonl --pause
```

Recordings store the config, the seed and the action sequence — nothing else. The
simulator holds no RNG in its step function, so replaying those actions
reconstructs every intermediate state exactly.

By default the live views draw once per **committed turn**, since a turn is the
meaningful unit of play and mid-turn frames are mostly one tile moving. Pass
`--render-on step` to see every micro-action instead — the workbench filling, the
table temporarily in pieces — which is what you want when reasoning about the
action space.

To record with Gymnasium's own tooling instead, `render_mode="rgb_array"` returns
a tuple of frames per the vector convention, so `RecordVideo` works directly:

```python
from gymnasium.wrappers.vector import RecordVideo
from rummi.env.vector_env import RummiVectorEnv

env = RecordVideo(                                  # writes MP4; needs moviepy
    RummiVectorEnv(num_envs=1, cfg=STANDARD, render_mode="rgb_array"),
    video_folder="videos", name_prefix="rummi",
)
```

Use `num_envs=1`: Gymnasium tiles one frame per sub-env, and only
`render_env_index` is ever drawn — rendering all N games to show N copies of the
same thing would cost N times as much.

The animations at the top of this page are generated, not pasted, and
regenerating them is byte-identical:

```bash
python tools/render_docs.py --format gif --out docs/render     # both animations
python tools/render_docs.py --format png --out still.png       # one frame, side by side
```

## Playing a hand yourself

```bash
python -m rummi.render.play --opponent optimal
```

Drag a tile off your rack onto the table, or off one set onto another to
rearrange. A press picks the tile up and a release puts it down, so clicking
works too: click to take, click again where it should go. `UNDO` walks back a
mis-drag, `END TURN` commits.

The window says things rather than printing them: a pill per seat showing whose
turn it is and how many tiles are left, a bar for how close your opening meld is,
a score under each set, a red ring on any set that is not legal yet, and a blue
one on every set the tile in your hand could go to.

The reason this is worth more than the fun of it: **the same `action_mask` an
agent consumes drives the interface.** It decides which tiles can be lifted and
which sets light up as you drag, so an illegal move is not rejected — it is
inexpressible, and a test fires random clicks across a whole game asserting that
no gesture can produce an action the mask forbids. Every drag emits the same
micro-actions an agent has to emit, which makes the window a way to read the
action space rather than a separate way of playing.

Undo is worth a note too. The MDP has no "unplace" — a tile leaves the workbench
by being assigned or by `DRAW` abandoning the turn — so rather than add an action
to fix a human's mis-click, `UNDO` rewinds to the turn's opening state and replays
it one action short. The engine every agent sees is untouched.
