# Learned agents: what was tried, and what killed it

Every attempt at a learned agent strong enough to join the ladder, with the reason
each one stopped. The scores are the least useful column. **The issue is the point** —
most of these failed for a reason that took a day to find and five minutes to
describe, and rediscovering one costs a day again.

Two kinds of row, and the distinction matters:

- **Deterministic agents** are captured, so their numbers are reproducible:

  ```bash
  python tools/capture_experiments.py --suite standard-greedy --games 60
  ```

  writes `docs/data/experiments.json`. `capture_agents.py` asserts `name in REGISTRY`
  and rightly refuses these, hence the sibling.

- **Training runs** cannot be captured. Each is **one seed of one recipe**, and
  reproducing one is a training job. Treat every number below as n=1.

Unless stated, all figures are the `standard-greedy` suite: `standard` config, two
players, opponent `greedy`, seat-rotated, n=120. `greedy`'s 50.0% / +0.00 self-match
is exact by rotation, not measured, which is what makes it a usable reference.

## The ladder, and where the experiments sit in it

| agent | score | win | note |
|---|---|---|---|
| `random` | -442.32 | 0.0% | Never assembles a legal meld. |
| best learned, primitive actions | -230 | 0.0% | A day of PPO, cloning, DAgger, two architectures. |
| `macro/first_legal` | +21.87 | 74.2% | Plays the lowest-indexed legal set. |
| `greedy` | +0.00 | 50.0% | Reference. Self-match is exact. |
| **`macro/by_value`** | **+24.28** | **80.8%** | **Best agent produced. Hand-written, not learned.** |
| `hybrid/macro_first` | +24.28 | 80.8% | Identical to the above by construction; a test asserts it. |
| `rearrange` | +31.92 | 85.0% | Steals exactly one tile. |
| `optimal` | +44.57 | 100% | CP-SAT. Never stalemates. |

`macro/by_value` scores **+25.23 at n=240**, which is the figure to quote against the
published ladder.

## Capability, not policy

`by_value` as each capability entered the macro action space. No learning anywhere in
this table.

| macro space contains | score | win |
|---|---|---|
| new sets only | -141.5 | 0.8% |
| + lay-offs onto existing sets | -65.4 | 11.7% |
| + joker substitution | -18.4 | 34.2% |
| + steal one tile off the table | **+24.3** | **80.8%** |

Four capabilities were worth ~255 points. Two trained networks were worth ~8. The
diagnostic that found the largest: **84.2% of `greedy`'s `ASSIGN`s land on an
already-occupied slot**, so a space that could only create new sets was missing the
commonest move in the game, and no policy inside it could have recovered that.

## Training attempts

| attempt | score | win | the issue |
|---|---|---|---|
| PPO from scratch, primitive actions | -441 | 0.0% | Never melds, so no reward ever distinguishes one action from another. Converges on passing. |
| clone `greedy`, then unanchored PPO | -442 | 0.0% | `--kl-coef` defaults to **0.0**; PPO returns the clone to the random floor. |
| clone `greedy`, anchored PPO | -232 | 0.8% | Works, and stops there: 74% agreement on the teacher's states, **5% on its own**. |
| factored softmax | -230 | 0.0% | `end_turn` unmoved (0.53% vs 0.49%). Buys melding (+8pp); does not buy turn completion. |
| DAgger on `greedy` | -442 | 0.0% | Agreement 3.2% → 59.5% **while the score stayed at the floor**. 84% of labels on the student's own states are `DRAW`. |
| bilinear head, primitive actions | -442.3 | 0.0% | Exactly `random`. Entropy still 2.6–3.2 after 1.2M steps: never found a gradient. |
| two-action delegate (`DRAW` or a planner's turn) | mirror | 50.0% | Converged to always-play at every inner strength. No cross-turn strategy from board state alone. |
| macro space, flat head | +19.93 | 70.0% | Each of 730 macros is an independent output row; nothing learned about one set transfers. |
| macro space, pointer head | +23.91 | 76.7% | Best head. Still below `by_value`. |
| macro space, pointer, 300 updates | **+26.72** | 80.8% | **Beat `by_value` once. Did not replicate — see below.** |
| macro space, same recipe, 713 layout | +22.48 | 75.8% | The replication attempt. Below the heuristic. |
| macro space, cloned from `by_value` | +22.59 | 80.0% | Cloning **hurts** here: it arrives confident, so RL settles near the teacher. |
| macro space, `--epochs 4` | +7.53 | 50.8% | Reusing a noisy bootstrapped advantage amplifies its error. |
| macro space, one step per minibatch | -393.95 | 0.0% | Four noisy small-batch steps are far worse than one averaged step. |
| hybrid space, any recipe | ~-1.0 term. | 0.0% | Collapses to stalling. A macro is on offer in **4.9%** of decisions. |

**The one apparent win did not replicate.** +26.72 was one seed on the 730-action
layout; the same recipe on the current 713-action layout scored +22.48, below
`by_value`. A learned policy matching the heuristic is established; beating it is
not. Note also that a layout change re-baselines everything — scores are not
comparable across it.

## Settings measured as harmful

| setting | effect | why |
|---|---|---|
| `--entropy-coef 0.05` | -11.07 | Exploration wants to go **down** in the macro space. The 0.01 default is inherited from the primitive recipe and is too high; 0.003 was best. |
| `--epochs > 1` | +7.53 | See above. Defaults to 1. |
| `--clone` (macro space) | +22.59 | Helps on the primitive space, hurts here. |
| `--kl-coef` (macro space) | -0.4 | Mandatory on the primitive space, a constraint here: it pinned entropy to 0.001. |

**The recipes do not transfer between action spaces.** Cloning is mandatory in one
and harmful in the other; the same inversion holds for the anchor, for exploration
and for batch reuse. Re-derive rather than inherit.

## Why `optimal` is out of reach in the macro space

At the states where `macro/by_value` draws because it cannot play:

- `optimal` could still play in **47.7%** of them,
- **100%** of those plays dissolve at least one set,
- **70.5%** dissolve more than one.

Choosing a multi-set repartition *is* the NP-hard partition problem, so no fixed
template list expresses it. That is why `optimal` needs CP-SAT and why `rearrange`
stops at one tile. The ceiling here is `rearrange`'s +31.9; +44.6 needs a solver in
the loop.

## The hybrid space does not work yet

`rummi/agents/hybrid.py` offers the 2400 primitives alongside the macros so that any
legal turn is expressible. It fails, and for a reason in its own design: macros
require a clean workbench, because a macro's expansion balances the board against
tiles played from the rack and `to_actions.plan` refuses a plan with tiles already
held. Measured under an untrained policy:

| | hybrid | macro space |
|---|---|---|
| workbench dirty (tiles held mid-turn) | 94.5% | — |
| `END_TURN` legal | 0.5% | ~9% |
| any macro on offer | **4.9%** | 100% |

So the escape hatch is shut exactly when it is needed, and training collapses to
stalling. A bigger batch, `--rack-shaping` and `--micro-step-cost` were each tried;
none can make an illegal action attractive.

**The fix:** let a macro consume the held tiles — judge `playable` against
`rack + workbench` with the workbench required to be used, and teach
`to_actions.plan` to account for tiles already held rather than only for what leaves
the rack.

## Scope: one suite proves one thing

`macro/by_value` across the whole frozen protocol, once each:

| suite | seats | opponent | win | even | score |
|---|---|---|---|---|---|
| `tiny` | 2 | `greedy` | **43.3%** | 50.0% | -1.12 |
| `standard-greedy` | 2 | `greedy` | 80.8% | 50.0% | +24.28 |
| `standard-optimal` | 2 | `optimal` | 1.2% | 50.0% | -35.83 |
| `standard-3p` | 3 | `greedy` | 58.3% | 33.3% | +37.37 |
| `standard-4p` | 4 | `greedy` | 37.5% | 25.0% | +26.13 |

It generalises across seat counts, is crushed by `optimal`, and is **worse than
`greedy` on `tiny`** — where the table holds 4 slots against 35, and both
slot-consuming macros gate on a free slot while `greedy`'s `ASSIGN` never needs a
spare. That last row is un-diagnosed, and it is why a claim from one suite should not
be stated as a claim about the agent.

## Method notes that earned their place

- **A metric that is not the objective will mislead you.** On-policy agreement went
  3.2% → 59.5% while the score sat at the random floor, because agreeing to `DRAW` is
  cheap. Agreement, teacher-state or on-policy, must be read against score.
- **Two measurements of one quantity disagreeing is what catches bugs.** `END_TURN`
  chosen 0 times in 6,286 decisions against 731 `END_TURN` actions emitted is what
  exposed the macro expansion committing the turn — a bug that had already produced a
  plausible wrong conclusion and a test asserting an artifact of itself.
- **Reachability before scoring.** A pointer head is worth +4 in a space where the
  reward is reachable and exactly nothing in one where it is not.
