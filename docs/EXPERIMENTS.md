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
| `greedy` | +0.00 | 50.0% | Reference. Self-match is exact. |
| `macro/first_legal` | +26.79 | 80.0% | Plays the lowest-indexed legal macro. |
| **`macro/by_value`** | **+29.60** | **85.4%** | **Best agent produced. Hand-written, not learned.** |
| `hybrid/macro_first` | +29.60 | 85.4% | Identical to the above by construction; a test asserts it. |
| `rearrange` | +31.92 | 85.0% | Steals exactly one tile. |
| `optimal` | +44.57 | 100% | CP-SAT. Never stalemates. |

The experimental rows are the n=240 capture in `docs/data/experiments.json`, taken
after the joker capability below; `by_value` now matches `rearrange`'s win rate and
sits 2.3 points under its score.

## Capability, not policy

`by_value` as each capability entered the macro action space. No learning anywhere in
this table.

| macro space contains | score | win |
|---|---|---|
| new sets only | -141.5 | 0.8% |
| + lay-offs onto existing sets | -65.4 | 11.7% |
| + joker substitution | -18.4 | 34.2% |
| + steal one tile off the table | +24.3 | 80.8% |
| + lay off onto joker sets, and the rack's joker | **+29.6** | **85.4%** |

Five capabilities were worth ~260 points. Two trained networks were worth ~8. The
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
| macro space, same recipe, 3 seeds on 713 | +22.48 / +25.06 / +24.12 | 74–79% | Every seed below `by_value`'s +25.23. |
| + jokers (`45287a7`), same recipe, 1 seed | +17.38 | 62.9% | The capability that lifted the heuristic +4.4 cost the fixed-budget learner ~8: `EXTEND` is legal far more often, every decision is wider, and 300 updates were tuned on the narrower space. Below `first_legal`. |
| + jokers, pre-joker weights **zero-shot**, seeds 0/1 | **+32.99** / +22.11 | **89.2%** / 74.2% | Transfer is a seed coin-toss decided in states neither net ever saw: s0 ranks the new joker appends above `DRAW` (takes 41% of them), s1 below (draws past 36%). **s0 zero-shot sits above `rearrange`** — the strongest agent produced, and the two nets were one point apart on the space they were trained on. |
| + jokers, s0 warm-started, 300 more updates vs `greedy` | +20.17 | 68.3% | Finetuning destroyed the transfer bonus: 13 points below the same weights untouched. Unanchored RL walks off the lucky extrapolation ridge, and the sampled-policy terminal reward (~0.0) never shows it. |
| macro space, cloned from `by_value` | +22.59 | 80.0% | Cloning **hurts** here: it arrives confident, so RL settles near the teacher. |
| macro space, `--epochs 4` | +7.53 | 50.8% | Reusing a noisy bootstrapped advantage amplifies its error. |
| macro space, one step per minibatch | -393.95 | 0.0% | Four noisy small-batch steps are far worse than one averaged step. |
| hybrid space, any recipe | ~-1.0 term. | 0.0% | Collapses to stalling. A macro is on offer in **4.9%** of decisions. |
| self-play A/B control: `--opponent greedy`, 3 seeds | -19.59 / +9.82 / **+30.51** | 33.5-84.6% | A **50-point spread** on one recipe, where the `3 seeds on 713` row above spread 2.6. One seed of three beat `by_value`. |
| **scored at update 40**, 5 seeds | **+30.84 / +29.20 / +29.76 / +25.25 / +27.04** | 76.5-85.6% | Mean **+28.42** against `by_value`'s +28.01. The same recipe scored at 300 averages +10. **The bar is reached in 40 updates and then training destroys it.** |
| self-play A/B: `--opponent greedy,self`, 3 seeds | -6.66 / **-72.24** / -17.71 | 9.8-42.5% | Worse on two seeds of three -- and the control's own spread is larger than the gap between the arms, so the sign is not evidence. **Inconclusive.** |

**The one apparent win did not replicate, now measured across seeds.** +26.72 was
one seed on the 730-action layout. On the current 713 layout the same recipe scored
**+22.48, +25.06 and +24.12** (mean +23.9), every seed below `by_value`'s +25.23 and
none above 78.8% win against its 80.8%. From scratch, the policy reliably lands in
the heuristic's neighbourhood; beating it did not survive replication, and the run
was stopped after the third seed because no remaining seed could flip that verdict.
So the capability-over-policy ratio holds all the way up: **the next win is a
capability** — the joker gap measured below, or the hybrid fix — not a policy
tweak. Note also that a layout change re-baselines everything — scores are not
comparable across it. The checkpoints behind these rows load with
`tools/eval_macro.py`; `by_value` reproducing its published +25.23 / 80.8% inside
every training eval is the check that the harness did not move under them.

**Self-play is unresolved, and 300 updates is why.** Six runs, three seeds per arm,
matched on everything but `--opponent`, scored on `standard-greedy` at n=480 with
`by_value` measured in the same run at **+28.01 / 82.1%** -- the same agent as the
published +29.60 / 85.4% (n=240), on twice the deals. The A/B answers nothing,
because the curves say neither arm was still training by the time it was scored:

```bash
python tools/plot_league.py runs/*.json --out league.png
```

Every run rises to a terminal reward of ~+0.05 against `greedy` by **update 60-80
and then degrades** -- five of six end between -0.02 and -0.12, and the only seed
that stayed flat is the only one that beat `by_value`. Entropy collapses 1.2 -> 0.35
by update 100 in both arms. **A 300-update score is the far side of the peak**,
which is a caveat on every 300-update row above this one as much as on these two:
they record where a run ended up, not what the recipe reaches. Checkpoint and score
every ~20 updates before reading any of them as a ceiling.

The league panel is its own result. Promotions cluster in updates ~25-75 and every
later one is refused, because `--snapshot-gate` will not promote a learner that has
stopped improving against its pool. That is the gate working, and it means the self
arm spent three quarters of training against a frozen early snapshot rather than
against itself -- so this A/B is not yet a fair test of self-play either.

One methodological check worth recording, since the whole table rests on it:
`eval_macro.py` argmaxes, and sampling the same checkpoints instead scores them
**worse** (-19.59 -> -32.24, +30.51 -> +16.62). The spread between seeds is real
policy quality, not an artifact of taking the mode.

**The recipe peaks at update 40 and the endpoint is a lottery ticket.** Five seeds
checkpointed every 20 updates and scored at each (`--checkpoint-every`, plots from
the CSV). Every seed reaches `by_value`'s level by **update 40**; three of five then
collapse -- one to **-105** -- and partially recover, while two sit flat at ~+28 for
the remaining 260 updates. A 300-update score records where in that collapse the run
happened to stop.

At u40, chosen once for all five seeds and re-scored at n=480: **+28.42 mean against
`by_value`'s +28.01**, three seeds above it. Quote that number and not the per-seed
maximum (+30.02): a max over fifteen noisy n=240 evals is selection on noise.

**This does not overturn "no learned policy beats `by_value`" -- it changes what is
wrong.** +28.42 against +28.01 is a tie, and three of five above is inside the
noise. But the rows above say the policy *lands below* the heuristic and stops
there, and that is not what happens: it reaches the heuristic in **40 updates**, at
one seventh the compute, and then training takes it apart. Those are different
failures with different fixes, and every 300-update row conflates them. The
instability is also not inevitable -- two seeds never fall -- so it is trainable,
and the u40-u100 window is where to look.

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

## The hybrid space: the consumption fix is built, and it is not enough

`rummi/agents/hybrid.py` offers the 2400 primitives alongside the macros so that any
legal turn is expressible. As first built it failed by design: macros required a
clean workbench, because `to_actions.plan` balanced the board against tiles played
from the rack alone. The specified fix — **a macro consumes the held tiles** — is
now implemented: feasibility is judged against `rack + workbench`, every held tile
must be laid by the macro itself, and `plan(held=...)` emits no `PLACE` for a tile
already in hand (the parameter defaults off, so `optimal` and the pure macro space
are untouched). Every macro offered on a dirty workbench provably clears it; a test
replays hundreds of such expansions against the env mask.

Measured under a uniform-random policy, before and after:

| | clean-workbench rule | consumes held tiles |
|---|---|---|
| workbench dirty | 94.5% | 94.1% |
| `END_TURN` legal | 0.5% | 0.6% |
| any macro on offer | **4.9%** | **14.7%** |

**And training still collapses** — `end` stays at 0.0% over a training smoke,
indistinguishable from the old gate. The mechanism is structural, not a bug in the
rule: **one macro plays one set, so it can only absorb a workbench that fits inside
one** — a macro is on offer at 44.5% of one-tile workbenches, 7.7% at two, ~0% past
four, and a uniform policy lifts a tile per step straight out of range. The blocker
has moved from expressiveness to exploration: the hatch is open wherever a set fits
around the held tiles, but a near-uniform policy over 2400 primitives never finds
it. Consistent with that, `--micro-step-cost 0.01` — useless when ending was
*illegal* — is the first arm where the finish counter ever moved (6–7 finished
episodes per update against ~1). If this space is picked up again, that is the
thread to pull, not the consumption rule.

## Scope: one suite proves one thing

`macro/by_value` across the whole frozen protocol, once each:

| suite | seats | opponent | before jokers | after jokers |
|---|---|---|---|---|
| `tiny` | 2 | `greedy` | **43.3%** / -1.12 | 51.5% / +0.14 |
| `standard-greedy` | 2 | `greedy` | 80.8% / +24.28 | 85.4% / +29.60 |
| `standard-optimal` | 2 | `optimal` | 1.2% / -35.83 | 4.0% / -30.29 |
| `standard-3p` | 3 | `greedy` | 58.3% / +37.37 | 73.8% / +61.00 |
| `standard-4p` | 4 | `greedy` | 37.5% / +26.13 | 52.3% / +53.29 |

Before the joker capability it was **worse than `greedy` on `tiny`** — which is why
a claim from one suite should not be stated as a claim about the agent. After it,
every suite moved: the `tiny` deficit closed to level, the multi-seat scores jumped
by 24-27 points, and only `optimal` still wins — the repartition ceiling above.

A learned net reaches only two of these five suites: the 3p/4p observations are
wider than `standard`'s (572/574 features against 570) and `tiny`'s macro table is
its own (41 actions), so a `standard`-trained checkpoint cannot be scored on the
rest — `tools/eval_macro.py` checks and skips with the reason printed. Where the
best replication seed can be scored, it has the heuristic's shape:

| suite (pre-joker space, both columns) | learned, seed 0 | `by_value` |
|---|---|---|
| `standard-greedy` | +25.06 / 78.8% (n=240) | +25.23 / 80.8% (n=240) |
| `standard-optimal` | -32.91 / 3.0% (n=200) | -35.83 / 1.2% (n=200) |

## The `tiny` regression, diagnosed — and the joker gap it exposed

The first hypothesis for that row — slot scarcity, 4 table slots against 35 with
both slot-consuming macros gating on a free one — is **refuted, measured**: over
48,224 decisions of `macro/by_value` on `tiny_groups` and 30,506 on `standard`
(`python tools/diagnose_stuck.py`), the free-slot gate never fired once. No
tile-feasible macro was ever blocked only by slots, on either config.

What the same sweep found instead, at the decisions where only `DRAW` was legal:

| | `tiny_groups` | `standard` |
|---|---|---|
| stuck (only `DRAW` available) | 75.0% of decisions | 67.2% |
| stuck, but `greedy` would act | **7.4%** of stuck states | **28.5%** |
| — append onto a joker-holding set | 6.9% | 27.7% |
| — lay off the rack's own joker | 0.5% | 0.9% |
| — plain append / new set (consistency check) | 0.0% | 0.0% |
| `EXTEND` + `STEAL` share of choices | 4.1% | 23.5% |

Two findings, one per config:

- **The macro space cannot touch a joker once it lands.** `extensions` and
  `removals` refuse joker-holding sets because the joker's role in them is
  ambiguous, and a rack joker can be spent only inside a new template. On
  `standard` that accounts for **every** stuck state where `greedy` still plays —
  28.5% of them — making it the largest unexpressed capability since `STEAL`.
  `greedy`'s own `_appendable` shows the shape of a fix: it validates a grown set
  through `evaluate_slots` instead of by arithmetic, which resolves the ambiguity
  the same way the env itself does.
- **`tiny`'s regression is geometry, not slots.** With 3 colors a group is complete
  at birth — nothing to extend, nothing to steal — and the 13-tile single-copy deck
  makes runs rare, so `EXTEND`+`STEAL` fall from 23.5% of choices to 4.1%. The two
  capabilities the arc above credits with ~119 points are simply absent from the
  config, and what remains is close to the "new sets + jokers" rung, which that
  same arc measured at **-18 against `greedy`**. The residual deficit candidate is
  the joker gap above (7.4% of stuck states), not slots.

**The gap is closed.** `EXTEND` now shares `greedy`'s own feasibility test —
`greedy_agent.appendable`, which grows the slot row and validates it through the
env's `evaluate_slots`, the only way a joker's role is answerable — so a set holding
a joker takes tiles and the rack's joker lays off wherever there is room. Re-run at
the same size (46,640 / 29,283 decisions), **"stuck but greedy would act" reads
0.0% on both configs**: the macro space expresses every single-tile move greedy can
make, and the diagnostic's columns stay in place as the drift alarm between the two.
`STEAL` still refuses joker-holding sets — what such a set can spare is undetermined
by arithmetic (`(R1,R2,R3,*)` can spare `R3`, a "middle" tile, because the joker
slides down) — and that residue is part of what `optimal` still wins with.

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
