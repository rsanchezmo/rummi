# What was tried, and what killed it

Every attempt at an agent strong enough to join the ladder, learned or hand-written,
with the reason each one stopped. The scores are the least useful column. **The issue is the point** —
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

**How to read this file.** It is a lab notebook, in the order things happened:
training runs first, then hand-written arms, then the hindsight measurements that
bound what was left. The index below is the same content in
the order of the *argument*, which is how [the README](../README.md#what-the-benchmark-found)
tells it.

## Questions this file answers

| question | answer | headline |
|---|---|---|
| [What is a game of standard Rummikub made of?](../README.md#the-game-in-numbers) | Mostly forced draws, decided by two tiles, and the seat matters more than the agent. | the seat decides **68%** of deals |
| [Is playing the best available turn the ceiling?](#rack-potential-the-asymmetric-half-and-why-a-recurring-decision-closes-it) | Yes, three ways: CP-SAT, its macro-space rendering, and a 233k clone of that rendering are tied -- the ceiling paragraph inside that section. | **48.7%** head to head, n=600 |
| [How much is left above it for any strategy to find?](#oracle-one-step-regret-the-bound-the-nulls-were-missing) | Nothing. 39,960 alternative whole turns rolled out to the end against the real deck; every deviation type is at or below zero. | same tiles, other table: **-0.2% +-0.5%** |
| [Can you shut an opponent out of going out?](#endgame-denial-the-cell-that-survived-the-oracle-and-the-reason-it-is-empty) | The sign is there and the cell is empty: a solved reply reads the table's tile multiset, not its arrangement, so 91.5% of equal-tile alternatives provably cannot move the door. | +11 / +20 / +18pp, fired 0.03-0.11 times per seat-game -- worth **0.2-0.5pp** |
| [Does the table you leave behind pay?](#board-shaping-the-axis-the-oracle-never-bounded-and-why-it-has-no-sign) | Null by mechanism: the table is common property, so `left - met` is 0.00 +-0.00 and any restriction pays its own cost. | **+1.06pp +-1.57** against a coin flip over the same ties |
| [Is a rack worth more than its size?](#rack-potential-the-asymmetric-half-and-why-a-recurring-decision-closes-it) | Null by mechanism: a decision recurs after every set, so what a turn "keeps" it plays before ending. | the axis bounded at **~1.8pp** |
| [Is the opening different, since it does not recur?](#the-opening-was-the-one-cell-that-moved-and-it-does-not-survive-being-played) | The oracle's one positive cell out of fourteen, tested as nine pre-registered policies. Null; the mechanism is present at every step and gone within two turns. | best arm **+0.91% +-1.04**, control **-0.45% +-0.50** pooled |
| [Does any of it change at three or four seats?](#three-seats-the-argument-that-killed-it-at-two-does-not-apply-and-the-answer-is-the-same) | No, and 3p breaks the symmetry argument that closed 2p, so it is an independent test rather than a repeat. | 3p **-1.2% +-1.1%**, [4p](#four-seats-and-the-number-the-three-runs-agree-on) **+0.4% +-1.3%** |
| [Is there information about the opponent worth having?](#training-attempts) | Null twice by mechanism (belief features, LSTM) and once by bound: handed the opponent's true rack, a policy learns to discard it. | zeroing the block flips **0.0%** of ~9,500 argmax decisions |
| [Can a network beat the heuristics?](#training-attempts) | It clones one exactly and never exceeds it. The strongest learned agent in the repo is a DAgger clone, and it is the ladder's `learned` rung. | **+47.32 / 99.6%**, even with `optimal` |
| [What moved the score, then?](#capability-not-policy) | Capability, not policy -- what the action space can express, not what the policy chooses inside it. | five capabilities **~260 points**, two trained nets **~8** |
| [Why did no training run resolve any of this?](#settings-measured-as-harmful) | The signal sits an order of magnitude under one update's own noise, and argmax scoring confounds policy quality with policy concentration. | [**0.012** normalised units of signal against **0.073** of per-episode noise](#training-attempts) |
| [What is still open?](#what-this-bounds) | Multi-turn plans, adaptation against a population, configs that force a residue, and league margin. Everything one turn wide is closed. | see also [what this leaves](#what-this-leaves) |

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
| hybrid + held-tile consumption, `--micro-step-cost 0.01` | -441.94 | 0.0% | The finish counter was a false positive: the cost is charged per micro-action, never on committing, so the optimum is "never touch a tile" -- `draw` 4% -> 100% by u40, entropy 0.000, 354 instant passes per update at terminal ~-1.05. |
| hybrid + held-tile consumption, unshaped control | -441.94 | 0.0% | Fails the opposite way: 98.9% of decisions in the primitive block, turns held open, `END_TURN` chosen ~109 times in 2.4M decisions and 0.000% from u45. |
| hybrid, warm-started from macro u40 (+30.84) | -442.32 | 0.0% | Transfer exact and inert. Arrives putting 55.9% of its mass on the macro block where one is on offer -- but one is on offer in **1.6%** of decisions, because a policy that draws 0.38% runs every turn to the micro cap. Macro use peaks 27.6% at u20, 0.0% from u70. |
| self-play A/B control: `--opponent greedy`, 3 seeds | -19.59 / +9.82 / **+30.51** | 33.5-84.6% | A **50-point spread** on one recipe, where the `3 seeds on 713` row above spread 2.6. One seed of three beat `by_value`. |
| **scored at update 40**, 5 seeds | **+30.84 / +29.20 / +29.76 / +25.25 / +27.04** | 76.5-85.6% | Mean **+28.42** against `by_value`'s +28.01. The same recipe scored at 300 averages +10. **The bar is reached in 40 updates and then training destroys it.** |
| self-play A/B: `--opponent greedy,self`, 3 seeds | -6.66 / **-72.24** / -17.71 | 9.8-42.5% | Worse on two seeds of three -- and the control's own spread is larger than the gap between the arms, so the sign is not evidence. **Inconclusive.** |
| opponent-history belief features (`--history`), 3 seeds | +27.67 / +24.75 / n.c. | ≤84% | **Null.** The inference is sharp -- a declined lay-off proves the opponent holds zero of that kind, 2,469/2,469 checks -- but entropy-matched at u60 the converged seeds land inside the control's argmax range, and one seed never converged. The 111 extra inputs delay the entropy collapse ~20 updates, so arms of different width must be matched on entropy, never on update count. |
| LSTM memory (`--memory lstm`, full BPTT), 3 seeds | +29.86 / +30.28 / -61.92 | 20-85% | **Null the same way.** Two seeds converge at the top of the control's same-day range (+22.92 / +29.00 / +27.70, n=240 each); the third hits the known late collapse. Sampled ties too (+11.36/+15.32 vs +9.89-+15.25). Zeroing the trained cell state flips **0.0% of 9,625 argmax decisions**: the policy converges memoryless. |
| oracle: opponent's TRUE rack as input (`--oracle-rack`), 3 seeds | 82.2% / 82.2% / n.c. win, in-env | -- | **The upper bound on the whole channel, and it is ~zero.** Contract-breaking by design, so it is scored arm-vs-arm through `--diagnostic-games` on identical deals: converged oracle seeds 82.2%/82.2% against the control's 75.0%/81.2%/80.3% (n=400 each). Zeroing the block flips **0.0% of ~9,500 argmax decisions** -- handed the answer outright, the policy learns to discard it. |
| clone `by_value+repartition` (`--repartition --clone`, DAgger 300k) | **+47.32** | **99.6%** | **The cloning ceiling, and the strongest learned agent produced.** Agreement 68.3% cold -> 99.4% on the student's own states; the clone scores its optimal-tier teacher's +48.01 to within noise, protocol-legal and observation-only. The residual 0.6% disagreement costs ~0.7 points -- no information story, exactly as the sufficiency claim predicts -- and it has a shape: over 15k teacher-labelled decisions on the clone's own states, `repartition`, `end` and `draw` disagree **0.00%**, and 74 of 75 misses are within-block substitutions (a different new-set/steal/lay-off of the same kind) -- exact value-ranking arithmetic approximated fuzzily, the one thing a bigger net would buy, worth a fraction of a point. Head-to-head with `optimal` itself: **48.7% at n=600** -- statistically even. Nothing about `optimal` resists a network except the NP-hard *construction*, which stays in the solver where it belongs: the net decides *when* to repartition, CP-SAT decides *how*. |

**Promoted.** These weights are the ladder's `learned` rung
(`rummi/agents/learned/clone.py`) -- the first agent with weights in it to clear the
bar. The recipe transfers to the multiplayer ladders unchanged and lands on its
teacher at both: **+92.34 / 99.0%** on standard-3p against the teacher's +91.67 /
99.0%, and **+140.25 / 100.0%** on standard-4p against +140.78 / 100.0%, each
reaching 99.4-99.6% agreement on its own states. So the cloning ceiling is a
property of the recipe, not of two seats, and one file of weights ships per preset
because the observation widens with the table (570 / 572 / 574).

| self-play from the clone (`--repartition --init-from`, greedy + 4-snapshot pool), 2 seeds | +47.51 / +47.75 | **100.0%** | **Stable, not improved.** 100 updates of PPO from the +47.32 clone with a 10-update critic warmup: entropy never leaves 0.13-0.14, terminal-vs-greedy never erodes, the gate promotes 2-3 snapshots and holds 1-2, and the policy ends 2.6% of decisions away from the clone -- sideways, since head-to-head with `optimal` reads 48.0%/45.5% against the clone's 46.5% (n=200 each). With the delegate row above, two very different recipes now find no strategy beyond per-turn maximization at optimal tier. Untested: exploration forced open from the strong start (entropy up, KL-anchored to the clone). |

| exploration from the clone, entropy coef 0.01 / 0.03 / 0.1, KL-anchored | -- | -- | **The entropy bonus cannot reopen a converged policy.** Its gradient on a suppressed action scales with that action's probability, so it vanishes at the one-hot corner it exists to escape: 33x the recipe's coefficient moved H 0.134 -> 0.148 in forty live updates. All arms killed at the flat-by-u40 guardrail. The working lever is behavioural -- `--explore-eps` mixes uniform-over-legal into the rollout sampler and stores the mixture's log-prob for the ratio to importance-correct. |
| best-response vs `optimal` from the clone, +-memory (120 u, `--explore-eps 0.05`) | +48.26 / +48.58 | **100.0%** | **No exploit of `optimal` that this finds.** The first training runs ever pointed at `optimal`: terminal never leaves +-0.013 in either arm despite forced exploration; vs `optimal` 46.0% / 46.5% (n=200) against the clone's own 48.7% (n=600) -- all statistically even, and the greedy suite shows zero erosion. The memory arm's cell, grown from exactly zero influence by the `--init-memory-from` transfer, ends at **0.3% argmax flips**: handed a fixed deterministic opponent, real exploration and 120 updates, memory still buys nothing. |

| `by_value+repartition` on the multiplayer ladders (no training) | +91.67 / +140.78 | 99.0% / 100.0% | **The per-turn ceiling holds in multiplayer.** On standard-3p and standard-4p the heuristic lands at `optimal`'s published level (+93.97 / +139.43) and erases the stalemate mass (90-98% -> 0.0%), exactly as in 2p. Head-to-head against two `optimal` seats at 3p (ad-hoc suite, seed_base 90k, n=180): **33.9% where even is 33.3%**. So multiplayer does not, by itself, open a gap that per-turn play misses -- whatever room 3p/4p hold lives in information or in play the heuristic family cannot express. |

| oracle at 3p: BOTH opponents' racks as input, 2 seeds + 2 control | 63.5% / 65.8% vs ctrl 55.8-69.8%, in-env | -- | **The information bound is null where the hidden mass is largest.** At 3p, `unseen` hides roughly twice the tiles, and `--oracle-rack` now hands the net one block per opponent. On identical deals the oracle seeds land inside the control's checkpoint range, and zeroing the blocks flips **0.0% of 8,409 argmax decisions** -- handed every hidden rack in the game, the policy still learns to discard them. (One control seed's u100 hit the known late collapse; its u040/u060 stand in.) |

| value net over afterstates, Monte Carlo target (u200) | +4.69 | 61.3% | **The estimator, not the idea.** Every decision of an episode shares one target, so only between-episode variance is explainable, and there is almost none to explain: train EV +0.94 against episode-split holdout EV **-0.35** (112k rows) -- memorised episodes, not ranked afterstates. Weight decay buys generalisation only by flattening to the mean. |
| value net over afterstates, TD(0), 3 seeds x 600 updates | +27.02 / +22.23 / +27.13 at u600, n=240 | 76-81% | **The first learner in this file with no policy head, no imitation and no collapse.** Greedy over `V(afterstate)` from outcomes alone -- the teacher drives only a 5-update critic warmup -- reaches the heuristic band by u80 on two seeds of three and every late checkpoint stays in it: twelve samples over u260-u600 span **+18.6 to +30.9** around `first_legal`'s +26.79, touching `by_value`'s +29.60 (+29.75 / +30.59 / +30.93) without holding it. Mid-run troughs (+7.3, -24.1, one seed opening at -138) all recover, unlike the macro trainer's collapse. |
| turn-completion search over the SWA nets (`tools/search_afterstate.py`, eval-only) | paired vs myopic: **+0.71 +-2.07** over 3 seeds | -- | **Null: search inherits its evaluator's resolution.** Per seed +4.66 / -0.15 / -2.37 on identical deals; the between-seed spread triples (3.2 -> 10.2 points) while the mean stands still, `--beam 3` buys nothing over beam 1, and the one seed that cleared `by_value` on score is *worse* than `by_value` on `standard-optimal` (2.0% vs 4.0%). |
| solver-free repartition: template picker cloned from CP-SAT (`tools/train_repartition.py`), beam 16 / beam 4 / greedy | **+47.51** / +45.26 / +42.31 | 99.8% / 99.8% / 98.5% | **99% / 88% / 73% of the repartition gap recovered with no solver in the loop.** All arms in one n=240 run with `by_value` (+28.01) and CP-SAT (+47.71) beside them; zero illegal actions; where it plays it plays 1.57 tiles against CP-SAT's 1.58 -- the same turn by a different construction. Greedy decodes at **8.5x** CP-SAT's speed, beam 16 at parity (0.98x). (`by_value` reads +28.01 rather than the published +29.60 because n=240 runs past the frozen suite's 200 deals; every arm here shares the same 240.) |
| RL fine-tune of the one-phase picker, self-critical PG best-of-4 (`tools/finetune_repartition.py`), greedy decode | **+44.05** | 99.2% | **Stage two of the AlphaStar recipe, and it moves the cheap arm.** Holdout playable 27.8% -> **35.9%** at greedy and 46.5% -> **55.4%** at beam 4 -- 43% of the greedy-to-beam-4 gap, at the greedy decode's own cost. On the suite the move is real but unresolvable (paired +1.75, 95% CI [-0.66, +4.22] at n=240): the offline metric resolves what 240 deals cannot. |
| **two-phase + RL fine-tune, composed** (`tools/finetune_two_phase.py`), greedy decode | **+45.63** | 99.8% | **89.4% of the solver's points at 1.7ms against its 20.7ms.** Holdout playable 39.7% greedy / 59.6% beam 4 -- the strongest greedy decode in the repo, from parents at 35.5% and 35.9%. The two fixes **partially overlap**: the fine-tune is worth +8.1pp alone and +4.2pp here, so 52% survives; adding would have predicted 43.6%. |
| two-phase: break slots, then cover the freed tiles (`tools/train_two_phase.py`), beam 16 / beam 4 / greedy | **+48.31** / +45.86 / +43.88 | 99.4% / 99.8% / 99.2% | **103% / 91% / 81% of the gap, and every arm is faster than the one-phase arm it replaces.** 8.88 decisions (3.89 of ~12 slots, 4.99 of ~330 templates) against ~13.6 of ~330: whole-sequence exact match goes 2.0% -> 38.2% and holdout playable 27.8 / 46.5 / 66.4% -> 35.5 / 56.2 / 72.4%. The >=90% claim moves from beam 16 to **beam 4** -- 5.2ms against 24.6ms, a 4.7x cost cut for the same result -- and beam 16 at 12.4ms *exceeds* CP-SAT's 17.4ms and its score. |
| **primitive vocabulary, the picker recipe end to end** (`tools/train_primitive_turn.py`), arm B beam 4 / greedy | **-28.28** / -196.96 | 38.1% / 1.9% | **The best the primitive action space has ever scored, and still under `greedy`.** A `PlanningAgent` decoding every turn boundary out of `PLACE`/`PICK`/`DISSOLVE`/`ASSIGN`/`END_TURN`, cloned from `frugal`'s own 60,005 turns and decoded against the env's mask on a simulated copy of the position: zero illegal actions in 960 games and 2.19 tiles where it commits, exactly the teacher's 2.19 -- but it commits only 34.8% of its turns against the teacher's 42.7%. The previous best here was -230, so the sequence recipe is worth ~200 points and leaves the space 28 under `greedy`. |
| primitive vocabulary as the stuck-state repartition (arm A), beam 4 / greedy | **+36.13** / +31.03 | 92.9% / 87.7% | **The one-variable cell: 41.2% / 15.3% of the solver's contribution against the template picker's 87.6% / 72.6%**, same teacher, same warm start, same beam, same trunk width, same ruler in the same n=240 run (`by_value` +28.01, `by_value+repartition` +47.71, picker +42.31 / +45.26). Offline on 1,500 held-out stuck states, 8.9% / 23.9% / 37.9% playable at greedy / beam 4 / beam 16 against the picker's 29.5% / 50.4% / 70.6% -- and 1.62 tiles where it plays against CP-SAT's 1.59. The vocabulary costs a factor of 2-3, not the quality of the turn. |
| + self-critical PG, best-of-4, KL-anchored (`tools/finetune_primitive_turn.py`), arm B beam 4 / greedy; arm A beam 4 / greedy | **-20.57** / -151.16; **+38.15** / +32.39 | 41.2% / 3.8%; 93.5% / 88.8% | **A fifth of the effect on the offline metric and the largest move on the suite.** Held-out committed 26.9% -> 28.6% at greedy and 54.9% -> 54.6% at beam 4, against +8.1pp / +8.9pp for the same stage two over templates, and what moves is the long repartition buckets. On the suite it is worth +45.8 to arm B at greedy, +7.7 at beam 4, and takes arm A's share of the solver's contribution from 41.2% to **51.5%**. Kept at update 100 of 300 and falling after -- the same early peak the macro trainer has. |
| + `DRAW` given a label (cell 2a, `tools/collect_turns.py`), arm B beam 4 / greedy; arm A beam 4 / greedy | **-74.29** / -274.45; **+37.13** / +29.43 | 17.3% / 0.2%; 93.1% / 85.0% | **The abandonment is cured and the playing goes with it.** 72,667 clean declines labelled beside the same 60,005 played turns; the greedy decode's failures go from **0.0% deliberate `DRAW` and 100.0% budget-out at depth 155** to **61.6% / 38.4% at depth 59.6**, and it declines 100.0% of the held-out declines -- but held-out committed falls 26.9% -> **13.4%** at greedy and 54.9% -> 49.3% at beam 4. The mechanism is the calibrated prior: 55% of boundary labels are `DRAW` while the 45% that plays is spread over fifty-three `PLACE`s, so the argmax at a *played* boundary is `DRAW` **53.1%** of the time against the unlabelled arm's 0.0%. No reweighting -- the label is 54.8% of turns but 7.3% of steps, so the loss was never dominated by it. |
| + the lay-off rendered in three actions, not seven (cell 2b, both levers, `rummi/solver/to_actions.plan`), arm B beam 4 / greedy; arm A beam 4 / greedy | **-146.11** / -322.63; **+33.39** / +28.20 | 7.9% / 0.0%; 89.4% / 84.0% | **Every published score identical, a third of every sequence gone, and the offline metric back above where it started.** Matching standing slots to target sets by overlap -- superset by `ASSIGN`, subset by `PICK` -- takes `frugal`'s expansions from 6.78 primitives to **4.59** with the `PLACE` and `END_TURN` counts unchanged, an ordinary turn 12.92 -> **8.64** and a repartition 27.44 -> **16.45**; all three `docs/data/agents*.json`, `experiments.json` and the n=240 ruler re-capture **identical on every field**. Held-out committed 13.4% -> **24.6%** at greedy and 49.3% -> **55.6%** at beam 4, which is *above* the unlabelled cell's 54.9% while still playing nothing on 98.6% of the positions the teacher declined. |
| + stage two on 2b, same self-critical PG (`tools/finetune_primitive_turn.py`), arm B beam 4 / greedy; arm A beam 4 / greedy | **-118.42** / -279.25; **+32.20** / +29.81 | 13.1% / 0.2%; 88.1% / 86.2% | **Seven times the offline move, bought by giving the decline back.** Held-out committed 28.1% -> **40.4%** against +1.7pp for the same stage two over the original rendering -- and the one-step measurement says what changed: the argmax at a *declined* boundary goes from `DRAW` 84.5% to **15.7%**. The reward is tiles alone with no term for declining and `--kl-coef 0.1` does not hold it, so stage two trades lever 1 away for commitment. |
| **solver-free composition: afterstate chooser over the two-phase picker** (`tools/eval_solver_free.py`), beam 4, 3 SWA seeds | **+45.05** / +45.00 / +38.62 | 99.4% / 99.0% / 98.5% | **Both halves learned, no solver at inference -- and the chooser is the free half.** Given a real solve the same value net scores **+47.31 / +47.39** against `frugal`'s +47.71 and the clone rung's +47.23, inside one standard error, so the whole of the +19 the `REPARTITION` macro is worth was already available to a policy trained with no imitation past a five-update warmup. What the composition gives up is the constructor's: 2.66 points from `frugal` at 2.52 ms per decision against its 6.33. Then the caveat that matters, and its attribution: head-to-head against `optimal` it wins **15.0%** (n=200) where `frugal` wins 53.0% on the same deals -- but the chooser swapped alone is **even** (49.0%) and the constructor swapped alone is 23.5%, so the duel loss is the picker's, and the suite's *best* picker arm (beam 1, +45.63) is its worst duellist (14.5%). A 99.4% win rate against `greedy` hides that because a declined stuck turn costs almost nothing there and is a turn a peer converts. |

**Afterstate value learning works, and its ceiling is the outcome's noise floor.**
Every macro's afterstate is deterministic and computable without stepping the env
(`rummi/agents/learned/afterstate.py`; the drift test replays real games and holds
the analytic afterstate to the env's own encoding, 409/409 field-exact, and an
injected off-by-one is caught 409/409). `tools/train_afterstate.py` scores the ~20
legal afterstates and plays the argmax, so the whole policy-side machinery this
file fights disappears: no 713-way head, no argmax-vs-sampled gap (the agent is
deterministic by construction), and `END_TURN` is a value comparison rather than a
starved logit. What stops it at the band is measured on both sides. Monte Carlo
fails in the data, not the fit, per the row above. TD(0) plays at heuristic level
while its explained variance sits at ~0.00 -- which is the diagnosis in one line:
the ranking that separates the rungs lives below the outcome's per-episode noise
(`first_legal` -> `rearrange` spans 0.012 normalized units against 0.073 noise per
episode), so the regression finds the band in ~80 updates and cannot lock the last
three points; annealing exploration from 0.10 to 0.02 over 600 updates does not
close the gap. The oscillation itself is solved for free: averaging each seed's
sixteen u300-u600 checkpoints (`tools/average_checkpoints.py`, plain float32 weight
averaging -- the net has no normalisation layers, and the shipped `-swa.pt` files
reproduce bit for bit) scores **+28.37 / +27.59 / +25.16 at n=240** against
endpoints of +27.02 / +22.23 / +27.13, seed 1 gaining five points over its own
endpoint. So the band is noise around a mean the averaging extracts -- and the
mean is ~+27, not `by_value`'s +29.60. **The residual gap is level, not
instability**, which is what retires a target network as the next lever.
`--repartition` runs (145 dec/s against ~1,100 without, CP-SAT) and is off in
these rows so the `by_value` comparison stays like-for-like.

**Lookahead over these nets is null, and the mechanism is the same noise floor.**
The search values a candidate macro by greedily completing the turn in
simulation: `afterstate_obs` returns the observation the env would report next,
so completion is recursion with no env stepping, and the fidelity test holds 442
predicted decisions field-exact against real games, 276 of them past depth one,
completions up to eleven macros long. The lookahead *fires* -- it changes
4.6-16.3% of decisions per seed, and 36% of one seed's pre-meld decisions get the
multi-set opening turn the myopic policy cannot see. It ties anyway, per the row
above, and the three signatures agree on why: added information narrows a spread
where amplified selection noise widens it (3.2 -> 10.2 while the mean moves
+0.7), a wider beam selects over strictly more completions and buys nothing, and
the apparent winner does not survive a change of opponent. A maximum over more of
V's outputs is a maximum over more of V's noise -- with explained variance at
~0.00 there is nothing at the leaf to propagate. Search is not the lever until an
evaluator has resolution, and that is a training problem, not a search one.

**The NP-hard construction amortises.** A repartition decomposes into one
~330-way template choice at a time over the rack-plus-table multiset -- `STOP`
legal once every table tile is re-covered -- and each step is maskable by plain
feasibility, so the sequence is legal by construction where a whole-repartition
head is intractable at mask time -- the same wall that rightly kept whole-turn
macros out of `rummi/agents/macro.py`.
Trained by imitation on 48,002 CP-SAT solutions from 6,183 games (100%
representable in template space, 79.3% tile-for-tile), the picker's holdout
numbers say the learned skill is *construction*, not mimicry: step accuracy 77%
and exact-match 2%, while beam-16 decoding emits a fully valid repartition in
92.1% of held-out stuck states and one that plays in 66.4%. Two settings carried
it, both measured against controls. Colour relabelling (24 free readings per
state) *lowers* step accuracy 80.6% -> 77.0% and doubles the greedy playable
rate 14.2% -> 27.8% -- predicting the teacher's next set and constructing a
legal repartition are different skills, and only the second survives thirteen
steps. And the beam ranks finished candidates by tiles played, not likelihood:
every finished beam is legal, so likelihood is the wrong tie-break. Two
corrections to the record while measuring it: with the REPARTITION macro **on**,
CP-SAT plays in **20.9%** of the states the gate asks about, not the 47.7%
measured with it off -- the macro eats its own opportunities -- and the mean
solve re-cuts nearly the whole table (12.6 sets) to absorb 1.59 tiles. What
separates greedy from beam is sequence length, not capacity: 68% of the chosen
sets are already on the table, so a two-phase space (slots to break, then covers
for the freed tiles) is ~9 decisions where this one takes 13x330.

**Sequence length was the right diagnosis, and the fix pays where the cost is.**
Splitting the construction -- pick slots to dissolve, then cover what they freed
-- decomposes **100%** of the same 48,002 solutions (92.4% tile-for-tile), which
is what the 68% figure predicted: 8.71 of 11.60 occupied slots are kept
untouched, so the real decision is 2.89 breaks and 3.99 builds. The shorter
sequence is worth 19x on whole-sequence exact match (2.0% -> 38.2%) and lifts
every decode width's playable rate, per the row above. Two consequences worth
separating. The **cheap arm improves but does not arrive**: greedy goes 72.6% ->
80.6% of the gap, and the >=90% threshold moves from beam 16 to beam 4, which is
a **4.7x cost cut for the same claim** rather than a free greedy decode. And
where the cheap arm still loses is the **cover, not the break** -- the break
head's holdout accuracy is 58.8%, `--weight-decay 1e-4` repairs its overfitting
exactly (59.7% held instead of decaying to 52.8%) and moves the deliverable not
at all, while widening phase A alone buys 1.4pp for 3.2x the time. The
decomposition itself is most of the gain: trained from scratch, with no warm
start from the one-phase weights, it still reaches 30.2% / 48.3%. One asymmetry
to note, because it inverts a metric: two-phase greedy has a slightly *lower*
valid rate than one-phase greedy and a much higher playable one, since a decode
that dissolves nothing is trivially valid and worth nothing -- validity is the
guardrail, playability is the objective.

**What RL adds, and the trap it walks past.** The picker is an imitation policy, so
stage two is policy gradient on top of it with a KL anchor back to the clone --
AlphaStar's own recipe. Self-critical is the fit: sample a construction, decode
greedily as the baseline, push up whatever beat it, which optimises the *greedy*
decode because that is the arm worth having. Three findings, each measured against
a control. **A validity bonus is actively harmful**: paying 0.1 for a valid
construction buys 23pp of validity and *costs* playability (29.6% against 32.2%),
because the nearest reward from a failing rollout is the identity cover -- keep
every set where it is, always valid, always worthless. **One sample per state is
the wall**: a lone deviation loses to greedy three times as often as it wins, so
the signed update mostly teaches "do not deviate" and four different recipes stall
at 32-33%; taking the best of four draws breaks it to 35.9%, because *the beam's
whole value is selection among finished constructions* and one sample gives the
update nothing to select from. Consistent with that, dropping the negative half of
the advantage matches full signed updates while barely drifting (KL 0.04 against
0.28), and the signed arm's entropy collapse (0.89 -> 0.37 nats) buys nothing --
so the stall was never the anchor binding. **And unlabelled states are usable but
inert**: 20,001 gate states collected in 116s, 79% of them ones CP-SAT declined,
cost 1.9pp, because under a tiles-only reward nothing plays, every sample and the
baseline score zero, and an exactly-zero advantage teaches nothing. RL needs no
labels; it does need two answers that differ. Declining, meanwhile, is structural
rather than learned -- `STOP` is masked until the table is covered, so a decode
covers or returns nothing.

**And beam 16 scores above CP-SAT, which is a lead rather than a result.**
+48.31 against +47.71 at n=480 is inside the noise this suite affords, and the
arm answers fewer asks than the solver does (19.0% against 20.9%), so it is not
a claim that the network out-solves CP-SAT. What is not noise is the mechanism
available to it: the two decode the *same* set of optima and rank them
differently -- the beam by tiles played, the solver by its own keep-weight
tie-break -- so choosing among equal-tile repartitions is a free parameter that
nothing in this repo has ever deliberately set. That is the cheapest standing
test of whether anything above per-turn maximisation exists at all: same tiles
shed, different table left behind.

**Composed, the two fixes partially overlap -- and the cheap decode arrives.**
Shortening the sequence and improving selection were designed against different
diagnoses, and each independently took the greedy decode from 72.6% of the gap to
~81%. Fine-tuning the two-phase net (`tools/finetune_two_phase.py`, phase B
*importing* the one-phase differentiable decode rather than restating it, so the
reward, advantage and anchor are the same code) reaches **39.7% holdout playable
at greedy and +45.63 on the suite: 89.4% of CP-SAT's points at 1.7ms against its
20.7ms.** That is the claim beam 4 used to carry, at a twelfth of the solver's
cost, and it is the strongest greedy decode the project has.

The gains do not add: the fine-tune is worth +8.1pp on the one-phase picker and
+4.2pp here, so **52% survives**, where adding would have predicted 43.6%. The
headroom barely narrowed -- best-of-4 still beats its own greedy arm on 19.9% of
states against 22.6% before -- so what shrank is how much a shorter sequence had
left to collect, not the method's reach.

Two things worth recording because they contradict what the design expected.
**There is no phase asymmetry.** The reward is the cover's, so the break head
looked likely to starve; instead it is credited on 67% of its decisions against
the cover's 83%, carries a comparable gradient, and frozen one at a time the heads
are worth +2.4pp and +2.0pp of the composed +4.2pp -- nearly additive *within* the
composition. They differ in *what* they move, not how much: the cover head lifts
validity 49.7% -> 56.2% while the break head barely touches it (51.3%) and moves
playability just as far. **And the break-nothing trap never fired** -- zero times
in 1.15M greedy rollouts. It moved the other way, greedy break count 2.29 -> 2.79
toward the teacher's own 2.89, with 4-or-more breaks going 14.2% -> 25.8%: nothing
paid for validity and validity rose anyway, as a by-product of shedding more.
Imitation accuracy meanwhile *falls* (break 71.3% -> 70.4%, cover 82.7% -> 81.9%),
which is the same signature the one-phase augmentation left -- fitting the
teacher's next set and constructing a good repartition are not the same skill.

## The picker recipe over primitive actions: the same turns, found a third as often

Every conclusion in this file about the primitive action space -- PPO never melds,
a cloned policy "fiddles and gives up" at a median of 7 micro-actions, `DRAW`
reverts what it half-built -- was reached with **per-step imitation and PPO**, and
all of it predates `tools/train_repartition.py`. What that established is that the
right objective for *construction* is an outcome-scored **sequence**: legal by mask
at every step, decoded as a beam, finished candidates ranked on tiles played, and
per-step accuracy explicitly not the deliverable. It had only ever been run over
the ~331 set templates. This is the same recipe over the env's own vocabulary,
**changing one variable** -- the same teacher (`frugal`), the same imitation warm
start, the same self-critical fine-tune with best-of-4 and a KL anchor, and the
ruler (`by_value`, `by_value+repartition`, the template picker at matched beams)
re-measured in the same run. The decoder emits `PLACE`/`PICK`/`DISSOLVE`/`ASSIGN`/
`END_TURN` instead of templates.

Two pieces of machinery make it a fair fight rather than a new set of rules.
`rummi/agents/learned/turn_sim.py` rebuilds a position as a `BatchState` **from the
observation alone**, so the beam's legality oracle is `legal_actions` itself and
the network reads what the env would actually hand it; the unknowable split of
`unseen` into pool and opponents' racks is invented so that `rack_sizes` and
`pool_size` come out right, which is all the mask and the encoder read of it. Over
**924,178** recorded teacher steps the simulator refused zero, and
`tests/test_turn_sim.py` holds it to real games field by field. The scorer is
`TorchPolicy` -- the same 2400-way net `tools/train_ppo.py` trains -- so the network
is not a second variable either.

**The data.** 60,005 committed turns over 4,020 games and 31,039 gate states CP-SAT
answered, four seeds of `frugal` against `greedy` (`tools/collect_turns.py`).
**57.3%** of the teacher's turns are draws and carry no sequence at all; the gate
answers **20.8%** of its firings, reproducing the 20.9% already on record. The
lengths are the axis under test: an ordinary turn is a mean of 12.9 primitives, a
turn with a `REPARTITION` in it 27.4, and the shortest turn the teacher ever
commits is **seven**.

**Stage one arrives on quality and not on frequency.** Held-out step accuracy 76.5%
against 90.5% on train, whole-turn exact match 8.8% -- neither is the deliverable.
Decoding from the boundary with no teacher commits a turn in **26.9% at greedy and
54.9% at beam 4** of the turns the teacher played, and where it commits it sheds
**2.22 tiles against the teacher's 2.19**; on stuck states 1.62 against CP-SAT's
1.59. That is the picker's own signature exactly: the turn it finds is as good as
the teacher's, it just finds one far less often.

**The one-variable cell.** Asked the question the picker is asked -- the gate has
fired, `by_value` has nothing, what do you play -- on 1,500 held-out gate states:

| decoder | plays, greedy | beam 4 | beam 16 | tiles where it plays | decodes/s, greedy |
|---|---|---|---|---|---|
| **primitive actions** | **8.9%** | **23.9%** | **37.9%** | 1.62 | 368 |
| templates, one-phase | 29.5% | 50.4% | 70.6% | 1.69 | 304 |
| templates, two-phase | 47.2% | 67.1% | 81.4% | 1.59 | 428 |
| CP-SAT | 100% | 100% | 100% | 1.59 | -- |

**The vocabulary costs 3.3x at greedy, 2.1x at beam 4 and 1.9x at beam 16**, on the
same states, from the same teacher, at matched capacity and matched beam -- and the
ratio narrows with the beam, which is search paying for what the vocabulary costs.
On `standard-greedy` at n=240, dropped into `by_value+repartition` in place of the
solve, it recovers **15.3% of the solver's 19.70-point contribution at greedy and
41.2% at beam 4** (+31.03 and +36.13 against `by_value`'s +28.01), where the
one-phase picker recovers 72.6% and 87.6% (+42.31 and +45.26) in the same run. In
that run it answers **4.9% of the gate's firings at greedy and 9.2% at beam 4**,
against the picker's 12.8% and 16.5% and CP-SAT's 20.8%: what a wider beam buys is
how often there is an answer, not what the answer is worth.

**The length breakdown, and why the obvious diagnosis is only half of it.**
Bucketed by the teacher's own plan length, over the 9,029 held-out turns:

| the teacher's turn | n | committed, greedy | committed, beam 4 |
|---|---|---|---|
| ordinary, 7-10 primitives | 3,811 | 32.5% | 66.5% |
| ordinary, 11-20 | 2,724 | 21.7% | 42.1% |
| ordinary, 21-35 | 810 | 9.3% | 19.5% |
| ordinary, 36+ | 139 | 5.8% | 12.2% |
| with a repartition, 11-20 | 424 | 36.6% | 77.4% |
| with a repartition, 21-35 | 836 | 34.7% | 71.8% |
| with a repartition, 36+ | 285 | 23.5% | 62.8% |

Within a class, length is the whole story -- 32.5% -> 5.8% at greedy and
66.5% -> 12.2% at beam 4, monotone across a factor of five in length. **Across
classes it inverts, and that kills the specific prediction.** At matched length the
repartition turns are far *easier*: 34.7% against 9.3% in the 21-35 bucket, 62.8%
against 12.2% at beam 4 in the 36+ one. So "heuristic level on ordinary turns, short
of it on repartitions" is exactly backwards, and the reason is what a long turn is
made of on each side. A long ordinary turn is a pre-meld opening or a multi-set turn
where every tile has to leave the rack and land right, and there is essentially one
target; a long repartition happens post-meld on a full table where a great many
committable turns exist and the decode only has to reach one of them. **Length bites
where the target is narrow, not where the sequence is long.**

**The failure is abandonment, and it is structural.** Of the held-out turns the
greedy decode does not commit, **0.0% end in `DRAW` and 72.5% run the turn's micro
budget out** -- a mean decode length of **115.3 primitives against the teacher's
15.3**. Counted the stricter way the follow-up below uses, where the forced `DRAW` of
a spent budget is a wander and not a decline, it is 100% at 155.0: every failure runs
the clock out. That is "fiddle and give up" again at fifteen times the length, and the
sequence recipe does not touch it, because in template space the ability to decline
is not learned: `STOP` is masked until the table is covered, so a decode covers or
returns nothing. Here nothing masks a turn that is going nowhere -- `PLACE` is legal
whenever a tile is in the rack -- so the mask cannot express the one thing that made
the template arm safe. The *agent* is safe either way, because a decode that never
reaches `END_TURN` returns no plan and the turn is a clean `DRAW` rather than a
half-built table, which `tests/test_primitive_turn.py` holds it to. What it is not
is cheap: the wandering is why beam 4 costs 12.5 ms a turn against greedy's 2.2.

Two smaller findings, both cheap levers that were deliberately not pulled because
either would have moved a second variable. **The teacher's own rendering inflates
the commonest move**: `to_actions.plan` keeps a set only where the target reproduces
it exactly, so a lay-off dissolves the receiving slot and rebuilds it -- laying one
tile onto a three-run is `DISSOLVE`, `PLACE`, four `ASSIGN`s and `END_TURN`, seven
primitives where the mask allows three (`PLACE`, `ASSIGN` onto the occupied slot,
`END_TURN`). 84.2% of `greedy`'s `ASSIGN`s land on an occupied slot, so a good part
of every length in this section is the rendering rather than the move, and the
student is being taught the long way round. **And `DRAW` has no label**: the dataset
is the turns the teacher *played*, so the 57.3% it drew are absent and the student is
never shown a position whose answer is "do not" -- which is exactly the decision the
mask cannot make for it.

**Stage two barely moves the offline metric and moves the suite a great deal.** The
same self-critical policy gradient -- sample four whole turns, keep the best-shedding,
push against the greedy decode, KL-anchor to the clone, reward in tiles with no
validity bonus -- lifts held-out committed from **26.9% to 28.6%** at greedy and
leaves beam 4 at 54.6% against 54.9%. On the template picker the same stage two was
worth +8.1pp and +8.9pp; here it is +1.7pp and nothing, and what it does move is
concentrated in the long repartition buckets (21-35: 34.7% -> 40.7%, 36+:
23.5% -> 34.0% at greedy). It peaks at update 100 of 300 and falls after, the same
early peak the macro trainer has. On the suite it is worth far more than that:
**+45.8** points to the whole-turn agent at greedy and +7.7 at beam 4, and it takes
the drop-in repartition from +31.03 to **+32.39** at greedy and +36.13 to **+38.15**
at beam 4 -- 41.2% of the solver's contribution to **51.5%**, the largest single move
in this section. That is despite the fine-tune not transferring on the offline
metric for that arm's own population, which it was not trained on: held-out stuck
states go
8.9% -> 7.9% at greedy and 23.9% -> 24.7% at beam 4. Both readings are one seed.

**Arm B is the honest arm, and it is the best the primitive space has ever scored.**
A `PlanningAgent` that decodes every turn boundary with no macro vocabulary, no
templates and no solver anywhere scores **-28.28 / 38.1% at beam 4** and
-196.96 / 1.9% at greedy, and after the fine-tune **-20.57 / 41.2%** and
-151.16 / 3.8%. The previous best in this file for the primitive action space is
**-230**; `greedy` is +0.00 / 50.0% and `by_value` +28.01 / 82.1%. The best arm
commits **36.5%** of its turn boundaries against the teacher's 42.7% and sheds 2.08
tiles where it commits against the teacher's 2.19, and across all four arm-B runs
(1,920 games) it attempted **zero** illegal actions -- the legality property holds
exactly, and what the arm costs is the turns it declines rather than the turns it
gets wrong. So the sequence recipe is worth about **210 points** in the primitive
space, and leaves it 21 under `greedy` and 49 under `by_value`: the same verdict as
the notebook's, by a much smaller margin and for a different reason.

Every arm here is decoded **deterministically** -- argmax at beam 1, best-of-beam
ranked on tiles above it -- so the argmax-versus-sampled caveat elsewhere in this
file does not apply to any of these numbers. Each row is one seed of one recipe.

### The two levers, pulled: the wandering stops, and the playing stops with it

The section above named two cheap fixes and left them alone, because either would
have moved a second variable. Both are built now, and measured as two further cells
against the same ruler: **2a** labels the teacher's declines, **2b** does that *and*
stops the teacher rendering a lay-off as seven actions. Everything else -- teacher,
seeds, targets, net, trunk, epochs, selection rule, stage two -- is where it was.

**Lever 1: `DRAW` gets a label.** `tools/collect_turns.py` records the boundaries
`frugal` drew from as the one-step sequence `DRAW`: **72,667** of them beside the
60,005 turns it played, and the played half comes out identical to the digit --
60,005 turns, 31,039 answered gate states, 12.92 and 27.44 mean primitives -- so the
declines are the only thing that changed. Only a *clean* decline is labelled: **9.6%**
of the teacher's draws revert a turn it had already begun lifting tiles for, and
calling that boundary "do not" teaches the opposite of what happened, so those are
counted and dropped.

**The class balance is left exactly as it falls, and that is the finding rather than
an oversight.** A decline is one label and a played turn 15.4, so the declines are
**54.8% of turns but 7.3% of steps**, and the loss -- summed over steps -- puts
nowhere near a majority of its mass on them. What cross-entropy learns at a boundary
is then the honest 55/45, and *that* is the problem: the argmax of a correctly
calibrated distribution is `DRAW`, because the 45% that plays is spread over
fifty-three `PLACE`s while the 55% that does not is one action. Measured one step at
a time over the played held-out boundaries, the argmax is `DRAW` in **53.1%** of them
where the original arm's was **0.0%**. Reweighting would be a thumb on the scale
rather than a fix; the principled correction is the one the recipe already has -- a
beam, which spends one hypothesis on the decline and follows the others.

**Lever 2: a lay-off is three actions, not seven.** `rummi/solver/to_actions.plan`
matches the standing slots to the target sets by overlap rather than by equality: a
target that is a *superset* of a slot is reached by `ASSIGN`ing the difference onto
it, a *subset* by `PICK`ing the surplus off, and only a target that is neither -- or
a subset so much shorter that picking costs more than rebuilding it -- dissolves. The
allocation is one function, `to_actions.allocate_slots`, because
`learned/afterstate.py` mirrors where a set lands and a second copy of that decision
is a silent, shape-clean drift in every `slot_features` column; it reads the
allocation now instead of restating it.

**What it does to the teacher, and what it does not.** Over 20 games of `frugal` with
every expansion replayed against `legal_actions` as it is queued, the same 3,046
expansions come out **32.4% shorter** -- 20,665 primitives to 13,974, a mean of 6.78
to 4.59 -- with the `PLACE` count and the `END_TURN` count identical to the tile: the
same turns, said in fewer words. `DISSOLVE` falls 73% and `ASSIGN` 47%, and 1,416
`PICK`s appear where this agent had never emitted one. On the teacher's own collection
an ordinary turn goes **12.92 primitives to 8.64** and one with a repartition **27.44
to 16.45**, the median 13 to 7, and the shortest turn `frugal` ever commits **7 to 3**.

**Nothing it scores moves.** Over 832 solver targets on all three configs both
renderings reach the same committed table position for position, every emitted step
is legal under the real mask, and the plan is 22.8% shorter. Re-captured to a scratch
path, `docs/data/agents.json`, `agents-standard-3p.json` and `agents-standard-4p.json`
are **identical on every rung and every field**, and so is `docs/data/experiments.json`.
The n=240 ruler is identical too: `by_value` +28.01, `by_value+repartition` +47.71,
the template picker +42.31 and +45.26, to the second decimal. That is the point --
the target sets never changed, only the sentence that reaches them -- and it is why
the two cells below compare what the *student* is taught rather than what the teacher
plays.

**Cell 2a: the abandonment is cured, and the cure is what stops the playing.** Stage
one fits better and arrives less often. Held-out step accuracy 75.5% against 76.5%
and whole-turn exact match **50.6%** against 8.8% -- a decline is a turn of one action
and it gets it right -- while the decode commits **13.4%** of held-out played turns at
greedy against 26.9%, and **49.3%** at beam 4 against 54.9%. It plays nothing on
**100.0%** of the 10,966 positions the teacher declined at greedy and 98.4% at beam
4 -- and the abandonment table below is what says those are *decisions* rather than
failures to find anything, which is the distinction the original arm could not make.

The failure diagnostic is the deliverable, and it inverts. Of the played held-out
turns the greedy decode does not commit, **61.6% now end in a deliberate `DRAW`** with
the micro budget still open and 38.4% run it out, at a mean decode length of **59.6**
primitives against the teacher's 15.6. The same measurement on the cell above reads
**0.0% and 100.0% at 155.0** -- the full budget, every time. (Read that way rather
than the section above's 72.5% / 115.3: a spent budget masks everything but `DRAW`, so
*every* failure emits it in the end, and only depth separates a decision to decline
from a wander that ran out of room. Both numbers here are on the same checkpoint and
holdout, under that accounting.) At beam 4 the diagnostic does not move at all --
0.0% / 100.0% / 155.0 -- because one hypothesis declining does not stop the other
three, which is exactly what a beam is for.

On the one-variable stuck-state population -- 1,500 held-out gate states, the gate has
fired, `by_value` has nothing -- the label costs a little at the cheap decodes and
nothing at the wide one:

| decoder | plays, greedy | beam 2 | beam 4 | beam 16 | tiles where it plays |
|---|---|---|---|---|---|
| primitive, `DRAW` labelled (2a) | **6.1%** | 15.1% | **20.9%** | **37.9%** | 1.65 |
| primitive, the cell above | 8.9% | 16.9% | 23.9% | 37.9% | 1.62 |
| primitive, both levers (2b) | 4.4% | 12.8% | 18.7% | 28.9% | 1.92 |
| templates, one-phase | 29.5% | -- | 48.2% | 68.2% | 1.64 |
| templates, two-phase | 47.2% | -- | 66.7% | 79.3% | 1.57 |
| CP-SAT | 100% | 100% | 100% | 100% | 1.57 |

The template rows are the published ones re-measured on each cell's own gate states;
the nets behind them are untouched, and they move a point or two because the sample
does.

**Cell 2b: the shorter rendering pays for the label, and a little more.** Same recipe
on sequences a third shorter -- 670,221 primitive steps against 996,845 for the same
60,006 turns -- kept at epoch 56. The decode commits **24.6%** of held-out played turns
at greedy against 2a's 13.4% and the original 26.9%, and **55.6%** at beam 4 against
49.3% and **54.9%**. So at beam 4 the two levers together sit *above* the cell they
came from while playing nothing on **98.6%** of what the teacher declined, and 99.9%
at greedy. Where it plays it sheds 1.94 tiles against the teacher's 2.18, against the
original's 2.22: smaller turns, and at beam 4 more of them. The abandonment reads the same way
as 2a's -- 63.5% deliberate declines, 36.5% budget, mean depth 57.2 against a teacher
length of 10.2 -- and the same 0.0% / 100.0% / 155.0 at beam 4.

**What the cure is unambiguously worth is time.** On the same 2,000 played held-out
boundaries, one process at a time, the greedy decode costs **3.50 ms** unlabelled,
2.05 ms in 2a and **1.46 ms** in 2b -- a **2.4x** cut at the same commit rate, because
the wandering is what it was paying for. Beam 4 barely moves (17.99 / 18.04 / 16.83
ms), for the reason the beam-4 abandonment gives: with four hypotheses one of them
always runs to the budget, so nothing is saved.

**The length breakdown is where the mechanism shows.** Lever 2 puts mass in buckets
the old rendering had nothing in -- 1,925 held-out turns of three primitives or fewer,
which is a lay-off said the short way -- and inside the ordinary class the monotone
decline with length survives exactly:

| the teacher's turn | n | committed, greedy | committed, beam 4 |
|---|---|---|---|
| ordinary, 1-3 primitives | 1,925 | 34.9% | 76.3% |
| ordinary, 4-6 | 217 | 24.4% | 52.5% |
| ordinary, 7-10 | 3,389 | 20.6% | 51.1% |
| ordinary, 11-20 | 1,622 | 14.0% | 31.4% |
| ordinary, 21-35 | 375 | 4.8% | 11.5% |
| ordinary, 36+ | 23 | 0.0% | 8.7% |
| with a repartition, 7-10 | 423 | 36.9% | 77.5% |
| with a repartition, 11-20 | 741 | 37.7% | 77.6% |
| with a repartition, 21-35 | 370 | 34.1% | 73.8% |
| with a repartition, 36+ | 34 | 44.1% | 76.5% |

The inversion the section above found survives and gets *stronger*: with a
repartition in them the turns are **flat** in length -- 34-44% at greedy, 74-78% at
beam 4, across a factor of five -- where the ordinary ones fall off a cliff.
Shortening the sequence moves everything up and changes nothing about that: length
bites where the target is narrow, not where the sequence is long.

**Stage two moves the offline metric seven times as much, and it buys it by giving
the decline back.** The same self-critical PG, unchanged, lifts held-out committed
from **28.1% to 40.4%** on the probe slice, kept at update 200 -- against +1.7pp for
the same stage two over the original rendering. What it actually did is visible in the
one-step measurement: the argmax at a played boundary goes from `DRAW` 46.5% to
**2.3%**, and at a *declined* boundary from 84.5% to **15.7%**. The reward is tiles
alone, with no term for declining, and a KL coefficient of 0.1 does not hold it: stage
two trades lever 1 away for commitment, which is why it is worth so much more here
than it was there. The abandonment goes back with it: deliberate declines fall
63.5% -> **4.4%** and the mean failed decode goes 57.2 -> **148.4** primitives, which
is the wandering the label had just cured.

Stage one and stage two are also where the offline metric and the population part
company: the probe reads +12.3pp while the *stuck-state* population -- the arm-A
question -- moves 4.4% -> 7.1% at greedy and 18.7% -> **17.5%** at beam 4, so the
fine-tune helps where it is asked to commit and not where it is asked to construct.

**On the suite the two levers are a loss, and the reason is the one the offline
metric cannot see.**

| arm, `standard-greedy` at n=240 | score | win | answers |
|---|---|---|---|
| `by_value` | +28.01 | 82.1% | -- |
| `by_value` + CP-SAT `REPARTITION` | +47.71 | 99.8% | 20.8% |
| template picker, beam 1 / beam 4 | +42.31 / +45.26 | 98.5% / 99.8% | 12.8% / 16.5% |
| arm A, the cell above | +31.03 / +36.13 | 87.7% / 92.9% | 4.9% / 9.2% |
| arm A, 2a | +29.43 / **+37.13** | 85.0% / 93.1% | 3.7% / 8.5% |
| arm A, 2b | +28.20 / +33.39 | 84.0% / 89.4% | 2.4% / 5.6% |
| arm A, 2b + stage two | +29.81 / +32.20 | 86.2% / 88.1% | 3.1% / 6.2% |
| arm B, the cell above | -196.96 / **-28.28** | 1.9% / 38.1% | 34.8% of turns |
| arm B, 2a | -274.45 / -74.29 | 0.2% / 17.3% | 10.2% / 30.5% |
| arm B, 2b | -322.63 / -146.11 | 0.0% / 7.9% | 8.0% / 25.4% |
| arm B, 2b + stage two | -279.25 / -118.42 | 0.2% / 13.1% | 12.2% / 27.4% |

Sixteen arms, zero illegal actions in any of them. Arm A at beam 4 is the one cell
that *gains*: 2a takes **46.3%** of the solver's 19.70-point contribution against the
cell above's 41.2%, so where the decode is asked a question the teacher answered, the
decline label is worth a point. Everywhere else the levers cost, and arm B costs a
lot -- 46 points at beam 4 in 2a, 118 in 2b. The reason is the one the offline metric
structurally cannot see. Offline the question is *given a turn the teacher played, do
you find one*; in a game the question is *is playing better than drawing*, and for
this agent the answer is almost always yes even when `frugal`'s answer was no.
`frugal` declines and then draws from a position it can still win from; the primitive
agent declines and draws itself into the random-play trap. Its in-game commit rate
falls from the unlabelled arm's 34.8% of turn boundaries to 30.5% / 25.4% / 27.4%,
and every point of that is a turn it could have played badly and chose not to play
at all. **Imitating a decline imitates a decision that only pays for the rest of the
player.**

**What this does to the verdict.** The diagnosis in the section above was right about
the mechanism and wrong about the remedy. Abandonment *was* the failure -- the greedy
decode ran the micro budget out on 100% of the turns it did not commit, at a mean of
155 primitives against a teacher's 16 -- and labelling `DRAW` cures it outright:
61.6% of the failures become deliberate declines, the mean decode length more than
halves, and the decline itself is learned to 100.0% on the held-out declines. The
lay-off rendering was also inflating everything it touched, and fixing it at the root
takes a third of every sequence out of the dataset while leaving every published score
identical to the second decimal. Both levers do exactly what they were predicted to
do. **Neither makes the primitive space competitive**, and together they leave it
further from `greedy` than the cell above did: the space's best arm is still that
cell's beam-4 whole-turn agent at **-20.57 / 41.2%**, which neither lever came near,
against `greedy`'s +0.00 / 50.0% and `by_value`'s +28.01. So "the primitive space is
closed by measurement" survives, and is now closed for a reason rather than by a
margin -- the two named repairs were the cheapest things left to try, both worked as
designed on the thing they targeted, and the space did not move. What is left is not
a rendering or a label but the decode itself: at beam 16 the vocabulary still reaches
only 29-38% of the states CP-SAT answers where a template picker reaches 74-81%, on
the same states from the same teacher, and search is the only axis on which that gap
has ever narrowed.

Every arm here is one seed, decoded deterministically, on the same protocol. **Stage
two was run on 2b only**: 2a is the isolating cell for lever 1 and its stage two
would have measured the same trade the 2b row already names, at another twenty
minutes of solver-free rollouts. Cell 2a is otherwise complete -- its own collection,
its own stage one, its own stuck-state and length tables, and its own four suite arms
on the *committed* renderer, since 2a is by construction the pre-lever-2 cell.

## No solver at inference: composing the two learned halves

Two results above were measured apart and never together. The afterstate value net
ranks ordinary macros **from outcomes alone** -- TD(0) over positions, the teacher
driving five updates of critic warmup and nothing else. The two-phase template
picker answers `REPARTITION` with no CP-SAT: cloned from 48,002 of the solver's own
solutions and then RL fine-tuned, it recovers ~90% of the solver's points. `frugal`
is the pair the other way round -- a hand-written chooser over a real solve -- and
it is the ceiling this ladder has.

`tools/eval_solver_free.py` crosses them. Every arm is scored in one process on the
same 240 deals played once per seat (n=480), every decode deterministic -- argmax
over afterstates, a fixed-width beam over templates, nothing sampled -- with
`by_value` reproducing +28.01, `frugal` +47.71 and both picker arms reproducing
their published +45.63 / +44.05 to the cent, which is the check that the ruler did
not move under the new rows.

| chooser | no repartition | CP-SAT | picker beam 1 | picker beam 4 |
|---|---|---|---|---|
| `by_value` | +28.01 / 82.1% | **+47.71** / 99.8% | +45.63 / 99.8% | +44.05 / 99.4% |
| afterstate s0 | -- | +47.31 / 99.6% | +41.85 / 98.5% | **+45.05** / 99.4% |
| afterstate s1 | -- | +47.39 / 99.6% | +43.07 / 98.5% | +45.00 / 99.0% |
| afterstate s2 | -- | +44.74 / 99.4% | +34.08 / 97.5% | +38.62 / 98.5% |

**The chooser is the free half.** Handed a solve, the value net *is* optimal tier:
+47.31 and +47.39 against `frugal`'s +47.71, inside one standard error (+-1.0), and
level with the `learned` clone rung's +47.23 in the same run. Those are the same
weights whose published myopic scores are +28.37 / +27.59, so the entire +19 the
`REPARTITION` macro is worth was already available to them -- ranking ordinary
moves and knowing when a turn is worth a solve needed no teacher beyond the warmup.
What the composed arm gives up is therefore the constructor's: `by_value` loses
3.66 points swapping CP-SAT for the beam-4 decode, and the composition loses 2.26
from its own CP-SAT arm.

**Cost is the other half of the claim.** 2.52 ms per decision at beam 4 and 1.42 at
beam 1, against `frugal`'s 6.33 and the clone rung's 7.09 -- the solve is the only
millisecond-scale thing either of those does. So dropping it buys **2.5x** for 2.66
points, or 4.5x for 5.86. (Re-measured on an idle machine at 30 deals: 6.39 / 2.66
/ 1.48, so the run's own figures are not contention.)

**And the suite is flattering it -- entirely on the constructor's account.**
Head-to-head against `optimal`, 100 deals from both seats, every arm on the same
deals (`--head-to-head optimal --h2h-arms`):

| arm | vs `optimal`, n=200 | its answer rate |
|---|---|---|
| `frugal` | 53.0% (106-94) | 20.5% |
| `learned` clone rung | 53.0% (106-94) | 20.5% |
| afterstate s0 + CP-SAT | **49.0%** (98-102) | 25.4% |
| `by_value` + picker b4 | 23.5% (47-153) | 17.6% |
| afterstate s0 + picker b4 | **15.0%** (30-170) | 23.0% |
| `by_value` + picker b1 | 14.5% (29-171) | 14.8% |

The single substitutions attribute it, and they do not split it evenly. Swapping the
**chooser** and keeping the solve is even -- 49.0% against the ceiling's 53.0%, well
inside the +-3.5pp n=200 affords -- so nothing the value net does is exposed by a
peer that a `greedy` opponent hid. Swapping the **constructor** and keeping
`by_value` falls to 23.5%, and the composition sits 8.5pp under that again, about
two standard errors: the loss is the picker's, with a weak negative interaction on
top. Holding the chooser at `by_value`, the duel then tracks how often the backend
answers the gate rather than how well it scores -- CP-SAT answers 20.5% and wins
53.0%, beam 4 answers 17.6% and wins 23.5%, beam 1 answers 14.8% and wins 14.5% --
because a declined stuck turn costs almost nothing against `greedy` and is a turn a
peer converts.

**So the suite cannot rank constructors.** Beam 1 is the best picker arm on the
suite (+45.63, above beam 4's +44.05) and the worst of every arm here in the duel
(14.5% against 23.5%): the ordering inverts outright. The "~90% of the solver's
points" the picker sections claim is therefore a statement about `standard-greedy`
and does not survive a peer opponent -- and it is the answer rate, not the score,
that predicts which way a repartition backend goes when one arrives.

**The spread is the interaction, not either part.** s2 is the weakest of the three
nets (+25.16 myopic) and it is the arm that falls apart -- 6.4 points below s0 at
beam 4, where its CP-SAT arm trails by only 2.57. Its gate fires 8,643 times
against s0's 15,578: a worse chooser reaches fewer of the states the picker answers
well, so the two errors multiply. The beam ordering moves with it: `by_value`
prefers beam 1 on the suite where every value-chooser seed prefers beam 4, and the
duel prefers beam 4 by 9pp where the suite puts it 1.6 points behind. The beam
ranks finished candidates by tiles played, so a wider one answers more of the gate
-- which is what a chooser ending turns on a value comparison rather than on a rule
is the first to feel.

**What is learned from what, plainly.** The chooser is learned from outcomes. The
constructor is cloned from CP-SAT and RL fine-tuned against its own greedy decode.
So there is no solver at inference, and **this is not an agent learned from scratch
by RL**: the NP-hard construction was learned *from the solver*, and the honest
label is solver-free at inference rather than solver-free end to end. It is not
offered as a rung either. Its win rate does sit inside the `rearrange` (85%) to
`optimal` (100%) band the bar names, but it carries two files of weights, spreads
6.4 points over three seeds, and loses the duel to the rung it would displace,
15.0% against its 53.0%.

## Board shaping: the axis the oracle never bounded, and why it has no sign

Every agent here ranks candidate plays by points before the opening meld and tile
count after, and **none of them has ever considered what the table it leaves gives
the opponent** -- a blue 5*6*7 hands over two lay-off doors, a group of three 7s
exactly one, for the same three tiles shed. That is not the information channel
the oracle arm closed: those arms asked whether a policy can *read* an opponent
(handed the true racks it flips 0.0% of its decisions), where this asks whether it
can *restrict* one, which needs nothing hidden. And it had a fresh hint going in --
the two-phase beam ranks the same set of optimal repartitions differently from
CP-SAT and edges its score.

Run as a hand-written tie-break with no learning and no reward anywhere
(`tools/denial_ab.py`), so that a null cannot be blamed on a training signal:
identical to `by_value` except that among plays it ranks *equal* it prefers the one
leaving the least permeable table, permeability being the unseen-weighted count of
lay-off doors from `greedy_agent.appendable` -- the same matrix `legal_macros` gates
a lay-off on.

| arm, head to head vs plain `by_value`, 1600 deals from both seats | win | paired vs the null control |
|---|---|---|
| `deny` | 50.84% +-1.38 | **+1.06pp +-1.57** |
| `deny+steal` | 50.59% +-1.35 | +0.81pp +-1.57 |
| `value` (positive control) | 50.59% +-1.30 | +0.81pp +-1.56 |
| `open` (sign control -- prefer the *most* permeable) | 49.97% +-1.09 | -- |
| `null` (seeded coin flip over the same ties) | 49.78% +-1.36 | -- |
| `base` mirrored against itself | exactly 50.00% +-0.00 | -- |

**Null, and the bound is wider than denial.** Power was sized from a pilot before
running (per-deal sd 0.305, so 894 deals for a 2pp half-width) and run at 1600;
the self-mirror reading *exactly* 50.00% is the check that the seat rotation is
exact. On `standard-greedy` at n=800 the coin flip is level with the arm and ahead
of it on score. The positive control -- preferring face value inside a tile-count
tie, which `by_value` post-meld is genuinely blind to -- came back null too, so
the honest statement is not "denial is worth nothing" but **no tie-break over
`by_value`'s indifference set is worth more than ~1.6pp**.

Three measurements explain it, and the third is the one that generalises:

- **The tie-break can barely act.** Ties are 12.4% of decisions, and in **58.9% of
  them the permeability spread is exactly zero** -- the candidates leave equally
  permeable tables. Net, the arm plays differently in **3.7%** of decisions.
- **Its effect on the table is real but tiny.** Mean permeability handed over goes
  13.17 (`base`) -> 12.84 (`deny`) -> 13.20 (`open`), so the axis does move in the
  intended direction -- but the coin flip alone reaches 13.00, so only 1.2% of the
  level is denial rather than perturbation.
- **And it is not differential: `left - met` is 0.00 +-0.00 for every arm.** The
  table is common property and both players draw one pool, so a door closed on the
  opponent is closed on the closer by the same amount. That is not a fact about
  this tie-break but about the game: **any table-shaping rule pays its own cost**,
  which no amount of search or learning removes, and it is why the axis has no sign
  to find. The oracle section prices the same choice with the future deck visible
  and closes it at **-0.2% +-0.5%**: shedding the same tiles onto another table
  changes the winner in 3.6% of 15,801 tries.

## Rack potential: the asymmetric half, and why a recurring decision closes it

The symmetry above does not reach a rack, because a rack is **private** -- improving
one's own finishing potential costs an opponent nothing in return. And the premise
is real: every agent here maximises rack *size* while the win condition is emptying
the rack *first*. Seven combining tiles are closer to done than five that do not; a
held joker guarantees a finish that spending it forfeits.

Tested as a **weighted objective** rather than a tie-break, deliberately -- the
section above bounds the whole indifference set at ~1.6pp, so the hypothesis only
has teeth where it *overrides* the ranking and sheds fewer tiles on purpose:
`score(play) = by_value's rank + w * potential(rack left behind)`, with **`w = 0`
exactly `by_value`** down to `argmax`'s first-maximum order (pinned by test, and the
`w=0` arm mirrors to exactly 50.00%). Potential comes off `shortfall` two ways:
`ready`, the largest set the remaining rack could still lay -- which prices the
joker inside the definition, since holding one makes every one-away template
playable and spending it collapses `ready` from 3 to 0 -- and `reach`, the
probability the next draw completes a set, deduplicated by kind (red 5*6*7*8 has two
real doors, not five) and gated on unseen availability. Two-away templates are
excluded on measurement, not taste: a rack at a turn boundary is 8.1 tiles with 5.7
templates one away and **35.3 two away**, and 63.3% of unseen copies complete *some*
two-away set, so the term cannot separate two candidates. Five arms (`ready`,
`reach`, `both`, `stop` -- which offers `END_TURN` at rank 0 -- and a hard
`joker`-holding rule), each against a twin that keeps the arm's own potential values
and **permutes them across the candidates**: same magnitude, same firing rate, wrong
owners.

| arm, head to head vs `by_value`, 1600 deals both seats | win | vs its OWN null, paired |
|---|---|---|
| `ready` | 50.41% +-0.53 | +0.28pp +-0.79 |
| `reach` | 49.84% +-1.13 | -0.06pp +-1.36 |
| `both` | 50.25% +-1.19 | +0.62pp +-1.48 |
| `stop` | 50.31% +-1.46 | +1.09pp +-1.75 |
| `joker` | **50.75% +-0.76** | +0.53pp +-1.02 |
| `base` (`w=0`) mirrored | exactly 50.00% +-0.00 | -- |

**Null at n=1600, and the axis is bounded at ~1.8pp.** No arm beats its own control,
every paired interval covers zero, and on `standard-greedy` three of five nulls sit
*ahead* of their arm. The sweep's best cell -- `both` at `w=0.5`, 51.75% on 400
deals -- reads **49.78% +-1.15** on 1600 disjoint deals at another `seed_base`, which
is why the sweep was required to confirm on a held-out seed: a maximum over a grid of
noisy cells is selection on noise, not a finding.

Three mechanisms, and the third is the one that generalises:

- **The residue is already the residue.** Mean `ready` of the rack at a decision is
  **0.83 tiles against a rack of 8.1** -- usually no complete set remains, because
  `by_value` has already played whatever it could. There is very little to protect.
- **The candidates barely differ on it.** `reach` sits at 0.15 with a spread of
  **0.01** across candidate plays -- the same shape as denial's zero-spread ties,
  arrived at from the private side of the game.
- **The "lookahead" is inside the turn, so it is not a lookahead at all.** A
  decision recurs after *every set*, not once per turn, so a set "kept" in the rack
  is simply played later in the same turn: replaying both picks forward, **83.7% of
  the first-of-turn decisions the arm moves end that turn with the identical rack
  and the identical table**. What actually survives to the next turn is what no
  partition could place -- and choosing among tiles that are all stuck is choosing
  nothing. `stop` is the proof by exaggeration: handed `END_TURN` at rank 0 it takes
  it in 1.42% of decisions and changes **75.8% of deals**, and still scores even.

So per-turn maximisation is **self-correcting** on this axis: the game re-asks after
every set, which is what leaves rack shaping nothing to do. What stays open is
narrow and config-shaped -- rack potential is an axis only where a turn boundary
*forces* the residue to be kept: a tight micro budget, a one-set-per-turn cap, or a
rack too large to drain in one turn.

**The 2p ceiling, as measured.** Per-turn play is tied three ways (`optimal`, its
macro-space rendering, and the 233k-parameter clone of that rendering, 48.7% at
n=600), and the oracle section prices that tie per decision rather than per match:
where `frugal` and CP-SAT's optimum pick different turns -- 15.2% of decisions --
the disagreement is worth **+0.0% +-2.3%** with the whole future in view. Cross-turn strategy is null three ways (the delegate at every inner
strength, self-play from the clone, best-response against `optimal` with forced
exploration). Information is null three ways and bounded above by the oracle arm;
memory stays behaviourally inert in every arena it was given. **Both hand-written
strategy axes are null by *mechanism*, not by resolution** -- board shaping because
a shared table makes any restriction symmetric, rack shaping because a decision
recurs after every set, so what a turn "keeps" it plays before ending. At the
resolution
these suites afford (~+-2pp at n=600), two-player standard Rummikub behaves as a
per-turn game: play the best turn available, every turn, and nothing measurable
remains above that. What this does NOT close: an edge smaller than the noise
floor, and search behind a better evaluator -- turn-completion search over the
afterstate value nets ties (the rows above), and the measurement narrows the
opening rather than shutting it: search inherits its evaluator's resolution, so
the axis stays open only behind a leaf evaluator with non-zero explained
variance. And **adaptation against a population**, which is the one reading of
"history" the oracle arm never bounded: knowing what an opponent *holds* is not
knowing how they *play*, and against a single fixed deterministic opponent there
is nothing to adapt to at all -- a best response to a fixed policy is itself a
fixed policy, which is the setting every history, memory and oracle arm ran in.
`tools/train_macro.py` refuses all three flags alongside a `self` opponent
(L725/L735/L764), so the arena is not merely untested but unimplemented. The
cheap precondition to measure first is whether adaptation has anywhere to go: if
the best response to `greedy` and to `optimal` are the same policy, identifying
which one you face is worth nothing. The 3p/4p suites, once the promising
unknown, are measured now: the per-turn ceiling holds there, the information
bound is null at 3p (the two rows above), and the hindsight sweep re-run at both
seat counts puts every deviation type at or below zero there as well --
[three seats](#three-seats-the-argument-that-killed-it-at-two-does-not-apply-and-the-answer-is-the-same),
[four seats](#four-seats-and-the-number-the-three-runs-agree-on). The one cell any
of this ever pointed at, the opening, is closed the same way: nine opening
policies over 1600 paired deals, null in every arena
([the opening](#the-opening-was-the-one-cell-that-moved-and-it-does-not-survive-being-played)).
So the verdict above now reads at three and four seats too, and not only at two.

**The information channel is closed against `greedy` -- twice by mechanism, once
by bound.** The observation merges the pool and opponents' racks into `unseen` on
purpose, so the only hidden information is which unseen tiles the opponent holds,
and the only path to it is cross-step history -- nothing in a snapshot can say what
an opponent declined to play. Both ways of reading that channel are measured:
engineered belief features (`--history`, exact negative evidence against a
deterministic opponent) and learned memory (`--memory lstm`, an LSTMCell stepped
once per decision and trained with full BPTT over each env's decision sequence --
the one-step truncation cannot learn to *store* anything, so it would not have
been a test). Both train to policies indistinguishable from the memoryless control
under argmax and under sampling, and both leave their memory behaviourally inert
under a direct probe. Either null could still have been blamed on the mechanism
failing to extract the signal, which is what the `--oracle-rack` arm removes: a
policy handed the opponent's true rack -- no inference left to do -- ties the same
control on the same deals and ignores the block. What history could at best have
reconstructed is worth nothing here even reconstructed perfectly. That is
consistent with the delegating result above -- no cross-turn strategy was findable
at any inner strength -- and with the recurring scoreboard: capability moves
points, information does not. The remaining caveat is the opponent, not the
mechanism: against deterministic `greedy` there may simply be nothing to exploit,
and the arena where information *should* matter is self-play, which none of the
three arms is wired for yet (`--history` has no tracker for the snapshot seat,
`--memory` and `--oracle-rack` no per-env state for one). Wire that before
concluding anything about information in general.

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

## Oracle one-step regret: the bound the nulls were missing

Every arm above tested one candidate strategy, and a null reads the same whether
the axis is empty or merely out of reach. Nothing had measured how much there is
to find. `tools/oracle_regret.py` measures that ceiling by hindsight instead of by
learning: for every turn `frugal` played against itself, replay the rest of the
game from every *other* whole turn the rules admit and count what one different
choice would have changed.

It is exact rather than sampled, which is what makes it affordable. A step holds
no randomness -- the deck is permuted once at reset and drawing only advances a
pointer -- so a state cloned at a turn boundary and continued with deterministic
agents replays the rest of the game tile for tile. The game that was played is
therefore the baseline rollout for every decision in it, and there is no variance
to average away. `solve_turn` grew four optional restrictions to enumerate the
rivals -- `tiles_min`/`tiles_cap`, `exclude` (no-good cuts on the set-count
vector) and `freeze_table` -- and each one is played through `to_actions.plan`,
one primitive action at a time against the mask, so an alternative is a turn the
env itself accepts and not a perturbation: CP-SAT's own optimum, k-best *other*
tables shedding the same number of tiles, turns shedding max-1, max-2 and one
tile, the best turn that takes no existing set apart, and `DRAW`.

300 deals, `frugal` vs `frugal` on `STANDARD` at seed base 91,000: 65.0 turns per
deal, 42% of turn boundaries a real decision and the rest a forced `DRAW`, 8,185
decisions, 2.32 tiles shed per playing turn, **39,960 alternatives each rolled out
to the end** in 35 minutes on ten cores. No stalemate, no truncation and no
harness failure in any of them.

| headline, per game per seat | |
|---|---|
| lost games a single deviation would have **won** | **91.3%** |
| won games a single deviation would have **lost** | **93.0%** |
| the same, if every deviation were an independent coin flip | 99.5% / 99.8% |

**The headline is not a bound on strategy, and the second row is why.** A
deviation is picked by a maximum over ~120 rollouts per seat, and every one of
them changes what the shared pool hands out from that turn on, so the two seats
diverge and the result is close to re-decided. Rescuing 91.3% of losses while
throwing 93.0% of wins is a deviation set with **no direction at all**, and it is
already below what treating each one as an independent flip would predict. So the
answer needs the deviation measured *against the turn it replaced* rather than
maximised over -- alt win rate minus base win rate at the same decisions, with the
interval clustered by deal because alternatives inside one share its deck:

| alternative type | alternatives | changes the winner | delta vs the played turn |
|---|---|---|---|
| `cpsat_max` (a different optimum) | 1,248 | 11.4% | +0.0% +-2.3% |
| **`same_tiles_other_table`** | **15,801** | **3.6%** | **-0.2% +-0.5%** |
| `fewer_tiles` (max-1, max-2, one) | 5,886 | 27.5% | -3.1% +-1.5% |
| `frozen_table` (no rearrangement) | 818 | 26.9% | -6.1% +-3.8% |
| `draw` (hold everything) | 8,185 | 26.3% | -0.4% +-1.1% |
| `__base_replay__` (the played turn, re-derived) | 8,022 | **0.0%** | +0.0% +-0.0% |

**The free parameter every board-shaping arm aimed at is worth nothing, and this
is the tightest interval in this document.** Shedding the same number of tiles
onto a different table changes the winner in **3.6%** of 15,801 tries and is worth
**-0.2% +-0.5%** -- with the future deck and the opponent's rack both visible. The
denial section bounded that axis at ~1.6pp by exhausting a tie-break; the oracle
closes it at half a point by exhausting the *choice*.

Two rows are the positive control that makes that null readable. `fewer_tiles` and
`frozen_table` change the winner eight times as often and lose, -3.1% and -6.1%,
so the harness does detect a worse turn when handed one. And `__base_replay__` is
the exactness check: 8,022 alternatives that CP-SAT re-derived to the same table
and the same played tiles, every one of them reproducing the baseline outcome
exactly.

Three measurements of the structure, and the last is the lead:

- **CP-SAT's optimum is usually not unique, and the choice still does not
  matter.** A single max-tiles table exists at only **24.3%** of decisions, the
  median decision has **4**, and 28.4% have six or more (the k-best width is 4, so
  59.7% of enumerations are provably complete). So the indifference set is large
  and the null above is not "there was nothing to choose between".
- **Where `frugal` and `optimal` disagree, the disagreement is free.**
  `cpsat_max` differs from the turn `frugal` actually played at **15.2%** of
  decisions; where it does it flips a loss to a win 11.0% of the time and a win to
  a loss 11.7%, net **+0.0% +-2.3%**. That is the head-to-head tie between `frugal`
  and `optimal` re-derived per decision instead of per match, and it says the tie
  is not a small edge hidden by noise -- the two turns are worth the same.
- **The one cell that moves is the opening.** Pre-meld, an alternative shedding
  *fewer* tiles than the maximum is worth **+9.3% +-5.7%** (428 alternatives over
  242 deals), against -1.2% +-2.4% midgame and -8.1% +-2.6% in the endgame; the
  same-tiles row is +6.1% +-8.6% pre-meld and covers zero. Pre-meld is 7.3% of
  decisions and one turn per seat per game, so this is a narrow window, and one
  3.2-sigma cell out of fourteen is not a finding -- but it is the only direction
  this measurement points in, and the mechanism is available: `by_value` opens on
  the dearest template and the oracle prefers keeping tiles back, which is the rack
  potential the section above found nothing to protect *because a decision recurs
  after every set*. The opening is the one decision that does not recur. Tested
  as a policy it is null as well, five times tighter than this cell and excluding
  it -- [the opening](#the-opening-was-the-one-cell-that-moved-and-it-does-not-survive-being-played),
  two sections below.

Two things this cannot bound, both by construction. It is perfect hindsight -- the
enumeration is scored on the real future of a fixed deck against a fixed opponent,
so no policy can reach it and the numbers are ceilings, not achievable rates. And
it deviates on **one** turn: a strategy that gives something up now to collect it
over several turns is outside the construction entirely, which is exactly where
the cross-turn and adaptation arms already sit.

## Endgame denial: the cell that survived the oracle, and the reason it is empty

The oracle section closed board shaping at **-0.2% +-0.5%** over 15,801
alternatives, and that is a mean over a *type*. A mean cannot see a targeted effect
inside one, and one was left standing: the opponent is a tile or two from
finishing, and among the tables that shed the same tiles, one closes the door its
last tile needs. On the back of an envelope that is worth at most a point -- and
15,801 alternatives, almost none of them in that position, would read null whether
the cell were worth a point or nothing at all.

The recorded JSON could not answer it (`kind`, `tiles`, `won` per alternative and
nothing about the table), so every turn now records what it leaves behind, **the
played turn included**, so the deviation is paired against it rather than pooled:

- **the opponent's best reply** -- `solve_turn` against its *true* rack and its own
  melded flag. One solve answers both halves, because it is the *maximum* it could
  shed: zero means the table offers it nothing, and its own rack size means its
  exit is open. An oracle feature is legitimate here for the reason the whole
  construction is -- this prices a ceiling, not a policy.
- **what the opponent actually shed** on its next turn in the continuation, and
  whether that is the turn it won on.
- **permeability**, imported from `tools/denial_ab.py` rather than recomputed: the
  unseen-weighted lay-off doors that arm was scored on, plus the door count
  unweighted. Nothing hidden, so every oracle cell can be read against the signal a
  real agent would have to find it with.

Two checks make those numbers usable, and both print with the tables.

- **The reply is a bound on the rollout, and a tight one.** Nothing intervenes
  between a turn and the next seat's answer to it, so that seat can never shed more
  than CP-SAT says it could, and a table the reply calls dead must leave it
  drawing. Over **46,400** turns compared: **0** exceeded the reply, **0** played
  where the reply was nothing. And the reply is not hypothetical -- the next seat
  takes **the whole** of it in **97.3%** of the 24,158 turns it is offered one, so a
  door the oracle sees is a door `frugal` walks through.
- **The two readings of a next turn agree.** A rollout carries its own boundaries;
  the played turn's continuation is read off the baseline game's boundary list
  instead, and `--check` compares the two derivations at every decision. 106
  continuations over four deals, no disagreement, on top of the outcome and
  turn-count checks that were already there.

The 300 deals are the *same* 300 (seed base 91,000, k-best 4), and the whole of the
previous section's output is reproduced line for line: 8,185 decisions, 39,960
alternatives, 91.3% / 93.0% headline, every type and phase cell. The two summaries
differ only by the sections added below. 3,457 s on six workers.

### The door does not move

Over the 14,789 (alternative, opponent) pairs -- the 15,801 same-tiles alternatives
less 1,012 that end the game and so leave no door to close:

| the alternative leaves a table where the opponent... | pairs | delta vs the played turn |
|---|---|---|
| **cannot go out, where it could before** | **22** | **+11.1% +-21.8%** |
| ...unchanged | 14,740 | -0.2% +-0.6% |
| ...can go out, where it could not before | 27 | -9.1% +-17.8% |
| cannot play at all, where it could before | 162 | +0.8% +-12.5% |
| ...can play, where it could not before | 250 | -4.3% +-10.6% |
| actually shed fewer next turn | 335 | +2.6% +-8.8% |
| actually shed more next turn | 547 | -2.1% +-7.1% |

**The signs are the hypothesis's and the sample is one event.** Of the 22
exit-closing pairs, exactly **one** turned a loss into a win and none did the
reverse; of the 27 exit-*opening* pairs, five threw a win and none rescued one. So
+11.1% is one flip over nine decisions, and the reason the interval is +-21.8% is
that there is nothing else in the cell.

The frequency is the finding, not the delta. **The oracle door is unchanged in
99.67% of pairs** -- 97.21% for "can play at all", and even the realised shed is
unchanged in 94.03%. Restricting to decisions where there is a door to close at all
(the opponent could go out against the table actually left): **736 pairs over 188
decisions, 0.63 per deal, and an equal-tile table takes it away in 22 of them, 3.0%**.
At the headline cell -- opponent at two tiles or fewer -- it is **12 pairs over 4
decisions, 0.01 per deal**.

That closes the axis by arithmetic rather than by an interval. Nine decisions over
300 deals is **0.015 opportunities per seat-game**, so a policy that shut every exit
it could would move its own win rate by 0.015 x 11.1% = **0.2pp**, or 0.5pp at the
top of that cell's interval. The denial arm bounded this axis at ~1.6pp by
exhausting a tie-break and the oracle section at half a point by exhausting the
choice; this says *why* both readings are that size.

### The observable proxy moves two hundred times more often than the door does

| the alternative's table is... | pairs | delta vs the played turn |
|---|---|---|
| less permeable | 4,169 | -1.3% +-1.4% |
| equally permeable | 5,825 | +0.2% +-0.5% |
| more permeable | 4,795 | -0.1% +-1.4% |

Permeability is **not** equal in 60.6% of pairs, mean absolute change 1.12 against a
level of 19.7, while the oracle door moves in 0.33% of them. It does point the right
way where the door moves -- mean change -0.318 where the exit closes against +0.096
where it does not, -0.414 where play closes -- but it correlates with the oracle at
**+0.052** against what the opponent actually shed and **+0.017** against the
outcome. A denial rule reading permeability is therefore acting on a signal that
fires two hundred times more often than the thing it stands for, which is the
mechanism behind that arm's 3.7% firing rate buying 1.2% of its own level.

### The reason, and it is not about this measurement

**A solved turn is a function of the multiset the table holds, not of its
arrangement.** Post-meld the solver may repartition the whole table, so three runs
and three groups of the same nine tiles are the same problem --
`test_the_optimum_reads_the_multiset_and_not_the_arrangement` pins that on
`solve_turn` itself, and pre-meld the table is off limits entirely, so the reply is
arrangement-invariant there too. Among alternatives that shed the same *number* of
tiles, then, the door can only move where the alternative sheds a different *set*
of tiles -- never by rearranging what is already there.

And they almost always are rearranging. Over 15 deals of the same sweep, enumeration
only (`multiset_check.py` beside this file), **692 of 756 equal-tile alternatives --
91.5% -- shed exactly the same tiles**, and the reply moved for *none* of them, which
is the invariance measured rather than argued. Of the 64 that shed a different set,
it moved for 38. So the 99.67% figure above is not a small sample of a rare event: it
is 91.5% of the pairs where the door provably cannot move, plus a 60% hit rate on the
8.5% where it can.

That is why the denial cell is nearly empty, and it does not depend on the
opponent's strength: it holds against any opponent that can rearrange. The half that
*is* arrangement-sensitive is what a lay-off-only opponent actually does -- `frugal`
sheds differently in 5.97% of pairs -- and that half is null on the outcome as well
(+2.6% +-8.8% / -2.1% +-7.1%).

### Three seats: the argument that killed it at two does not apply, and the answer is the same

The 2p verdict rests partly on symmetry -- the table is common property, so a door
closed on the opponent is closed on the closer. With two opponents that is different:
a door costs the closer once and the opposition twice, and there is a *leader* to
aim at. So the whole sweep was re-run on `STANDARD_3P`, deviations examined at every
seat, the outcome read from the acting seat's own view: 200 deals over two pooled
runs of 100 (seed bases 73,000 and 74,000, k-best 4), 3,822 decisions, 20,104
alternatives, 1,352 s of the two runs together on six workers. Every check holds
there too: 3,712 re-derived turns reproducing their baselines exactly, 22,810 turns
compared against the oracle reply with 0 exceeding it, and the next seat taking the
whole reply in 96.0% of the turns it is offered one.

| delta vs the played turn | 2p, 300 deals | 3p, 200 deals |
|---|---|---|
| `cpsat_max` | +0.0% +-2.3% (1,248) | -1.3% +-3.0% (769) |
| `same_tiles_other_table` | -0.2% +-0.5% (15,801) | **-1.2% +-1.1% (7,280)** |
| `fewer_tiles` | -3.1% +-1.5% (5,886) | -1.0% +-1.6% (3,820) |
| `frozen_table` | -6.1% +-3.8% (818) | -5.8% +-4.2% (701) |
| `draw` | -0.4% +-1.1% (8,185) | -1.2% +-1.7% (3,822) |

The two-seat column is the re-measure under the solver's freeze fix -- `freeze_table`
used to let a joker take a real tile out of a frozen set in about one solve in eight,
so the arm was not quite frozen -- and it moved the row from 768 / -5.9% +-3.9% to
818 / -6.1% +-3.8%, inside its own interval. The three- and four-seat `frozen_table`
cells predate that fix and stand as measured; every other type is untouched by it.

**Every type is at or below zero at three seats as well, and the free parameter is
further below it than at two.** By phase, `same_tiles_other_table` reads -7.3%
pre-meld (261), +0.4% midgame (1,729), -0.3% endgame (5,290) -- the pre-meld cell
that read +6.4% at two seats reverses. The one cell that *does* survive the seat
count is the opening one, and only in the tiles split: an alternative shedding
**fewer** tiles than the played turn is worth **+5.4% +-5.8%** pre-meld at three
seats (369 alternatives) against **+8.1% +-6.2%** at two (384). Same sign, same
size, both about 1.5 sigma; everywhere else pre-meld the two seat counts disagree.

The denial cells, and this is the part the symmetry argument said might differ:

| the alternative leaves a table where the opponent... | 3p pairs | delta |
|---|---|---|
| **cannot go out, where it could before** | **28** | **+20.0% +-21.0%** |
| ...unchanged | 13,190 | -1.4% +-1.2% |
| ...can go out, where it could not before | 20 | -33.3% +-32.7% |
| cannot play at all, where it could before | 183 | -4.9% +-12.5% |
| actually shed fewer next turn | 457 | +0.8% +-8.7% |
| actually shed more next turn | 503 | -4.2% +-7.8% |

The exit cell is the same shape as at two seats, a little larger and just as rare:
**15 decisions in 200 deals, 0.07 per deal**, against 231 decisions (1.16 per deal)
where the opponent could go out against the table actually left -- so an equal-tile
table takes the exit away in **3.3%** of the pairs where there is one to take. And
the door belongs to the leader almost by definition: 27 of the 28 exit-closing pairs
are the seat with the fewest tiles (+21.4% +-22.3%), one is not. Split by *whose*
turn is next -- the only seat that meets the table as the deviation left it -- the
next seat's half reads +10.0% +-19.6% (14) and the later seat's +40.0% +-48.0% (14),
and the later seat's half is not a causal cell at all: two more turns are played onto
that table before it gets there.

Two 3p differences worth recording, neither of them denial. `play closed` is
**negative** at three seats (-4.9% +-12.5%) and so is its mirror (-8.4% +-12.1%):
both directions losing means that split is not measuring denial but "this
alternative differs from the played turn in a way that costs", which is the sanity
check the 2p cell also passes (+0.8% / -4.3%). And the observable proxy stops
pointing the right way entirely: mean permeability change is **+0.071** where the
exit closes against +0.116 where it does not, so at three seats permeability does
not even distinguish the cells, let alone score them (correlation with what the
opponent shed: +0.029; with the outcome: -0.010).

### Four seats, and the number the three runs agree on

`STANDARD_4P` at 150 deals (seed base 76,000, 2,330 decisions, 12,665 alternatives,
640 s) says the same thing again, and it is worth having because it is where the
per-turn maximum should be *weakest*: three opponents draw from one pool and a turn
of yours is three turns away from mattering. `same_tiles_other_table` reads **+0.4%
+-1.3%** over 4,111 alternatives -- the only positive point estimate in the three
runs and still covering zero -- while `fewer_tiles` (-8.3% +-2.0%) and
`frozen_table` (-12.1% +-3.6%) lose more heavily than anywhere else, so the harness
still has resolution at four seats. 14,050 turns compared against the oracle reply,
0 exceeding it, the next seat taking the whole reply 94.7% of the time.

The exit cell, all three seat counts side by side:

| exit-closing pairs | 2p (300 deals) | 3p (200) | 4p (150) |
|---|---|---|---|
| delta vs the played turn | +11.1% +-21.8% | +20.0% +-21.0% | +17.6% +-18.7% |
| pairs / decisions | 22 / 9 | 28 / 15 | 39 / 17 |
| decisions per deal | 0.03 | 0.07 | 0.11 |
| decisions where the exit was open at all | 188 (0.63/deal) | 231 (1.16) | 224 (1.49) |
| of those pairs, share an equal-tile table shuts | 3.0% | 3.3% | 3.4% |

**Three independent runs, +11% / +20% / +18%, and the same 3% availability.** That
consistency is the reason to believe the sign at all -- each interval alone covers
zero -- and the frequency is what closes it: per seat-game the cell is reachable
0.015 (2p), 0.025 (3p) and 0.028 (4p) times, so **an oracle that shut every exit it
could would move its own win rate by 0.3-0.5pp, and 1pp at the top of the
intervals**. At four seats the door stops being the leader's alone (16 leader pairs
against 23 that are not, +22.2% +-28.8% and +12.5% +-24.5%), which is the one thing
the seat count changes.

`play closed` is the control that shows how little of this is denial as such: it
reads +0.8% +-12.5% at two seats, **-4.9%** +-12.5% at three and **+13.2%** +-14.0%
at four. A cell that changes sign twice across seat counts, with its own mirror
losing in two of the three runs, is measuring "this alternative differs from the
played turn" and not a door.

### What this bounds

The same two limits as the section above, and two of its own.

It is perfect hindsight and it deviates on one turn. The door it prices is the door
as **CP-SAT** sees it, so an opponent whose search is weaker can be shut out of a
play CP-SAT would have found; the 97.3% / 96.0% / 94.7% figures say how little room
that leaves against `frugal` at two, three and four seats, but a weaker opponent, a
tighter micro budget or a rule that forbids rearrangement is a different question
and this does not answer it.

And the exit cell is **not "the same turn with the door shut"**. Because the reply is
arrangement-invariant, an alternative that shuts an exit has to shed a *different
set* of tiles of the same size -- so the +17% is the value of that whole different
turn, of which the shut door is one part. It is an upper bound on denial for the
same reason it is a bound on everything else here: the maximum is taken with the
future visible.

What is left open is not this axis. `same_tiles_other_table` is now measured at all
three seat counts (-0.2% +-0.5%, -1.2% +-1.1%, +0.4% +-1.3%), its targeted cells are
rare by structure rather than by sample size, and the observable proxy for them
carries +0.05, +0.03 and +0.05 of correlation with what the opponent could do.
**Board shaping is closed at every seat count the repo scores.** The nearest thing
to a surviving cell is the one the previous section already named -- shedding fewer
tiles than the played turn *pre-meld*: +8.1% +-6.2% at two seats, +5.4% +-5.8% at
three, and +1.3% +-5.7% at four, so two runs of three and a shrinking one. That is
an opening question, not a table question, and it is where the next arm belonged.
It ran, and it is null: nine opening policies over 1600 paired deals, best arm
+0.91% +-1.04, the negative control the only sign that repeats across seat counts
-- [the opening](#the-opening-was-the-one-cell-that-moved-and-it-does-not-survive-being-played),
the section below.

## The opening was the one cell that moved, and it does not survive being played

`tools/oracle_regret.py` priced every single-turn deviation in hindsight and found
none worth anything -- except one cell of fourteen. Pre-meld, an alternative
shedding **fewer** tiles than the maximum was worth **+9.3% +-5.7%**, and split
against what `frugal` actually opened with, shedding fewer was **+6.2** (n=384)
where CP-SAT's max-tiles opening was -1.1 (n=90); midgame and endgame the same
deviation is -0.8 and -8.3. The mechanism was available and specific: **the
opening is the one decision that does not recur.** Before melding the table is
untouchable, so the sets an agent opens with are handed to the opponent as
*rigid* sets, while a tile kept back is played later with full rearrangement
rights -- and every other decision in a turn recurs, which is exactly why
`rack_potential_ab` found nothing to protect (a set kept out of the rack is
played in the same turn anyway).

Tested directly, pre-sized, and pre-registered (`runs/opening-ab/`). Nine arms in
`rummi/agents/opening.py`, each `frugal` with the opening turn rebuilt and **every
post-meld decision delegated to `by_value` untouched**, so a score difference is
attributable to the opening alone -- `tests/test_opening.py` holds every arm to
`by_value`'s macro on post-meld states. Pre-meld the macro space collapses to
something plannable exactly: lay-offs and steals are illegal under
`strict_initial_meld` and `REPARTITION` is offered only after melding, so an
opening *is* a sequence of templates drawn from the rack.

Head to head against plain `frugal`, `STANDARD`, 1600 deals played from both
seats (3200 games per arm), paired per deal, seed base 93,000:

| arm | win | paired delta | opened tiles | value | sets | doors handed over | reply sheds | moves |
|---|---|---|---|---|---|---|---|---|
| `base` (frugal mirrored) | **exactly 50.00% +-0.00** | +0.00 | 5.71 | 43.86 | 1.75 | 6.42 of 8.80 | 2.58 | -- |
| `full` (by_value's opening, replanned) | **exactly 50.00% +-0.00** | +0.00 | 5.71 | 43.86 | 1.75 | 6.42 of 8.80 | 2.58 | 0.00% |
| `runs_first` | 50.91% +-1.04 | +0.91 +-1.04 | 4.47 | 37.68 | 1.36 | 5.88 of 8.26 | 2.45 | 8.31% |
| `min_sets` (stop at 30) | 50.56% +-1.02 | +0.56 +-1.02 | 4.13 | 36.88 | 1.25 | 5.18 of 7.59 | 2.40 | 5.98% |
| `minus_one` | 50.50% +-1.01 | +0.50 +-1.01 | 4.35 | 38.03 | 1.32 | 5.36 of 7.76 | 2.44 | 5.92% |
| `no_joker` | 50.25% +-0.49 | +0.25 +-0.49 | 5.61 | 42.81 | 1.74 | 6.37 of 8.75 | 2.55 | 2.12% |
| `min_tiles` (CP-SAT, fewest) | 50.12% +-1.22 | +0.12 +-1.22 | 3.96 | 35.50 | 1.29 | 5.18 of 7.55 | 2.36 | 12.54% |
| `cheap` | 49.69% +-1.02 | -0.31 +-1.02 | 5.62 | 40.77 | 1.82 | 6.77 of 9.11 | 2.52 | 10.80% |
| `groups_first` | 49.66% +-1.25 | -0.34 +-1.25 | 4.43 | 37.15 | 1.40 | 4.99 of 7.38 | 2.43 | 10.04% |
| `max_tiles` (negative control) | 49.47% +-0.89 | -0.53 +-0.89 | 5.95 | 44.87 | 1.88 | 6.81 of 9.15 | 2.57 | 9.92% |

**Null.** The pre-registered bar was a paired delta > +2pp with the interval
excluding zero, *and* the same sign against `optimal`; no arm reaches it and no
interval excludes zero. The whole ladder spans 1.44pp, from the negative control
at -0.53 to `runs_first` at +0.91, and the widest within-mechanism contrasts are
covered too: `min_sets` - `max_tiles` is **+1.09% +-1.34**, `min_tiles` -
`max_tiles` **+0.66% +-1.36**, `runs_first` - `groups_first` **+1.25% +-1.34**.
On `standard-greedy` at n=800 every arm reads 99.5-99.9% and +46.81 to +47.53
(+-1.5), which resolves nothing: `frugal` is already at 99.6% there.

**And no sign survives changing the arena.** Every remaining arena is the same
rotation, paired the same way:

| arm, paired delta | 2p vs `frugal` (1600) | 2p vs `optimal` (800) | 3p (800) | 4p (800) | pooled, 2p/3p/4p |
|---|---|---|---|---|---|
| `min_sets` | +0.56 +-1.02 | -0.56 +-1.51 | -0.67 +-1.14 | +0.56 +-0.84 | +0.26 +-0.56 |
| `minus_one` | +0.50 +-1.01 | -0.38 +-1.50 | -0.54 +-1.09 | +0.72 +-0.83 | +0.33 +-0.55 |
| `runs_first` | +0.91 +-1.04 | +0.44 +-1.56 | -0.46 +-1.17 | -- | +0.30 +-0.78 |
| `max_tiles` | -0.53 +-0.89 | -- | -0.29 +-0.95 | -0.50 +-0.79 | **-0.45 +-0.50** |
| `base` / `full` | exactly 0.00 | exactly 0.00 | exactly 0.00 | exactly 0.00 | -- |

`min_sets` and `minus_one` change sign between two seats and three, which is the
second half of the pre-registered criterion failing outright. **The three seats
matter for one specific reason: the hypothesis predicts the effect should get
*bigger* there**, since an opening is handed to two opponents rather than one, and
it goes the other way. The only sign that repeats in every arena is the negative
control's -- shedding *more* at the opening is mildly bad, three times out of
three -- and pooled across the three configs even that is **-0.45pp +-0.50**,
which is where this axis is bounded.

Two exactness controls carry the reading. `base` mirrored against itself is
*exactly* 50.00% and +0.00, so the rotation cancels turn order; and `full` --
`by_value`'s own opening rebuilt by the arms' own planner -- is exactly 50.00%
too, over **24,490** pre-meld decisions with 0.00% of them moved. So the planner
reproduces `by_value` byte for byte and the other arms differ from `frugal` in
their rule and in nothing else.

Four measurements say why, and the last two are the transferable part:

- **The arms are not inert.** They move 6.0-12.5% of pre-meld decisions and
  **change the outcome of 17.1-25.9% of deals** (`min_sets` 17.5%, `groups_first`
  25.9%, `no_joker` 4.0%). That caps what any of them could move the win rate by
  at ~9pp and rules out the reading that closed the two preceding sections: there
  *is* something to choose here, and choosing it differently is worth nothing.
- **The opening really does get smaller, and the opponent really does get less.**
  `min_sets` opens on 4.13 tiles against 5.71, declares 36.88 against 43.86, in
  1.25 sets against 1.75, and hands over **5.18 lay-off doors of the table's 7.59
  against 6.42 of 8.80**. The next seat then sheds **2.40 tiles against 2.58**, a
  7% reduction. The intended mechanism is present and measurable at every step.
- **And it pays for itself inside two turns.** Per opponent turn *after* the
  reply the effect is gone -- 1.141 tiles shed against `base`'s 1.146, across
  every arm -- while the arm's own shedding rises to **0.951 per turn against
  0.902** and the deal lengthens from 63.8 turns to 64.1. The kept tiles are
  played a turn or two later by the same hand that kept them, which is
  `rack_potential`'s recurring-decision finding reappearing one turn out rather
  than zero: the opening forecloses the *turn*, not the game.
- **Permeability does not order the result, so denial is not the channel.**
  `groups_first` hands over the tightest opening measured -- 4.99 doors, 1.4
  fewer than `frugal` -- and scores **-0.34**, while `runs_first` hands over 5.88
  and scores the highest at +0.91. That is the board-shaping section's conclusion
  arriving from the other side: a table-shaping rule pays its own cost, and here
  the sets you deny the opponent are the sets you were going to extend yourself.

One confound is controlled by construction and worth stating because it is not
free. `min_sets`, `minus_one` and `no_joker` open on **turn 3.58**, the identical
turn `frugal` does, because reordering within dearest-first cannot change whether
30 points are reachable -- so they are pure shape arms. The solver and reordering
arms open at 3.50-3.56: CP-SAT and cheapest-first find openings the dearest-first
greedy misses, which is a **capability** difference on top of the shape one. That
is why `min_tiles` - `max_tiles` (+0.66% +-1.36) is the clean contrast inside the
solver pair -- same capability, opposite tile count -- and it is null as well.

**What this does to the oracle's cell.** The hindsight measurement said an opening
shedding fewer tiles was worth +6.2pp over 384 alternatives spread across 242
deals; the same class of deviation, played as a policy over 1600 deals, is
+0.12% +-1.22 (`min_tiles`) and +0.56% +-1.02 (`min_sets`). The null is roughly
five times tighter than the positive cell and excludes it, so **the cell was one
of fourteen and it was noise** -- which is what the oracle section said about it
before this ran, and the reason this arm existed at all. The oracle bound stands:
no single-turn deviation from per-turn maximisation is worth anything on the
standard config, the opening included.

Caveats, all by construction. The arms are deterministic rules, so each deviates
on *every* opening -- a state-dependent rule that shrank only the openings worth
shrinking is not excluded by this, only bounded by the 17.5%-of-deals ceiling
above. And nothing here tests a *multi-turn* opening plan, holding tiles across
the meld for a combination two turns out: that is the cross-turn arm, and it sits
outside the one-turn construction exactly as the oracle's does.

## Attention over kinds: the routing is learned, used, and still loses

The first attention architecture tried in this repo, aimed where the measurement
said to aim it -- the cover head, which the beam buys and the break head does not.
The cover trunk reads `[need, avail, avail - need, scalars]` flat, so kind 7 and
kind 8 arrive at the first weight matrix unrelated, while a template asks for
exactly a relation between kinds. `set_encoder` reads that vector back as one token
per kind plus a summary and lets them attend before the pointer head scores.

Three arms, 90 epochs, `--seed 0`, cold start, the same eight shards, holdout split
and colour augmentation. **The break head's training loss is bit-identical across
all three at every epoch** (1.0805 at 10, 0.9721 at 23, 0.2791 at 90), so the arms
differ in the cover encoder and nothing else -- shown, not asserted.

| arm | params | greedy valid | greedy playable | beam 4 playable | greedy ms | kept |
|---|---|---|---|---|---|---|
| `enc-mlp-s0` | 879,577 | 42.5% | **31.9%** | **52.8%** | 2.78 | 34 |
| `enc-wide-s0` (`--cover-hidden 768`) | 1,283,545 | 41.1% | 29.6% | 47.8% | 2.78 | 63 |
| `enc-attn-s0` (2 layers, 4 heads, dim 128) | 870,105 | 36.0% | 27.9% | 49.1% | 4.24 | 19 |

On the 7,276-state holdout, so 4.0pp at greedy is outside the probe noise that
makes any single epoch unreadable.

**The mechanism is not inert, and that is the informative part.** Every information
null in this repo was confirmed by an ablation flipping 0.0% of argmax decisions --
the LSTM's cell state, the oracle's rack block. This one does not read like those
(1,200 decodes, 20,000 teacher-forced decisions):

| ablation | flips | teacher-forced | playable | ms |
|---|---|---|---|---|
| `attend` | -- | 73.7% | 29.0% | 4.24 |
| `uniform` (routing removed, tokens kept) | 35.0% | 60.4% | 20.2% | 4.47 |
| `self` (mixing removed; a floor, not a control) | 50.4% | 49.6% | 14.2% | 4.67 |
| vs `enc-mlp-s0` | 24.3% | 77.9% | 32.3% | 2.78 |

Content-based routing between kinds is **learned and load-bearing** -- removing it
alone flips 35.0% of cover decisions and costs 13.3pp of teacher-forced accuracy
and 8.8pp of playable, the largest ablation delta measured on any mechanism here.
The encoder does the thing it was built to do. It still loses to a flat MLP over
the same three count vectors, by 4.0pp at 1.53x the decode cost.

**It is not underfitting, and it is not the budget.** By epoch 90 attention has the
*lower* training loss (0.4230 against 0.4373) and the *worse* holdout step accuracy
(76.5% against 79.0%): it fits slower early (22.5% behind at epoch 23, 5.0% at 66),
overtakes, and generalises worse throughout. Its best decode came at **epoch 19 of
90**, with 30.8% at 33 and at 59 -- flat with noise, not climbing.

**Capacity is bracketed on both sides**, which is what makes this a statement about
the prior rather than about size. The attention arm is *smaller* than the plain
baseline (870k against 880k), and the 1.28M wide MLP fits best of the three (train
loss 0.3675) while scoring worst at greedy (29.6%). Width buys fit and loses the
deliverable, exactly as the `table_sets` null did.

**What it bounds.** The cover head's bottleneck is not how the multiset is read.
Two priors and a third arm with 46% more capacity land within 4pp of each other at
greedy, while beam 4 moves a single checkpoint by 20.9pp (31.9% -> 52.8%). The
resolution left in this component is in the decode and the labels, not the trunk.

Two limits worth stating. The `--init-cover` warm start was dropped for
comparability -- an attention cover cannot load a one-phase MLP state dict, so
warm-starting only the controls would have been the confound -- which costs ~8pp
absolute against the published 39.7% / 59.6%; these numbers rank arms and do not
restate the deliverable. And the picker is trained on 48,002 states with labels
purchasable at 20.7ms each, so **nothing here separates any arm from "more data
would have done the same"**. That cuts against an encoder *win*, which is why the
data-scaling curve gates one -- it does not rescue this loss, but a prior with more
capacity needing more data to pay is the one reading this run cannot exclude.

## The mixed-pool matrix measured nothing, and finding out why cost more than the matrix

**Adaptation only has content against a population.** A best response to one fixed
deterministic opponent is itself a fixed policy, so the arena for "does a policy
counter *this* opponent and not *that* one" is a pool. `--opponent` already takes a
comma-separated one and reports every rate per member, so the experiment needed no
plumbing: four best-response arms, identical seed and recipe, differing only in
whom they faced (`greedy` / `rearrange` / `frugal` / `optimal`), cross-scored on all
four columns. `frugal` earns its place in `OPPONENTS` here -- it is the only member
even with `optimal` head-to-head (48.7%, n=600) while getting there by a different
mechanism, so it separates "counters that opponent" from "that opponent is weaker".

`eval_macro --vs` seats an opponent in a frozen suite's own deals rather than
editing a suite, so the columns are paired on deals and no published score is
touched. Read at u40, argmax, entropy comparable (H 0.005-0.007), n=400:

| arm | vs `greedy` | vs `rearrange` | vs `frugal` | vs `optimal` |
|---|---|---|---|---|
| BR vs greedy | 99.5% / +48.43 | 98.5% / +33.17 | 51.5% / +0.48 | 51.7% / +1.20 |
| BR vs rearrange | 99.8% / +48.29 | 98.8% / +33.45 | 50.5% / +0.39 | 51.2% / +1.32 |
| clone (baseline) | 99.8% / +48.24 | 98.8% / +33.47 | 49.8% / -0.14 | 51.2% / +0.78 |

**Flat -- and worthless, because the arms are one policy.** `policy_divergence.py`
scores every arm's argmax over one driver's recorded rows: the four arms agree with
each other on **99.0% of 3,327 decisions**, and each sits **0.5-0.6% from its own
exact anchor** (reproduced by re-running the clone at the same seed, nll 0.0041
identical). Movement does not grow with training -- 0.5% at u20, 0.5% at u40, 0.6%
at u60 -- so it is not a matter of more updates. A flat row here says nothing about
adaptation; it says four copies of `by_value` score the same, which they must.

**This is the trap the denial writeup names** ("either it is worthless, or that
learner was signal-limited"), caught only because the movement control was
pre-registered. Without it the matrix would have shipped as a null.

### `--kl-coef` is not what pins it

The obvious suspect was the anchor: RL is KL-anchored to a clone that agrees with
`by_value` 99.9% of the time. Five rungs from a shared anchor, `--init-from` so
none re-pays the 200k-state gather:

| rung | moves from anchor | argmax vs `greedy` |
|---|---|---|
| kl 0.1, eps 0.05 | 0.5% | -- |
| kl 0.03 | 0.9% | -- |
| kl 0.01 | 0.3% | -- |
| kl 0.003 | 1.1% | -- |
| kl 0 (unanchored) | 1.1% | 100.0% / +47.90 |

Deleting the anchor entirely moves 1.1% and is **not monotone** (0.01 moves less
than 0.03), so the whole ladder is one noise band. Nothing collapsed either, which
qualifies the "unanchored PPO walks to the random floor" note above: that was the
primitive space: from a repartition-space clone, kl=0 scores +47.90 against the
anchor's +48.67.

### `--explore-eps` moves it, and past 0.2 it destroys it -- invisibly

| rung (kl 0) | moves | argmax vs `greedy` | H |
|---|---|---|---|
| eps 0.05 | 1.1% | 100.0% / +47.90 | 0.005 |
| eps 0.2 | 1.6% | 100.0% / +47.66 | 0.007 |
| eps 0.5 | 3.3% | **0.0% / -442.32, 100% stalemate** | 0.101 |

**The eps 0.5 arms are destroyed, and every number the trainer printed said
otherwise**: terminal +0.117 (the best of any rung), meld 90.0%, end 25%. Because
`--explore-eps` mixes uniform-over-legal into the rollout sampler, at 0.5 half of
every action is random and `terminal`/`meld`/`end` describe **the mixture, not the
policy**. A policy can rot to 100% stalemate with the training log improving. Nor
is it the diffuse-argmax artifact this file warns about elsewhere -- H is 0.089-0.101,
a mode carrying ~97% of the mass, and scored `--sample` the same checkpoint reads
59.2% / -113.06 with 40.5% stalemate against the anchor's 99.8% / +48.70. Sharpest
of all: sampled against **its own training opponent** it reads **26.8% / -135.44**
where the anchor reads 51.2%. Sixty updates of best response against `optimal`
made it worse at `optimal` than the policy it started from.

So the same recipe against the two hard opponents produced movement without
specialisation: vs `optimal` **11.6% moved, KL 0.262**, terminal -0.040 -> -0.037
over 60 updates; vs `frugal` 1.2% moved, KL 0.115, -0.035 -> -0.011. Equal
headroom (the anchor is 50.5% / 51.2% against them), very different divergence, no
gain in either -- and the larger divergence is the *degraded* arm, so it measures
rot, not adaptation.

### What this leaves

**The specialisation question is untested, not answered**, and the blocker is now
named: **everything that keeps the policy competent moves it at most 1.6%, and the
first setting that moves more destroys it.** The window has no measured middle. The
clone sits at a sharp optimum, and exploration large enough to leave its mode is
large enough to poison the update.

Two things to carry, both cheap and both learned the expensive way:
- **A flat training arm needs proof it moved** before its score is a null.
  `policy_divergence.py` is that proof, and it reports entropy and mean KL beside
  the flip rate because an argmax flip rate is the right instrument for a
  concentrated policy and a poor one for a diffuse one.
- **Any run with `--explore-eps` above a few percent must be scored by its argmax
  policy per checkpoint, never by the training terminal.** Three of four rungs here
  were unreadable without it.

## Settings measured as harmful

| setting | effect | why |
|---|---|---|
| `--entropy-coef 0.05` | -11.07 | Exploration wants to go **down** in the macro space. The 0.01 default is inherited from the primitive recipe and is too high; 0.003 was best. |
| `--epochs > 1` | +7.53 | See above. Defaults to 1. |
| `--clone` (macro space) | +22.59 | Helps on the primitive space, hurts here. |
| `--kl-coef` (macro space) | -0.4 | Mandatory on the primitive space, a constraint here: it pinned entropy to 0.001. |
| `--lr-decay` | **no effect** | Linear anneal to zero over 300 updates, five seeds against five without: +29.31 vs +29.10 at u40, -19.62 vs -24.71 at u100, -4.83 vs +10.13 at u300. Inside seed noise everywhere, and the collapse happens anyway. Kept because it is the first thing anyone asks of a PPO loop; the schedule is also gentle where it matters (87% of `--lr` still live at u40) so this is weak evidence against step size, not a refutation of it. |
| `--envs 1024` (macro space) | **no effect** | 16x the batch, three seeds. At u18 entropy is 1.17/1.17/1.29, indistinguishable from the 64-env runs at u20 (1.10-1.27) -- a policy given sixteen times the data per update is exactly as diffuse after the same number of updates. Concentration tracks **updates, not samples**, so the batch is 16x the cost per update for nothing. Scores from that run are not quotable: every checkpoint was in the entropy band where argmax is unreliable, below. |
| `--backend jax` (macro space) | **no effect** | 1332 against NumPy's 1362 dec/s at `--envs 256`. This trainer scores one env per forward pass and `legal_macros` is 28.5% of runtime against the policy's 13%, so the simulator is not what a larger batch waits on. The 1.6x in `CLAUDE.md` is under `train_ppo.py`, whose policy is batched. |

**Three explanations for the collapse, all refuted.** An entropy floor is
*backwards*: the two seeds that never fall are the ones whose entropy settles
**lowest** (0.26-0.29 against 0.46-0.69 for the three that do, one spiking to 1.225
at u200 as its policy comes apart). Step size does nothing, above. And it is not a
scoring artifact -- sampling the checkpoints instead of argmaxing them scores them
*worse*. `end_rate` and `draw_rate` are identical across all five seeds to within a
point, so no behaviour aggregate separates them either; only `terminal` moves, which
means the trainer can already see the collapse and simply acts on nothing. The
surviving hypothesis is the advantage's own noise: it is normalised over ~30 closed
episodes per update.

**Argmax scoring confounds how good a policy is with how concentrated it is.**
Every number in this file comes from `MacroAgent` taking the mode. Scored both ways
across the sweep (five seeds, n=480):

| update | argmax | sampled | entropy |
|---|---|---|---|
| u20 | -67.02 | **-3.73** | 1.163 |
| u40 | **+28.42** | +12.20 | 0.547 |
| u60 | +24.36 | +11.69 | 0.454 |
| u80 | +5.97 | -1.63 | 0.452 |
| u100 | -24.45 | -22.47 | 0.455 |
| u140 | -28.43 | -28.93 | 0.404 |
| u300 | +10.51 | -5.36 | 0.320 |

Above entropy ~1 the mode is nearly arbitrary -- ~20 legal macros, near-uniform --
and taking it *every* time loses every game while sampling the same distribution
plays normally. At u20 that is worth **63 points**, and a checkpoint reading exactly
`random`'s -441.94 is usually this rather than a policy that failed to learn. Below
~0.5 the sign flips and argmax leads by 8-16. Two consequences: **do not compare
checkpoints by argmax score unless their entropy is comparable and low**, and a
score at the random floor is a claim to check, not to believe.

**The collapse is real, though.** Sampled scoring falls the same way -- +12.20 at
u40 to -28.93 at u140 -- so it is behaviour and not measurement, and it remains
unexplained after four refuted causes.

**And the sampled policy never exceeds +12.20**, under half of `by_value`'s +28.01.
Every "the learned policy matches the heuristic" row above rests on the mode. A
deterministic agent is a legitimate agent, so +28.42 stands as its score -- but the
accurate statement is narrower than it looks: *the single most likely action is
usually as good as `by_value`'s pick, while the distribution around it is much
worse.* The policy has not learned to play at the heuristic's level; it has learned
a mode that does.

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

**That thread is pulled, and it was a false positive (three arms, checkpointed
every 20 and scored to u160).** `micro_step_cost` is charged on every
`PLACE`/`PICK`/`DISSOLVE`/`ASSIGN` and never on a committing action, so it prices
playing exactly as it prices dithering -- a macro expands into 5-7 charged
micro-actions while `DRAW` costs nothing and reverts the turn. The optimum it
creates is "never touch a tile", and the arm walks straight to it: `draw` 4% ->
100% by u40, entropy 0.000, and the 354 episodes "finishing" per update are all
instant passes at terminal ~-1.05. The unshaped control fails the opposite way --
98.9% of decisions in the primitive block, turns held open to the budget. Every
checkpoint of both arms scores exactly `random`'s -441.94 at 100% stalemate.

**Warm-starting from a trained macro net is exact, and inert.** `--init-from-macro`
maps a macro-space pointer checkpoint into the hybrid net -- trunk, query and
critic whole, the action-shaped tensors by `macro_to_hybrid_actions` -- and the
warm-started net's logits over all 713 macro-space actions are bitwise the
source's (a parity test pins it; the softmax denominator, now shared with 2398
primitives, is the one unavoidable difference). The policy demonstrably arrives
knowing how to play a macro: 55.9% of its mass sits on the macro block where one
is on offer. It scores -442.32 at every checkpoint anyway, because **a macro is on
offer in 1.6% of its decisions** against the 14.7% measured under uniform random:
uniform play draws ~2.9% of the time and keeps workbenches shallow, while this
policy draws 0.38%, runs every turn to the 155-micro cap, and a workbench that
deep fits inside no single macro. The teacher's states stop occurring, the
untrained primitives are all that is left, and PPO walks the macro preference off
by u70. A rack-shaping arm keeps the preference alive (57.1% at u60) and still
scores the floor -- preserving macro competence is not the binding constraint.

**What actually blocks the space is a conjunction the mask makes visible.**
`end_turn & playable` means a turn that burns its micro budget has `END_TURN`
masked and can only `DRAW`; the exploration target is clear-the-workbench *and*
reach 30 points *and* commit, which a near-uniform policy over 2400 primitives
hits ~0.05% of the time -- too rare to reinforce before the entropy bonus dies.
Two threads survive, neither yet tried: a term that pays for *committing* rather
than taxing every tile touched (SPEC section 7's `micro_step_cost` cannot separate
the two; a trainer-side bonus on `END_TURN` needs no env change), and keeping
macro-bearing states reachable at all -- a macro spanning more than one set, a
workbench-clearing move, or a shorter micro budget.

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
