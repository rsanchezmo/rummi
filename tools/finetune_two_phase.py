"""Stage two on the two-phase picker: the same self-critical fine-tune, over both phases.

    python tools/train_two_phase.py --data checkpoints/repartitions-s*.npz ...   # stage one
    python tools/finetune_two_phase.py --data checkpoints/repartitions-s*.npz \
        --init checkpoints/twophase-init.pt --updates 1500 --batch 256 \
        --valid-bonus 0 --advantage positive --samples 4 --beam 1 4 --eval-games 240

Two independent fixes took the greedy repartition decode from 72.6% of CP-SAT's
contribution to ~81%, and they were designed against different diagnoses.
`tools/train_two_phase.py` shortened the sequence -- break ~3 slots, then cover the
freed tiles with ~4 templates, 8.88 decisions against 13.6. `tools/finetune_repartition.py`
improved selection -- self-critical policy gradient with the best of four sampled
constructions kept, which puts the beam's own comparison inside the weights. Neither
had been applied to the other. This composes them.

It is a sibling rather than a flag on `finetune_repartition.py` for two reasons. The
recipe is imported, not restated: `run_decode`, `score_sequences`, `Starts` and
`tile_price` come from that file, so phase B *is* the one-phase differentiable
decode, and reward, advantage handling and the KL anchor are the same code rather
than a second copy of it. And what changes is the rollout and the report -- eight
branch points in a `main` whose one-phase numbers are published.

What the composition has to get right is the rollout: a break and a cover are one
trajectory carrying one reward, so both phases are sampled with log-probs tracked
and the advantage lands on both. `run_break` is phase A's differentiable twin, and
`test_finetune_two_phase.py` holds the pair to `decode_two_phase` at beam 1 -- the
same parity the one-phase tool rests on.

Three carried findings, none re-measured here:

- **No validity bonus.** In the two-phase space the identity cover wears a different
  hat -- phase A can emit `STOP` immediately, and "break nothing" is then a valid
  repartition that sheds nothing. `--valid-bonus 0` is the default here (it is 0.1
  there), and the break-count distribution is reported at every probe so a collapse
  onto it would be visible rather than inferred.
- **`--samples 4`.** One sample per state stalled four different one-phase recipes at
  32-33%; selection among finished constructions is the whole mechanism, and one
  sample gives the update nothing to select from.
- **`--advantage positive`.** Dropping the negative half matched full signed updates
  at a sixth of the drift.

`--stuck` is not carried over. Unlabelled gate states are collectable but inert
under a tiles-only reward -- nothing plays, so every sample and the baseline score
zero and the advantage is exactly zero -- and the two-phase reward is the same one.

**What it measured.** 1,500 updates of 256 states from `checkpoints/twophase-init.pt`,
kept at update 1,425, on the same shards and the same game-split holdout both parents
report (7,276 states); one seed per row. The two-phase parent's rows are re-measured
here with `--updates 0` and come back at its published figures to the decimal, so the
composed rows are read against a control this harness produced:

| arm | valid | plays | valid, beam 4 | plays, beam 4 |
|---|---|---|---|---|
| one-phase imitation | 49.8% | 27.8% | 75.9% | 46.5% |
| one-phase + RL best-of-4 | 57.2% | 35.9% | 80.5% | 55.4% |
| two-phase imitation | 49.7% | 35.5% | 86.4% | 56.2% |
| **two-phase + RL best-of-4** | 56.2% | **39.7%** | 89.4% | **59.6%** |
| ... cover head only, `--freeze break` | 56.2% | 37.9% | 90.3% | 58.1% |
| ... break head only, `--freeze cover` | 51.3% | 37.5% | 86.6% | 57.1% |

`checkpoints/twophase-rl4.pt` is the composed row; the two controls are
`twophase-rl4-cover.pt` and `twophase-rl4-break.pt`. On `standard-greedy` at n=480
(`--eval-games 240`), every arm scored in this run:

| arm | score | win | ms/state | of CP-SAT's 19.70 points |
|---|---|---|---|---|
| `by_value` | +28.01 | 82.1% | -- | 0% |
| two-phase imitation, greedy | +43.88 | 99.2% | 1.6 | 80.6% |
| **two-phase + RL, greedy** | **+45.63** | 99.8% | 1.7 | **89.4%** |
| two-phase imitation, beam 4 | +45.86 | 99.8% | 3.8 | 90.6% |
| two-phase + RL, beam 4 | +44.05 | 99.4% | 4.0 | 81.4% |
| `by_value+repartition` (CP-SAT) | +47.71 | 99.8% | -- | 100% |

**The two gains partially overlap; they do not add.** The fine-tune is worth +8.1pp
of greedy playability on the one-phase picker (27.8% -> 35.9%) and +4.2pp on the
two-phase one (35.5% -> 39.7%): **52% of its standalone effect survives** the
shorter sequence, and adding would have predicted 43.6%. Beam 4 keeps less again,
+8.9pp against +3.4pp, or 38%. What overlaps is what both fixes were aiming at --
the composed rollout still finds a better-shedding sample than its own greedy arm on
**19.9%** of states against the one-phase run's 22.6%, so the *headroom* barely
narrowed; what narrowed is how much of it a shorter sequence had left to collect.

`train_two_phase.py --epochs 0 --init-from` scores the composed checkpoint at the
same 56.2% / 39.7%, so the number is the decoder's and not this harness's -- and it
adds the signature the repo has seen before: imitation step accuracy *falls* while
the decode improves, break 71.3% -> 70.4% and cover 82.7% -> 81.9%. Predicting the
teacher's next decision and constructing a repartition remain different skills.

It is still the strongest greedy repartition decode measured. 39.7% beats both
parents by ~4pp, and on the ruler the greedy arm reaches 89.4% of the solver's
contribution -- just short of the >=90% that previously took beam 4, at 2.2x its
speed. The beam-4 row moving the other way (+45.86 -> +44.05 while its holdout
playability rises 56.2% -> 59.6%) is the suite's resolution, not a regression: a
paired two-point move is inside its 95% CI at 240 deals, and 7,276 states are many
times their own noise. Read the ruler as consistent with the holdout.

**Phase A and phase B do not want different treatment, and that is the surprise.**
The reward is the cover's -- the kept slots shed nothing by definition -- so the
break head is trained only by what the cover made of the tiles it freed, and the
obvious failure mode was for that signal to vanish. It does not. The break head is
credited on **67%** of its decisions against the cover's 83%, carries a comparable
gradient (0.36 against 0.46), and frozen one at a time the two heads are worth
**+2.4pp** (cover) and **+2.0pp** (break) of the composed **+4.2pp** -- so *within*
the composition the heads are very nearly additive. They differ in what they buy,
not in how much: the cover head moves validity 49.7% -> 56.2% and the break head
barely moves it (51.3%) while moving playability just as far. And the break head is
the *less* decided of the two, 0.94 nats per decision against the cover's 0.46.

**Break-nothing never happens.** The trap is real in this space -- phase A can emit
`STOP` before any slot, phase B can then `STOP` on an uncovered-nothing table, and
`build` accepts it as a valid repartition worth exactly zero -- and across the three
runs' 1.15M greedy rollouts it fired **zero** times. The fine-tune moves the other
way: on 2,000 holdout states the greedy break count goes 2.29 -> 2.79 and the
distribution shifts off its floor (1 break 33.3% -> 22.3%, 4-or-more 14.2% ->
25.8%), toward the 2.89 the teacher's own decomposition breaks. Nothing pays for
validity here, and validity rose anyway, as a by-product of shedding more.
"""

from __future__ import annotations

import argparse
import copy
import json
import pathlib
import sys
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from rummi.agents.learned.two_phase_net import (
    BreakNet,
    TwoPhaseNet,
    TwoPhaseScorer,
    break_dynamic,
    break_feasible,
    break_state_features,
    decode_two_phase,
    slot_counts,
    slot_present,
    slot_static,
    stop_break,
    two_phase_from_checkpoint,
)
from rummi.agents.macro import MacroAgent, by_value
from rummi.evaluate.protocol import evaluate, suite_for
from rummi.rules.config import CONFIG_BY_NAME, RummiConfig

# Read from where they are defined rather than restated: the holdout metric has to
# be the number both parents reported, the reward and the anchor have to be the
# one-phase fine-tune's own, and the scored arm has to differ from stage one's only
# in the weights.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from finetune_repartition import (
    Starts,
    Trajectories,
    run_decode,
    score_sequences,
    tile_price,
)
from train_repartition import Steps
from train_two_phase import TwoPhaseRepartition, playable_rate

Decisions = tuple[list[tuple[int, ...]] | None, list[tuple[int, ...]] | None]
"""One rollout's decisions, phase A then phase B -- what a baseline or a replay needs.

Either half may be `None`, which leaves that phase to decode itself: forcing the
break and letting the cover answer it is how the credit rule is tested."""


@dataclass(frozen=True, slots=True)
class Slots:
    """The table as slots, derived once per update: phase A's whole candidate block.

    Everything here is a function of the rack and the board alone, so the greedy
    arm, the `--samples` draws and the replay of the winner all read one copy. It is
    per batch rather than per pool because `present` is a template count per slot --
    the whole training pool would be gigabytes of it.
    """

    rack: np.ndarray
    """`(B, K)`."""
    counts: np.ndarray
    """`(B, S, K)` tiles per slot."""
    static: np.ndarray
    """`(B, S + 1, K + 6)` slot descriptions, `STOP` last."""
    present: np.ndarray
    """`(B, S, T)` which templates each slot already reads as."""
    table: np.ndarray
    """`(B, K)` the whole table."""
    occupied: np.ndarray
    """`(B,)` slots holding anything."""

    def __len__(self) -> int:
        return len(self.rack)


def break_starts(cfg: RummiConfig, racks: np.ndarray, boards: np.ndarray) -> Slots:
    counts = slot_counts(cfg, boards).astype(np.int64)
    return Slots(
        rack=np.asarray(racks).astype(np.int64),
        counts=counts,
        static=slot_static(cfg, counts),
        present=slot_present(cfg, boards),
        table=counts.sum(1),
        occupied=(counts.sum(-1) > 0).sum(-1),
    )


def cover_starts(slots: Slots, subsets: list[tuple[int, ...]]) -> Starts:
    """Phase B's construction start, from the subsets phase A dissolved.

    `decode_two_phase` derives this per subset with `freed_counts` and
    `present_counts`; over a batch every term is a masked sum of the slot block, so
    it is one gather rather than a Python loop per rollout. `base` is the kept
    slots, which is what makes `max_sets` count the whole table.
    """
    mask = np.zeros(slots.counts.shape[:2], dtype=bool)
    for row, broken in enumerate(subsets):
        if broken:
            mask[row, list(broken)] = True
    taken = mask[:, :, None]
    need = (slots.counts * taken).sum(1)
    return Starts(
        need=need,
        avail=need + slots.rack,
        present=(slots.present * taken).sum(1),
        base=slots.occupied - mask.sum(1),
    )


def run_break(
    cfg: RummiConfig,
    net: BreakNet,
    slots: Slots,
    *,
    sample: bool,
    generator: torch.Generator | None = None,
    grad: bool = False,
    clone: BreakNet | None = None,
    baseline: list[tuple[int, ...]] | None = None,
    temperature: float = 1.0,
    forced: list[tuple[int, ...]] | None = None,
) -> Trajectories:
    """Dissolve slots, one per round per state, all of them in lockstep.

    The differentiable twin of `two_phase_net._break_round` at beam 1: same
    features, same mask, same `last`-at-start convention, so `sample=False`
    reproduces the subset a beam-1 `decode_two_phase` breaks. Every argument means
    what it means in `run_decode`, which this mirrors deliberately -- the two loops
    are one rollout and the composition reads better for their being the same shape.

    One thing is simpler than the cover's loop: `break_feasible` always leaves
    `STOP` legal, so no row can be closed by the mask and there is no abandoned
    construction to drop before the forward pass.
    """
    stop = stop_break(cfg)
    size = len(slots)
    freed = np.zeros((size, cfg.n_kinds), dtype=np.int64)
    last = np.full(size, -1, dtype=np.int64)
    n_broken = np.zeros(size, dtype=np.int64)
    alive = np.ones(size, dtype=bool)
    sequences: list[list[int]] = [[] for _ in range(size)]
    actions: list[list[int]] = [[] for _ in range(size)]
    if baseline is None:
        left = np.ones(size, dtype=bool)
    else:
        left = np.zeros(size, dtype=bool)

    picked_rows: list[torch.Tensor] = []
    picked_logp: list[torch.Tensor] = []
    entropy_sum = torch.zeros((), dtype=torch.float32)
    kl_sum = torch.zeros((), dtype=torch.float32)
    steps = credited = 0

    for depth in range(cfg.max_sets + 1):
        if forced is not None:
            alive &= np.array([depth < len(row) for row in forced], dtype=bool)
        rows = np.flatnonzero(alive)
        if rows.size == 0:
            break
        counts = slots.counts[rows]
        racks = slots.rack[rows]
        pool = freed[rows]
        where = last[rows]
        state = break_state_features(
            cfg, racks, slots.table[rows], pool, slots.occupied[rows], n_broken[rows], where
        )
        static = slots.static[rows]
        dynamic = break_dynamic(cfg, counts, racks, pool, where)
        legal = break_feasible(cfg, counts, where)

        state_t = torch.from_numpy(state)
        static_t = torch.from_numpy(static)
        dynamic_t = torch.from_numpy(dynamic)
        legal_t = torch.from_numpy(legal)
        with torch.set_grad_enabled(grad):
            logits = net(state_t, static_t, dynamic_t, legal_t)
            logp = torch.log_softmax(logits, dim=-1)
            probs = logp.exp()
            entropy_sum = entropy_sum + -(probs * logp).sum(-1).sum()
            if clone is not None:
                with torch.no_grad():
                    reference = torch.log_softmax(
                        clone(state_t, static_t, dynamic_t, legal_t), dim=-1
                    )
                kl_sum = kl_sum + (probs * (logp - reference)).sum(-1).sum()

        if forced is not None:
            action = torch.from_numpy(np.array([forced[row][depth] for row in rows]))
        elif sample:
            draw = (
                probs.detach()
                if temperature == 1.0
                else torch.softmax(logits.detach() / temperature, -1)
            )
            action = torch.multinomial(draw, 1, generator=generator).squeeze(-1)
        else:
            action = logp.detach().argmax(-1)
        chosen = action.numpy()
        steps += len(rows)

        if baseline is not None:
            taken = np.array(
                [baseline[row][depth] if depth < len(baseline[row]) else -1 for row in rows]
            )
            left[rows] |= chosen != taken
        credit = left[rows]
        if credit.any():
            keep = torch.from_numpy(np.flatnonzero(credit))
            picked_rows.append(torch.from_numpy(rows[credit]))
            picked_logp.append(logp[keep].gather(1, action[keep, None]).squeeze(-1))
            credited += int(credit.sum())

        for local, (row, slot) in enumerate(zip(rows.tolist(), chosen.tolist(), strict=True)):
            actions[row].append(slot)
            if slot == stop:
                alive[row] = False
                continue
            sequences[row].append(slot)
            freed[row] = freed[row] + counts[local, slot]
            last[row] = slot
            n_broken[row] += 1

    total = torch.zeros(size, dtype=torch.float32)
    if picked_rows:
        total = total.index_add(0, torch.cat(picked_rows), torch.cat(picked_logp))
    divisor = max(steps, 1)
    return Trajectories(
        sequences=[tuple(s) for s in sequences],
        actions=[tuple(a) for a in actions],
        logp=total,
        entropy=entropy_sum / divisor,
        kl=kl_sum / divisor,
        steps=steps,
        credited=credited,
        diverged=left,
    )


@dataclass(slots=True)
class Roll:
    """One break-then-cover rollout: two decision sequences, one reward.

    The reward is the cover's -- the kept slots shed nothing by definition -- so the
    break head is trained entirely by what the cover it hands over managed to do
    with the freed tiles. That is the composition's one structural asymmetry, and
    `credited_a` is what measures whether it carries signal.
    """

    brk: Trajectories
    cover: Trajectories
    starts: Starts

    @property
    def decisions(self) -> Decisions:
        return self.brk.actions, self.cover.actions

    @property
    def logp(self) -> torch.Tensor:
        return self.brk.logp + self.cover.logp

    @property
    def steps(self) -> int:
        return self.brk.steps + self.cover.steps

    @property
    def entropy(self) -> torch.Tensor:
        """Per decision over both phases, which is what the one-phase run reports."""
        pooled = self.brk.entropy * self.brk.steps + self.cover.entropy * self.cover.steps
        return pooled / max(self.steps, 1)

    @property
    def kl(self) -> torch.Tensor:
        pooled = self.brk.kl * self.brk.steps + self.cover.kl * self.cover.steps
        return pooled / max(self.steps, 1)

    @property
    def credited_fraction(self) -> float:
        return (self.brk.credited + self.cover.credited) / max(self.steps, 1)


def head_norm(head: nn.Module) -> float:
    """One head's gradient norm, read before the global clip clamps them together."""
    total = sum(float(q.grad.pow(2).sum()) for q in head.parameters() if q.grad is not None)
    return total**0.5


def run_two_phase(
    cfg: RummiConfig,
    net: TwoPhaseNet,
    slots: Slots,
    *,
    monotone: bool,
    sample: bool,
    generator: torch.Generator | None = None,
    grad: bool = False,
    clone: TwoPhaseNet | None = None,
    baseline: Decisions | None = None,
    temperature: float = 1.0,
    forced: Decisions | None = None,
) -> Roll:
    """Break, then cover what it freed -- one trajectory across the two heads.

    Phase A's divergence from the baseline is handed to phase B, so a row that broke
    a different subset credits its whole cover: every template it then picks is
    conditioned on tiles the greedy arm never freed.
    """
    brk = run_break(
        cfg,
        net.breaker,
        slots,
        sample=sample,
        generator=generator,
        grad=grad,
        clone=None if clone is None else clone.breaker,
        baseline=None if baseline is None else baseline[0],
        temperature=temperature,
        forced=None if forced is None else forced[0],
    )
    starts = cover_starts(slots, brk.sequences)
    cover = run_decode(
        cfg,
        net.cover,
        starts,
        monotone=monotone,
        sample=sample,
        generator=generator,
        grad=grad,
        clone=None if clone is None else clone.cover,
        baseline=None if baseline is None else baseline[1],
        temperature=temperature,
        forced=None if forced is None else forced[1],
        diverged=brk.diverged,
    )
    return Roll(brk, cover, starts)


def break_profile(
    cfg: RummiConfig,
    data: Steps,
    scorer: TwoPhaseScorer,
    states: np.ndarray,
    beam: int,
    monotone: bool,
) -> dict[str, float]:
    """How much of the table the decode actually dissolves -- the break-nothing check.

    A repartition that breaks nothing is trivially valid and worth exactly zero, so
    it is the reward's nearest bad answer and the shape a collapse would take. The
    counts come out of the returned `Repartition`: its `templates` are the sets phase
    B built and the rest of its `sets` are the slots phase A kept.
    """
    broken: list[int] = []
    built: list[int] = []
    for i in states.tolist():
        found = decode_two_phase(cfg, scorer, data.rack[i], data.board[i], beam, monotone)
        if found is None:
            continue
        kept = len(found.sets) - len(found.templates)
        occupied = int((np.asarray(data.board[i]) >= 0).any(-1).sum())
        broken.append(occupied - kept)
        built.append(len(found.templates))
    counts = np.asarray(broken, dtype=np.float32)
    return {
        "valid": len(broken) / max(len(states), 1),
        "breaks": float(counts.mean()) if len(counts) else 0.0,
        "breaks_zero": float((counts == 0).mean()) if len(counts) else 0.0,
        "builds": float(np.mean(built)) if built else 0.0,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="standard", choices=sorted(CONFIG_BY_NAME))
    p.add_argument("--data", type=pathlib.Path, nargs="+", required=True)
    p.add_argument("--init", type=pathlib.Path, required=True, help="the two-phase checkpoint")
    p.add_argument(
        "--updates", type=int, default=400,
        help="0 scores the checkpoint as it stands, which is how the parent's row of "
             "the table is produced in this harness rather than a different one",
    )
    p.add_argument("--batch", type=int, default=256, help="states rolled out per update")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--kl-coef", type=float, default=0.1)
    p.add_argument("--entropy-coef", type=float, default=0.001)
    p.add_argument("--reward", default="value", choices=("value", "count"))
    p.add_argument(
        "--valid-bonus", type=float, default=0.0,
        help="0 by default, where the one-phase tool defaults to 0.1: paying for a "
             "valid construction buys the identity cover, and in this space phase A "
             "can reach it in one decision by breaking nothing",
    )
    p.add_argument(
        "--samples", type=int, default=4,
        help="rollouts drawn per state, of which the best-shedding one is the "
             "sample. Both phases are re-drawn, because a break is only worth what "
             "the cover after it manages",
    )
    p.add_argument("--advantage", default="positive", choices=("signed", "positive"))
    p.add_argument("--sample-temp", type=float, default=1.0)
    p.add_argument(
        "--freeze", default="none", choices=("none", "break", "cover"),
        help="hold one head at its imitation weights, still sampled and still part "
             "of the rollout. The reward is the cover's, so which head the gain "
             "comes from is a control rather than an inference from a gradient norm",
    )
    p.add_argument(
        "--credit", default="diverged", choices=("diverged", "all"),
        help="`diverged` drops the prefix the greedy arm also walked, across both "
             "phases: a break the greedy arm also made cannot have caused the "
             "difference in reward, and neither can a cover that followed it",
    )
    p.add_argument("--beam", type=int, nargs="+", default=[1, 4])
    p.add_argument("--probe", type=int, default=400, help="holdout states decoded per probe")
    p.add_argument("--probe-every", type=int, default=25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-games", type=int, default=0)
    p.add_argument("--baseline-arms", action="store_true", help="score by_value and CP-SAT too")
    p.add_argument("--out", type=pathlib.Path, default=None)
    p.add_argument("--log-json", type=pathlib.Path, default=None)
    args = p.parse_args()
    if args.samples < 1:
        p.error("--samples is the number of rollouts drawn per state, so at least one")

    cfg = CONFIG_BY_NAME[args.config]
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    price = tile_price(cfg)

    checkpoint = torch.load(args.init, weights_only=True)
    monotone = bool(checkpoint["monotone"])
    net = two_phase_from_checkpoint(cfg, checkpoint)
    clone = copy.deepcopy(net).eval()
    for q in clone.parameters():
        q.requires_grad_(False)
    scorer = TwoPhaseScorer(net)
    held = {"break": net.breaker, "cover": net.cover}.get(args.freeze)
    if held is not None:
        for q in held.parameters():
            q.requires_grad_(False)
    opt = torch.optim.Adam([q for q in net.parameters() if q.requires_grad], lr=args.lr)

    began = time.perf_counter()
    # `Steps` is the one-phase dataset, and it is the right one: the holdout split is
    # the same game split both parents report, and the two-phase probe reads only the
    # rack, the board and the solver's own tile count off it.
    data = Steps(cfg, args.config, list(args.data), monotone)
    train_states = np.flatnonzero(~data.holdout)
    held_states = np.flatnonzero(data.holdout)
    racks = data.rack[train_states]
    boards = data.board[train_states]
    print(
        f"config={args.config} init={args.init} order={'template' if monotone else 'free'}\n"
        f"train {len(train_states):,} states  holdout {len(held_states):,} states  "
        f"({time.perf_counter() - began:.0f}s to materialise)",
        flush=True,
    )

    def probe(rows: np.ndarray, beam: int) -> dict[str, float]:
        net.eval()
        out = playable_rate(cfg, data, scorer, rows, beam, monotone)
        out.update(break_profile(cfg, data, scorer, rows, beam, monotone))
        net.train()
        return out

    # One slice, drawn once: every probe then measures the same states, so the
    # trajectory is a paired comparison rather than 400 fresh states of noise per
    # reading -- and the update kept is chosen on a difference that is real.
    probe_rows = rng.permutation(held_states)[: args.probe]
    baseline = probe(probe_rows, 1)
    print(
        f"before  probe valid {baseline['valid']:.1%}  playable {baseline['playable']:.1%}  "
        f"breaks {baseline['breaks']:.2f} ({baseline['breaks_zero']:.1%} none)",
        flush=True,
    )

    history: list[dict] = []
    best_probe, best_update = baseline["playable"], 0
    best_state = {k: v.clone() for k, v in net.state_dict().items()}
    order = rng.permutation(len(racks))
    cursor = 0
    net.train()
    started = time.perf_counter()
    for update in range(1, args.updates + 1):
        if cursor + args.batch > len(order):
            order, cursor = rng.permutation(len(racks)), 0
        index = order[cursor : cursor + args.batch]
        cursor += args.batch
        slots = break_starts(cfg, racks[index], boards[index])

        with torch.no_grad():
            greedy = run_two_phase(cfg, net, slots, monotone=monotone, sample=False)
        r_greedy, valid_greedy, played_greedy = score_sequences(
            cfg, greedy.starts, greedy.cover.sequences,
            mode=args.reward, valid_bonus=args.valid_bonus, price=price,
        )
        anchor = greedy.decisions if args.credit == "diverged" else None

        # The beam's own trick, moved inside the weights: draw several whole
        # rollouts and keep the one that sheds most. Only the winner is replayed
        # with a tape, because K tapes buy one gradient. A reward is never negative,
        # so the -1 start makes the first draw win every row without a special case.
        best_a: list[tuple[int, ...]] = [()] * len(slots)
        best_b: list[tuple[int, ...]] = [()] * len(slots)
        r_sample = np.full(len(slots), -1.0, dtype=np.float32)
        valid_sample = np.zeros(len(slots), dtype=bool)
        played_sample = np.zeros(len(slots), dtype=np.int64)
        for _ in range(args.samples):
            with torch.no_grad():
                drawn = run_two_phase(
                    cfg, net, slots, monotone=monotone, sample=True,
                    generator=generator, temperature=args.sample_temp,
                )
            reward, valid, played = score_sequences(
                cfg, drawn.starts, drawn.cover.sequences,
                mode=args.reward, valid_bonus=args.valid_bonus, price=price,
            )
            better = reward > r_sample
            for i in np.flatnonzero(better).tolist():
                best_a[i] = drawn.brk.actions[i]
                best_b[i] = drawn.cover.actions[i]
            r_sample = np.where(better, reward, r_sample)
            valid_sample = np.where(better, valid, valid_sample)
            played_sample = np.where(better, played, played_sample)
        sampled = run_two_phase(
            cfg, net, slots, monotone=monotone, sample=False,
            grad=True, clone=clone, baseline=anchor, forced=(best_a, best_b),
        )

        advantage = torch.from_numpy(r_sample - r_greedy)
        weight = advantage.clamp(min=0.0) if args.advantage == "positive" else advantage

        policy = -(weight * sampled.logp).mean()
        loss = policy + args.kl_coef * sampled.kl - args.entropy_coef * sampled.entropy
        opt.zero_grad(set_to_none=True)
        loss.backward()
        # Per head before the global clip: the reward is the cover's, so whether any
        # of it reaches the break head is the composition's open question.
        norm_a = head_norm(net.breaker)
        norm_b = head_norm(net.cover)
        grad_norm = float(nn.utils.clip_grad_norm_(net.parameters(), 1.0))
        opt.step()

        broke_greedy = np.array([len(s) for s in greedy.brk.sequences])
        broke_sample = np.array([len(s) for s in sampled.brk.sequences])
        row = {
            "update": update,
            "loss": float(loss.detach()),
            "policy_loss": float(policy.detach()),
            "entropy": float(sampled.entropy.detach()),
            "entropy_break": float(sampled.brk.entropy.detach()),
            "entropy_cover": float(sampled.cover.entropy.detach()),
            "kl": float(sampled.kl.detach()),
            "kl_break": float(sampled.brk.kl.detach()),
            "kl_cover": float(sampled.cover.kl.detach()),
            "grad_norm": grad_norm,
            "grad_norm_break": norm_a,
            "grad_norm_cover": norm_b,
            "reward_sample": float(r_sample.mean()),
            "reward_greedy": float(r_greedy.mean()),
            "advantage": float(advantage.mean()),
            "advantage_std": float(advantage.std()),
            "batch_valid_greedy": float(valid_greedy.mean()),
            "batch_playable_greedy": float((played_greedy >= 1).mean()),
            "batch_playable_sample": float((played_sample >= 1).mean()),
            "credited": sampled.credited_fraction,
            "credited_break": sampled.brk.credited / max(sampled.brk.steps, 1),
            "credited_cover": sampled.cover.credited / max(sampled.cover.steps, 1),
            "breaks_greedy": float(broke_greedy.mean()),
            "breaks_sample": float(broke_sample.mean()),
            "breaks_zero_greedy": float((broke_greedy == 0).mean()),
            "same_break": float(
                np.mean([a == b for a, b in zip(greedy.brk.sequences, sampled.brk.sequences, strict=True)])
            ),
            "won": float((advantage > 0).float().mean()),
            "lost": float((advantage < 0).float().mean()),
        }
        if update % 5 == 0 or update == 1:
            print(
                f"update {update:>4}  loss {row['loss']:>8.4f}  "
                f"R {row['reward_sample']:.3f}/{row['reward_greedy']:.3f}  "
                f"adv {row['advantage']:>+7.4f}+-{row['advantage_std']:.3f}  "
                f"H {row['entropy_break']:>5.3f}a/{row['entropy_cover']:>5.3f}b  "
                f"KL {row['kl']:>7.4f}  |g| {norm_a:>6.3f}a/{norm_b:>6.3f}b  "
                f"won {row['won']:>5.1%}  brk {row['breaks_greedy']:>4.2f}g/"
                f"{row['breaks_sample']:>4.2f}s same {row['same_break']:>5.1%}  "
                f"play {row['batch_playable_greedy']:>5.1%}g/{row['batch_playable_sample']:>5.1%}s  "
                f"{(time.perf_counter() - started) / update:>4.1f}s/upd",
                flush=True,
            )
        if update % args.probe_every == 0 or update == args.updates:
            probed = probe(probe_rows, 1)
            row.update({f"holdout_{k}": v for k, v in probed.items()})
            print(
                f"  probe   valid {probed['valid']:>6.1%}  playable {probed['playable']:>6.1%}  "
                f"tiles {probed['tiles']:.2f} vs {probed['solver_tiles']:.2f}  "
                f"breaks {probed['breaks']:.2f} ({probed['breaks_zero']:.1%} none)",
                flush=True,
            )
            # Selected on the decode, exactly as stage one selects: the training
            # objective is a proxy for it and the two peak at different updates.
            if probed["playable"] > best_probe:
                best_probe, best_update = probed["playable"], update
                best_state = {k: v.clone() for k, v in net.state_dict().items()}
                if args.out:
                    args.out.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {
                            "cfg": args.config,
                            "hidden": checkpoint["hidden"],
                            "key": checkpoint["key"],
                            "cover_hidden": checkpoint.get("cover_hidden"),
                            "cover_encoder": checkpoint.get("cover_encoder"),
                            "monotone": monotone,
                            "epoch": update,
                            "state": best_state,
                        },
                        args.out,
                    )
        history.append(row)

    net.load_state_dict(best_state)
    print(f"\nkept update {best_update} (probe playable {best_probe:.1%})", flush=True)

    final = {}
    for beam in args.beam:
        net.eval()
        final[beam] = playable_rate(cfg, data, scorer, held_states, beam, monotone)
        final[beam].update(break_profile(cfg, data, scorer, held_states, beam, monotone))
        net.train()
        print(
            f"holdout ({len(held_states):,} states, beam {beam})  "
            f"valid {final[beam]['valid']:.1%}  playable {final[beam]['playable']:.1%}  "
            f"tiles {final[beam]['tiles']:.2f} against CP-SAT's "
            f"{final[beam]['solver_tiles']:.2f}  "
            f"breaks {final[beam]['breaks']:.2f} ({final[beam]['breaks_zero']:.1%} none) "
            f"builds {final[beam]['builds']:.2f}  "
            f"{1000 / final[beam]['decodes_per_second']:.1f} ms/state",
            flush=True,
        )

    scores: list[dict] = []
    if args.eval_games:
        suite = suite_for(args.config)
        net.eval()
        for beam in args.beam:
            label = f"two-phase rl (beam {beam})"
            agent = TwoPhaseRepartition(cfg, scorer, beam=beam, monotone=monotone)
            elapsed = time.perf_counter()
            result = evaluate(label, suite, build_agent=lambda c, a=agent: a, games=args.eval_games)
            print(
                f"  {label:28s} win {result.win_rate:>6.1%}  score {result.mean_score:>+8.2f}  "
                f"illegal {result.illegal_attempts}  n={result.games}  "
                f"asked {agent.asked:,} answered {agent.answered:,}  "
                f"{time.perf_counter() - elapsed:.0f}s",
                flush=True,
            )
            scores.append(
                {
                    "label": label,
                    "win_rate": result.win_rate,
                    "mean_score": result.mean_score,
                    "illegal_attempts": result.illegal_attempts,
                    "games": result.games,
                    "asked": agent.asked,
                    "answered": agent.answered,
                }
            )
        # The two ends of the ruler, re-measured inside this run so the middle is not
        # compared against numbers a different harness produced.
        for label, repartition in (("by_value", False), ("by_value+repartition", True)):
            if not args.baseline_arms:
                break
            result = evaluate(
                label,
                suite,
                build_agent=lambda c, r=repartition: MacroAgent(c, choose=by_value(c), repartition=r),
                games=args.eval_games,
            )
            print(
                f"  {label:28s} win {result.win_rate:>6.1%}  score {result.mean_score:>+8.2f}  "
                f"illegal {result.illegal_attempts}  n={result.games}",
                flush=True,
            )
            scores.append(
                {
                    "label": label,
                    "win_rate": result.win_rate,
                    "mean_score": result.mean_score,
                    "illegal_attempts": result.illegal_attempts,
                    "games": result.games,
                }
            )

    if args.log_json:
        args.log_json.parent.mkdir(parents=True, exist_ok=True)
        args.log_json.write_text(
            json.dumps(
                {
                    "config": args.config,
                    "init": str(args.init),
                    "data": [str(path) for path in args.data],
                    "monotone": monotone,
                    "updates": args.updates,
                    "batch": args.batch,
                    "lr": args.lr,
                    "kl_coef": args.kl_coef,
                    "entropy_coef": args.entropy_coef,
                    "reward": args.reward,
                    "valid_bonus": args.valid_bonus,
                    "advantage": args.advantage,
                    "samples": args.samples,
                    "sample_temp": args.sample_temp,
                    "credit": args.credit,
                    "freeze": args.freeze,
                    "seed": args.seed,
                    "before": baseline,
                    "best_update": best_update,
                    "holdout_by_beam": {str(k): v for k, v in final.items()},
                    "history": history,
                    "eval": scores,
                },
                indent=2,
            )
        )
        print(f"wrote {args.log_json}", flush=True)


if __name__ == "__main__":
    main()
