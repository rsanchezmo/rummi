"""Stage two on the repartition picker: self-critical policy gradient, anchored to the clone.

    python tools/train_repartition.py --data checkpoints/repartitions-s*.npz ...   # stage one
    python tools/finetune_repartition.py --data checkpoints/repartitions-s*.npz \
        --init checkpoints/repart-aug2.pt --updates 1500 --batch 256 \
        --valid-bonus 0 --advantage positive --samples 4 --eval-games 240

Imitation aimed the network at CP-SAT's answer; the deliverable is a *decode*, and
the two are not the same objective -- exact match sits at 2% while beam 16 emits a
playable repartition in 66.4% of held-out stuck states. What separates the beam
from greedy is that the beam gets to compare finished constructions on tiles
played, which is the reward, evaluated after the fact. Policy gradient can put
that comparison inside the weights instead of inside the search: sample a whole
construction, score it with `build`, and push against what the greedy decode of
the same state would have scored.

Everything here rests on one property of the action space -- a finished sequence
is legal by construction, and `build` says what it is worth with no env and no
solver. So the reward is exact, one rollout is ~13 masked 331-way choices, and the
baseline is free: the greedy arm is the arm being measured anyway.

The loss is `-(R(sample) - R(greedy)) * logp(sample) + kl * KL(pi || pi_clone) -
ent * H`. The KL anchor is against a frozen deepcopy of the initial net, which is
what keeps a policy that has stopped being scored on likelihood from wandering off
the manifold imitation put it on.

**What it measured.** 1,500 updates of 256 states from `repart-aug2.pt`, scored on
the same game-split holdout stage one reports (7,276 states); `--reward value`
unless the row says otherwise, and every row is one seed:

| arm | valid | plays | valid, beam 4 | plays, beam 4 |
|---|---|---|---|---|
| imitation, as trained | 49.8% | 27.8% | 75.9% | 46.5% |
| self-critical, `--valid-bonus 0.1` | 73.0% | 29.6% | 89.4% | 44.0% |
| ... `--reward count` | 72.8% | 29.3% | 89.6% | 44.2% |
| ... `--credit all` | 72.5% | 29.7% | 89.5% | 44.9% |
| self-critical, no validity bonus | 62.0% | 32.2% | 82.9% | 49.4% |
| ... `--reward count` | 62.0% | 33.0% | 82.6% | 49.5% |
| `--advantage positive`, bonus 0.1 | 59.0% | 32.1% | 82.6% | 50.7% |
| `--advantage positive`, no bonus | 54.2% | 32.1% | 78.9% | 51.3% |
| **no bonus, `--samples 4 --advantage positive`** | 57.2% | **35.9%** | 80.5% | **55.4%** |
| ... `--stuck 20000` | 56.0% | 34.0% | 79.7% | 53.3% |

The best-of-4 row is `checkpoints/repart-rl4.pt` and the single-sample no-bonus one
is `checkpoints/repart-rl1.pt`; both load through `--init`.

One sample per state hits a wall at **32-33%** and four different recipes land on
it. Four samples with the best kept goes to **35.9%**, +8.1pp -- 43% of the 18.7pp
greedy-to-beam-4 gap -- at **3.1 ms/state**, which is the greedy decode's own cost,
and it lifts beam 4 by the same 8.9pp. That is the whole finding: what separates
greedy from a beam is *selection among finished constructions*, and selection is
worth what it is worth whether it happens at decode time or at training time.

`--samples 8` tracks 4 probe for probe through 400 updates (32.2 / 34.5 / 33.8 /
34.0 / 33.5 / 33.2 / 32.8 / 34.8 against 32.2 / 33.2 / 35.5 / 33.2 / 34.5 / 34.0 /
34.0 / 33.8) at twice the cost per update, so the knob saturates early.

On `standard-greedy` at n=240 the best-of-4 arm's greedy decode scores **+44.05 /
99.2%** against the clone's +42.31 / 98.5%; the two single-sample no-bonus
checkpoints score +44.25 (value) and +43.10 (count) on the same suite. Paired per
game the best-of-4 delta is +1.75 with a 95% CI of [-0.66, +4.22]: the suite cannot
resolve a two-point move at 240 deals, which the offline metric can -- +8.1pp on
7,276 states is many times its own noise. Read the ruler as consistent with the
holdout, not as confirmation of it.

**Two traps, both measured, and they are the same trap.** Paying anything for a
valid construction (`--valid-bonus`) buys +11pp of validity and *costs* 5-6pp of
playability at beam 4: from a failing rollout the nearest reward is the identity
cover -- keep every set where it is -- which is always valid and always sheds
nothing. The signed advantage is the same mistake from the other side. A sampled
construction loses to the greedy one three times as often as it wins, so the update
is mostly "do not deviate": entropy falls 0.89 -> 0.37 nats per decision, KL runs
to 0.28, and the argmax path sharpens instead of moving. `--advantage positive`
holds entropy at 0.80 and KL at 0.04 and reaches the same playable rate, so the
extra drift was buying nothing -- the fine-tune that stalled did not stall against
the anchor. Value against count is a wash, ~1pp either way.

**Unlabelled states are usable and do not help.** `--stuck` collects gate states by
rolling the arm forward -- 20,001 in 116s -- and 79% of them are states CP-SAT
declined, which imitation could not use at all. Under a tiles-only reward they
carry no reward *variance* either: nothing plays, so every sample and the baseline
score 0 and the advantage is exactly zero. Adding them costs 1.9pp by diluting the
third of the batch that could have taught something. RL needs no labels; it does
need two answers that differ.

**Declining is structural, not learned.** `STOP` is masked until the table is
covered, so a decode cannot emit an invalid repartition -- it either covers or
returns nothing, and `NeuralRepartition` falls through to `DRAW`. What the
fine-tune moves is the other direction: the arm answers 14.9% of the gate's firings
against the clone's 12.8%, both under CP-SAT's 20.9%.

`tools/finetune_two_phase.py` runs this same recipe over the break-then-cover space,
importing `run_decode`, `score_sequences` and `Starts` from here so phase B *is* this
decode. It measures how much of the +8.1pp above survives a shorter sequence.
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

from rummi.agents.base import Observation, table
from rummi.agents.learned.repartition_net import (
    RepartitionNet,
    Scorer,
    apply_template,
    build,
    candidate_features,
    feasible,
    initial_counts,
    present_counts,
    state_features,
    stop_action,
)
from rummi.agents.macro import MacroAgent, by_value
from rummi.evaluate.protocol import evaluate, suite_for
from rummi.rules.config import CONFIG_BY_NAME, RummiConfig
from rummi.rules.encoding import tables

# Read from where they are defined rather than restated: the holdout metric has to
# be the number stage one reported, and the scored arm has to differ from stage
# one's only in the weights.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from train_repartition import NeuralRepartition, Steps, playable_rate

VALUE_SCALE = 30.0
COUNT_SCALE = 4.0
"""Divisors that put a typical solve near 0.4. CP-SAT sheds a mean of 1.59 tiles
worth 11.2 points over the 48k labelled states, so neither reward saturates and a
big turn is still allowed to be worth more than a small one."""


def tile_price(cfg: RummiConfig) -> np.ndarray:
    """What shedding each kind is worth, priced as `macro.by_value` prices a lay-off.

    `tables().value` scores the joker 0 -- its face value is positional -- so what
    it costs to keep is `joker_penalty`, and that is what playing it saves.
    """
    price = tables(cfg).value.astype(np.float32).copy()
    price[cfg.joker_kind] = float(cfg.joker_penalty)
    return price


@dataclass(frozen=True, slots=True)
class Starts:
    """The construction's starting multiset for a set of states, precomputed.

    `present_counts` walks 35 slots per board and `initial_counts` the tiles on
    them; doing it inside the rollout put a numpy loop in front of every forward
    pass, and the same states come round every epoch.
    """

    need: np.ndarray
    avail: np.ndarray
    present: np.ndarray
    base: np.ndarray | None = None
    """`(B,)` sets standing outside the construction, or `None` where it covers the
    whole table. `tools/finetune_two_phase.py` sets it to the slots phase A kept, so
    `max_sets` and `build`'s `reserved` are measured against the whole table."""

    def take(self, index: np.ndarray) -> Starts:
        return Starts(
            self.need[index],
            self.avail[index],
            self.present[index],
            None if self.base is None else self.base[index],
        )

    def __len__(self) -> int:
        return len(self.need)


def starting_states(cfg: RummiConfig, racks: np.ndarray, boards: np.ndarray) -> Starts:
    need, avail, present = [], [], []
    for rack, board in zip(racks, boards, strict=True):
        n, a = initial_counts(cfg, rack, board)
        need.append(n.astype(np.int16))
        avail.append(a.astype(np.int16))
        present.append(np.minimum(present_counts(cfg, board), 255.0).astype(np.uint8))
    return Starts(np.stack(need), np.stack(avail), np.stack(present))


@dataclass(slots=True)
class Trajectories:
    """One decode of a batch of states, and what the loss needs from it."""

    sequences: list[tuple[int, ...]]
    actions: list[tuple[int, ...]]
    """Every decision including the terminal `STOP`, which `sequences` drops."""
    logp: torch.Tensor
    """`(B,)` summed log-probability of the sampled sequence, with grad."""
    entropy: torch.Tensor
    """Mean per-decision entropy over every step of the batch."""
    kl: torch.Tensor
    """Mean per-decision `KL(pi || pi_clone)` over the same steps."""
    steps: int
    credited: int
    diverged: np.ndarray | None = None
    """`(B,)` which rows had left the baseline arm by the end, so a second phase
    decoded after this one can go on crediting from where this one departed."""


def run_decode(
    cfg: RummiConfig,
    net: RepartitionNet,
    starts: Starts,
    *,
    monotone: bool,
    sample: bool,
    generator: torch.Generator | None = None,
    grad: bool = False,
    clone: RepartitionNet | None = None,
    baseline: list[tuple[int, ...]] | None = None,
    temperature: float = 1.0,
    forced: list[tuple[int, ...]] | None = None,
    diverged: np.ndarray | None = None,
) -> Trajectories:
    """Construct one repartition per state, all of them in lockstep.

    The differentiable twin of `repartition_net.decode` at beam 1: same features,
    same mask, same `last`-at-start convention, so `sample=False` reproduces its
    sequence exactly and the sampled arm cannot be measuring a different space.
    Rows drop out as they reach `STOP` -- or as the mask closes on them, which is
    an abandoned construction and `build` rejects it.

    `baseline` is the greedy arm's decisions, and where it is given the returned
    `logp` covers only the steps from the first departure from it onwards. A whole
    sequence carries one reward, so a shared prefix would otherwise be pushed in
    the direction of a difference it did not cause -- and the shared prefix is most
    of the sequence, so that term is the loud one.

    `temperature` widens the *draw* only; `logp` stays the policy's own, so above 1
    the objective is exploratory self-imitation rather than an unbiased gradient --
    which is why it is only worth turning on beside `--advantage positive`.

    `forced` replays decisions already taken and returns their log-probability with
    a tape attached. That is how the best of several samples is scored: holding a
    tape per sample would cost K times the memory to produce one gradient.

    `diverged` carries a divergence state in from a phase decoded before this one:
    a row that already left the baseline arm keeps crediting here, because every
    decision after a departure is conditioned on it.
    """
    stop = stop_action(cfg)
    size = len(starts)
    need = starts.need.astype(np.int64)
    avail = starts.avail.astype(np.int64)
    present = starts.present.astype(np.float32)
    last = np.full(size, 0 if monotone else -1, dtype=np.int64)
    n_sets = (
        np.zeros(size, dtype=np.int64) if starts.base is None else starts.base.astype(np.int64)
    )
    alive = np.ones(size, dtype=bool)
    sequences: list[list[int]] = [[] for _ in range(size)]
    actions: list[list[int]] = [[] for _ in range(size)]
    if baseline is None:
        left = np.ones(size, dtype=bool)
    elif diverged is None:
        left = np.zeros(size, dtype=bool)
    else:
        left = np.asarray(diverged).copy()

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
        dynamic, short = candidate_features(cfg, need[rows], avail[rows], present[rows], last[rows])
        legal = feasible(cfg, need[rows], avail[rows], n_sets[rows], last[rows], short, monotone)
        # A row the mask has closed would soft-max over nothing but `MASKED` and
        # hand the sampler a uniform draw over illegal templates.
        open_rows = legal.any(-1)
        if not open_rows.all():
            alive[rows[~open_rows]] = False
            rows = rows[open_rows]
            if rows.size == 0:
                break
            dynamic, legal, short = dynamic[open_rows], legal[open_rows], short[open_rows]
        state = state_features(cfg, need[rows], avail[rows], n_sets[rows], last[rows], present[rows])

        state_t = torch.from_numpy(state)
        dynamic_t = torch.from_numpy(dynamic)
        legal_t = torch.from_numpy(legal)
        with torch.set_grad_enabled(grad):
            logits = net(state_t, dynamic_t, legal_t)
            logp = torch.log_softmax(logits, dim=-1)
            probs = logp.exp()
            entropy_sum = entropy_sum + -(probs * logp).sum(-1).sum()
            if clone is not None:
                with torch.no_grad():
                    reference = torch.log_softmax(clone(state_t, dynamic_t, legal_t), dim=-1)
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

        for row, template in zip(rows.tolist(), chosen.tolist(), strict=True):
            actions[row].append(template)
            if template == stop:
                alive[row] = False
                continue
            sequences[row].append(template)
            need[row], avail[row] = apply_template(cfg, need[row], avail[row], template)
            present[row, template] = max(present[row, template] - 1.0, 0.0)
            last[row] = template
            n_sets[row] += 1

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


def score_sequences(
    cfg: RummiConfig,
    starts: Starts,
    sequences: list[tuple[int, ...]],
    *,
    mode: str,
    valid_bonus: float,
    price: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`(reward, valid, played)` per state, from `build` alone -- no env, no solver.

    A sequence that leaves a table tile uncovered, or that `build` rejects for any
    other reason, is worth zero. Validity is worth a token amount on its own so the
    gradient has a rung between "not a repartition" and "a repartition that sheds
    nothing" -- the greedy baseline collects it too, so it cannot be farmed.
    """
    reward = np.zeros(len(sequences), dtype=np.float32)
    valid = np.zeros(len(sequences), dtype=bool)
    played = np.zeros(len(sequences), dtype=np.int64)
    for i, sequence in enumerate(sequences):
        found = build(
            cfg,
            starts.need[i].astype(np.int64),
            starts.avail[i].astype(np.int64),
            list(sequence),
            reserved=0 if starts.base is None else int(starts.base[i]),
        )
        if found is None:
            continue
        valid[i] = True
        played[i] = found.tiles_played
        shed = (
            float(found.played.astype(np.float32) @ price) / VALUE_SCALE
            if mode == "value"
            else found.tiles_played / COUNT_SCALE
        )
        reward[i] = valid_bonus + shed
    return reward, valid, played


class GateRecorder(NeuralRepartition):
    """`NeuralRepartition`, keeping every state its own gate fires on.

    The collector stores only the states CP-SAT *answered*: 79% of the gate's
    firings are declines and are counted, not written. Those are exactly the states
    imitation could not use and RL can, so they are re-collected here -- under the
    net that will be trained on them, so the distribution is the one it will meet.
    """

    def __init__(self, cfg: RummiConfig, scorer: Scorer, monotone: bool) -> None:
        super().__init__(cfg, scorer, beam=1, monotone=monotone)
        self.racks: list[np.ndarray] = []
        self.boards: list[np.ndarray] = []

    def _repartition(self, obs: Observation, env: int) -> list[int]:
        self.racks.append(np.asarray(obs["rack"][env]).astype(np.int16))
        self.boards.append(np.asarray(table(obs)[env]).astype(np.int16))
        return super()._repartition(obs, env)


def collect_gate_states(
    cfg: RummiConfig, scorer: Scorer, monotone: bool, target: int, envs: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Roll the arm forward and keep every state the repartition gate asks about."""
    from rummi.env.fixed_opponent import FixedOpponentEnv

    env = FixedOpponentEnv(num_envs=envs, cfg=cfg, seed=seed, opponent="greedy")
    agent = GateRecorder(cfg, scorer, monotone)
    agent.reset(envs)
    obs, info = env.reset()
    began = time.perf_counter()
    steps = 0
    while len(agent.racks) < target and steps < 200_000:
        actions = agent.act(obs, np.asarray(info["action_mask"]))
        obs, _, _, _, info = env.step(actions)
        steps += 1
    env.close()
    print(
        f"collected {len(agent.racks):,} unlabelled gate states in "
        f"{time.perf_counter() - began:.0f}s over {steps:,} steps "
        f"({agent.answered / max(agent.asked, 1):.1%} answered by the net, against "
        f"CP-SAT's 20.9% on the same gate)",
        flush=True,
    )
    return np.stack(agent.racks), np.stack(agent.boards)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="standard", choices=sorted(CONFIG_BY_NAME))
    p.add_argument("--data", type=pathlib.Path, nargs="+", required=True)
    p.add_argument("--init", type=pathlib.Path, required=True, help="the imitation checkpoint")
    p.add_argument("--updates", type=int, default=400)
    p.add_argument("--batch", type=int, default=192, help="states rolled out per update")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--kl-coef", type=float, default=0.1)
    p.add_argument("--entropy-coef", type=float, default=0.001)
    p.add_argument("--reward", default="value", choices=("value", "count"))
    p.add_argument("--valid-bonus", type=float, default=0.1)
    p.add_argument(
        "--samples", type=int, default=1,
        help="constructions drawn per state, of which the best-shedding one is "
             "the sample. Above 1 this is the beam's selection done at training "
             "time rather than at decode time, which is the whole point: the "
             "greedy arm is what has to get better",
    )
    p.add_argument(
        "--advantage", default="signed", choices=("signed", "positive"),
        help="`positive` drops the negative half, so a rollout that lost to the "
             "greedy arm is ignored rather than pushed down. Self-critical "
             "sampling loses three times as often as it wins, and pushing every "
             "loss down sharpens the decode it already has",
    )
    p.add_argument(
        "--sample-temp", type=float, default=1.0,
        help="temperature of the draw only. Above 1 the objective is exploratory "
             "self-imitation, so it belongs with --advantage positive",
    )
    p.add_argument(
        "--credit", default="diverged", choices=("diverged", "all"),
        help="which steps of the sampled sequence carry the advantage. `diverged` "
             "drops the prefix the greedy arm also walked, which cannot have "
             "caused the difference the reward is measuring",
    )
    p.add_argument(
        "--stuck", type=int, default=0,
        help="unlabelled gate states to collect and add to the training pool. The "
             "labelled set is the 21%% of firings CP-SAT answered; RL needs no "
             "label, so this is the rest of the population",
    )
    p.add_argument("--stuck-envs", type=int, default=32)
    p.add_argument("--beam", type=int, nargs="+", default=[1, 4])
    p.add_argument("--probe", type=int, default=400, help="holdout states decoded per probe")
    p.add_argument("--probe-every", type=int, default=25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-games", type=int, default=0)
    p.add_argument("--baseline-arms", action="store_true", help="score by_value and CP-SAT too")
    p.add_argument("--out", type=pathlib.Path, default=None)
    p.add_argument("--log-json", type=pathlib.Path, default=None)
    args = p.parse_args()

    cfg = CONFIG_BY_NAME[args.config]
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    price = tile_price(cfg)

    checkpoint = torch.load(args.init, weights_only=True)
    monotone = bool(checkpoint["monotone"])
    net = RepartitionNet(cfg, hidden=checkpoint["hidden"], key=checkpoint["key"])
    net.load_state_dict(checkpoint["state"])
    clone = copy.deepcopy(net).eval()
    for q in clone.parameters():
        q.requires_grad_(False)
    scorer = Scorer(net)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    began = time.perf_counter()
    data = Steps(cfg, args.config, list(args.data), monotone)
    train_states = np.flatnonzero(~data.holdout)
    held_states = np.flatnonzero(data.holdout)
    pool = starting_states(cfg, data.rack[train_states], data.board[train_states])
    print(
        f"config={args.config} init={args.init} order={'template' if monotone else 'free'}\n"
        f"train {len(train_states):,} states  holdout {len(held_states):,} states  "
        f"({time.perf_counter() - began:.0f}s to materialise)",
        flush=True,
    )

    if args.stuck:
        racks, boards = collect_gate_states(
            cfg, scorer, monotone, args.stuck, args.stuck_envs, args.seed
        )
        extra = starting_states(cfg, racks, boards)
        pool = Starts(
            np.concatenate([pool.need, extra.need]),
            np.concatenate([pool.avail, extra.avail]),
            np.concatenate([pool.present, extra.present]),
        )
        print(f"training pool {len(pool):,} states ({len(extra):,} unlabelled)", flush=True)

    def probe(rows: np.ndarray, beam: int) -> dict[str, float]:
        net.eval()
        out = playable_rate(cfg, data, scorer, rows, beam, monotone)
        net.train()
        return out

    # One slice, drawn once: every probe then measures the same states, so the
    # trajectory is a paired comparison rather than 400 fresh states of noise per
    # reading -- and the update kept is chosen on a difference that is real.
    probe_rows = rng.permutation(held_states)[: args.probe]
    baseline = probe(probe_rows, 1)
    print(
        f"before  probe valid {baseline['valid']:.1%}  playable {baseline['playable']:.1%}",
        flush=True,
    )

    history: list[dict] = []
    best_probe, best_update = baseline["playable"], 0
    best_state = {k: v.clone() for k, v in net.state_dict().items()}
    order = rng.permutation(len(pool))
    cursor = 0
    net.train()
    started = time.perf_counter()
    for update in range(1, args.updates + 1):
        if cursor + args.batch > len(order):
            order, cursor = rng.permutation(len(pool)), 0
        index = order[cursor : cursor + args.batch]
        cursor += args.batch
        batch = pool.take(index)

        with torch.no_grad():
            greedy = run_decode(cfg, net, batch, monotone=monotone, sample=False)
        r_greedy, valid_greedy, played_greedy = score_sequences(
            cfg, batch, greedy.sequences,
            mode=args.reward, valid_bonus=args.valid_bonus, price=price,
        )
        anchor = greedy.actions if args.credit == "diverged" else None

        if args.samples == 1:
            sampled = run_decode(
                cfg, net, batch, monotone=monotone, sample=True,
                generator=generator, grad=True, clone=clone,
                baseline=anchor, temperature=args.sample_temp,
            )
            r_sample, valid_sample, played_sample = score_sequences(
                cfg, batch, sampled.sequences,
                mode=args.reward, valid_bonus=args.valid_bonus, price=price,
            )
        else:
            # The beam's own trick, moved inside the weights: draw several whole
            # constructions and keep the one that sheds most. Only the winner is
            # replayed with a tape, because K tapes buy one gradient.
            best: list[tuple[int, ...]] | None = None
            r_sample = np.zeros(len(batch), dtype=np.float32)
            valid_sample = np.zeros(len(batch), dtype=bool)
            played_sample = np.zeros(len(batch), dtype=np.int64)
            for _ in range(args.samples):
                with torch.no_grad():
                    drawn = run_decode(
                        cfg, net, batch, monotone=monotone, sample=True,
                        generator=generator, temperature=args.sample_temp,
                    )
                reward, valid, played = score_sequences(
                    cfg, batch, drawn.sequences,
                    mode=args.reward, valid_bonus=args.valid_bonus, price=price,
                )
                better = reward > r_sample if best is not None else np.ones(len(batch), bool)
                best = [
                    drawn.actions[i] if better[i] else best[i]  # type: ignore[index]
                    for i in range(len(batch))
                ]
                r_sample = np.where(better, reward, r_sample)
                valid_sample = np.where(better, valid, valid_sample)
                played_sample = np.where(better, played, played_sample)
            sampled = run_decode(
                cfg, net, batch, monotone=monotone, sample=False,
                grad=True, clone=clone, baseline=anchor, forced=best,
            )
        advantage = torch.from_numpy(r_sample - r_greedy)
        weight = advantage.clamp(min=0.0) if args.advantage == "positive" else advantage

        policy = -(weight * sampled.logp).mean()
        loss = policy + args.kl_coef * sampled.kl - args.entropy_coef * sampled.entropy
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(nn.utils.clip_grad_norm_(net.parameters(), 1.0))
        opt.step()

        row = {
            "update": update,
            "loss": float(loss.detach()),
            "policy_loss": float(policy.detach()),
            "entropy": float(sampled.entropy.detach()),
            "kl": float(sampled.kl.detach()),
            "grad_norm": grad_norm,
            "reward_sample": float(r_sample.mean()),
            "reward_greedy": float(r_greedy.mean()),
            "advantage": float(advantage.mean()),
            "advantage_std": float(advantage.std()),
            "batch_valid_greedy": float(valid_greedy.mean()),
            "batch_playable_greedy": float((played_greedy >= 1).mean()),
            "batch_playable_sample": float((played_sample >= 1).mean()),
            "credited": sampled.credited / max(sampled.steps, 1),
            "won": float((advantage > 0).float().mean()),
            "lost": float((advantage < 0).float().mean()),
        }
        if update % 5 == 0 or update == 1:
            print(
                f"update {update:>4}  loss {row['loss']:>8.4f}  "
                f"R {row['reward_sample']:.3f}/{row['reward_greedy']:.3f}  "
                f"adv {row['advantage']:>+7.4f}+-{row['advantage_std']:.3f}  "
                f"H {row['entropy']:>6.3f}  KL {row['kl']:>7.4f}  "
                f"|g| {grad_norm:>6.3f}  won {row['won']:>5.1%}/lost {row['lost']:>5.1%}  "
                f"play {row['batch_playable_greedy']:>5.1%}g/{row['batch_playable_sample']:>5.1%}s  "
                f"{(time.perf_counter() - started) / update:>4.1f}s/upd",
                flush=True,
            )
        if update % args.probe_every == 0 or update == args.updates:
            probed = probe(probe_rows, 1)
            row.update({f"holdout_{k}": v for k, v in probed.items()})
            print(
                f"  probe   valid {probed['valid']:>6.1%}  playable {probed['playable']:>6.1%}  "
                f"tiles {probed['tiles']:.2f} vs {probed['solver_tiles']:.2f}",
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
        final[beam] = probe(held_states, beam)
        print(
            f"holdout ({len(held_states):,} states, beam {beam})  "
            f"valid {final[beam]['valid']:.1%}  playable {final[beam]['playable']:.1%}  "
            f"tiles {final[beam]['tiles']:.2f} against CP-SAT's "
            f"{final[beam]['solver_tiles']:.2f}  "
            f"{1000 / final[beam]['decodes_per_second']:.1f} ms/state",
            flush=True,
        )

    scores: list[dict] = []
    if args.eval_games:
        suite = suite_for(args.config)
        net.eval()
        arms: list[tuple[str, object]] = [
            (f"rl-repartition (beam {beam})", NeuralRepartition(cfg, scorer, beam=beam, monotone=monotone))
            for beam in args.beam
        ]
        for label, agent in arms:
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
                    "stuck": args.stuck,
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
