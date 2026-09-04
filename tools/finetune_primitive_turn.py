"""Stage two on the primitive turn decoder: self-critical PG, anchored to the clone.

    python tools/train_primitive_turn.py --data checkpoints/turns-s*.npz \
        --out checkpoints/primitive-turn.pt                                  # stage one
    python tools/finetune_primitive_turn.py --data checkpoints/turns-s*.npz \
        --init checkpoints/primitive-turn.pt --updates 400 --samples 4

The same stage two `tools/finetune_repartition.py` runs over templates, with the
vocabulary swapped and nothing else: sample whole turns from the policy with the
simulator enforcing legality, reward them by the tiles they commit, and push
against what the *greedy* decode of the same position scored -- because the greedy
decode is the arm worth having. The KL anchor is a frozen copy of the stage-one
weights.

Three measured lessons carried over rather than rediscovered. A **validity bonus is
harmful** -- the nearest reward from a failing rollout is the do-nothing turn, and
paying for it buys validity and costs playability -- so the reward here is tiles
alone. **One sample per position is the wall**: a lone deviation loses to greedy
three times as often as it wins, so `--samples 4` draws four whole turns and keeps
the one that sheds most, which is the beam's selection moved inside the weights.
And `--advantage positive` drops the negative half, which reached the same rate
without the drift.

The one thing this space adds is that a rollout can *wander*: unlike a template
sequence, whose mask closes on it, a primitive rollout can keep lifting tiles until
the micro budget runs out. Those score zero, which is what the greedy baseline also
scores where it declines, so they cost time and teach nothing -- `--max-depth` caps
them. A sampled `DRAW` ends its rollout there and scores zero for the same reason,
so a position the teacher declined draws no policy gradient at all -- it stays in
the batch only for the KL and entropy terms, which is what holds the decline the
imitation stage taught.

**What it measured.** 300 updates of 128 turn boundaries from the stage-one
checkpoint, `--samples 4 --advantage positive`, kept at update 100 and falling
after -- the same early peak the macro trainer has. Held-out committed goes
**26.9% -> 28.6%** at greedy and 54.9% -> 54.6% at beam 4, against +8.1pp and
+8.9pp for the same stage two over templates, and what does move is concentrated in
the long repartition buckets. The suite moves far more than that: on `standard-greedy`
at n=240 it is worth +45.8 points to the whole-turn agent at greedy and +7.7 at beam
4, and takes the drop-in repartition arm's share of the solver's contribution from
41.2% to 51.5%.
"""

from __future__ import annotations

import argparse
import copy
import json
import pathlib
import sys
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from rummi.agents.learned.architecture import Architecture
from rummi.agents.learned.primitive_turn import Scorer
from rummi.agents.learned.torch_net import TorchPolicy
from rummi.agents.learned.turn_sim import TurnStart, to_state
from rummi.env.numpy.engine import step as engine_step
from rummi.env.numpy.masks import legal_actions
from rummi.env.numpy.sets import summarize
from rummi.env.observation import encode
from rummi.rules.config import CONFIG_BY_NAME, RummiConfig

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from train_primitive_turn import (
    Turns,
    decode_report,
    length_table,
    print_abandonment,
    print_length_table,
)

TILE_SCALE = 4.0
"""Divisor that puts a typical committed turn near 0.5. `frugal` sheds a mean of
2.2 tiles per turn it plays, so the reward does not saturate and a big turn is
still allowed to be worth more than a small one."""


@dataclass(slots=True)
class Rollouts:
    """One turn decoded per position, and what the loss needs from it."""

    actions: list[list[int]] = field(default_factory=list)
    committed: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    tiles: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    logp: torch.Tensor = field(default_factory=lambda: torch.zeros(0))
    entropy: torch.Tensor = field(default_factory=lambda: torch.zeros(()))
    kl: torch.Tensor = field(default_factory=lambda: torch.zeros(()))
    steps: int = 0
    credited: int = 0


def roll(
    cfg: RummiConfig,
    net: TorchPolicy,
    starts: TurnStart,
    *,
    sample: bool,
    generator: torch.Generator | None = None,
    grad: bool = False,
    clone: TorchPolicy | None = None,
    baseline: list[list[int]] | None = None,
    forced: list[list[int]] | None = None,
    max_depth: int | None = None,
) -> Rollouts:
    """Decode one turn per position in lockstep, sampling or greedily.

    The differentiable twin of `primitive_turn.decode_turns` at beam 1: same
    simulator, same mask, same terminal rules, so `sample=False` reproduces its
    sequence exactly and the sampled arm cannot be measuring a different space.

    `baseline` is the greedy arm's actions, and where it is given the returned
    `logp` covers only the steps from the first departure onwards. A whole turn
    carries one reward, so a shared prefix would otherwise be pushed in the
    direction of a difference it did not cause -- and the prefix is most of the
    turn.

    `forced` replays decisions already taken and returns their log-probability with
    a tape attached, which is how the best of several samples is scored without
    holding a tape per sample.
    """
    n = len(starts)
    state = to_state(cfg, starts)
    rows = np.arange(n)
    opening = starts.placed.sum(-1).astype(np.int64)
    out = Rollouts(
        actions=[[] for _ in range(n)],
        committed=np.zeros(n, dtype=bool),
        tiles=np.zeros(n, dtype=np.int64),
    )
    left = np.zeros(n, dtype=bool) if baseline is not None else np.ones(n, dtype=bool)
    picked_rows: list[torch.Tensor] = []
    picked_logp: list[torch.Tensor] = []
    entropy_sum = torch.zeros((), dtype=torch.float32)
    kl_sum = torch.zeros((), dtype=torch.float32)

    budget = int((cfg.max_micro_per_turn - starts.micro_count).max()) if n else 0
    for depth in range(budget if max_depth is None else min(budget, max_depth)):
        if forced is not None:
            live = np.flatnonzero([depth < len(forced[row]) for row in rows])
            if not live.size:
                break
            state, rows = state.select(live), rows[live]
        if not rows.size:
            break
        summary = summarize(cfg, state.table_sets)
        mask = legal_actions(state, summary)
        obs = encode(state, summary)
        mask_t = torch.from_numpy(mask)
        with torch.set_grad_enabled(grad):
            logits, _ = net(obs, mask_t)
            logp = torch.log_softmax(logits, dim=-1)
            probs = logp.exp()
            entropy_sum = entropy_sum + -(probs * logp).sum(-1).sum()
            if clone is not None:
                with torch.no_grad():
                    reference = torch.log_softmax(clone(obs, mask_t)[0], dim=-1)
                kl_sum = kl_sum + (probs * (logp - reference)).sum(-1).sum()

        if forced is not None:
            action = torch.from_numpy(
                np.asarray([forced[row][depth] for row in rows], dtype=np.int64)
            )
        elif sample:
            action = torch.multinomial(probs.detach(), 1, generator=generator).squeeze(-1)
        else:
            action = logp.detach().argmax(-1)
        chosen = action.numpy()
        out.steps += len(rows)

        if baseline is not None:
            taken = np.asarray(
                [baseline[row][depth] if depth < len(baseline[row]) else -1 for row in rows]
            )
            left[rows] |= chosen != taken
        credit = left[rows]
        if credit.any():
            keep = torch.from_numpy(np.flatnonzero(credit))
            picked_rows.append(torch.from_numpy(rows[credit]))
            picked_logp.append(logp[keep].gather(1, action[keep, None]).squeeze(-1))
            out.credited += int(credit.sum())

        placed = state.placed_rack.sum(-1).astype(np.int64)
        going: list[int] = []
        for k, row in enumerate(rows.tolist()):
            act = int(chosen[k])
            out.actions[row].append(act)
            if act == cfg.end_turn_action:
                out.committed[row] = True
                out.tiles[row] = int(placed[k]) - int(opening[row])
            elif act != cfg.draw_action:
                going.append(k)
        if not going:
            break
        cont = np.asarray(going)
        state = state.select(cont)
        engine_step(state, chosen[cont], mask[cont])
        rows = rows[cont]

    total = torch.zeros(n, dtype=torch.float32)
    if picked_rows:
        total = total.index_add(0, torch.cat(picked_rows), torch.cat(picked_logp))
    out.logp = total
    out.entropy = entropy_sum / max(out.steps, 1)
    out.kl = kl_sum / max(out.steps, 1)
    return out


def reward_of(rolled: Rollouts) -> np.ndarray:
    """Tiles committed, scaled. A declined or abandoned turn is worth exactly zero.

    No validity term: `tools/finetune_repartition.py` measured what paying for one
    costs, and the failure mode is the same here -- the cheapest valid turn is the
    one that sheds least.
    """
    return np.where(rolled.committed, rolled.tiles, 0).astype(np.float32) / TILE_SCALE


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="standard", choices=sorted(CONFIG_BY_NAME))
    p.add_argument("--data", type=pathlib.Path, nargs="+", required=True)
    p.add_argument("--init", type=pathlib.Path, required=True, help="the imitation checkpoint")
    p.add_argument("--prefix", default="turn", choices=("turn", "gate"))
    p.add_argument("--updates", type=int, default=400)
    p.add_argument("--batch", type=int, default=128, help="positions rolled out per update")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--kl-coef", type=float, default=0.1)
    p.add_argument("--entropy-coef", type=float, default=0.001)
    p.add_argument(
        "--samples", type=int, default=4,
        help="turns drawn per position, of which the best-shedding one is the "
             "sample. Above 1 this is the beam's selection done at training time "
             "rather than at decode time, which is the whole point",
    )
    p.add_argument("--advantage", default="positive", choices=("signed", "positive"))
    p.add_argument(
        "--max-depth", type=int, default=60,
        help="primitives a rollout may spend before it is abandoned. A wandering "
             "rollout scores zero either way; this stops it costing a whole turn's "
             "micro budget to find that out",
    )
    p.add_argument("--beam", type=int, nargs="+", default=[1, 4])
    p.add_argument("--probe", type=int, default=384, help="holdout positions decoded per probe")
    p.add_argument("--probe-every", type=int, default=25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=pathlib.Path, default=None)
    p.add_argument("--log-json", type=pathlib.Path, default=None)
    args = p.parse_args()

    cfg = CONFIG_BY_NAME[args.config]
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    generator = torch.Generator().manual_seed(args.seed)

    checkpoint = torch.load(args.init, weights_only=True)
    arch = Architecture(hidden=tuple(checkpoint["hidden"]))
    net = TorchPolicy(cfg, arch, seed=args.seed)
    net.load_state_dict(checkpoint["state"])
    clone = copy.deepcopy(net).eval()
    for q in clone.parameters():
        q.requires_grad_(False)
    scorer = Scorer(net)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    data = Turns(cfg, args.config, list(args.data), prefix=args.prefix)
    train = np.flatnonzero(~data.holdout)
    held = np.flatnonzero(data.holdout)
    print(
        f"config={args.config} init={args.init} population={args.prefix}\n"
        f"train {len(train):,} positions  holdout {len(held):,}  "
        f"declines {data.drawn.mean():.1%}",
        flush=True,
    )

    def probe(rows: np.ndarray, beam: int) -> dict[str, float]:
        net.eval()
        out = decode_report(cfg, scorer, data, rows, beam, 256).summary()
        net.train()
        return out

    # One slice, drawn once, so the trajectory is a paired comparison rather than
    # fresh noise per reading -- and out of the played holdout, because committing a
    # turn is not what a declined position is asking for.
    probe_rows = rng.permutation(held[~data.drawn[held]])[: args.probe]
    before = probe(probe_rows, 1)
    print(f"before  probe committed {before['committed']:.1%}", flush=True)

    history: list[dict] = []
    best_probe, best_update = before["committed"], 0
    best_state = {k: v.clone() for k, v in net.state_dict().items()}
    order = rng.permutation(train)
    cursor = 0
    net.train()
    started = time.perf_counter()
    for update in range(1, args.updates + 1):
        if cursor + args.batch > len(order):
            order, cursor = rng.permutation(train), 0
        index = order[cursor : cursor + args.batch]
        cursor += args.batch
        batch = data.starts.take(index)

        with torch.no_grad():
            greedy = roll(cfg, net, batch, sample=False, max_depth=args.max_depth)
        r_greedy = reward_of(greedy)

        best: list[list[int]] = greedy.actions
        r_sample = np.zeros(len(index), dtype=np.float32)
        played = np.zeros(len(index), dtype=bool)
        for draw in range(args.samples):
            with torch.no_grad():
                drawn = roll(
                    cfg, net, batch, sample=True, generator=generator, max_depth=args.max_depth
                )
            reward = reward_of(drawn)
            better = reward > r_sample if draw else np.ones(len(index), dtype=bool)
            best = [drawn.actions[i] if better[i] else best[i] for i in range(len(index))]
            r_sample = np.where(better, reward, r_sample)
            played = np.where(better, drawn.committed, played)
        sampled = roll(
            cfg, net, batch, sample=False, grad=True, clone=clone,
            baseline=greedy.actions, forced=best, max_depth=args.max_depth,
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
            "entropy": float(sampled.entropy.detach()),
            "kl": float(sampled.kl.detach()),
            "grad_norm": grad_norm,
            "reward_sample": float(r_sample.mean()),
            "reward_greedy": float(r_greedy.mean()),
            "advantage": float(advantage.mean()),
            "advantage_std": float(advantage.std()),
            "batch_committed_greedy": float(greedy.committed.mean()),
            "batch_committed_sample": float(played.mean()),
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
                f"won {row['won']:>5.1%}/lost {row['lost']:>5.1%}  "
                f"play {row['batch_committed_greedy']:>5.1%}g/{row['batch_committed_sample']:>5.1%}s  "
                f"{(time.perf_counter() - started) / update:>4.1f}s/upd",
                flush=True,
            )
        if update % args.probe_every == 0 or update == args.updates:
            probed = probe(probe_rows, 1)
            row.update({f"holdout_{k}": v for k, v in probed.items()})
            print(f"  probe   committed {probed['committed']:>6.1%}", flush=True)
            # Selected on the decode, exactly as stage one selects.
            if probed["committed"] > best_probe:
                best_probe, best_update = probed["committed"], update
                best_state = {k: v.clone() for k, v in net.state_dict().items()}
                if args.out:
                    args.out.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {
                            "cfg": args.config,
                            "hidden": list(arch.hidden),
                            "epoch": update,
                            "state": best_state,
                        },
                        args.out,
                    )
        history.append(row)

    net.load_state_dict(best_state)
    print(f"\nkept update {best_update} (probe committed {best_probe:.1%})", flush=True)

    net.eval()
    reports = {beam: decode_report(cfg, scorer, data, held, beam, 256) for beam in args.beam}
    for beam, report in reports.items():
        summary = report.summary()
        print(
            f"holdout ({len(held):,} positions, beam {beam})  "
            f"committed {summary['committed']:.1%}  tiles {summary['tiles']:.2f} "
            f"against the teacher's {summary['teacher_tiles']:.2f}  "
            f"declined {summary['declined']:.1%} of {summary['declines']:,}  "
            f"{summary['ms_per_turn']:.1f} ms/turn",
            flush=True,
        )
    print_abandonment(reports)
    rows = length_table(data, held, reports) if args.prefix == "turn" else []
    if rows:
        print_length_table(rows, args.beam)

    if args.log_json:
        args.log_json.parent.mkdir(parents=True, exist_ok=True)
        args.log_json.write_text(
            json.dumps(
                {
                    "config": args.config,
                    "init": str(args.init),
                    "prefix": args.prefix,
                    "updates": args.updates,
                    "batch": args.batch,
                    "lr": args.lr,
                    "kl_coef": args.kl_coef,
                    "entropy_coef": args.entropy_coef,
                    "samples": args.samples,
                    "advantage": args.advantage,
                    "max_depth": args.max_depth,
                    "seed": args.seed,
                    "before": before,
                    "best_update": best_update,
                    "history": history,
                    "holdout_by_beam": {
                        str(beam): report.summary() for beam, report in reports.items()
                    },
                    "abandonment": {
                        str(beam): report.abandonment() for beam, report in reports.items()
                    },
                    "length_breakdown": rows,
                },
                indent=2,
            )
        )
        print(f"wrote {args.log_json}", flush=True)


if __name__ == "__main__":
    main()
