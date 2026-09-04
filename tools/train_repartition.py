"""Imitating CP-SAT's repartitions, one template at a time.

    python tools/collect_repartitions.py --target 20000 --out checkpoints/repartitions.npz
    python tools/train_repartition.py --data checkpoints/repartitions.npz --epochs 20 --eval-games 240

The action space is `rummi/agents/learned/repartition_net.py`: pick one of the ~330
set templates, or `STOP`, under a feasibility mask over the remaining multiset.
Teacher forcing walks the solver's own answer in template order, so the target at
every step is a single label and the loss is a plain cross-entropy -- the NP-hard
choice is decomposed, not approximated.

Three numbers, and only the third is the deliverable:

- **step accuracy** -- how often the argmax is the teacher's next set. Flattering,
  because most steps of a solve are "keep the set that is already there".
- **exact match** -- the whole sequence, teacher-forced. Strict, and it fails on a
  repartition that is different from the solver's and just as good.
- **playable rate** -- greedy-decode from scratch, then check the construction:
  every tile on the table re-covered, every set legal, and at least one tile out
  of the rack. That is what the plug-in arm needs and what CP-SAT gets 100% of on
  the same states, by construction of the dataset.

`--eval-games` runs the arm that matters: `by_value` with the stuck-state solve
replaced by a decode, scored on `standard-greedy` beside `by_value` with no
repartition at all and with CP-SAT's.

**What it measured.** 48,002 labelled states over 6,183 games, 90 epochs,
`--free-order --augment --hidden 512 --key 128`, held-out step accuracy 77.0% and
exact match 2.0% -- and neither of those is the story:

| decode | holdout valid | holdout plays a tile | ms/state | `standard-greedy`, n=240 |
|---|---|---|---|---|
| greedy | 49.8% | 27.8% | 2.5 | +42.31 / 98.5% |
| beam 4 | 75.9% | 46.5% | 4.9 | +45.26 / 99.8% |
| beam 16 | 92.1% | 66.4% | 21.1 | **+47.51 / 99.8%** |
| CP-SAT | 100% | 100% | 20.7 | +47.71 / 99.8% |

against `by_value` with no repartition at +28.01 / 82.1% in the same run, so beam
16 recovers **99%** of the solver's 19.7-point contribution -- at the same cost per
state, while greedy recovers 73% at **8.5x** its speed. Exact match stays at 2%
throughout, which is the point: the network is not reproducing CP-SAT's answer, it
is finding a different repartition that plays the same 1.6 tiles.

Two settings carry most of that, and both were measured against their control.
**Colour relabelling** (`--augment`) *lowers* held-out step accuracy, 80.6% ->
77.0%, and nearly doubles the greedy playable rate, 14.2% -> 27.8%: fitting the
teacher's exact next set and constructing a legal repartition are different
skills, and only the second survives 13 steps. **Free order** beats holding the
decode to template order, 9.0% against 6.0% greedy-playable on the 24k-state
half-dataset, because a template chosen too early closes everything below it.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np
import torch
from torch import nn

from rummi.agents.base import Observation, table
from rummi.agents.learned.repartition_net import (
    RepartitionNet,
    Scorer,
    apply_template,
    candidate_features,
    colour_relabellings,
    decode,
    feasible,
    initial_counts,
    n_actions,
    present_counts,
    relabel_rows,
    state_dim,
    state_features,
    stop_action,
)
from rummi.agents.macro import MacroAgent, by_value
from rummi.evaluate.protocol import evaluate, suite_for
from rummi.rules.config import CONFIG_BY_NAME, RummiConfig


class Steps:
    """Every decision of every labelled repartition, as flat arrays.

    Materialised rather than replayed per batch: a prefix of the sequence has to
    be walked to know what is left, and walking it inside the training loop cost
    more than the forward pass it feeds. `present` is the wide one -- one count per
    template -- and it is still `uint8`, so the whole set of ~300k rows is a few
    hundred megabytes.
    """

    def __init__(
        self, cfg: RummiConfig, name: str, files: list[pathlib.Path], monotone: bool
    ) -> None:
        racks, boards, sequences, holdouts, played = [], [], [], [], []
        for path in files:
            with np.load(path, allow_pickle=False) as data:
                if str(data["config"]) != name:
                    raise SystemExit(f"{path} was collected on {data['config']}, not {name}")
                racks.append(data["rack"])
                boards.append(data["board"])
                played.append(data["played"])
                holdouts.append(data["holdout"])
                lengths = data["length"]
                rows = data["sequence"]
                sequences.extend(
                    rows[i, : lengths[i]].astype(np.int64) for i in range(len(lengths))
                )
        self.rack = np.concatenate(racks)
        self.board = np.concatenate(boards)
        self.played = np.concatenate(played)
        self.holdout = np.concatenate(holdouts)
        self.sequence = sequences
        self.cfg = cfg

        stop = stop_action(cfg)
        need_rows, avail_rows, present_rows = [], [], []
        sizes, lasts, targets, owners = [], [], [], []
        for i, sequence in enumerate(sequences):
            need, avail = initial_counts(cfg, self.rack[i], self.board[i])
            present = present_counts(cfg, self.board[i])
            last = 0 if monotone else -1
            for step in range(len(sequence) + 1):
                need_rows.append(need.astype(np.int8))
                avail_rows.append(avail.astype(np.int8))
                present_rows.append(np.minimum(present, 255).astype(np.uint8))
                sizes.append(step)
                lasts.append(last)
                owners.append(i)
                if step == len(sequence):
                    targets.append(stop)
                    break
                chosen = int(sequence[step])
                targets.append(chosen)
                need, avail = apply_template(cfg, need, avail, chosen)
                present = present.copy()
                present[chosen] = max(present[chosen] - 1.0, 0.0)
                last = chosen

        self.need = np.stack(need_rows)
        self.avail = np.stack(avail_rows)
        self.present = np.stack(present_rows)
        self.n_sets = np.asarray(sizes, dtype=np.int64)
        self.last = np.asarray(lasts, dtype=np.int64)
        self.target = np.asarray(targets, dtype=np.int64)
        self.owner = np.asarray(owners, dtype=np.int64)
        self.row_holdout = self.holdout[self.owner]
        self.offset = np.concatenate(
            [[0], np.cumsum([len(s) + 1 for s in sequences])]
        ).astype(np.int64)
        self.kind_to, self.template_to = colour_relabellings(cfg)

    def __len__(self) -> int:
        return len(self.target)

    def rows_of(self, states: np.ndarray) -> np.ndarray:
        """Every row of the given states, in order -- what an exact match needs."""
        return np.concatenate(
            [np.arange(self.offset[s], self.offset[s + 1]) for s in states.tolist()]
        )

    def batch(self, index: np.ndarray, monotone: bool, relabel: np.ndarray | None = None):
        """`(state, dynamic, legal, target)` for the rows at `index`, as tensors.

        `relabel` picks one colour relabelling per row. It is applied to the *rows*
        rather than by re-deriving the sequences, which is what makes 24x data cost
        two gathers -- and it is why it needs `--free-order`: the label order is by
        template index, and a relabelling permutes that.
        """
        cfg = self.cfg
        need = self.need[index].astype(np.int64)
        avail = self.avail[index].astype(np.int64)
        present = self.present[index].astype(np.float32)
        sizes, last = self.n_sets[index], self.last[index]
        target = self.target[index]
        if relabel is not None:
            need, avail, present, last, target = relabel_rows(
                cfg, relabel, need, avail, present, last, target
            )
        dynamic, short = candidate_features(cfg, need, avail, present, last)
        legal = feasible(cfg, need, avail, sizes, last, short, monotone)
        state = state_features(cfg, need, avail, sizes, last, present)
        return (
            torch.from_numpy(state),
            torch.from_numpy(dynamic),
            torch.from_numpy(legal),
            torch.from_numpy(target),
        )


class NeuralRepartition(MacroAgent):
    """`by_value`, with the stuck-state solve answered by the network.

    Nothing about *when* to repartition changes: `MacroAgent.legal_macros` decides
    that, and it is the one thing this overrides -- so the arm differs from
    `by_value+repartition` in exactly the component under test. A decode that comes
    back invalid, plays nothing, or overruns the turn's micro budget returns no
    actions, which is the same fall-through to `END_TURN`/`DRAW` that a declining
    solve produces.
    """

    def __init__(
        self, cfg: RummiConfig, scorer: Scorer, beam: int = 1, monotone: bool = True
    ) -> None:
        super().__init__(cfg, choose=by_value(cfg), repartition=True)
        self.scorer = scorer
        self.beam = beam
        self.monotone = monotone
        self.asked = 0
        self.answered = 0

    def _repartition(self, obs: Observation, env: int) -> list[int]:
        from rummi.solver.to_actions import plan

        cfg = self.cfg
        board = np.asarray(table(obs)[env])
        rack = np.asarray(obs["rack"][env]).astype(np.int64)
        need, avail = initial_counts(cfg, rack, board)
        self.asked += 1
        found = decode(
            cfg, self.scorer, need, avail, present_counts(cfg, board), self.beam, self.monotone
        )
        if found is None or found.tiles_played < 1:
            return []
        actions = self._repartition_plan(
            obs, env, plan(cfg, board, list(found.sets), found.played)
        )
        if not actions:
            return []
        self.answered += 1
        return actions


def playable_rate(
    cfg: RummiConfig, data: Steps, scorer: Scorer, rows: np.ndarray, beam: int, monotone: bool
) -> dict[str, float]:
    """Decode from scratch and check the construction, on `rows` of the dataset."""
    valid = plays = 0
    tiles = solver_tiles = 0
    started = time.perf_counter()
    for i in rows.tolist():
        need, avail = initial_counts(cfg, data.rack[i], data.board[i])
        found = decode(cfg, scorer, need, avail, present_counts(cfg, data.board[i]), beam, monotone)
        solver_tiles += int(data.played[i].sum())
        if found is None:
            continue
        valid += 1
        if found.tiles_played >= 1:
            plays += 1
            tiles += found.tiles_played
    n = max(len(rows), 1)
    return {
        "valid": valid / n,
        "playable": plays / n,
        "tiles": tiles / max(plays, 1),
        "solver_tiles": solver_tiles / n,
        "decodes_per_second": len(rows) / max(time.perf_counter() - started, 1e-9),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="standard", choices=sorted(CONFIG_BY_NAME))
    p.add_argument("--data", type=pathlib.Path, nargs="+", required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--key", type=int, default=64)
    p.add_argument(
        "--free-order", action="store_true",
        help="drop the template-order constraint from the mask. The target is "
             "still sorted, so this only widens what a decode may do -- and lets a "
             "wrong early pick be recovered from instead of closing everything "
             "below it",
    )
    p.add_argument(
        "--beam", type=int, nargs="+", default=[1],
        help="beam widths to report; 1 is greedy decoding. The first drives the "
             "per-epoch probe, and every one of them gets a final holdout figure "
             "and, under --eval-games, an arm of its own",
    )
    p.add_argument(
        "--augment", action="store_true",
        help="relabel the colours at random, one reading per row per epoch. Needs "
             "--free-order, because the label order is by template index and a "
             "relabelling permutes it",
    )
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--probe", type=int, default=400, help="holdout states decoded per epoch")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-games", type=int, default=0)
    p.add_argument("--out", type=pathlib.Path, default=None)
    p.add_argument("--init-from", type=pathlib.Path, default=None)
    p.add_argument("--log-json", type=pathlib.Path, default=None)
    args = p.parse_args()

    cfg = CONFIG_BY_NAME[args.config]
    monotone = not args.free_order
    if args.augment and monotone:
        p.error("--augment needs --free-order")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    started = time.perf_counter()
    data = Steps(cfg, args.config, list(args.data), monotone)
    train = np.flatnonzero(~data.row_holdout)
    test = np.flatnonzero(data.row_holdout)
    train_states = np.flatnonzero(~data.holdout)
    held_states = np.flatnonzero(data.holdout)
    print(
        f"config={args.config} states={len(data.sequence):,} "
        f"rows={len(data):,} holdout={len(test) / max(len(data), 1):.1%} "
        f"actions={n_actions(cfg)} state_dim={state_dim(cfg)} "
        f"order={'template' if monotone else 'free'} "
        f"({time.perf_counter() - started:.1f}s to materialise)",
        flush=True,
    )

    net = RepartitionNet(cfg, hidden=args.hidden, key=args.key)
    if args.init_from:
        net.load_state_dict(torch.load(args.init_from, weights_only=True)["state"])
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scorer = Scorer(net)
    print(f"params {sum(q.numel() for q in net.parameters()):,}", flush=True)

    def accuracy(states: np.ndarray) -> tuple[float, float]:
        """Step accuracy, and the fraction of sequences right at every step.

        Scored over **whole** sequences: a subsample of rows would let a state
        count as exactly matched on the two steps that happened to be drawn, which
        reads as a 55% exact match where the truth is a percent.
        """
        rows = data.rows_of(states)
        correct = np.zeros(len(rows), dtype=bool)
        net.eval()
        with torch.no_grad():
            for start in range(0, len(rows), 4096):
                chunk = slice(start, start + 4096)
                state, dynamic, legal, target = data.batch(rows[chunk], monotone)
                correct[chunk] = (net(state, dynamic, legal).argmax(-1) == target).numpy()
        net.train()
        whole = np.ones(len(data.sequence), dtype=bool)
        np.logical_and.at(whole, data.owner[rows], correct)
        return float(correct.mean()), float(whole[states].mean())

    history: list[dict] = []
    best_probe, best_epoch = -1.0, 0
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(train)
        loss_sum = steps = 0.0
        began = time.perf_counter()
        n_relabel = len(data.template_to)
        for start in range(0, len(order) - args.batch + 1, args.batch):
            rows = order[start : start + args.batch]
            relabel = rng.integers(n_relabel, size=len(rows)) if args.augment else None
            state, dynamic, legal, target = data.batch(rows, monotone, relabel)
            loss = nn.functional.cross_entropy(net(state, dynamic, legal), target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            loss_sum += float(loss.detach())
            steps += 1

        train_step, train_exact = accuracy(rng.permutation(train_states)[:2_000])
        test_step, test_exact = accuracy(held_states)
        probe = playable_rate(
            cfg, data, scorer, rng.permutation(held_states)[: args.probe], args.beam[0], monotone
        )
        print(
            f"epoch {epoch:>3}  loss {loss_sum / max(steps, 1):>7.4f}  "
            f"step {train_step:>6.1%}/{test_step:>6.1%}  "
            f"exact {train_exact:>6.1%}/{test_exact:>6.1%}  "
            f"valid {probe['valid']:>6.1%}  playable {probe['playable']:>6.1%}  "
            f"tiles {probe['tiles']:>4.2f} vs {probe['solver_tiles']:>4.2f}  "
            f"{probe['decodes_per_second']:>5.0f} dec/s  "
            f"{time.perf_counter() - began:>5.1f}s",
            flush=True,
        )
        history.append(
            {
                "epoch": epoch,
                "loss": loss_sum / max(steps, 1),
                "train_step": train_step,
                "train_exact": train_exact,
                "test_step": test_step,
                "test_exact": test_exact,
                **{f"holdout_{k}": v for k, v in probe.items()},
            }
        )
        # Selected on the probe, not on the last epoch: step accuracy keeps
        # improving on the training split long after the decode stops getting
        # better, and it is the decode that is the deliverable.
        if probe["playable"] > best_probe:
            best_probe, best_epoch = probe["playable"], epoch
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
            if args.out:
                args.out.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "cfg": args.config,
                        "hidden": args.hidden,
                        "key": args.key,
                        "monotone": monotone,
                        "epoch": epoch,
                        "state": best_state,
                    },
                    args.out,
                )

    if best_state is not None:
        net.load_state_dict(best_state)
        print(f"\nkept epoch {best_epoch} (probe playable {best_probe:.1%})", flush=True)

    final = {}
    for beam in args.beam:
        final[beam] = playable_rate(cfg, data, scorer, held_states, beam, monotone)
        print(
            f"\nholdout ({len(held_states):,} states, beam {beam})  "
            f"valid {final[beam]['valid']:.1%}  playable {final[beam]['playable']:.1%}  "
            f"tiles {final[beam]['tiles']:.2f} against CP-SAT's "
            f"{final[beam]['solver_tiles']:.2f}  "
            f"{final[beam]['decodes_per_second']:.0f} decodes/s",
            flush=True,
        )

    scores: list[dict] = []
    if args.eval_games:
        suite = suite_for(args.config)
        for beam in args.beam:
            label = f"neural-repartition (beam {beam})"
            agent = NeuralRepartition(cfg, scorer, beam=beam, monotone=monotone)
            began = time.perf_counter()
            result = evaluate(label, suite, build_agent=lambda c, a=agent: a, games=args.eval_games)
            print(
                f"  {label:28s} win {result.win_rate:>6.1%}  score {result.mean_score:>+8.2f}  "
                f"illegal {result.illegal_attempts}  n={result.games}  "
                f"asked {agent.asked:,} answered {agent.answered:,}  "
                f"{time.perf_counter() - began:.0f}s",
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
        # The two ends of the ruler, re-measured inside this run so the middle is
        # not compared against numbers a different harness produced.
        for label, repartition in (("by_value", False), ("by_value+repartition", True)):
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
                    "data": [str(path) for path in args.data],
                    "states": len(data.sequence),
                    "rows": len(data),
                    "monotone": monotone,
                    "beam": args.beam,
                    "holdout_by_beam": {str(k): v for k, v in final.items()},
                    "hidden": args.hidden,
                    "key": args.key,
                    "lr": args.lr,
                    "batch": args.batch,
                    "augment": bool(args.augment),
                    "weight_decay": args.weight_decay,
                    "seed": args.seed,
                    "best_epoch": best_epoch,
                    "history": history,
                    "eval": scores,
                },
                indent=2,
            )
        )
        print(f"wrote {args.log_json}", flush=True)


if __name__ == "__main__":
    main()
