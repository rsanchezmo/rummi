"""Imitating whole turns in the env's own vocabulary, one primitive at a time.

    python tools/collect_turns.py --seed 0 --out checkpoints/turns-s0.npz
    python tools/train_primitive_turn.py --data checkpoints/turns-s*.npz --epochs 20

The picker experiments established that the right objective for *construction* is
an outcome-scored sequence: legal by mask at every step, several candidates kept,
the finished ones ranked by tiles played. That recipe has only ever been run over
the ~331 set templates. This runs it over `PLACE`/`PICK`/`DISSOLVE`/`ASSIGN`/
`END_TURN` instead, changing one variable -- the vocabulary the decoder emits --
and leaving the teacher, the warm start and the ruler where they are.

Stage one is teacher forcing along `frugal`'s own turns, replayed on
`learned/turn_sim.py` so the network sees at every step exactly the observation
and mask the env would have handed it. `TorchPolicy` is the net, unchanged: it is
the primitive-space policy `tools/train_ppo.py` already trains, and using anything
else would add a second variable.

Three numbers, and as in `tools/train_repartition.py` only the third is the
deliverable:

- **step accuracy** -- how often the argmax is the teacher's next primitive.
- **exact match** -- the whole turn, teacher-forced.
- **committed rate** -- decode from the turn boundary with no teacher, and check
  whether what comes back is a turn that sheds a tile. That is what an agent needs;
  the other two only say how the fit is going.

The breakdown that matters is by **sequence length**, reported per bucket and split
by whether the teacher's turn used the `REPARTITION` macro: an ordinary turn is a
dozen primitives, one with a repartition in it two dozen, and the question this
experiment exists to answer is where along that axis the decode stops arriving.

**What it measured.** 60,005 turns from 4,020 games (924,178 primitive decisions,
of which the simulator refused zero), 70 epochs at `--batch 512 --hidden 512,512`,
kept at epoch 44 on the probe. Held-out step accuracy 76.5% against 90.5% on train
and whole-turn exact match 8.8% -- and neither is the story. Decoding from the
boundary with no teacher commits a turn in **26.9% of held-out turns at greedy and
54.9% at beam 4**, and where it commits it plays 2.22 tiles against the teacher's
2.19: the turn it finds is the teacher's turn, it just finds one far less often.
The length breakdown and what it says are in `docs/EXPERIMENTS.md`.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import time
from collections.abc import Iterator

import numpy as np
import torch
from torch import nn

from rummi.agents.learned.architecture import Architecture
from rummi.agents.learned.primitive_turn import Scorer, decode_turns
from rummi.agents.learned.torch_net import TorchPolicy
from rummi.agents.learned.turn_sim import TurnStart, to_state
from rummi.env.numpy.engine import step as engine_step
from rummi.env.numpy.masks import legal_actions
from rummi.env.numpy.sets import summarize
from rummi.env.observation import encode
from rummi.rules.config import CONFIG_BY_NAME, RummiConfig

BUCKETS: tuple[tuple[int, int], ...] = (
    (1, 3), (4, 6), (7, 10), (11, 20), (21, 35), (36, 10_000)
)
"""Teacher-plan lengths the length breakdown is reported over."""


def bucket_of(length: int) -> int:
    return next(i for i, (lo, hi) in enumerate(BUCKETS) if lo <= length <= hi)


def bucket_name(index: int) -> str:
    lo, hi = BUCKETS[index]
    return f"{lo}-{hi}" if hi < 10_000 else f"{lo}+"


class Turns:
    """The recorded turns of one or more collections, as one dataset.

    Nothing is materialised per *step*: a turn is replayed on the simulator when it
    is used, which costs about what the forward pass over it costs and avoids
    holding half a gigabyte of mid-turn tables. :func:`replay` is that one walk, and
    the loss, the accuracy and the tests all read it.

    `tag` is whatever the collection recorded beside each row -- the `REPARTITION`
    flag for turns, the solver's tile count for gate states -- because the two
    populations are otherwise the same shape and deserve the same loader.
    """

    def __init__(
        self, cfg: RummiConfig, name: str, files: list[pathlib.Path], prefix: str = "turn"
    ) -> None:
        from collect_turns import load_starts

        starts, plans, holdout, tags, drawn = [], [], [], [], []
        for path in files:
            with np.load(path, allow_pickle=False) as data:
                if str(data["config"]) != name:
                    raise SystemExit(f"{path} was collected on {data['config']}, not {name}")
                starts.append(load_starts(data, prefix))
                lengths = data[f"{prefix}_len"]
                rows = data[f"{prefix}_plan"]
                plans.extend(
                    rows[i, : lengths[i]].astype(np.int64).tolist()
                    for i in range(len(lengths))
                )
                holdout.append(data[f"{prefix}_holdout"])
                tags.append(
                    data[f"{prefix}_repart"] if prefix == "turn" else data[f"{prefix}_tiles"]
                )
                # A gate state is one the solver answered, so it is never a decline;
                # a collection made before declines were labelled carries none either.
                drawn.append(
                    data["turn_drawn"]
                    if prefix == "turn" and "turn_drawn" in data
                    else np.zeros(len(lengths), dtype=bool)
                )
        self.cfg = cfg
        self.starts = TurnStart.stack(starts)
        self.plans = plans
        self.length = np.asarray([len(p) for p in plans], dtype=np.int64)
        self.holdout = np.concatenate(holdout)
        self.tag = np.concatenate(tags)
        self.drawn = np.concatenate(drawn)
        """The teacher declined this position, and the one label is `DRAW`."""
        self.bucket = np.asarray([bucket_of(int(n)) for n in self.length])
        # What the teacher shed: a `PLACE` is the only action that moves a tile out
        # of the rack, and it is the lowest block of the layout.
        self.tiles = np.asarray(
            [sum(1 for a in plan if a < cfg.pick_offset) for plan in plans], dtype=np.int64
        )

    def __len__(self) -> int:
        return len(self.plans)

    def batch(self, index: np.ndarray) -> tuple[TurnStart, list[list[int]]]:
        return self.starts.take(index), [self.plans[i] for i in index.tolist()]


def replay(
    cfg: RummiConfig,
    starts: TurnStart,
    plans: list[list[int]],
    rejected: list[int] | None = None,
) -> Iterator[tuple[dict, np.ndarray, np.ndarray, np.ndarray]]:
    """Walk every plan on the simulator, yielding `(obs, mask, target, rows)` per depth.

    Every turn advances in lockstep, so a batch costs one mask and one forward pass
    per depth rather than one per turn. `rows` indexes into `plans`, which is what
    lets a consumer attribute a step to its turn -- a whole-sequence exact match
    needs that and a mean step accuracy does not.

    A recorded action the mask refuses drops its turn into `rejected`. That cannot
    happen for a plan the teacher actually played if this simulator is faithful, so
    the counter is a drift alarm rather than a tolerance, and it reads zero.
    """
    state = to_state(cfg, starts)
    rows = np.arange(len(plans))
    depth = 0
    while state.batch_size:
        summary = summarize(cfg, state.table_sets)
        mask = legal_actions(state, summary)
        target = np.asarray([plans[i][depth] for i in rows], dtype=np.int64)
        ok = mask[np.arange(len(rows)), target]
        if not ok.all():
            if rejected is None:
                raise ValueError(
                    f"recorded action {int(target[~ok][0])} is masked out at depth {depth}"
                )
            rejected.extend(rows[~ok].tolist())
            live = np.flatnonzero(ok)
            if not live.size:
                return
            state = state.select(live)
            rows, mask, target = rows[live], mask[live], target[live]
            summary = summarize(cfg, state.table_sets)

        yield encode(state, summary), mask, target, rows

        depth += 1
        more = np.flatnonzero([depth < len(plans[i]) for i in rows])
        if not more.size:
            return
        state = state.select(more)
        engine_step(state, target[more], mask[more])
        rows = rows[more]


@dataclasses.dataclass(slots=True)
class Forced:
    """What one teacher-forced pass over a batch of turns produced."""

    loss: torch.Tensor
    """Summed cross-entropy, with a tape when the pass asked for one."""
    steps: int
    correct: int
    exact: np.ndarray


def teacher_forced(
    cfg: RummiConfig,
    net: TorchPolicy,
    starts: TurnStart,
    plans: list[list[int]],
    grad: bool = False,
) -> Forced:
    total = torch.zeros((), dtype=torch.float32)
    steps = correct = 0
    exact = np.ones(len(plans), dtype=bool)
    for obs, mask, target, rows in replay(cfg, starts, plans):
        labels = torch.from_numpy(target)
        with torch.set_grad_enabled(grad):
            logits, _ = net(obs, torch.from_numpy(mask))
            total = total + nn.functional.cross_entropy(logits, labels, reduction="sum")
        hit = (logits.detach().argmax(-1) == labels).numpy()
        np.logical_and.at(exact, rows, hit)
        correct += int(hit.sum())
        steps += len(rows)
    return Forced(total, steps, correct, exact)


def forced_accuracy(
    cfg: RummiConfig, net: TorchPolicy, data: Turns, index: np.ndarray, batch: int
) -> tuple[float, float]:
    """Step accuracy and whole-turn exact match over `index`, teacher-forced.

    Scored over **whole** turns rather than a sample of steps, for the reason
    `tools/train_repartition.py` gives: a subsample lets a turn count as exactly
    matched on the two steps that happened to be drawn.
    """
    net.eval()
    steps = correct = 0
    exact = []
    for start in range(0, len(index), batch):
        forced = teacher_forced(cfg, net, *data.batch(index[start : start + batch]))
        steps += forced.steps
        correct += forced.correct
        exact.append(forced.exact)
    net.train()
    return correct / max(steps, 1), float(np.concatenate(exact).mean())


@dataclasses.dataclass(slots=True)
class Decodes:
    """What decoding a set of positions with no teacher came back with.

    Every rate here is over the positions the teacher **played**: a decline carries
    no turn to reproduce, so counting one as a failure to commit would move the
    metric by however many declines the collection holds rather than by anything
    the decoder did. `declined` is the other half, scored on the declines alone.
    """

    committed: np.ndarray
    tiles: np.ndarray
    teacher_tiles: np.ndarray
    ms_per_turn: float
    drawn: np.ndarray
    """The teacher declined this position."""
    depth: np.ndarray
    """Primitives the deepest hypothesis explored."""
    declined: np.ndarray
    """The decode emitted `DRAW` somewhere."""
    budget: np.ndarray
    """Micro-actions the turn had left, which is where wandering ends."""
    teacher_length: np.ndarray

    def summary(self) -> dict[str, float]:
        played = ~self.drawn
        hit = self.committed & played
        return {
            "committed": float(self.committed[played].mean()) if played.any() else 0.0,
            "tiles": float(self.tiles[hit].mean()) if hit.any() else 0.0,
            "teacher_tiles": float(self.teacher_tiles[played].mean()) if played.any() else 0.0,
            "ms_per_turn": self.ms_per_turn,
            "declines": int(self.drawn.sum()),
            "declined": (
                float((~self.committed[self.drawn]).mean()) if self.drawn.any() else 0.0
            ),
        }

    def abandonment(self) -> dict[str, float]:
        """Where the decodes that found nothing on a played position ended.

        A spent micro budget masks everything but `DRAW`, so *every* failure emits
        it in the end and the plan alone cannot tell a decision to decline from a
        wander that ran out of room. Depth is what separates them.
        """
        failed = ~self.committed & ~self.drawn
        if not failed.any():
            return {"n": 0, "declined": 0.0, "ran_out": 0.0, "depth": 0.0, "teacher": 0.0}
        ran_out = self.depth[failed] >= self.budget[failed]
        return {
            "n": int(failed.sum()),
            "declined": float((self.declined[failed] & ~ran_out).mean()),
            "ran_out": float(ran_out.mean()),
            "depth": float(self.depth[failed].mean()),
            "teacher": float(self.teacher_length[failed].mean()),
        }


def decode_report(
    cfg: RummiConfig, scorer, data: Turns, index: np.ndarray, beam: int, batch: int
) -> Decodes:
    """Decode from each position with no teacher, and say what came back."""
    started = time.perf_counter()
    committed = np.zeros(len(index), dtype=bool)
    tiles = np.zeros(len(index), dtype=np.int64)
    depth = np.zeros(len(index), dtype=np.int64)
    declined = np.zeros(len(index), dtype=bool)
    for start in range(0, len(index), batch):
        chunk = index[start : start + batch]
        for i, row in enumerate(decode_turns(cfg, scorer, data.starts.take(chunk), beam)):
            committed[start + i] = row.plays
            tiles[start + i] = row.tiles if row.plays else 0
            depth[start + i] = row.depth
            declined[start + i] = row.declined
    return Decodes(
        committed=committed,
        tiles=tiles,
        teacher_tiles=data.tiles[index],
        ms_per_turn=1000 * (time.perf_counter() - started) / max(len(index), 1),
        drawn=data.drawn[index],
        depth=depth,
        declined=declined,
        budget=cfg.max_micro_per_turn - data.starts.micro_count[index].astype(np.int64),
        teacher_length=data.length[index],
    )


def print_abandonment(reports: dict[int, Decodes]) -> None:
    """The failure diagnostic: what the decodes that came back empty were doing."""
    print("\nabandonment (played holdout turns the decode did not commit)")
    print(f"  {'beam':>4} {'n':>7} {'declined':>9} {'ran out':>8} {'depth':>7} {'teacher':>8}")
    for beam, report in reports.items():
        row = report.abandonment()
        print(
            f"  {beam:>4} {row['n']:>7,} {row['declined']:>8.1%} {row['ran_out']:>7.1%} "
            f"{row['depth']:>7.1f} {row['teacher']:>8.1f}",
            flush=True,
        )


def length_table(data: Turns, index: np.ndarray, reports: dict[int, Decodes]) -> list[dict]:
    """Committed rate per teacher-plan-length bucket, ordinary turns and stuck apart.

    Declines are left out: their one label is `DRAW`, so bucketing them by length
    would put every one of them in the shortest bucket and read as a construction
    that never arrives.
    """
    out: list[dict] = []
    stuck = data.tag.astype(bool)[index] & ~data.drawn[index]
    played = ~data.drawn[index]
    buckets = data.bucket[index]
    for kind, mine in (("ordinary", played & ~stuck), ("repartition", stuck)):
        for b in range(len(BUCKETS)):
            rows = mine & (buckets == b)
            if not rows.any():
                continue
            row: dict = {"kind": kind, "bucket": bucket_name(b), "n": int(rows.sum())}
            for beam, report in reports.items():
                row[f"beam{beam}"] = float(report.committed[rows].mean())
            out.append(row)
    return out


def print_length_table(rows: list[dict], beams: list[int]) -> None:
    header = "  ".join(f"beam{b:<2}" for b in beams)
    print(f"\nlength breakdown (holdout)\n  kind          bucket      n   {header}")
    for row in rows:
        cells = "  ".join(f"{row[f'beam{b}']:>6.1%}" for b in beams)
        print(f"  {row['kind']:<12} {row['bucket']:>7} {row['n']:>6}   {cells}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="standard", choices=sorted(CONFIG_BY_NAME))
    p.add_argument("--data", type=pathlib.Path, nargs="+", required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=256, help="turns per optimiser step")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--hidden", default="512,512",
        help="trunk widths. The default matches the template picker's trunk, so the "
             "two arms differ in vocabulary and not in capacity",
    )
    p.add_argument("--beam", type=int, nargs="+", default=[1, 4])
    p.add_argument("--probe", type=int, default=512, help="holdout turns decoded per epoch")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=pathlib.Path, default=None)
    p.add_argument("--log-json", type=pathlib.Path, default=None)
    args = p.parse_args()

    cfg = CONFIG_BY_NAME[args.config]
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    began = time.perf_counter()
    data = Turns(cfg, args.config, list(args.data))
    train = np.flatnonzero(~data.holdout)
    held = np.flatnonzero(data.holdout)
    kept = data.length[~data.drawn]
    print(
        f"config={args.config} turns={len(data):,} holdout={len(held):,} "
        f"steps={int(data.length.sum()):,} "
        f"declines={data.drawn.mean():.1%} of turns, "
        f"{data.drawn.sum() / max(int(data.length.sum()), 1):.1%} of steps "
        f"repartition={(data.tag.astype(bool) & ~data.drawn).mean():.1%} "
        f"length mean {kept.mean():.1f} median {np.median(kept):.0f} "
        f"max {kept.max()} ({time.perf_counter() - began:.1f}s to load)",
        flush=True,
    )

    arch = Architecture(hidden=tuple(int(w) for w in args.hidden.split(",")))
    net = TorchPolicy(cfg, arch, seed=args.seed)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    scorer = Scorer(net)
    print(f"params {sum(q.numel() for q in net.parameters()):,}", flush=True)

    # Drawn from the played holdout, so the selection metric is the same one the
    # cell without declines selected on.
    probe_rows = rng.permutation(held[~data.drawn[held]])[: args.probe]
    history: list[dict] = []
    best_probe, best_epoch = -1.0, 0
    best_state = {k: v.clone() for k, v in net.state_dict().items()}
    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(train)
        loss_sum = steps = 0.0
        started = time.perf_counter()
        net.train()
        for start in range(0, len(order) - args.batch + 1, args.batch):
            forced = teacher_forced(cfg, net, *data.batch(order[start : start + args.batch]), grad=True)
            loss = forced.loss / max(forced.steps, 1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            loss_sum += float(loss.detach()) * forced.steps
            steps += forced.steps

        train_step, train_exact = forced_accuracy(
            cfg, net, data, rng.permutation(train)[:2_000], args.batch
        )
        test_step, test_exact = forced_accuracy(cfg, net, data, held, args.batch)
        probe = decode_report(cfg, scorer, data, probe_rows, args.beam[0], args.batch).summary()
        print(
            f"epoch {epoch:>3}  loss {loss_sum / max(steps, 1):>7.4f}  "
            f"step {train_step:>6.1%}/{test_step:>6.1%}  "
            f"exact {train_exact:>6.1%}/{test_exact:>6.1%}  "
            f"committed {probe['committed']:>6.1%}  "
            f"tiles {probe['tiles']:>4.2f} vs {probe['teacher_tiles']:>4.2f}  "
            f"{probe['ms_per_turn']:>5.1f} ms/turn  "
            f"{time.perf_counter() - started:>5.0f}s",
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
                **probe,
            }
        )
        # Selected on the decode, not on the last epoch: step accuracy keeps
        # improving long after the decode stops getting better, and the decode is
        # the deliverable.
        if probe["committed"] > best_probe:
            best_probe, best_epoch = probe["committed"], epoch
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
            if args.out:
                args.out.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "cfg": args.config,
                        "hidden": list(arch.hidden),
                        "epoch": epoch,
                        "state": best_state,
                    },
                    args.out,
                )

    net.load_state_dict(best_state)
    print(f"\nkept epoch {best_epoch} (probe committed {best_probe:.1%})", flush=True)

    reports = {beam: decode_report(cfg, scorer, data, held, beam, args.batch) for beam in args.beam}
    for beam, report in reports.items():
        summary = report.summary()
        print(
            f"holdout ({len(held):,} turns, beam {beam})  "
            f"committed {summary['committed']:.1%}  tiles {summary['tiles']:.2f} "
            f"against the teacher's {summary['teacher_tiles']:.2f}  "
            f"declined {summary['declined']:.1%} of {summary['declines']:,}  "
            f"{summary['ms_per_turn']:.1f} ms/turn",
            flush=True,
        )
    print_abandonment(reports)
    rows = length_table(data, held, reports)
    print_length_table(rows, args.beam)

    if args.log_json:
        args.log_json.parent.mkdir(parents=True, exist_ok=True)
        args.log_json.write_text(
            json.dumps(
                {
                    "config": args.config,
                    "data": [str(path) for path in args.data],
                    "turns": len(data),
                    "steps": int(data.length.sum()),
                    "hidden": list(arch.hidden),
                    "lr": args.lr,
                    "batch": args.batch,
                    "seed": args.seed,
                    "best_epoch": best_epoch,
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
