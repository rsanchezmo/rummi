"""The same CP-SAT repartitions, imitated as break-then-cover.

    python tools/train_two_phase.py --data checkpoints/repartitions-s*.npz \
        --epochs 40 --free-order --augment --hidden 512 --key 128 --beam 1 4

`tools/train_repartition.py` learns a repartition as ~13 choices of ~330. The
measured wall there is sequence length, not capacity: 8.7 of the mean solve's 12.7
sets are already on the table, so most of those choices re-derive a partition the
solver only edited. `rummi/agents/learned/two_phase_net.py` splits the decision the
way the label actually decomposes -- **2.9 slots to break, 4.0 sets to build** --
and this trains both heads on the same shards, the same game-split holdout and the
same colour augmentation, so the playable rates are like for like.

The two heads share no parameters and no batch; they are trained in one loop
because they are one deliverable, and `--init-cover` starts the cover head from a
one-phase checkpoint, whose action space it is exactly.

The labels come from the same `.npz` files: `decompose` matches each solution set
against the board's slot contents, and only an exact content match counts as kept
-- the same rule `to_actions.plan` applies when it decides what to dissolve.

**What it measured.** All 48,002 states decompose -- 92.37% with every set
reproduced tile for tile, 7.63% with a joker resting elsewhere, none
unrepresentable -- into 3.89 break decisions of ~12 and 4.99 cover decisions of
~330, against the one-phase space's 13.6 of ~330. 90 epochs,
`--free-order --augment --hidden 512 --key 128 --init-cover repart-aug2.pt`, kept
at epoch 33: break 58.8% step / 20.6% whole-subset, cover 80.9% / 38.2%.

| decode | holdout valid | holdout plays a tile | `standard-greedy`, n=480 |
|---|---|---|---|
| greedy | 49.7% (49.8) | **35.5%** (27.8) | +43.88 / 99.2% (+42.31) |
| beam 4 | 86.4% (75.9) | **56.2%** (46.5) | +45.86 / 99.8% (+45.26) |
| beam 16 | 99.0% (92.1) | **72.4%** (66.4) | +48.31 / 99.4% (+47.51) |
| CP-SAT | 100% | 100% | +47.71 / 99.8% |

with the one-phase figure in brackets and `by_value` at +28.01 / 82.1% in the same
run, so beam 4 keeps **90.6%** of the solver's 19.70-point contribution where the
one-phase space needed beam 16 for 99%; on the same 1,200 states in one process
that is 5.2 ms against 24.6 ms, and 3.3x CP-SAT's own 17.4 ms. Greedy keeps 80.6%
at 2.4 ms, up from 72.6% -- better, and still short of 90%.

Three controls. Without `--init-cover` the same recipe reaches 30.2% / 48.3%, so
the decomposition carries most of it and the warm start is worth ~5pp. The break
head overfits from epoch ~20 (holdout step 59.9% -> 52.8% while its loss keeps
falling) and `--weight-decay 1e-4` fixes exactly that without moving the
deliverable at all: 34.2% / 55.5%. And widening phase A alone -- `breaks` at 8
under a greedy cover -- buys 37.1% -> 38.5% for 3.2x the time. **The break choice
is not what the beam is buying;** the cover is.
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
    apply_template,
    candidate_features,
    colour_relabellings,
    feasible,
    label_sequence,
    present_counts,
    relabel_rows,
    state_features,
    stop_action,
)
from rummi.agents.learned.set_encoder import EncoderSpec
from rummi.agents.learned.two_phase_net import (
    TwoPhaseNet,
    TwoPhaseScorer,
    break_dynamic,
    break_feasible,
    break_state_features,
    decode_two_phase,
    decompose,
    freed_counts,
    slot_counts,
    slot_static,
    stop_break,
    two_phase_from_checkpoint,
)
from rummi.agents.macro import MacroAgent, by_value
from rummi.evaluate.protocol import SUITE_BY_NAME, evaluate
from rummi.rules.config import CONFIG_BY_NAME, RummiConfig
from rummi.rules.observation import MICRO_COUNT


class TwoPhase:
    """Every decision of every labelled repartition, split into the two phases.

    Phase A's rows are tiny -- the freed pool and which slot was broken last -- so
    they are materialised whole; the slot block itself is rebuilt per batch from
    the stored board, because one `(35, 53)` description per state would cost more
    than the `bincount` that derives it.
    """

    def __init__(
        self, cfg: RummiConfig, name: str, files: list[pathlib.Path], monotone: bool
    ) -> None:
        racks, boards, solutions, played, holdouts = [], [], [], [], []
        for path in files:
            with np.load(path, allow_pickle=False) as data:
                if str(data["config"]) != name:
                    raise SystemExit(f"{path} was collected on {data['config']}, not {name}")
                racks.append(data["rack"])
                boards.append(data["board"])
                solutions.append(data["sets"])
                played.append(data["played"])
                holdouts.append(data["holdout"])
        self.cfg = cfg
        rack = np.concatenate(racks)
        board = np.concatenate(boards)
        solution = np.concatenate(solutions)
        every_played = np.concatenate(played)
        every_holdout = np.concatenate(holdouts)

        stop_a, stop_b = stop_break(cfg), stop_action(cfg)
        keep: list[int] = []
        breaks: list[tuple[int, ...]] = []
        covers: list[tuple[int, ...]] = []
        kept_counts: list[int] = []
        self.verdicts = {"exact": 0, "relaxed": 0, "none": 0}
        for i in range(len(rack)):
            sets = [
                tuple(int(k) for k in row if k >= 0) for row in solution[i] if (row >= 0).any()
            ]
            kept, broken, to_build = decompose(cfg, board[i], sets)
            need, avail = freed_counts(cfg, board[i], broken, rack[i])
            sequence, verdict = label_sequence(cfg, need, avail, to_build)
            self.verdicts[verdict] += 1
            if verdict == "none":
                continue
            keep.append(i)
            breaks.append(broken)
            covers.append(tuple(int(t) for t in sequence))
            kept_counts.append(len(kept))

        rows = np.asarray(keep)
        self.rack = rack[rows]
        self.board = board[rows]
        self.played = every_played[rows]
        self.holdout = every_holdout[rows]
        self.broken = breaks
        self.cover = covers
        self.kept = np.asarray(kept_counts, dtype=np.int64)

        # Phase A: one row per break plus the STOP that ends the subset.
        a_owner, a_freed, a_last, a_count, a_target = [], [], [], [], []
        for i, broken in enumerate(breaks):
            freed = np.zeros(cfg.n_kinds, dtype=np.int64)
            last = -1
            for step in range(len(broken) + 1):
                a_owner.append(i)
                a_freed.append(freed.astype(np.int8))
                a_last.append(last)
                a_count.append(step)
                if step == len(broken):
                    a_target.append(stop_a)
                    break
                slot = broken[step]
                a_target.append(slot)
                row = self.board[i][slot]
                freed = freed.copy()
                for kind in row[row >= 0]:
                    freed[int(kind)] += 1
                last = slot
        self.a_owner = np.asarray(a_owner, dtype=np.int64)
        self.a_freed = np.stack(a_freed)
        self.a_last = np.asarray(a_last, dtype=np.int64)
        self.a_count = np.asarray(a_count, dtype=np.int64)
        self.a_target = np.asarray(a_target, dtype=np.int64)
        self.a_offset = np.concatenate(
            [[0], np.cumsum([len(b) + 1 for b in breaks])]
        ).astype(np.int64)

        # Phase B: the one-phase construction, over the freed tiles alone.
        b_owner, b_need, b_avail, b_present = [], [], [], []
        b_sets, b_last, b_target = [], [], []
        for i, sequence in enumerate(covers):
            need, avail = freed_counts(cfg, self.board[i], breaks[i], self.rack[i])
            present = present_counts(cfg, self.board[i][list(breaks[i])])
            last = 0 if monotone else -1
            for step in range(len(sequence) + 1):
                b_owner.append(i)
                b_need.append(need.astype(np.int8))
                b_avail.append(avail.astype(np.int8))
                b_present.append(np.minimum(present, 255).astype(np.uint8))
                b_sets.append(self.kept[i] + step)
                b_last.append(last)
                if step == len(sequence):
                    b_target.append(stop_b)
                    break
                chosen = sequence[step]
                b_target.append(chosen)
                need, avail = apply_template(cfg, need, avail, chosen)
                present = present.copy()
                present[chosen] = max(present[chosen] - 1.0, 0.0)
                last = chosen
        self.b_owner = np.asarray(b_owner, dtype=np.int64)
        self.b_need = np.stack(b_need)
        self.b_avail = np.stack(b_avail)
        self.b_present = np.stack(b_present)
        self.b_sets = np.asarray(b_sets, dtype=np.int64)
        self.b_last = np.asarray(b_last, dtype=np.int64)
        self.b_target = np.asarray(b_target, dtype=np.int64)
        self.b_offset = np.concatenate(
            [[0], np.cumsum([len(s) + 1 for s in covers])]
        ).astype(np.int64)

        self.a_holdout = self.holdout[self.a_owner]
        self.b_holdout = self.holdout[self.b_owner]
        self.kind_to, self.template_to = colour_relabellings(cfg)

    def __len__(self) -> int:
        return len(self.rack)

    def rows_of(self, offset: np.ndarray, states: np.ndarray) -> np.ndarray:
        """Every row of the given states, in order -- what an exact match needs."""
        return np.concatenate(
            [np.arange(offset[s], offset[s + 1]) for s in states.tolist()]
        )

    def a_batch(self, index: np.ndarray, relabel: np.ndarray | None = None):
        """`(state, static, dynamic, legal, target)` for phase-A rows, as tensors.

        A relabelling permutes the kind columns of the slot block and of the pools;
        the target is a slot index, which it leaves alone -- which is the whole
        reason phase A is order-free where phase B is not.
        """
        cfg = self.cfg
        owner = self.a_owner[index]
        counts = slot_counts(cfg, self.board[owner]).astype(np.int64)
        rack = self.rack[owner].astype(np.int64)
        freed = self.a_freed[index].astype(np.int64)
        if relabel is not None:
            kind_back = np.argsort(self.kind_to, axis=1)[relabel]
            counts = np.take_along_axis(counts, kind_back[:, None, :], axis=2)
            rack = np.take_along_axis(rack, kind_back, axis=1)
            freed = np.take_along_axis(freed, kind_back, axis=1)
        last = self.a_last[index]
        state = break_state_features(
            cfg,
            rack,
            counts.sum(1),
            freed,
            (counts.sum(-1) > 0).sum(-1),
            self.a_count[index],
            last,
        )
        return (
            torch.from_numpy(state),
            torch.from_numpy(slot_static(cfg, counts)),
            torch.from_numpy(break_dynamic(cfg, counts, rack, freed, last)),
            torch.from_numpy(break_feasible(cfg, counts, last)),
            torch.from_numpy(self.a_target[index]),
        )

    def b_batch(self, index: np.ndarray, monotone: bool, relabel: np.ndarray | None = None):
        """`(state, dynamic, legal, target)` for phase-B rows, as tensors."""
        cfg = self.cfg
        need = self.b_need[index].astype(np.int64)
        avail = self.b_avail[index].astype(np.int64)
        present = self.b_present[index].astype(np.float32)
        sizes, last = self.b_sets[index], self.b_last[index]
        target = self.b_target[index]
        if relabel is not None:
            need, avail, present, last, target = relabel_rows(
                cfg, relabel, need, avail, present, last, target
            )
        dynamic, short = candidate_features(cfg, need, avail, present, last)
        return (
            torch.from_numpy(state_features(cfg, need, avail, sizes, last, present)),
            torch.from_numpy(dynamic),
            torch.from_numpy(feasible(cfg, need, avail, sizes, last, short, monotone)),
            torch.from_numpy(target),
        )


class TwoPhaseRepartition(MacroAgent):
    """`by_value`, with the stuck-state solve answered by the two-phase decode.

    The same arm as `train_repartition.NeuralRepartition` with the decoder swapped,
    so the ruler compares action spaces and nothing else.
    """

    def __init__(
        self, cfg: RummiConfig, scorer: TwoPhaseScorer, beam: int = 1, monotone: bool = True
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
        self.asked += 1
        found = decode_two_phase(cfg, self.scorer, rack, board, self.beam, self.monotone)
        if found is None or found.tiles_played < 1:
            return []
        actions = plan(cfg, board, list(found.sets), found.played)
        spent = int(np.asarray(obs["scalars"])[env, MICRO_COUNT])
        if len(actions) > cfg.max_micro_per_turn - spent:
            return []
        actions.pop()
        self.answered += 1
        return actions


def playable_rate(
    cfg: RummiConfig,
    data: TwoPhase,
    scorer: TwoPhaseScorer,
    states: np.ndarray,
    beam: int,
    monotone: bool,
) -> dict[str, float]:
    """Decode from scratch and check the construction, on `states` of the dataset."""
    valid = plays = 0
    tiles = solver_tiles = 0
    started = time.perf_counter()
    for i in states.tolist():
        found = decode_two_phase(cfg, scorer, data.rack[i], data.board[i], beam, monotone)
        solver_tiles += int(data.played[i].sum())
        if found is None:
            continue
        valid += 1
        if found.tiles_played >= 1:
            plays += 1
            tiles += found.tiles_played
    n = max(len(states), 1)
    return {
        "valid": valid / n,
        "playable": plays / n,
        "tiles": tiles / max(plays, 1),
        "solver_tiles": solver_tiles / n,
        "decodes_per_second": len(states) / max(time.perf_counter() - started, 1e-9),
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
        "--cover-encoder", default="mlp", choices=("mlp", "attention"),
        help="what produces the cover head's query: the flat MLP over the count "
             "vectors, or attention over one token per kind. The break head is "
             "untouched either way, and trains identically from the same seed",
    )
    p.add_argument(
        "--cover-hidden", type=int, default=None,
        help="the cover trunk's width, defaulting to --hidden. The capacity control "
             "for an attention arm: a plain wider MLP, nothing else changed",
    )
    p.add_argument("--attn-dim", type=int, default=128)
    p.add_argument("--attn-layers", type=int, default=2)
    p.add_argument("--attn-heads", type=int, default=4)
    p.add_argument("--attn-ffn", type=int, default=256)
    p.add_argument(
        "--free-order", action="store_true",
        help="drop the template-order constraint from the phase-B mask. Phase A is "
             "always in slot order, which costs nothing: every subset is expressible "
             "that way",
    )
    p.add_argument("--beam", type=int, nargs="+", default=[1])
    p.add_argument(
        "--augment", action="store_true",
        help="relabel the colours at random, one reading per row per epoch. Needs "
             "--free-order for the phase-B rows, whose label order is by template index",
    )
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--probe", type=int, default=400, help="holdout states decoded per epoch")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-games", type=int, default=0)
    p.add_argument("--out", type=pathlib.Path, default=None)
    p.add_argument(
        "--init-cover", type=pathlib.Path, default=None,
        help="a one-phase checkpoint to start the cover head from; its action space "
             "is exactly this one's phase B",
    )
    p.add_argument(
        "--init-from", type=pathlib.Path, default=None,
        help="a two-phase checkpoint to start from. With --epochs 0 this scores one "
             "that is already trained, rather than training a second copy to score it",
    )
    p.add_argument("--log-json", type=pathlib.Path, default=None)
    args = p.parse_args()

    cfg = CONFIG_BY_NAME[args.config]
    monotone = not args.free_order
    if args.augment and monotone:
        p.error("--augment needs --free-order")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    started = time.perf_counter()
    data = TwoPhase(cfg, args.config, list(args.data), monotone)
    total = sum(data.verdicts.values())
    train_states = np.flatnonzero(~data.holdout)
    held_states = np.flatnonzero(data.holdout)
    a_train = np.flatnonzero(~data.a_holdout)
    b_train = np.flatnonzero(~data.b_holdout)
    print(
        f"config={args.config} states={len(data):,} of {total:,} "
        f"(coverage {len(data) / max(total, 1):.2%}, "
        f"exact {data.verdicts['exact'] / max(total, 1):.2%}, "
        f"relaxed {data.verdicts['relaxed'] / max(total, 1):.2%})\n"
        f"  break rows {len(data.a_target):,} ({len(data.a_target) / len(data):.2f}/state)  "
        f"cover rows {len(data.b_target):,} ({len(data.b_target) / len(data):.2f}/state)  "
        f"kept {data.kept.mean():.2f}/state\n"
        f"  order={'template' if monotone else 'free'} "
        f"({time.perf_counter() - started:.1f}s to materialise)",
        flush=True,
    )

    encoder = EncoderSpec(
        kind=args.cover_encoder,
        dim=args.attn_dim,
        layers=args.attn_layers,
        heads=args.attn_heads,
        ffn=args.attn_ffn,
    )
    net = TwoPhaseNet(
        cfg,
        hidden=args.hidden,
        key=args.key,
        cover_hidden=args.cover_hidden,
        cover_encoder=encoder,
    )
    if args.init_cover is not None:
        loaded = torch.load(args.init_cover, weights_only=True)
        width = args.cover_hidden or args.hidden
        if loaded["hidden"] != width or loaded["key"] != args.key:
            raise SystemExit(f"{args.init_cover} is {loaded['hidden']}/{loaded['key']} wide")
        net.cover.load_state_dict(loaded["state"])
    if args.init_from is not None:
        loaded = torch.load(args.init_from, weights_only=True)
        net = two_phase_from_checkpoint(cfg, loaded)
        # The resumed weights decide the architecture, so what is saved has to
        # describe them rather than the flags: a resume that did not repeat
        # --cover-encoder would otherwise label an attention cover as the default
        # MLP, and no reader could rebuild it.
        args.hidden, args.key = loaded["hidden"], loaded["key"]
        args.cover_hidden = loaded.get("cover_hidden")
        spec = loaded.get("cover_encoder")
        encoder = EncoderSpec(**spec) if spec else EncoderSpec()
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scorer = TwoPhaseScorer(net)
    print(f"params {sum(q.numel() for q in net.parameters()):,}", flush=True)

    def accuracy(states: np.ndarray) -> tuple[float, float, float, float]:
        """Step accuracy and whole-sequence accuracy, per phase.

        Scored over **whole** sequences: a subsample of rows would let a state count
        as exactly matched on the steps that happened to be drawn.
        """
        net.eval()
        out: list[float] = []
        for offset, owner, head in (
            (data.a_offset, data.a_owner, "a"),
            (data.b_offset, data.b_owner, "b"),
        ):
            rows = data.rows_of(offset, states)
            correct = np.zeros(len(rows), dtype=bool)
            with torch.no_grad():
                for begin in range(0, len(rows), 4096):
                    chunk = slice(begin, begin + 4096)
                    if head == "a":
                        state, static, dynamic, legal, target = data.a_batch(rows[chunk])
                        logits = net.breaker(state, static, dynamic, legal)
                    else:
                        state, dynamic, legal, target = data.b_batch(rows[chunk], monotone)
                        logits = net.cover(state, dynamic, legal)
                    correct[chunk] = (logits.argmax(-1) == target).numpy()
            whole = np.ones(len(data), dtype=bool)
            np.logical_and.at(whole, owner[rows], correct)
            out += [float(correct.mean()), float(whole[states].mean())]
        net.train()
        return out[0], out[1], out[2], out[3]

    history: list[dict] = []
    best_probe, best_epoch = -1.0, 0
    best_state: dict[str, torch.Tensor] | None = None
    n_relabel = len(data.template_to)
    for epoch in range(1, args.epochs + 1):
        began = time.perf_counter()
        losses = {"a": 0.0, "b": 0.0}
        counts = {"a": 0, "b": 0}
        for head, rows_all in (("a", a_train), ("b", b_train)):
            order = rng.permutation(rows_all)
            for begin in range(0, len(order) - args.batch + 1, args.batch):
                rows = order[begin : begin + args.batch]
                relabel = rng.integers(n_relabel, size=len(rows)) if args.augment else None
                if head == "a":
                    state, static, dynamic, legal, target = data.a_batch(rows, relabel)
                    logits = net.breaker(state, static, dynamic, legal)
                else:
                    state, dynamic, legal, target = data.b_batch(rows, monotone, relabel)
                    logits = net.cover(state, dynamic, legal)
                loss = nn.functional.cross_entropy(logits, target)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                losses[head] += float(loss.detach())
                counts[head] += 1

        a_step, a_exact, b_step, b_exact = accuracy(held_states)
        probe = playable_rate(
            cfg, data, scorer, rng.permutation(held_states)[: args.probe], args.beam[0], monotone
        )
        print(
            f"epoch {epoch:>3}  loss {losses['a'] / max(counts['a'], 1):>6.4f}/"
            f"{losses['b'] / max(counts['b'], 1):>6.4f}  "
            f"break {a_step:>6.1%}/{a_exact:>6.1%}  cover {b_step:>6.1%}/{b_exact:>6.1%}  "
            f"valid {probe['valid']:>6.1%}  playable {probe['playable']:>6.1%}  "
            f"tiles {probe['tiles']:>4.2f} vs {probe['solver_tiles']:>4.2f}  "
            f"{probe['decodes_per_second']:>5.0f} dec/s  "
            f"{time.perf_counter() - began:>5.1f}s",
            flush=True,
        )
        history.append(
            {
                "epoch": epoch,
                "break_loss": losses["a"] / max(counts["a"], 1),
                "cover_loss": losses["b"] / max(counts["b"], 1),
                "break_step": a_step,
                "break_exact": a_exact,
                "cover_step": b_step,
                "cover_exact": b_exact,
                **{f"holdout_{k}": v for k, v in probe.items()},
            }
        )
        # Selected on the probe, not on the last epoch: step accuracy keeps
        # improving long after the decode stops getting better.
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
                        "cover_hidden": args.cover_hidden,
                        "cover_encoder": encoder.as_dict(),
                        "monotone": monotone,
                        "epoch": epoch,
                        "state": best_state,
                    },
                    args.out,
                )

    if best_state is not None:
        net.load_state_dict(best_state)
        print(f"\nkept epoch {best_epoch} (probe playable {best_probe:.1%})", flush=True)

    train_a, _, train_b, _ = accuracy(rng.permutation(train_states)[:2_000])
    print(f"train step  break {train_a:.1%}  cover {train_b:.1%}", flush=True)

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
        suite = SUITE_BY_NAME["standard-greedy" if args.config == "standard" else "tiny"]
        for beam in args.beam:
            label = f"two-phase (beam {beam})"
            agent = TwoPhaseRepartition(cfg, scorer, beam=beam, monotone=monotone)
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
                    "states": len(data),
                    "coverage": len(data) / max(total, 1),
                    "verdicts": data.verdicts,
                    "break_rows": len(data.a_target),
                    "cover_rows": len(data.b_target),
                    "monotone": monotone,
                    "beam": args.beam,
                    "holdout_by_beam": {str(k): v for k, v in final.items()},
                    "hidden": args.hidden,
                    "key": args.key,
                    "cover_hidden": args.cover_hidden,
                    "cover_encoder": encoder.as_dict(),
                    "lr": args.lr,
                    "batch": args.batch,
                    "augment": bool(args.augment),
                    "init_cover": str(args.init_cover) if args.init_cover else None,
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
