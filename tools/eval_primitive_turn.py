"""The primitive turn decoder against the template picker and against CP-SAT.

    python tools/eval_primitive_turn.py --data checkpoints/turns-s*.npz \
        --init checkpoints/primitive-turn.pt --games 240

Two comparisons, and they answer different questions.

**On held-out stuck states** every arm is asked the same thing -- the gate has
fired, `by_value` has nothing, what do you play -- from the same states, so the
answer is a like-for-like rate. The template arms are scored through
`train_repartition.playable_rate` and `train_two_phase.playable_rate` themselves
rather than through a restatement of them, so the numbers are defined exactly as
the ones already in `docs/EXPERIMENTS.md`. CP-SAT is 100% by construction: a state
is in this set because the solver answered it.

**On `standard-greedy`** the ruler runs inside the same process as the arms:
`by_value` with no repartition at one end, `by_value` plus CP-SAT at the other, and
between them the template picker and the two primitive arms -- the drop-in
repartition (arm A) and the agent that decodes every turn with no macro vocabulary
at all (arm B). Everything is decoded deterministically, greedily or by beam, so
the argmax-versus-sampled caveat elsewhere in that file does not apply.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch

from rummi.agents.learned.architecture import Architecture
from rummi.agents.learned.primitive_turn import (
    PrimitiveRepartition,
    PrimitiveTurnAgent,
    Scorer,
)
from rummi.agents.learned.repartition_net import RepartitionNet
from rummi.agents.learned.repartition_net import Scorer as TemplateScorer
from rummi.agents.learned.torch_net import TorchPolicy
from rummi.agents.learned.two_phase_net import TwoPhaseScorer, two_phase_from_checkpoint
from rummi.agents.macro import MacroAgent, by_value
from rummi.evaluate.protocol import SUITE_BY_NAME, evaluate
from rummi.rules.config import CONFIG_BY_NAME, RummiConfig

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from train_primitive_turn import (
    Turns,
    decode_report,
    length_table,
    print_abandonment,
    print_length_table,
)
from train_repartition import NeuralRepartition
from train_repartition import playable_rate as template_playable
from train_two_phase import playable_rate as two_phase_playable


class Shim:
    """The three fields both template `playable_rate`s read, from gate states.

    Borrowing their metric rather than restating it is the point: a rate defined
    twice is a rate that can differ, and these are the numbers `docs/EXPERIMENTS.md`
    already publishes for the same population.
    """

    def __init__(self, cfg: RummiConfig, data: Turns, index: np.ndarray) -> None:
        self.rack = data.starts.rack[index].astype(np.int64)
        self.board = data.starts.board[index]
        played = np.zeros((len(index), cfg.n_kinds), dtype=np.int64)
        for row, i in enumerate(index.tolist()):
            for action in data.plans[i]:
                if action < cfg.pick_offset:
                    played[row, action - cfg.place_offset] += 1
        self.played = played


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="standard", choices=sorted(CONFIG_BY_NAME))
    p.add_argument("--data", type=pathlib.Path, nargs="+", required=True)
    p.add_argument("--init", type=pathlib.Path, required=True, help="the primitive checkpoint")
    p.add_argument("--template", type=pathlib.Path, default=pathlib.Path("checkpoints/repart-aug2.pt"))
    p.add_argument("--two-phase", type=pathlib.Path, default=pathlib.Path("checkpoints/twophase-rl4.pt"))
    p.add_argument("--beam", type=int, nargs="+", default=[1, 4])
    p.add_argument("--template-beam", type=int, nargs="+", default=[1, 4])
    p.add_argument(
        "--stuck", type=int, default=1_500,
        help="held-out gate states compared; 0 skips the comparison, which is what a "
             "run splitting the suite arms across processes wants",
    )
    p.add_argument("--batch", type=int, default=256, help="gate states decoded in lockstep")
    p.add_argument(
        "--turns", type=int, default=0,
        help="held-out whole turns to decode for the length breakdown; 0 skips it. "
             "The same table `tools/train_primitive_turn.py` prints at the end of a "
             "run, so a checkpoint can be re-read without retraining",
    )
    p.add_argument("--games", type=int, default=0, help="deals per suite arm; 0 skips the suite")
    p.add_argument("--arms", default="all", help="comma-separated suite arms, or `all`")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-json", type=pathlib.Path, default=None)
    args = p.parse_args()

    cfg = CONFIG_BY_NAME[args.config]
    rng = np.random.default_rng(args.seed)

    checkpoint = torch.load(args.init, weights_only=True)
    arch = Architecture(hidden=tuple(checkpoint["hidden"]))
    net = TorchPolicy(cfg, arch, seed=args.seed)
    net.load_state_dict(checkpoint["state"])
    scorer = Scorer(net)

    template = torch.load(args.template, weights_only=True)
    template_net = RepartitionNet(cfg, hidden=template["hidden"], key=template["key"])
    template_net.load_state_dict(template["state"])
    template_scorer = TemplateScorer(template_net)
    monotone = bool(template["monotone"])

    two_phase = torch.load(args.two_phase, weights_only=True)
    two_phase_scorer = TwoPhaseScorer(two_phase_from_checkpoint(cfg, two_phase))
    two_phase_monotone = bool(two_phase["monotone"])

    stuck: list[dict] = []
    if args.stuck:
        gates = Turns(cfg, args.config, list(args.data), prefix="gate")
        held = np.flatnonzero(gates.holdout)
        rows = np.sort(rng.permutation(held)[: args.stuck])
        print(
            f"{len(gates):,} gate states, {len(held):,} held out, comparing on {len(rows):,}\n"
            f"CP-SAT answered every one of them by construction, shedding "
            f"{gates.tiles[rows].mean():.2f} tiles over "
            f"{gates.length[rows].mean():.1f} primitive actions\n",
            flush=True,
        )
        shim = Shim(cfg, gates, rows)
        every = np.arange(len(rows))
        for beam in args.beam:
            report = decode_report(cfg, scorer, gates, rows, beam, args.batch).summary()
            stuck.append(
                {
                    "arm": f"primitive (beam {beam})",
                    "playable": report["committed"],
                    "tiles": report["tiles"],
                    "decodes_per_second": 1000 / max(report["ms_per_turn"], 1e-9),
                }
            )
        for beam in args.template_beam:
            stuck.append({
                "arm": f"template one-phase (beam {beam})",
                **template_playable(cfg, shim, template_scorer, every, beam, monotone),
            })
            stuck.append({
                "arm": f"template two-phase (beam {beam})",
                **two_phase_playable(cfg, shim, two_phase_scorer, every, beam, two_phase_monotone),
            })
        stuck.append(
            {"arm": "CP-SAT", "playable": 1.0, "tiles": float(gates.tiles[rows].mean())}
        )

        print(f"{'arm':<30} {'plays':>7} {'tiles':>7} {'dec/s':>8}")
        for row in stuck:
            rate = row.get("decodes_per_second")
            speed = f"{rate:>8.1f}" if rate else f"{'--':>8}"
            print(
                f"{row['arm']:<30} {row['playable']:>6.1%} {row['tiles']:>7.2f} {speed}",
                flush=True,
            )

    lengths: list[dict] = []
    abandonment: dict[str, dict] = {}
    if args.turns:
        turns = Turns(cfg, args.config, list(args.data))
        held = np.flatnonzero(turns.holdout)
        rows = np.sort(rng.permutation(held)[: args.turns])
        reports = {beam: decode_report(cfg, scorer, turns, rows, beam, args.batch)
                   for beam in args.beam}
        for beam, report in reports.items():
            summary = report.summary()
            print(
                f"\nholdout ({len(rows):,} turns, beam {beam})  "
                f"committed {summary['committed']:.1%}  tiles {summary['tiles']:.2f} "
                f"against the teacher's {summary['teacher_tiles']:.2f}  "
                f"declined {summary['declined']:.1%} of {summary['declines']:,}  "
                f"{summary['ms_per_turn']:.1f} ms/turn",
                flush=True,
            )
        print_abandonment(reports)
        abandonment = {str(beam): r.abandonment() for beam, r in reports.items()}
        lengths = length_table(turns, rows, reports)
        print_length_table(lengths, args.beam)

    scores: list[dict] = []
    if args.games:
        suite = SUITE_BY_NAME["standard-greedy" if args.config == "standard" else "tiny"]
        wanted = None if args.arms == "all" else set(args.arms.split(","))

        def build(label: str, make) -> None:
            if wanted is not None and label not in wanted:
                return
            began = time.perf_counter()
            agent = make()
            result = evaluate(label, suite, build_agent=lambda c, a=agent: a, games=args.games)
            extra = ""
            if isinstance(agent, PrimitiveRepartition):
                extra = f"  asked {agent.asked:,} answered {agent.answered:,}"
            if isinstance(agent, PrimitiveTurnAgent):
                extra = (
                    f"  turns {agent.turns:,} committed "
                    f"{agent.committed / max(agent.turns, 1):.1%} "
                    f"tiles {agent.tiles / max(agent.committed, 1):.2f}"
                )
            print(
                f"  {label:<30} win {result.win_rate:>6.1%}  score {result.mean_score:>+8.2f}  "
                f"illegal {result.illegal_attempts}  n={result.games}{extra}  "
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
                    "stalemates": result.stalemates,
                    "asked": getattr(agent, "asked", None),
                    "answered": getattr(agent, "answered", None),
                    "turns": getattr(agent, "turns", None),
                    "committed": getattr(agent, "committed", None),
                    "tiles": getattr(agent, "tiles", None),
                }
            )

        build("by_value", lambda: MacroAgent(cfg, choose=by_value(cfg)))
        build("by_value+repartition", lambda: MacroAgent(cfg, choose=by_value(cfg), repartition=True))
        for beam in args.template_beam:
            build(
                f"template picker (beam {beam})",
                lambda b=beam: NeuralRepartition(cfg, template_scorer, beam=b, monotone=monotone),
            )
        for beam in args.beam:
            build(
                f"arm A primitive (beam {beam})",
                lambda b=beam: PrimitiveRepartition(cfg, scorer, beam=b),
            )
        for beam in args.beam:
            build(
                f"arm B primitive turn (beam {beam})",
                lambda b=beam: PrimitiveTurnAgent(cfg, scorer, beam=b),
            )

    if args.log_json:
        args.log_json.parent.mkdir(parents=True, exist_ok=True)
        args.log_json.write_text(
            json.dumps(
                {
                    "config": args.config,
                    "init": str(args.init),
                    "data": [str(path) for path in args.data],
                    "stuck": stuck,
                    "length_breakdown": lengths,
                    "abandonment": abandonment,
                    "games": args.games,
                    "eval": scores,
                },
                indent=2,
            )
        )
        print(f"wrote {args.log_json}", flush=True)


if __name__ == "__main__":
    main()
