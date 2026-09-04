"""The chooser and the constructor, crossed: every arm of the composition in one run.

    python tools/eval_solver_free.py --games 240 --log-json checkpoints/solver-free.json

Two learned pieces were measured apart and never together. The afterstate value net
ranks ordinary macros from outcomes alone; the two-phase template picker builds the
repartition CP-SAT would otherwise be asked for. `frugal` is the hand-written
chooser over a real solve, and it is the ceiling this ladder has.

So the table is a 2x2 -- `by_value` or the value net choosing, CP-SAT or the picker
constructing -- with the two ends of the ruler beside it. Crossing it is the point:
the composition's distance from `frugal` is one number, and which half of it belongs
to the chooser and which to the constructor is what the two single-substitution cells
say. Every arm is scored in the same process on the same deals, so `by_value`
reproducing +28.01 and `frugal` +47.71 is the check that the ruler did not move.

Every arm is deterministic -- argmax over the afterstates, a fixed-width beam over
the templates, no sampling anywhere -- so a repeat of this command is a repeat of
these numbers.

**What it measured**, `standard-greedy`, 240 deals played once per seat (n=480),
score / win rate, every arm in one run:

| chooser | no repartition | CP-SAT | picker beam 1 | picker beam 4 |
|---|---|---|---|---|
| `by_value` | +28.01 / 82.1% | **+47.71** / 99.8% | +45.63 / 99.8% | +44.05 / 99.4% |
| afterstate s0 | -- | +47.31 / 99.6% | +41.85 / 98.5% | **+45.05** / 99.4% |
| afterstate s1 | -- | +47.39 / 99.6% | +43.07 / 98.5% | +45.00 / 99.0% |
| afterstate s2 | -- | +44.74 / 99.4% | +34.08 / 97.5% | +38.62 / 98.5% |

`learned`, the clone rung, reads +47.23 / 99.8% here. Zero illegal actions in every
arm. Substituting the chooser is free -- given a solve the value net is optimal tier,
inside one standard error of `frugal` -- and substituting the constructor costs 2 to
4 points, which is where the whole gap lives. Per decision the composition costs
**2.52 ms** at beam 4 and 1.42 at beam 1 against `frugal`'s 6.33 and the clone's
7.09.

`--head-to-head optimal` is the caveat that matters, and `--h2h-arms` duels the
cells against each other on the same 100 deals from both seats (n=200), which is
what attributes a duel loss to one half or the other:

| arm | vs `optimal` | its answer rate |
|---|---|---|
| `frugal` | 53.0% | 20.5% |
| `learned` | 53.0% | 20.5% |
| afterstate s0 + CP-SAT | **49.0%** | 25.4% |
| `by_value` + picker b4 | 23.5% | 17.6% |
| afterstate s0 + picker b4 | **15.0%** | 23.0% |
| `by_value` + picker b1 | 14.5% | 14.8% |

The chooser swapped alone is even; the constructor swapped alone is not, and with the
chooser held at `by_value` the duel tracks how often the backend *answers* its gate
rather than how well it scores. Beam 1 is the best picker arm on the suite and the
worst here, so **the suite cannot rank constructors** -- a declined stuck turn costs
almost nothing against `greedy` and is a turn a peer converts. Run the duel.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time
from dataclasses import dataclass, field

import numpy as np

from rummi.agents import build
from rummi.agents.base import Observation
from rummi.agents.learned.afterstate_net import argmax_chooser, load_value_net, value_fn
from rummi.agents.learned.solver_free import PickerMacroAgent, SolverFreeAgent, load_picker
from rummi.agents.macro import MacroAgent, by_value
from rummi.evaluate.protocol import SUITE_BY_NAME, evaluate

VALUES = [pathlib.Path(f"checkpoints/afterstate-sweep-s{seed}-swa.pt") for seed in range(3)]
PICKER = pathlib.Path("checkpoints/twophase-rl4.pt")


class Meter:
    """What one arm spent, and what its repartition backend was asked.

    Both hooks are wrapped on the instance because they sit on opposite sides of
    `MacroAgent`: `choose` is a field, `_repartition` a method the mask path calls
    before any chooser sees the state. A cost claim about dropping the solver is
    about their sum, so neither can be timed alone.
    """

    def __init__(self, agent: MacroAgent) -> None:
        self.decisions = 0
        self.choose_seconds = 0.0
        self.asked = 0
        self.answered = 0
        self.repartition_seconds = 0.0
        chooser, backend = agent.choose, agent._repartition

        def choose(obs: Observation, env: int, legal: np.ndarray) -> int:
            started = time.perf_counter()
            try:
                return chooser(obs, env, legal)
            finally:
                self.choose_seconds += time.perf_counter() - started
                self.decisions += 1

        def repartition(obs: Observation, env: int) -> list[int]:
            started = time.perf_counter()
            try:
                actions = backend(obs, env)
            finally:
                self.repartition_seconds += time.perf_counter() - started
            self.asked += 1
            self.answered += bool(actions)
            return actions

        agent.choose = choose
        agent._repartition = repartition

    @property
    def ms_per_decision(self) -> float:
        total = self.choose_seconds + self.repartition_seconds
        return 1e3 * total / max(self.decisions, 1)


@dataclass
class Arm:
    """One agent, its label, and the meter wrapped around it.

    `key` is what `--h2h-arms` names: the labels read well in a table and badly on
    a command line, and a duel has to be able to ask for a cell by name.
    """

    key: str
    label: str
    agent: MacroAgent
    meter: Meter = field(init=False)

    def __post_init__(self) -> None:
        self.meter = Meter(self.agent)


def seed_label(path: pathlib.Path) -> str:
    """`s0` out of `afterstate-sweep-s0-swa`. The JSON carries the whole path."""
    for part in path.stem.split("-"):
        if part.startswith("s") and part[1:].isdigit():
            return part
    return path.stem


def arms(
    cfg,
    values: list[pathlib.Path],
    picker: pathlib.Path,
    beams: list[int],
    ruler_only: bool,
) -> list[Arm]:
    """The ruler, then the 2x2, seed by seed. Ordered so the run prints top-down."""
    out = [
        Arm("by_value", "by_value", MacroAgent(cfg, choose=by_value(cfg))),
        Arm(
            "frugal",
            "frugal = by_value + CP-SAT",
            MacroAgent(cfg, choose=by_value(cfg), repartition=True),
        ),
        Arm("learned", "learned (clone rung)", build("learned", cfg)),
    ]
    if ruler_only:
        return out

    scorer, monotone = load_picker(cfg, picker)
    for beam in beams:
        out.append(
            Arm(
                f"picker-b{beam}",
                f"by_value + picker b{beam}",
                PickerMacroAgent(cfg, scorer, choose=by_value(cfg), beam=beam, monotone=monotone),
            )
        )
    for path in values:
        seed = seed_label(path)
        net, _ = load_value_net(path, cfg)
        scored = value_fn(net)
        # The chooser reads back the macro layout it ranks, so it is bound after
        # the agent exists rather than passed to the constructor.
        solver_backed = MacroAgent(cfg, repartition=True)
        solver_backed.choose = argmax_chooser(cfg, solver_backed, scored)
        out.append(Arm(f"{seed}-cpsat", f"afterstate {seed} + CP-SAT", solver_backed))
        for beam in beams:
            out.append(
                Arm(
                    f"{seed}-b{beam}",
                    f"afterstate {seed} + picker b{beam}",
                    SolverFreeAgent(cfg, scorer, scored, beam=beam, monotone=monotone),
                )
            )
    return out


def head_to_head(
    suite, a: MacroAgent, b: MacroAgent, deals: int, seed_base: int
) -> dict:
    """`a` against `b` on identical deals, once from each seat.

    The shape `tools/game_structure.py` plays, taking built agents rather than
    registry names -- the composition is not a rung and has no name to look up.
    Seeded from its own base so it cannot reuse the suite's deals, which every
    arm above has already been tuned against by being chosen on them.
    """
    from rummi.agents.base import act_by_seat
    from rummi.env.numpy.deal import reset as deal_reset
    from rummi.env.numpy.deal import reset_envs
    from rummi.env.numpy.engine import step as engine_step
    from rummi.env.numpy.masks import legal_actions
    from rummi.env.numpy.sets import summarize
    from rummi.env.observation import encode

    cfg = suite.cfg
    seeds = [np.random.SeedSequence([seed_base, i]) for i in range(deals)]
    wins = np.zeros(2, dtype=np.int64)
    played = 0
    for seat in range(2):
        state = deal_reset(cfg, deals, seed=0)
        reset_envs(state, np.arange(deals), seeds)
        seats = [b, a] if seat else [a, b]
        for agent in seats:
            agent.reset(deals)
        for _ in range(suite.max_steps):
            if state.done.all():
                break
            summary = summarize(cfg, state.table_sets)
            mask = legal_actions(state, summary)
            actions, illegal = act_by_seat(
                seats, cfg, state.current, state.done, mask, encode(state, summary)
            )
            if illegal:
                raise RuntimeError(f"{illegal} illegal actions in the head-to-head")
            engine_step(state, actions, mask)
        for env in range(deals):
            if not state.done[env]:
                continue
            played += 1
            winner = int(state.winner[env])
            if winner in (0, 1):
                wins[winner ^ seat] += 1
    return {
        "games": played,
        "a_wins": int(wins[0]),
        "b_wins": int(wins[1]),
        "a_win_rate": float(wins[0] / max(played, 1)),
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--suite", default="standard-greedy", choices=sorted(SUITE_BY_NAME))
    p.add_argument(
        "--games", type=int, default=240,
        help="distinct deals; each is played once per seat, so n is this times the "
             "seat count -- 240 here is the n=480 the picker's rows were scored at",
    )
    p.add_argument("--value", type=pathlib.Path, nargs="+", default=VALUES)
    p.add_argument("--picker", type=pathlib.Path, default=PICKER)
    p.add_argument("--beam", type=int, nargs="+", default=[1, 4])
    p.add_argument(
        "--ruler-only", action="store_true",
        help="score the three reference arms and stop, which is how long the ruler "
             "itself takes and what a timing estimate is made from",
    )
    p.add_argument(
        "--head-to-head", default=None,
        help="after scoring, duel --h2h-arms against this registry agent on deals of "
             "their own. A suite score against `greedy` is mostly the loser's "
             "leftover rack; what a peer opponent separates it can hide",
    )
    p.add_argument("--h2h-deals", type=int, default=100)
    p.add_argument(
        "--h2h-arms", nargs="+", default=["s0-b4", "frugal"],
        help="arm keys to duel, from the table this run built: by_value, frugal, "
             "learned, picker-b<beam>, s<seed>-cpsat, s<seed>-b<beam>. Every one of "
             "them plays the same deals, so the single substitutions attribute a "
             "duel loss to the chooser, the constructor or their interaction",
    )
    p.add_argument(
        "--h2h-only", action="store_true",
        help="skip the suite table and duel only. The arms still have to be built, "
             "so this is how a decomposition is added without re-scoring 14 arms",
    )
    p.add_argument("--h2h-seed-base", type=int, default=91_000)
    p.add_argument("--log-json", type=pathlib.Path, default=None)
    args = p.parse_args()

    suite = SUITE_BY_NAME[args.suite]
    built = arms(suite.cfg, list(args.value), args.picker, list(args.beam), args.ruler_only)
    print(
        f"suite={suite.name} deals={args.games} n={args.games * suite.cfg.n_players} "
        f"picker={args.picker.name} arms={len(built)}",
        flush=True,
    )

    rows: list[dict] = []
    for arm in [] if args.h2h_only else built:
        began = time.perf_counter()
        result = evaluate(
            arm.label, suite, build_agent=lambda c, a=arm.agent: a, games=args.games
        )
        wall = time.perf_counter() - began
        # Printed beside the mean because the whole question is a few points of it,
        # and a mean over 480 games carries a couple of points of standard error.
        sem = float(np.std(result.scores, ddof=1) / np.sqrt(len(result.scores)))
        meter = arm.meter
        print(
            f"  {arm.label:34s} score {result.mean_score:>+7.2f} +-{sem:>4.2f}  "
            f"win {result.win_rate:>6.1%}  stale {result.stalemates / max(result.games, 1):>5.1%}  "
            f"left {result.mean_final_rack:>5.1f}  illegal {result.illegal_attempts}  "
            f"asked {meter.asked:>6,} answered {meter.answered:>6,}  "
            f"{meter.ms_per_decision:>5.2f} ms/dec  {wall:>5.0f}s",
            flush=True,
        )
        rows.append(
            {
                "label": arm.label,
                "suite": suite.name,
                "mean_score": result.mean_score,
                "score_sem": sem,
                "win_rate": result.win_rate,
                "stalemate_rate": result.stalemates / max(result.games, 1),
                "mean_final_rack": result.mean_final_rack,
                "illegal_attempts": result.illegal_attempts,
                "games": result.games,
                "decisions": meter.decisions,
                "asked": meter.asked,
                "answered": meter.answered,
                "choose_seconds": meter.choose_seconds,
                "repartition_seconds": meter.repartition_seconds,
                "ms_per_decision": meter.ms_per_decision,
                "wall_seconds": wall,
            }
        )

    duels: list[dict] = []
    if args.head_to_head:
        by_key = {arm.key: arm for arm in built}
        unknown = [key for key in args.h2h_arms if key not in by_key]
        if unknown:
            p.error(f"--h2h-arms {unknown} not built; this run has {sorted(by_key)}")
        # Every duel runs the same deals, so the arms are comparable to each other
        # and `frugal` among them is the ceiling's own reading -- a win rate here
        # means nothing beside a suite score that compresses what a peer separates.
        for key in args.h2h_arms:
            arm = by_key[key]
            began = time.perf_counter()
            duel = head_to_head(
                suite,
                arm.agent,
                build(args.head_to_head, suite.cfg),
                args.h2h_deals,
                args.h2h_seed_base,
            )
            duel["a"], duel["b"] = arm.label, args.head_to_head
            duels.append(duel)
            print(
                f"  {arm.label:34s} vs {args.head_to_head}: {duel['a_win_rate']:>6.1%} "
                f"({duel['a_wins']}-{duel['b_wins']}, n={duel['games']}, "
                f"{time.perf_counter() - began:.0f}s)",
                flush=True,
            )

    if args.log_json:
        args.log_json.parent.mkdir(parents=True, exist_ok=True)
        args.log_json.write_text(
            json.dumps(
                {
                    "suite": suite.name,
                    "games": args.games,
                    "value": [str(path) for path in args.value],
                    "picker": str(args.picker),
                    "beam": list(args.beam),
                    "arms": rows,
                    "head_to_head": duels,
                },
                indent=2,
            )
        )
        print(f"wrote {args.log_json}", flush=True)


if __name__ == "__main__":
    main()
