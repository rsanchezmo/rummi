"""One-turn lookahead over the afterstate value net: score finished turns, not moves.

    python tools/average_checkpoints.py --series checkpoints/afterstate-sweep-s0 \
        --updates 300 600 20 --out checkpoints/afterstate-sweep-s0-swa.pt
    python tools/search_afterstate.py checkpoints/afterstate-sweep-s0-swa.pt --games 120

The `-swa` checkpoint is not something `tools/train_afterstate.py` writes -- it is
the mean of that run's u300-u600 checkpoints, which is what
`tools/average_checkpoints.py` above is for, and reproducing it is the first line
of that command.

`tools/train_afterstate.py` learns V over afterstates and plays the argmax over
single macros. That policy is myopic in one exact way, and it is the way this file
tests: a turn holds several decisions -- one per set -- so the afterstate of the
*first* of them says nothing about what the rest of the turn can still play. Two
sets that reach the opening meld together are indistinguishable, before the first
is played, from one that does not.

Search removes that without touching the net. A macro's afterstate is a full
observation (`rummi/agents/learned/afterstate.py`), so `MacroAgent.legal_macros`
runs on it and *its* macros have afterstates of their own: completing a turn in
simulation is a recursion over the same predictor, and nothing is stepped. The
candidate's value is then V at the position the finished turn reaches, carrying the
`END_TURN` flag -- the same row the myopic policy would eventually score, reached
one turn earlier.

Three rules hold the simulation to the game the env would play.

**The leaf is a commitment.** A completed turn is valued at its `END_TURN` row; a
turn that cannot be closed -- nothing left to play and the opening meld unmet -- is
valued at its `DRAW` row, because `DRAW` is what the env offers there and it reverts
the turn (SPEC.md section 4).

**Before the meld there is nothing to stop for.** `END_TURN` is illegal until the
threshold is met, so the argmax runs over the plays alone and the completion keeps
going while anything plays. That is where the myopic blind spot lives.

**The micro budget is the depth bound.** Every tile-playing macro spends at least
one micro-action and a continuation that would leave none for the trailing
`END_TURN` is not offered -- the refusal `MacroAgent._repartition` already makes,
for the same reason: overrunning leaves the env offering only `DRAW`, which throws
the turn away.

`--myopic` runs the plain argmax-V chooser through this same harness, so the two
arms differ in the search and in nothing else.

**What it measured.** The three SWA nets of `docs/EXPERIMENTS.md`, `standard-greedy`,
both arms on the same deals so the difference is paired -- and `by_value` reproducing
its +29.60 / 85.4% inside every run as the check that the harness did not move:

| net | myopic | search, beam 1 | paired difference |
|---|---|---|---|
| s0 | +28.37 | **+33.03** | +4.66 +- 2.74 (n=240) |
| s0 | +28.43 | +31.62 | +3.19 +- 2.05 (n=400) |
| s1 | +27.59 | +27.44 | -0.15 +- 2.08 (n=240) |
| s2 | +25.16 | +22.79 | -2.37 +- 3.24 (n=240) |

Over the three seeds that is **+0.71 +- 2.07**: null. One seed clears `by_value` and
no seed clears its own control, which is the shape of a selection effect rather than
a capability -- and the rest of the evidence agrees. The mean barely moves while the
spread between seeds triples, 3.2 points to 10.2; `--beam 3` selects over more
completions still and buys nothing (+32.35 +- 2.05 against beam 1's +33.03); on
`standard-optimal` the same net scores -33.10 searching against -34.70 myopic, both
below `by_value`'s -30.29. The mechanism is not that the lookahead fails to fire --
it changes 4.6% of s0's decisions and 16.3% of s2's, and a fifth of its chosen turns
run past one macro. It is that a maximum over more of V's outputs is a maximum over
more of V's noise: explained variance for these nets is ~0.00, so there is no
ranking to propagate to the leaf. **Search inherits its evaluator's resolution and
cannot exceed it.**

Cost is what search costs and it is net-dependent, because a net that rarely prefers
ending expands a completion until nothing plays: 1,023 / 1,908 / 111 decisions per
second for s0 / s1 / s2, against 3,871-4,990 for the myopic arm.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import time
from collections.abc import Sequence

import numpy as np

from rummi.agents.base import Observation
from rummi.agents.learned.afterstate import (
    afterstate_dim,
    afterstate_obs,
    afterstate_rows,
    afterstate_view,
)
from rummi.agents.learned.afterstate_net import (
    Value,
    argmax_chooser,
    load_value_net,
    value_fn,
)
from rummi.agents.macro import Choose, MacroAgent, by_value, first_legal
from rummi.evaluate.protocol import SUITE_BY_NAME, evaluate
from rummi.rules.config import RummiConfig
from rummi.rules.observation import MICRO_COUNT


@dataclasses.dataclass(frozen=True, slots=True)
class Completion:
    """A whole turn as the search imagines it, from one candidate macro onwards."""

    macros: tuple[int, ...]
    views: tuple[Observation, ...]
    """The observation predicted after each macro, aligned with ``macros``."""
    value: float
    commits: bool
    """The leaf is ``END_TURN``; false means the turn reverts on ``DRAW``."""


@dataclasses.dataclass(slots=True)
class _Node:
    """One partially completed turn. ``start`` is the candidate it descends from."""

    start: int
    view: Observation
    macros: tuple[int, ...]
    views: tuple[Observation, ...]


class TurnSearch:
    """Rank each legal macro by the turn it can still finish.

    The simulation asks `legal_macros` about imagined states, and with
    `repartition` on that call caches a CP-SAT plan per env index. A second
    `MacroAgent` keeps that bookkeeping off the one actually playing, which shares
    both the macro layout and the templates and so decides identically.
    """

    def __init__(self, cfg: RummiConfig, agent: MacroAgent, value_of: Value, beam: int = 1) -> None:
        self.cfg = cfg
        self.agent = agent
        self.sim = MacroAgent(cfg, repartition=agent.repartition_macro is not None)
        self.value_of = value_of
        self.beam = beam
        self.rows_scored = 0

    def choose(self, obs: Observation, env: int, legal: np.ndarray) -> tuple[int, Completion | None]:
        """The macro to play, and the turn it was chosen for."""
        agent = self.agent
        if agent.repartition_macro is not None and legal[agent.repartition_macro]:
            # Offered only where nothing else plays, so there is nothing to search
            # and no afterstate to score it against.
            return int(agent.repartition_macro), None

        options = np.flatnonzero(legal).tolist()
        fields, kinds = afterstate_obs(self.cfg, obs, env, options, agent)
        values = self._score(afterstate_rows(self.cfg, fields, kinds))

        plays = [
            (i, macro)
            for i, macro in enumerate(options)
            if macro not in (agent.end_macro, agent.draw_macro)
        ]
        finished: dict[int, Completion] = {}
        if plays:
            completions = self._complete(
                [(macro, afterstate_view(fields, i)) for i, macro in plays]
            )
            finished = {macro: c for (_, macro), c in zip(plays, completions, strict=True)}

        # Ascending macro id and a strict improvement, so ties break exactly as the
        # myopic `argmax` over the same options would. Seeded with a real option
        # rather than a sentinel: a value that comes back NaN satisfies no `>`, and
        # `-1` would leave the loop as an action -- DRAW's mask column is always
        # true and `expand` reads it as the last template, so nothing catches it.
        best, best_value, chosen = options[0], -np.inf, None
        for i, macro in enumerate(options):
            completion = finished.get(macro)
            value = completion.value if completion is not None else float(values[i])
            if value > best_value:
                best, best_value, chosen = macro, value, completion
        return best, chosen

    def _score(self, rows: np.ndarray) -> np.ndarray:
        self.rows_scored += len(rows)
        return self.value_of(rows)

    def _complete(self, starts: Sequence[tuple[int, Observation]]) -> list[Completion]:
        """Finish the turn from each start; its value is the best leaf reached.

        Beam-width many continuations survive each depth per start, ranked by V.
        The best *leaf* rather than the last one, because the agent re-searches at
        every real decision and can therefore stop wherever the search says to.
        """
        cfg, sim, budget = self.cfg, self.sim, self.cfg.max_micro_per_turn
        best: list[Completion | None] = [None] * len(starts)
        frontier = [
            _Node(i, view, (macro,), (view,)) for i, (macro, view) in enumerate(starts)
        ]

        while frontier:
            expanded: list[tuple[_Node, list[int], bool, dict[str, np.ndarray]]] = []
            rows: list[np.ndarray] = []
            for node in frontier:
                legal = sim.legal_macros(node.view, 0)
                plays = [
                    macro
                    for macro in np.flatnonzero(legal[: sim.end_macro]).tolist()
                    if macro != sim.repartition_macro
                ]
                # Both endings come off the same `summarize` as the continuations,
                # so the leaf costs nothing beyond one more row.
                fields, kinds = afterstate_obs(
                    cfg, node.view, 0, [*plays, sim.end_macro, sim.draw_macro], sim
                )
                expanded.append((node, plays, bool(legal[sim.end_macro]), fields))
                rows.append(afterstate_rows(cfg, fields, kinds))

            values = self._score(np.concatenate(rows))
            successors: dict[int, list[tuple[float, _Node]]] = {}
            at = 0
            for node, plays, can_end, fields in expanded:
                n = len(plays)
                scores = values[at : at + n + 2]
                at += n + 2

                leaf = float(scores[n] if can_end else scores[n + 1])
                current = best[node.start]
                if current is None or leaf > current.value:
                    best[node.start] = Completion(node.macros, node.views, leaf, can_end)

                micro = np.asarray(fields["scalars"])[:n, MICRO_COUNT]
                offerable = [j for j in range(n) if micro[j] < budget]
                if not offerable:
                    continue
                offerable.sort(key=scores.__getitem__, reverse=True)
                # `END_TURN` winning the argmax is what stops a completion. Where it
                # is illegal the turn is unfinished and there is nothing to weigh a
                # play against, so the search keeps going -- which is the whole of
                # what a single afterstate cannot see before the opening meld.
                if can_end and scores[n] >= scores[offerable[0]]:
                    continue
                for j in offerable[: self.beam]:
                    view = afterstate_view(fields, j)
                    successors.setdefault(node.start, []).append(
                        (
                            float(scores[j]),
                            _Node(
                                node.start,
                                view,
                                (*node.macros, plays[j]),
                                (*node.views, view),
                            ),
                        )
                    )

            frontier = [
                node
                for options in successors.values()
                for _, node in sorted(options, key=lambda row: -row[0])[: self.beam]
            ]

        out: list[Completion] = []
        for completion in best:
            # Every start is a frontier node on the first pass, so every start has a
            # leaf: a turn that plays this macro and stops is always one of them.
            assert completion is not None
            out.append(completion)
        return out


class Timed:
    """A `Choose` that reports what it cost, so the two arms are comparable."""

    def __init__(self, inner: Choose) -> None:
        self.inner = inner
        self.decisions = 0
        self.seconds = 0.0

    def __call__(self, obs: Observation, env: int, legal: np.ndarray) -> int:
        started = time.perf_counter()
        try:
            return self.inner(obs, env, legal)
        finally:
            self.seconds += time.perf_counter() - started
            self.decisions += 1

    @property
    def rate(self) -> float:
        return self.decisions / self.seconds if self.seconds else float("nan")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint", type=pathlib.Path)
    p.add_argument("--suite", default="standard-greedy", choices=sorted(SUITE_BY_NAME))
    p.add_argument(
        "--games", type=int, default=120,
        help="distinct deals; each is played once per seat, so n is this times the "
             "seat count -- 120 here is the n=240 the published rows quote",
    )
    p.add_argument(
        "--beam", type=int, default=1,
        help="continuations kept per depth. 1 follows the argmax, which is the "
             "greedy completion the myopic policy would itself have played",
    )
    p.add_argument(
        "--myopic", action="store_true",
        help="argmax V over single afterstates instead, through this same harness",
    )
    p.add_argument(
        "--no-baselines", action="store_true",
        help="skip `by_value` and `first_legal`. They are the check that the harness "
             "did not move under the learned row, so drop them only when repeating",
    )
    p.add_argument("--log-json", type=pathlib.Path, default=None)
    args = p.parse_args()

    suite = SUITE_BY_NAME[args.suite]
    cfg = suite.cfg
    net, repartition = load_value_net(args.checkpoint, cfg)
    value_of = value_fn(net)

    label = "myopic" if args.myopic else f"search-b{args.beam}"
    played = MacroAgent(cfg, repartition=repartition)
    if args.myopic:
        chooser = Timed(argmax_chooser(cfg, played, value_of))
        search: TurnSearch | None = None
    else:
        search = TurnSearch(cfg, played, value_of, beam=args.beam)
        chooser = Timed(lambda obs, env, legal: search.choose(obs, env, legal)[0])
    played.choose = chooser

    print(
        f"{args.checkpoint.name}  suite={suite.name} games={args.games} "
        f"arm={label} repartition={repartition} dim={afterstate_dim(cfg)}",
        flush=True,
    )

    arms: list[tuple[str, MacroAgent]] = [(label, played)]
    if not args.no_baselines:
        arms += [
            ("by_value", MacroAgent(cfg, choose=by_value(cfg), repartition=repartition)),
            ("first_legal", MacroAgent(cfg, choose=first_legal, repartition=repartition)),
        ]

    scores: list[dict] = []
    for name, agent in arms:
        started = time.perf_counter()
        result = evaluate(name, suite, build_agent=lambda c, a=agent: a, games=args.games)
        wall = time.perf_counter() - started
        # Printed beside the mean because the whole question here is a few points of
        # it, and a mean score over 240 games carries several points of standard
        # error -- an arm "beating" another by less than this is not a result.
        sem = float(np.std(result.scores, ddof=1) / np.sqrt(len(result.scores)))
        print(
            f"  {name:12s} win {result.win_rate:>6.1%}  score {result.mean_score:>+8.2f}"
            f" +-{sem:>5.2f}  left {result.mean_final_rack:>5.1f}  "
            f"illegal {result.illegal_attempts}  n={result.games}  {wall:>6.1f}s",
            flush=True,
        )
        scores.append(
            {
                "label": name,
                "suite": suite.name,
                "win_rate": result.win_rate,
                "mean_score": result.mean_score,
                "score_sem": sem,
                "mean_final_rack": result.mean_final_rack,
                "illegal_attempts": result.illegal_attempts,
                "games": result.games,
                "wall_seconds": wall,
            }
        )

    print(
        f"  {label}: {chooser.decisions:,} decisions, {chooser.rate:>7.1f} dec/s"
        + (f", {search.rows_scored / max(chooser.decisions, 1):>6.1f} afterstates each" if search else ""),
        flush=True,
    )

    if args.log_json:
        args.log_json.parent.mkdir(parents=True, exist_ok=True)
        args.log_json.write_text(
            json.dumps(
                {
                    "checkpoint": str(args.checkpoint),
                    "suite": suite.name,
                    "games": args.games,
                    "arm": label,
                    "beam": args.beam,
                    "decisions": chooser.decisions,
                    "decisions_per_second": chooser.rate,
                    "afterstates_per_decision": (
                        search.rows_scored / max(chooser.decisions, 1) if search else 1.0
                    ),
                    "eval": scores,
                },
                indent=2,
            )
        )
        print(f"wrote {args.log_json}", flush=True)


if __name__ == "__main__":
    main()
