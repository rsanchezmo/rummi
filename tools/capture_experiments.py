"""Score the experimental agents -- the ones `REGISTRY` deliberately does not hold.

    python tools/capture_experiments.py --suite standard-greedy --games 60

`capture_agents.py` asserts every name is in `REGISTRY`, and that is right: it feeds
the published ladder, where a rung has to earn its place. This is its sibling for
agents that have not earned one, so their numbers are reproducible rather than typed
into a table by hand.

**Only deterministic agents are captured.** The training runs recorded in
`docs/EXPERIMENTS.md` cannot be: each is one seed of one recipe, and reproducing one
is a training job rather than a capture. Which rows are which is the most useful
thing that file says.

`hybrid/macro_first` is here even though it should equal `macro/by_value` exactly --
`tests/test_benchmark.py` asserts the two agree action for action, and having both
rows makes that visible in the data. If they ever diverge, the hybrid wrapper has
started doing something of its own.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rummi.agents.hybrid import HybridAgent, macro_first
from rummi.agents.macro import MacroAgent, by_value, first_legal
from rummi.evaluate.protocol import PROTOCOL_VERSION, SUITE_BY_NAME, evaluate

AGENTS = {
    "macro/first_legal": lambda cfg: MacroAgent(cfg, choose=first_legal),
    "macro/by_value": lambda cfg: MacroAgent(cfg, choose=by_value(cfg)),
    "hybrid/macro_first": lambda cfg: HybridAgent(cfg, choose=macro_first(cfg)),
}

REPARTITION = {
    # The same chooser, so the row is the value of the extra macro and nothing else.
    # Behind a flag because it is the only agent here that solves: every solve is
    # milliseconds against a matrix comparison, so a capture with it runs far longer.
    "macro/by_value+repartition": lambda cfg: MacroAgent(
        cfg, choose=by_value(cfg), repartition=True
    ),
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", default="standard-greedy", choices=sorted(SUITE_BY_NAME))
    p.add_argument("--games", type=int, default=60)
    p.add_argument("--out", type=Path, default=Path("docs/data/experiments.json"))
    p.add_argument(
        "--repartition", action="store_true",
        help="also score the macro agent with the solver-backed REPARTITION macro",
    )
    args = p.parse_args()

    suite = SUITE_BY_NAME[args.suite]
    agents = []
    for name, build in {**AGENTS, **(REPARTITION if args.repartition else {})}.items():
        agent = build(suite.cfg)
        result = evaluate(name, suite, build_agent=lambda c, a=agent: a, games=args.games)
        assert not result.disqualified, f"{name} was disqualified"
        assert result.illegal_attempts == 0, f"{name} proposed a masked-out action"
        agents.append(
            {
                "name": name,
                "win_rate": result.win_rate,
                "mean_score": result.mean_score,
                "mean_turns": result.mean_turns,
                "mean_final_rack": result.mean_final_rack,
                "stalemate_rate": result.stalemates / max(1, result.games),
                "games": result.games,
            }
        )
        print(
            f"{name:20s} win {result.win_rate:>6.1%}  score {result.mean_score:>+8.2f}  "
            f"n={result.games}",
            flush=True,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "protocol": PROTOCOL_VERSION,
                "suite": suite.name,
                "opponent": suite.opponent,
                "players": suite.cfg.n_players,
                "games": agents[0]["games"] if agents else 0,
                "agents": agents,
            },
            indent=1,
        )
        + "\n"
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
