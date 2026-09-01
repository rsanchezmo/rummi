"""Score every bundled agent and save the result for the README charts.

Committed alongside the charts so the figures are reproducible rather than
screenshots of a number somebody once saw.

    python tools/capture_agents.py --suite standard-greedy --games 60
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from rummi.agents import REGISTRY
from rummi.evaluate.protocol import PROTOCOL_VERSION, SUITE_BY_NAME, evaluate

# Weakest to strongest. The order is the chart's ordinal axis, so it is asserted
# rather than assumed -- see the check at the end.
LADDER = ["random", "weighted-random", "greedy", "rearrange", "frugal", "optimal"]

TIE = 0.02
"""How far a rung may dip below its predecessor before the ladder is out of
order. Strict order stopped being assertable when `frugal` landed: it is
statistically even with `optimal` (48.7% head-to-head at n=600), so which of
the two tops a given suite is that suite's sampling noise."""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", default="standard-greedy", choices=sorted(SUITE_BY_NAME))
    p.add_argument("--games", type=int, default=60)
    p.add_argument("--out", type=Path, default=Path("docs/data/agents.json"))
    args = p.parse_args()

    suite = SUITE_BY_NAME[args.suite]
    agents = []
    for name in LADDER:
        assert name in REGISTRY, name
        result = evaluate(name, suite, games=args.games)
        agents.append(
            {
                "name": name,
                "win_rate": result.win_rate,
                "mean_score": result.mean_score,
                "mean_turns": result.mean_turns,
                "mean_final_rack": result.mean_final_rack,
                "stalemate_rate": result.stalemates / max(1, result.games),
                "games": result.games,
                "disqualified": result.disqualified,
            }
        )
        print(f"{name:16s} win {result.win_rate:>6.1%}  score {result.mean_score:>+7.1f}")

    for below, above in itertools.pairwise(agents):
        if above["win_rate"] < below["win_rate"] - TIE:
            raise SystemExit(
                f"ladder is out of order: {below['name']} {below['win_rate']:.1%} "
                f"sits above {above['name']} {above['win_rate']:.1%}"
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "protocol": PROTOCOL_VERSION,
                "suite": suite.name,
                "opponent": suite.opponent,
                "games": agents[0]["games"],
                "agents": agents,
            },
            indent=2,
        )
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
