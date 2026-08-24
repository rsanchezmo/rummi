"""Run the benchmark. ``python -m rummi.benchmark.run --agent greedy``"""

from __future__ import annotations

import argparse
import json
import time

from rummi.agents.reference import REGISTRY
from rummi.benchmark.protocol import (
    PROTOCOL_VERSION,
    SUITE_BY_NAME,
    SUITES,
    evaluate,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--agent", default="greedy", choices=sorted(REGISTRY))
    p.add_argument("--suites", nargs="+", default=[s.name for s in SUITES], choices=sorted(SUITE_BY_NAME))
    p.add_argument("--games", type=int, default=None, help="override for a quick look")
    p.add_argument("--json", action="store_true", help="emit machine-readable results")
    args = p.parse_args()

    rows = []
    if not args.json:
        print(f"rummi benchmark v{PROTOCOL_VERSION}   agent={args.agent}")
        print(f"{'suite':<18} {'win':>7} {'score':>8} {'turns':>7} {'left':>6} {'stale':>7}")
    for name in args.suites:
        suite = SUITE_BY_NAME[name]
        t0 = time.perf_counter()
        result = evaluate(args.agent, suite, games=args.games)
        elapsed = time.perf_counter() - t0
        rows.append(
            {
                "protocol": PROTOCOL_VERSION,
                "suite": name,
                "agent": args.agent,
                "opponent": suite.opponent,
                "games": result.games,
                "disqualified": result.disqualified,
                "illegal_attempts": result.illegal_attempts,
                "win_rate": result.win_rate,
                "mean_score": result.mean_score,
                "mean_turns": result.mean_turns,
                "mean_final_rack": result.mean_final_rack,
                "stalemate_rate": result.stalemates / max(1, result.games),
                "seconds": round(elapsed, 1),
            }
        )
        if not args.json:
            print(result.report() + f"  [{elapsed:.0f}s]")
    if args.json:
        print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
