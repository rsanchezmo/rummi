"""Throughput of the Gymnasium env, per backend.

Distinct from :mod:`rummi.bench.bench_backends`, which measures the *simulator*.
This measures what a training loop actually gets: one `RummiVectorEnv.step`, so
mask, transition, observation encoding and next-step autoreset all included. That
gap is the whole reason the torch and jax observation encoders exist -- without
them a device backend copies its observation to the host every step.

Actions are constant `DRAW`, which is always legal, so the figure is the env and
not the cost of sampling from a `(B, n_actions)` mask. That sampling is a real
cost for a policy; it is just not the env's.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np

from rummi.rules.config import CONFIG_BY_NAME, RummiConfig
from rummi.env.api import available


def rate(
    backend: str, cfg: RummiConfig, batch_size: int, iters: int, repeats: int = 3
) -> float:
    from rummi.env.vector_env import RummiVectorEnv

    actions = np.full(batch_size, cfg.draw_action, dtype=np.int64)
    if backend.endswith("+compile"):
        import torch

        # Dynamo's recompile limit is per code object and shared across every env
        # built here, so past it `compile` falls back to eager without saying so and
        # the cell reports an eager number under a compiled name.
        torch._dynamo.reset()
    best = 0.0
    for _ in range(repeats):
        # A fresh env each repeat: the state advances while it is timed and the
        # table fills as it goes, so reusing one measures a later, cheaper phase.
        env = RummiVectorEnv(num_envs=batch_size, cfg=cfg, seed=0, backend=backend)
        env.reset()
        # Two throwaway steps keep tracing out of the timed region: `reset` builds
        # the observation graph, the first step the transition's.
        env.step(actions)
        env.step(actions)
        t0 = time.perf_counter()
        for _ in range(iters):
            env.step(actions)
        best = max(best, iters * batch_size / (time.perf_counter() - t0))
        env.close()
    return best


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", choices=sorted(CONFIG_BY_NAME), default="standard")
    p.add_argument("--backends", nargs="+", default=None, help="default: everything installed")
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1024, 4096])
    p.add_argument("--iters", type=int, default=80)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--json", type=pathlib.Path, default=None)
    p.add_argument(
        "--compile", action="store_true", help="also time the torch backends compiled"
    )
    args = p.parse_args()

    cfg = CONFIG_BY_NAME[args.config]
    backends = args.backends or available()
    if args.compile:
        backends = backends + [f"{b}+compile" for b in backends if b.startswith("torch")]
    rows: dict[str, dict[int, float]] = {}

    width = max(12, max(len(b) for b in backends) + 2)
    header = "backend".ljust(width) + "".join(f"{b:>12}" for b in args.batch_sizes)
    print(header)
    print("-" * len(header))
    for name in backends:
        rows[name] = {}
        line = name.ljust(width)
        for b in args.batch_sizes:
            r = rate(name, cfg, b, args.iters, args.repeats)
            rows[name][b] = r
            line += f"{r / 1000:>11.1f}k"
        print(line, flush=True)

    baseline = rows.get("numpy", {})
    if baseline:
        print("\nvs numpy at the same batch size:")
        for name, by_batch in rows.items():
            speedups = "  ".join(
                f"{b}: {by_batch[b] / baseline[b]:.1f}x" for b in args.batch_sizes if baseline.get(b)
            )
            print(f"  {name.ljust(width)}{speedups}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    "config": args.config,
                    "iters": args.iters,
                    "repeats": args.repeats,
                    "env_steps_per_second": {
                        name: {str(b): v for b, v in by_batch.items()}
                        for name, by_batch in rows.items()
                    },
                },
                indent=2,
            )
        )
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
