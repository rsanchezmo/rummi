"""NumPy vs torch, across devices and batch sizes.

Both backends are driven identically and by the cheapest possible action choice
(the first legal action), so the number measured is the simulator and not a
policy. Conformance is verified separately in ``tests/test_backends.py`` --
these figures only mean something because the two implementations are known to
agree.

MPS and CUDA are asynchronous, so each timed region is synchronised before the
clock is read; without that the loop would measure enqueue time.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
import pathlib
import time

import numpy as np

from rummi.rules.config import CONFIG_BY_NAME, RummiConfig



def _best(samples: list[float]) -> float:
    """Throughput noise is one-sided -- interference only ever slows a run -- so
    the maximum of several identical measurements is the honest estimate."""
    return max(samples)


def bench_numpy(
    cfg: RummiConfig, batch_size: int, iters: int, warmup: int = 20, repeats: int = 3
) -> float:
    from rummi.env.numpy.deal import reset
    from rummi.env.numpy.engine import step
    from rummi.env.numpy.masks import legal_actions

    def advance(state, n):
        for _ in range(n):
            m = legal_actions(state)
            step(state, np.argmax(m, axis=-1), m)

    samples = []
    for _ in range(repeats):
        # A fresh state each repeat, advanced to the same point. The state moves
        # while it is being timed and the table fills as it goes, so reusing one
        # across repeats measures a different, later phase of the game each time.
        state = reset(cfg, batch_size, seed=0)
        advance(state, warmup)
        t0 = time.perf_counter()
        advance(state, iters)
        samples.append(batch_size * iters / (time.perf_counter() - t0))
    return _best(samples)


def bench_torch(
    cfg: RummiConfig, batch_size: int, iters: int, device: str, compile_it: bool = False,
    warmup: int = 20, validate: bool = True, repeats: int = 3,
) -> float:
    import torch

    from rummi.env.torch import sim

    dev = torch.device(device)
    if compile_it:
        # Dynamo's recompile limit is per code object and shared across every cell
        # of the sweep, so past the eighth shape `compile` silently falls back to
        # eager -- publishing an eager number under a compiled name. Resetting
        # gives each cell its own cache.
        torch._dynamo.reset()
    legal = torch.compile(sim.legal_actions, dynamic=False) if compile_it else sim.legal_actions

    def sync():
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()

    def advance(state, n):
        for _ in range(n):
            m = legal(state)
            # Validation reads a boolean off the device, a host sync every step;
            # the env exposes it as a switch for exactly this reason.
            sim.step(state, m.to(torch.int8).argmax(dim=-1), m if validate else None)

    # One throwaway pass so compilation is not inside a timed region.
    advance(sim.reset(cfg, batch_size, seed=0, device=dev), warmup)
    sync()

    samples = []
    for _ in range(repeats):
        state = sim.reset(cfg, batch_size, seed=0, device=dev)
        advance(state, warmup)
        sync()
        t0 = time.perf_counter()
        advance(state, iters)
        sync()
        samples.append(batch_size * iters / (time.perf_counter() - t0))
    return _best(samples)


def bench_jax(
    cfg: RummiConfig, batch_size: int, iters: int, mode: str = "fused", warmup: int = 20,
    repeats: int = 3,
) -> float:
    """``mode`` selects how much is handed to the compiler.

    ``step`` jits the mask and the transition separately, so the Python loop sits
    between them. ``fused`` jits mask-choose-step as one function. ``scan`` puts
    the whole rollout inside ``lax.scan``, the idiomatic JAX shape, which removes
    the Python loop entirely.
    """
    from functools import partial

    import jax
    import jax.numpy as jnp

    from rummi.env.jax import sim

    def pick(mask):
        return jnp.argmax(mask.astype(jnp.int8), axis=-1)

    @partial(jax.jit, static_argnums=0)
    def once(cfg_, s):
        m = sim.legal_actions(cfg_, s)
        s, _ = sim.step(cfg_, s, pick(m))
        return s

    @partial(jax.jit, static_argnums=(0, 1))
    def rollout(cfg_, n, s):
        def body(carry, _):
            m = sim.legal_actions(cfg_, carry)
            nxt, _ = sim.step(cfg_, carry, pick(m))
            return nxt, None

        out, _ = jax.lax.scan(body, s, None, length=n)
        return out

    def advance(state, n):
        if mode == "scan":
            return rollout(cfg, n, state)
        for _ in range(n):
            state = once(cfg, state)
        return state

    warm = jax.block_until_ready(advance(sim.reset(cfg, batch_size, seed=0), warmup))
    # `rollout` is jitted on the scan length, so the timed advance is a *different*
    # graph from the warm-up's. Tracing it here is what keeps a compile out of the
    # clock: `_best` hides one only while there is a later repeat to beat it, so
    # without this `--repeats 1` publishes a compile-polluted figure.
    jax.block_until_ready(advance(warm, iters))

    samples = []
    for _ in range(repeats):
        state = jax.block_until_ready(advance(sim.reset(cfg, batch_size, seed=0), warmup))
        t0 = time.perf_counter()
        state = jax.block_until_ready(advance(state, iters))
        samples.append(batch_size * iters / (time.perf_counter() - t0))
    return _best(samples)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", choices=sorted(CONFIG_BY_NAME), default="standard")
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[64, 256, 1024, 4096])
    p.add_argument("--iters", type=int, default=60)
    p.add_argument("--compile", action="store_true", help="also time torch.compile")
    p.add_argument(
        "--no-validate", action="store_true",
        help="also time with action validation off, which removes a per-step host sync",
    )
    p.add_argument("--json", type=str, default=None, help="also write results here")
    p.add_argument(
        "--repeats", type=int, default=3,
        help="timed runs per cell; the best is reported, since a slow run means "
             "interference and a fast one cannot be luck",
    )
    args = p.parse_args()

    cfg = CONFIG_BY_NAME[args.config]
    import torch

    reps = args.repeats
    backends: list[tuple[str, Callable[[int, int], float]]] = [
        ("numpy", lambda b, n: bench_numpy(cfg, b, n, repeats=reps))
    ]
    backends.append(("torch-cpu", lambda b, n: bench_torch(cfg, b, n, "cpu", repeats=reps)))
    if torch.backends.mps.is_available():
        backends.append(("torch-mps", lambda b, n: bench_torch(cfg, b, n, "mps", repeats=reps)))
    if torch.cuda.is_available():
        backends.append(("torch-cuda", lambda b, n: bench_torch(cfg, b, n, "cuda", repeats=reps)))
    if args.compile:
        backends.append(("torch-cpu+compile", lambda b, n: bench_torch(cfg, b, n, "cpu", True, repeats=reps)))
        if torch.backends.mps.is_available():
            backends.append(("torch-mps+compile", lambda b, n: bench_torch(cfg, b, n, "mps", True, repeats=reps)))
    try:
        import jax  # noqa: F401

        backends.append(("jax-step", lambda b, n: bench_jax(cfg, b, n, "step", repeats=reps)))
        backends.append(("jax-fused", lambda b, n: bench_jax(cfg, b, n, "fused", repeats=reps)))
        backends.append(("jax-scan", lambda b, n: bench_jax(cfg, b, n, "scan", repeats=reps)))
    except ModuleNotFoundError:
        pass
    if args.no_validate:
        backends.append(
            ("torch-cpu+comp-noval", lambda b, n: bench_torch(cfg, b, n, "cpu", True, validate=False, repeats=reps))
        )
        if torch.backends.mps.is_available():
            backends.append(
                ("torch-mps+comp-noval", lambda b, n: bench_torch(cfg, b, n, "mps", True, validate=False, repeats=reps))
            )

    rows: list[dict] = []
    print(f"config={args.config}  A={cfg.n_actions}  S={cfg.max_sets}  K={cfg.n_kinds}")
    print(f"{'backend':<20}" + "".join(f"{b:>13}" for b in args.batch_sizes))
    print(f"{'':<20}" + "".join(f"{'env-steps/s':>13}" for _ in args.batch_sizes))
    baseline: dict[int, float] = {}
    for name, fn in backends:
        cells = []
        for b in args.batch_sizes:
            rate = fn(b, args.iters)
            baseline.setdefault(b, rate)
            speedup = rate / baseline[b]
            rows.append(
                {
                    "backend": name,
                    "batch_size": b,
                    "env_steps_per_sec": rate,
                    "speedup": speedup,
                }
            )
            cells.append(f"{rate:>10,.0f}" + (f" {speedup:>.1f}x" if name != "numpy" else "     "))
        print(f"{name:<20}" + "".join(f"{c:>13}" for c in cells))

    if args.json:
        import json
        import platform

        payload = {
            "config": args.config,
            "n_actions": cfg.n_actions,
            "iters": args.iters,
            # Recorded because a throughput number without the machine is a
            # number nobody can reproduce or argue with.
            "machine": f"{platform.machine()} / {platform.system()}",
            "rows": rows,
        }
        pathlib.Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.json).write_text(json.dumps(payload, indent=2))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
