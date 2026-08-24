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
import time

import numpy as np

from rummi.rules.config import STANDARD, TINY, TINY_GROUPS, RummiConfig

CONFIGS = {"standard": STANDARD, "tiny": TINY, "tiny_groups": TINY_GROUPS}


def bench_numpy(cfg: RummiConfig, batch_size: int, iters: int, warmup: int = 20) -> float:
    from rummi.env.numpy.deal import reset
    from rummi.env.numpy.engine import step
    from rummi.env.numpy.masks import legal_actions

    state = reset(cfg, batch_size, seed=0)
    for _ in range(warmup):
        m = legal_actions(state)
        step(state, np.argmax(m, axis=-1), m)

    t0 = time.perf_counter()
    for _ in range(iters):
        m = legal_actions(state)
        step(state, np.argmax(m, axis=-1), m)
    return batch_size * iters / (time.perf_counter() - t0)


def bench_torch(
    cfg: RummiConfig, batch_size: int, iters: int, device: str, compile_it: bool = False,
    warmup: int = 20, validate: bool = True,
) -> float:
    import torch

    from rummi.env.torch import sim

    dev = torch.device(device)
    state = sim.reset(cfg, batch_size, seed=0, device=dev)

    legal = sim.legal_actions
    if compile_it:
        legal = torch.compile(sim.legal_actions, dynamic=False)

    def sync():
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()

    def once():
        m = legal(state)
        # Validation reads a boolean off the device, which is a host sync every
        # step; the env exposes it as a switch for exactly this reason.
        sim.step(state, m.to(torch.int8).argmax(dim=-1), m if validate else None)

    for _ in range(warmup):
        once()
    sync()

    t0 = time.perf_counter()
    for _ in range(iters):
        once()
    sync()
    return batch_size * iters / (time.perf_counter() - t0)


def bench_jax(
    cfg: RummiConfig, batch_size: int, iters: int, mode: str = "fused", warmup: int = 3
) -> float:
    """``mode`` selects how much is handed to the compiler.

    ``step`` jits the mask and the transition separately, so the Python loop sits
    between them. ``fused`` jits mask-choose-step as one function. ``scan`` puts
    the whole rollout inside ``lax.scan``, which is the idiomatic JAX shape and
    removes the Python loop entirely.
    """
    from functools import partial

    import jax
    import jax.numpy as jnp

    from rummi.env.jax import sim

    state = sim.reset(cfg, batch_size, seed=0)

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

    if mode == "scan":
        state = jax.block_until_ready(rollout(cfg, iters, state))
        t0 = time.perf_counter()
        state = jax.block_until_ready(rollout(cfg, iters, state))
        return batch_size * iters / (time.perf_counter() - t0)

    body = once if mode == "fused" else None
    for _ in range(warmup):
        if body is None:
            m = sim.legal_actions(cfg, state)
            state, _ = sim.step(cfg, state, pick(m))
        else:
            state = body(cfg, state)
    jax.block_until_ready(state)

    t0 = time.perf_counter()
    for _ in range(iters):
        if body is None:
            m = sim.legal_actions(cfg, state)
            state, _ = sim.step(cfg, state, pick(m))
        else:
            state = body(cfg, state)
    jax.block_until_ready(state)
    return batch_size * iters / (time.perf_counter() - t0)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", choices=sorted(CONFIGS), default="standard")
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[64, 256, 1024, 4096])
    p.add_argument("--iters", type=int, default=60)
    p.add_argument("--compile", action="store_true", help="also time torch.compile")
    p.add_argument(
        "--no-validate", action="store_true",
        help="also time with action validation off, which removes a per-step host sync",
    )
    args = p.parse_args()

    cfg = CONFIGS[args.config]
    import torch

    backends: list[tuple[str, callable]] = [("numpy", lambda b, n: bench_numpy(cfg, b, n))]
    backends.append(("torch-cpu", lambda b, n: bench_torch(cfg, b, n, "cpu")))
    if torch.backends.mps.is_available():
        backends.append(("torch-mps", lambda b, n: bench_torch(cfg, b, n, "mps")))
    if torch.cuda.is_available():
        backends.append(("torch-cuda", lambda b, n: bench_torch(cfg, b, n, "cuda")))
    if args.compile:
        backends.append(("torch-cpu+compile", lambda b, n: bench_torch(cfg, b, n, "cpu", True)))
        if torch.backends.mps.is_available():
            backends.append(("torch-mps+compile", lambda b, n: bench_torch(cfg, b, n, "mps", True)))
    try:
        import jax  # noqa: F401

        backends.append(("jax-step", lambda b, n: bench_jax(cfg, b, n, "step")))
        backends.append(("jax-fused", lambda b, n: bench_jax(cfg, b, n, "fused")))
        backends.append(("jax-scan", lambda b, n: bench_jax(cfg, b, n, "scan")))
    except ModuleNotFoundError:
        pass
    if args.no_validate:
        backends.append(
            ("torch-cpu+comp-noval", lambda b, n: bench_torch(cfg, b, n, "cpu", True, validate=False))
        )
        if torch.backends.mps.is_available():
            backends.append(
                ("torch-mps+comp-noval", lambda b, n: bench_torch(cfg, b, n, "mps", True, validate=False))
            )

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
            cells.append(f"{rate:>10,.0f}" + (f" {speedup:>.1f}x" if name != "numpy" else "     "))
        print(f"{name:<20}" + "".join(f"{c:>13}" for c in cells))


if __name__ == "__main__":
    main()
