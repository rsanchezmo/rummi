"""NumPy vs torch, across devices and batch sizes.

Both backends are driven identically and by the cheapest possible action choice
(the first legal action), so the number measured is the simulator and not a
policy. Conformance is verified separately in ``tests/test_torch_backend.py`` --
these figures only mean something because the two implementations are known to
agree.

MPS and CUDA are asynchronous, so each timed region is synchronised before the
clock is read; without that the loop would measure enqueue time.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from rummi.core.config import STANDARD, TINY, TINY_GROUPS, RummiConfig

CONFIGS = {"standard": STANDARD, "tiny": TINY, "tiny_groups": TINY_GROUPS}


def bench_numpy(cfg: RummiConfig, batch_size: int, iters: int, warmup: int = 20) -> float:
    from rummi.core.deal import reset
    from rummi.core.engine import step
    from rummi.core.masks import legal_actions

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

    from rummi.backends.torch_backend import sim

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
