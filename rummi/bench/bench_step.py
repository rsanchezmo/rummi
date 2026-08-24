"""Throughput benchmark: how many envs run in parallel, and how fast.

Measures the *simulator*, not a policy: actions come from a fully vectorised
uniform-over-legal sampler, and the per-stage breakdown shows where the time
goes. Baselines that plan in Python (greedy, CP-SAT) are reported separately
because their cost is per-turn and scales with the batch, not with the arrays.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from rummi.rules.config import STANDARD, TINY, TINY_GROUPS, RummiConfig
from rummi.env.numpy.deal import reset
from rummi.env.numpy.engine import step
from rummi.env.numpy.masks import legal_actions
from rummi.env.numpy.sets import evaluate_slots, slot_stats
from rummi.env.numpy.state import BatchState

CONFIGS = {"standard": STANDARD, "tiny": TINY, "tiny_groups": TINY_GROUPS}


def sample_legal(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Uniform over legal actions, vectorised over the batch."""
    return np.argmax(np.where(mask, rng.random(mask.shape), -1.0), axis=-1)


def _warm(cfg: RummiConfig, batch_size: int, seed: int) -> tuple[BatchState, np.ndarray]:
    """A state driven a little way in, so slots are populated rather than empty."""
    rng = np.random.default_rng(seed)
    state = reset(cfg, batch_size, seed=seed)
    for _ in range(50):
        mask = legal_actions(state)
        step(state, sample_legal(mask, rng), mask)
    return state, legal_actions(state)


def time_it(fn, iters: int) -> float:
    """Seconds per call, best of three short runs to shake off scheduler noise."""
    best = float("inf")
    for _ in range(3):
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        best = min(best, (time.perf_counter() - t0) / iters)
    return best


def bench(cfg: RummiConfig, batch_size: int, iters: int, seed: int = 0) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    state, mask = _warm(cfg, batch_size, seed)

    stages = {
        "slot_stats": lambda: slot_stats(cfg, state.table_sets),
        "evaluate": lambda: evaluate_slots(cfg, state.table_sets),
        "legal_actions": lambda: legal_actions(state),
        "sample": lambda: sample_legal(mask, rng),
    }
    out = {name: time_it(fn, iters) for name, fn in stages.items()}

    # The full loop has to be measured on a live state, so it advances as it runs.
    live = reset(cfg, batch_size, seed=seed + 1)
    live_rng = np.random.default_rng(seed + 1)

    def full_step():
        m = legal_actions(live)
        step(live, sample_legal(m, live_rng), m)

    t0 = time.perf_counter()
    for _ in range(iters):
        full_step()
    out["step_total"] = (time.perf_counter() - t0) / iters
    out["env_steps_per_sec"] = batch_size / out["step_total"]
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", choices=sorted(CONFIGS), default="standard")
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 64, 256, 1024, 4096])
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    cfg = CONFIGS[args.config]
    print(f"config={args.config}  A={cfg.n_actions}  S={cfg.max_sets}  K={cfg.n_kinds}")
    header = f"{'B':>6} {'env-steps/s':>12} {'ms/step':>9} " + " ".join(
        f"{n:>13}" for n in ("slot_stats", "evaluate", "legal_actions", "sample")
    )
    print(header)
    for b in args.batch_sizes:
        r = bench(cfg, b, args.iters, args.seed)
        cells = " ".join(f"{r[n] * 1e3:>12.3f}m" for n in ("slot_stats", "evaluate", "legal_actions", "sample"))
        print(f"{b:>6} {r['env_steps_per_sec']:>12,.0f} {r['step_total'] * 1e3:>9.3f} {cells}")


if __name__ == "__main__":
    main()
