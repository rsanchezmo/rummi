"""Throughput benchmark: how many envs run in parallel, and how fast.

Measures the *simulator*, not a policy. The actions are a script -- uniform over
legal, recorded once before the clock starts and replayed inside it -- because
choosing one costs `rng.random((B, A))`, which on the standard config is three
quarters of the timed loop and would be most of the headline. That is the same
discipline :mod:`rummi.bench.bench_env` follows for the same reason.

The per-stage breakdown shows where a step's time goes. Baselines that plan in
Python (greedy, CP-SAT) are reported separately because their cost is per-turn
and scales with the batch, not with the arrays.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from rummi.rules.config import CONFIG_BY_NAME, RummiConfig
from rummi.env.numpy.deal import reset
from rummi.env.numpy.engine import step
from rummi.env.numpy.masks import legal_actions
from rummi.env.numpy.sets import evaluate_slots, slot_stats
from rummi.env.numpy.state import BatchState

REPEATS = 3
"""Both the stages and the headline are best of this many, on a state advanced to
the same point each time -- throughput noise is one-sided, and the two halves of
the printed table are only comparable if they were measured the same way."""

STAGES = ("slot_stats", "evaluate", "legal_actions")


def sample_legal(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Uniform over legal actions, vectorised over the batch."""
    return np.argmax(np.where(mask, rng.random(mask.shape), -1.0), axis=-1)


def script(cfg: RummiConfig, batch_size: int, steps: int, seed: int) -> list[np.ndarray]:
    """The actions to replay, recorded by actually playing them.

    Recorded outside every timed region, so the sampler that chose them is not
    measured. Replaying them on a state advanced to the same point reproduces the
    same game, and `step` validates each against the mask, so a drift raises
    rather than quietly measuring something else.
    """
    rng = np.random.default_rng(seed)
    state = reset(cfg, batch_size, seed=seed)
    out = []
    for _ in range(steps):
        mask = legal_actions(state)
        actions = sample_legal(mask, rng)
        out.append(actions)
        step(state, actions, mask)
    return out


def _advance(
    cfg: RummiConfig, batch_size: int, seed: int, actions: list[np.ndarray]
) -> BatchState:
    """A fresh state with ``actions`` played into it.

    Rebuilt per repeat rather than reused: the state advances while it is timed
    and the work per step is not constant across a run, so a reused state measures
    a later phase of the game each time.
    """
    state = reset(cfg, batch_size, seed=seed)
    for step_actions in actions:
        step(state, step_actions, legal_actions(state))
    return state


def time_it(fn, iters: int) -> float:
    """Seconds per call, best of :data:`REPEATS` short runs."""
    best = float("inf")
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        best = min(best, (time.perf_counter() - t0) / iters)
    return best


def bench(
    cfg: RummiConfig, batch_size: int, iters: int, seed: int = 0, warmup: int = 50
) -> dict[str, float]:
    plan = script(cfg, batch_size, warmup + iters, seed)
    warm = _advance(cfg, batch_size, seed, plan[:warmup])

    stages = {
        "slot_stats": lambda: slot_stats(cfg, warm.table_sets),
        "evaluate": lambda: evaluate_slots(cfg, warm.table_sets),
        "legal_actions": lambda: legal_actions(warm),
    }
    out = {name: time_it(fn, iters) for name, fn in stages.items()}

    best = float("inf")
    for _ in range(REPEATS):
        live = _advance(cfg, batch_size, seed, plan[:warmup])
        t0 = time.perf_counter()
        for step_actions in plan[warmup:]:
            step(live, step_actions, legal_actions(live))
        best = min(best, (time.perf_counter() - t0) / iters)
    out["step_total"] = best
    out["env_steps_per_sec"] = batch_size / best
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", choices=sorted(CONFIG_BY_NAME), default="standard")
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 64, 256, 1024, 4096])
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    cfg = CONFIG_BY_NAME[args.config]
    print(f"config={args.config}  A={cfg.n_actions}  S={cfg.max_sets}  K={cfg.n_kinds}")
    print(
        f"{'B':>6} {'env-steps/s':>12} {'ms/step':>9} "
        + " ".join(f"{n:>16}" for n in STAGES)
    )
    for b in args.batch_sizes:
        r = bench(cfg, b, args.iters, args.seed)
        cells = " ".join(f"{r[n] * 1e3:>13.3f} ms" for n in STAGES)
        print(f"{b:>6} {r['env_steps_per_sec']:>12,.0f} {r['step_total'] * 1e3:>9.3f} {cells}")


if __name__ == "__main__":
    main()
