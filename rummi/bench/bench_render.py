"""Prove the render path is bounded: switched off it must cost nothing, and
switched on it must be capped by ``render_fps`` rather than by the step rate.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from rummi.bench.bench_step import CONFIGS, sample_legal
from rummi.rules.config import RummiConfig
from rummi.env.numpy.deal import reset
from rummi.env.numpy.engine import step
from rummi.env.numpy.masks import legal_actions
from rummi.render.driver import RenderMode, Renderer


def run(cfg: RummiConfig, batch_size: int, steps: int, renderer: Renderer | None) -> float:
    rng = np.random.default_rng(0)
    state = reset(cfg, batch_size, seed=0)
    t0 = time.perf_counter()
    for _ in range(steps):
        mask = legal_actions(state)
        step(state, sample_legal(mask, rng), mask)
        if renderer is not None:
            renderer.render(state, mask)
    dt = time.perf_counter() - t0
    if renderer is not None:
        renderer.close()
    return batch_size * steps / dt


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", choices=sorted(CONFIGS), default="standard")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--steps", type=int, default=200)
    args = p.parse_args()
    cfg = CONFIGS[args.config]

    baseline = run(cfg, args.batch_size, args.steps, None)
    print(f"{'no renderer at all':<34} {baseline:>12,.0f} env-steps/s")
    import io

    def ansi(**kw):
        # Render to a detached buffer so terminal I/O is not what we are timing.
        r = Renderer(cfg, RenderMode.ANSI, **kw)
        from rummi.render.text import TerminalView

        r._terminal = TerminalView(stream=io.StringIO(), color=True, live=True)
        return r

    for label, renderer in (
        ("render_mode=none", Renderer(cfg, RenderMode.NONE)),
        ("rgb_array, every step", Renderer(cfg, RenderMode.RGB_ARRAY)),
        ("rgb_array, capped 12 fps", Renderer(cfg, RenderMode.RGB_ARRAY, fps=12.0)),
        ("ansi, every step", ansi()),
        ("ansi, capped 12 fps", ansi(fps=12.0)),
    ):
        rate = run(cfg, args.batch_size, args.steps, renderer)
        print(f"{label:<34} {rate:>12,.0f} env-steps/s  ({rate / baseline:.0%} of baseline)")


if __name__ == "__main__":
    main()
