"""Record and replay games, and watch one live.

A recording stores the config, the seed and the action sequence -- nothing more.
The simulator is deterministic and carries no RNG in its step function, so
replaying those actions reconstructs every intermediate state exactly. That makes
recordings tiny and makes the format immune to state-layout changes.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np

from rummi.rules.config import STANDARD, TINY, TINY_GROUPS, RewardMode, RummiConfig
from rummi.env.numpy.deal import reset
from rummi.env.numpy.engine import step
from rummi.env.numpy.masks import legal_actions
from rummi.render.driver import RenderMode, Renderer

CONFIGS = {"standard": STANDARD, "tiny": TINY, "tiny_groups": TINY_GROUPS}


def config_to_json(cfg: RummiConfig) -> dict:
    out = asdict(cfg)
    out["reward_mode"] = cfg.reward_mode.value
    return out


def config_from_json(data: dict) -> RummiConfig:
    known = {f.name for f in fields(RummiConfig)}
    kwargs = {k: v for k, v in data.items() if k in known}
    kwargs["reward_mode"] = RewardMode(kwargs["reward_mode"])
    return RummiConfig(**kwargs)


class Recorder:
    """Append a single env's action stream to a JSONL file."""

    def __init__(self, path: Path, cfg: RummiConfig, seed: int, env_index: int = 0) -> None:
        self.path = Path(path)
        self.env_index = env_index
        self.handle = self.path.open("w")
        header = {"config": config_to_json(cfg), "seed": seed, "env_index": env_index}
        self.handle.write(json.dumps(header) + "\n")

    def record(self, actions: np.ndarray) -> None:
        self.handle.write(json.dumps({"a": int(actions[self.env_index])}) + "\n")

    def close(self) -> None:
        self.handle.close()

    def __enter__(self) -> "Recorder":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def load(path: Path) -> tuple[RummiConfig, int, list[int]]:
    lines = Path(path).read_text().splitlines()
    header = json.loads(lines[0])
    return (
        config_from_json(header["config"]),
        int(header["seed"]),
        [json.loads(line)["a"] for line in lines[1:]],
    )


def replay(path: Path, renderer: Renderer, pause: bool = False):
    """Re-simulate a recording, rendering each step, and return the final state.

    Every recorded action is re-validated against the live mask, so a replay that
    completes is itself proof that the run was reproducible.
    """
    cfg, seed, actions = load(path)
    state = reset(cfg, 1, seed=seed)
    renderer.render(state, legal_actions(state))
    for action in actions:
        mask = legal_actions(state)
        step(state, np.array([action]), mask)
        renderer.render(state, legal_actions(state))
        if pause:
            input()
    renderer.close()
    return state


def play(
    cfg: RummiConfig,
    policy_name: str,
    seed: int,
    renderer: Renderer,
    max_steps: int = 20_000,
    recorder: Recorder | None = None,
) -> None:
    from rummi.bench.fuzz import make_policy

    policy = make_policy(cfg, policy_name, seed)
    state = reset(cfg, 1, seed=seed)
    renderer.render(state, legal_actions(state))
    for _ in range(max_steps):
        mask = legal_actions(state)
        actions = policy(state, mask)
        if recorder is not None:
            recorder.record(actions)
        step(state, actions, mask)
        renderer.render(state, legal_actions(state))
        if state.done.all():
            break
    renderer.render(state, legal_actions(state))
    renderer.close()
    outcome = "truncated" if state.truncated[0] else f"seat {int(state.winner[0])} wins"
    print(f"\nfinished after {int(state.turn_count[0])} turns: {outcome}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--play", action="store_true", help="run a fresh game instead of replaying")
    p.add_argument("--replay", type=Path, help="path to a recorded .jsonl game")
    p.add_argument("--out", type=Path, help="record the played game to this path")
    p.add_argument("--config", choices=sorted(CONFIGS), default="standard")
    p.add_argument("--policy", choices=["random", "greedy", "optimal"], default="greedy")
    p.add_argument("--render-mode", choices=[m.value for m in RenderMode], default="ansi")
    p.add_argument("--fps", type=float, default=12.0)
    p.add_argument("--every", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pause", action="store_true", help="wait for Enter between frames")
    args = p.parse_args()

    if args.replay:
        cfg, _, _ = load(args.replay)
        final = replay(
            args.replay, Renderer(cfg, args.render_mode, fps=args.fps, every=args.every), args.pause
        )
        outcome = "truncated" if final.truncated[0] else f"seat {int(final.winner[0])} wins"
        print(f"\nreplayed {int(final.turn_count[0])} turns: {outcome}")
        return

    cfg = CONFIGS[args.config]
    renderer = Renderer(cfg, args.render_mode, fps=args.fps, every=args.every)
    recorder = Recorder(args.out, cfg, args.seed) if args.out else None
    try:
        play(cfg, args.policy, args.seed, renderer, recorder=recorder)
    finally:
        if recorder:
            recorder.close()


if __name__ == "__main__":
    main()
