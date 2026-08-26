"""RL over set templates: which complete set to play, when to stop, when to draw.

    python tools/train_macro.py --updates 200 --eval-games 60

The action space is `rummi/agents/macro.py`'s -- 329 sets, `END_TURN`, `DRAW` --
and every action leaves the table whole, so the half-built invalid workbench that
defeated the primitive-action learner is unreachable rather than penalised.

What that buys, measured before any learning: turns are bounded at 7 micro-actions
against the primitive policy's 71, and a hand-written `by_value` ordering scores
**-143** where the best cloned-then-PPO'd primitive policy scored -230. So the
decision this trains is the one that matters: `first_legal` melds in 29.6% of steps
and `by_value` in 78.5%, from nothing but *which* set to play.

Trained with the same bootstrapped actor-critic over decision transitions as
`train_delegate.py`, for the reason recorded there: handing every decision in an
episode that episode's outcome is unbiased but teaches only a global bias, and
situational play needs per-decision credit. Decisions are per *set* here rather
than per turn, so an episode yields several times more of them.
"""

from __future__ import annotations

import argparse
import dataclasses
import pathlib
import time

import numpy as np
import torch
from torch import nn

from rummi.agents.learned.features import FEATURE_FIELDS, feature_dim, feature_scale
from rummi.agents.learned.torch_net import MASKED
from rummi.agents.macro import MacroAgent, by_value, first_legal, set_templates
from rummi.env.fixed_opponent import FixedOpponentEnv
from rummi.evaluate.protocol import SUITE_BY_NAME, evaluate
from rummi.rules.config import CONFIG_BY_NAME, RewardMode, RummiConfig


class MacroNet(nn.Module):
    """Logits over the macro actions, and a value."""

    def __init__(self, cfg: RummiConfig, n_macros: int, hidden: int = 256) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(feature_dim(cfg), hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.pi = nn.Linear(hidden, n_macros)
        self.v = nn.Linear(hidden, 1)
        # 0.01, as in `learned/architecture.py`: with a handful of the 331 macros
        # legal at a time, a confident wrong start is slow to unlearn.
        nn.init.orthogonal_(self.pi.weight, 0.01)
        nn.init.zeros_(self.pi.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        return self.pi(h), self.v(h).squeeze(-1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="standard", choices=sorted(CONFIG_BY_NAME))
    p.add_argument("--opponent", default="greedy", choices=["greedy", "rearrange", "optimal"])
    p.add_argument("--envs", type=int, default=64)
    p.add_argument("--horizon", type=int, default=256)
    p.add_argument("--updates", type=int, default=200)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-games", type=int, default=0)
    p.add_argument("--out", type=pathlib.Path, default=None)
    args = p.parse_args()

    cfg = dataclasses.replace(
        CONFIG_BY_NAME[args.config], reward_mode=RewardMode.SCORE_NORMALIZED
    )
    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    scale = feature_scale(cfg)
    n_macros = len(set_templates(cfg)) + 2
    net = MacroNet(cfg, n_macros, args.hidden)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    env = FixedOpponentEnv(
        num_envs=args.envs, cfg=cfg, seed=args.seed, opponent=args.opponent
    )
    obs, info = env.reset()

    open_choice: list[tuple[np.ndarray, np.ndarray, int] | None] = [None] * args.envs
    accrued = np.zeros(args.envs, dtype=np.float32)
    steps: list[tuple[np.ndarray, np.ndarray, int, float, np.ndarray, float]] = []
    tally: dict[str, int] = {}

    def features(o, e: int) -> np.ndarray:
        row = np.concatenate(
            [np.asarray(o[f])[e].reshape(-1) for f in FEATURE_FIELDS]
        ).astype(np.float32)
        return row / scale

    def choose(o, e: int, legal: np.ndarray) -> int:
        x = features(o, e)
        logits, _ = net(torch.as_tensor(x)[None])
        masked = torch.where(
            torch.as_tensor(legal)[None], logits, torch.full_like(logits, MASKED)
        )
        macro = int(
            torch.multinomial(torch.softmax(masked[0], -1), 1, generator=generator)
        )
        if open_choice[e] is not None:
            prev_x, prev_legal, prev_a = open_choice[e]
            steps.append((prev_x, prev_legal, prev_a, float(accrued[e]), x, 0.0))
        accrued[e] = 0.0
        open_choice[e] = (x, legal.copy(), macro)
        tally["n"] = tally.get("n", 0) + 1
        tally["end"] = tally.get("end", 0) + int(macro == n_macros - 2)
        tally["draw"] = tally.get("draw", 0) + int(macro == n_macros - 1)
        return macro

    agent = MacroAgent(cfg, choose=choose)
    agent.reset(args.envs)
    print(
        f"config={args.config} opponent={args.opponent} macros={n_macros} "
        f"params={sum(q.numel() for q in net.parameters()):,}",
        flush=True,
    )
    started = time.perf_counter()

    for update in range(1, args.updates + 1):
        steps.clear()
        tally.clear()
        finished = 0

        for _ in range(args.horizon):
            mask = np.asarray(info["action_mask"])
            actions = agent.act(obs, mask)
            obs, reward, term, trunc, info = env.step(actions)
            accrued += np.asarray(reward, dtype=np.float32)
            done = np.asarray(term) | np.asarray(trunc)
            for e in np.flatnonzero(done):
                if open_choice[e] is not None:
                    prev_x, prev_legal, prev_a = open_choice[e]
                    steps.append(
                        (prev_x, prev_legal, prev_a, float(accrued[e]), prev_x, 1.0)
                    )
                    open_choice[e] = None
                accrued[e] = 0.0
                finished += 1

        if not steps:
            print(f"update {update:>4}  no decision closed; raise --horizon", flush=True)
            continue

        x = torch.as_tensor(np.stack([s[0] for s in steps]))
        legal = torch.as_tensor(np.stack([s[1] for s in steps]))
        a = torch.as_tensor(np.asarray([s[2] for s in steps]))
        r = torch.as_tensor(np.asarray([s[3] for s in steps], dtype=np.float32))
        nxt = torch.as_tensor(np.stack([s[4] for s in steps]))
        terminal = torch.as_tensor(np.asarray([s[5] for s in steps], dtype=np.float32))

        logits, value = net(x)
        logits = torch.where(legal, logits, torch.full_like(logits, MASKED))
        with torch.no_grad():
            _, next_value = net(nxt)
            target = r + args.gamma * (1.0 - terminal) * next_value
        logp_all = torch.log_softmax(logits, -1)
        logp = logp_all.gather(1, a[:, None])[:, 0]
        advantage = target - value.detach()
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        entropy = -(logp_all.exp() * logp_all).sum(-1).mean()
        loss = -(logp * advantage).mean() + 0.5 * (value - target).pow(2).mean()
        loss = loss - args.entropy_coef * entropy

        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 0.5)
        opt.step()

        n = max(tally.get("n", 1), 1)
        print(
            f"update {update:>4}  episodes {finished:>4}  decisions {len(steps):>6,}  "
            f"end {tally.get('end', 0) / n:>5.1%}  draw {tally.get('draw', 0) / n:>5.1%}  "
            f"terminal {r[terminal > 0].mean().item():>+7.3f}  H {entropy.item():>5.3f}  "
            f"{len(steps) / (time.perf_counter() - started):>5.0f} dec/s",
            flush=True,
        )
        started = time.perf_counter()

    env.close()

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"cfg": args.config, "hidden": args.hidden, "state": net.state_dict()}, args.out)
        print(f"wrote {args.out}")

    if args.eval_games:
        def greedy_choose(o, e: int, legal: np.ndarray) -> int:
            with torch.no_grad():
                logits, _ = net(torch.as_tensor(features(o, e))[None])
            logits = torch.where(
                torch.as_tensor(legal)[None], logits, torch.full_like(logits, MASKED)
            )
            return int(logits[0].argmax())

        suite = SUITE_BY_NAME[
            "standard-greedy" if args.config == "standard" else "tiny"
        ]
        for label, ch in (
            ("learned", greedy_choose),
            ("by_value", by_value(cfg)),
            ("first_legal", first_legal),
        ):
            scored = MacroAgent(cfg, choose=ch)
            result = evaluate(label, suite, build_agent=lambda c, s=scored: s,
                              games=args.eval_games)
            print(
                f"  {label:12s} win {result.win_rate:>6.1%}  score {result.mean_score:>+8.2f}  "
                f"illegal {result.illegal_attempts}  n={result.games}"
            )


if __name__ == "__main__":
    main()
