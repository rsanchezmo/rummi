"""RL over two actions: play the turn a planner found, or hold and draw.

    python tools/train_delegate.py --inner greedy --updates 60
    python tools/train_delegate.py --inner optimal --updates 20 --envs 16

The question is whether **cross-turn strategy exists**. Every bundled agent
maximises the turn in front of it, and the table is shared -- tiles you place
become material any opponent may rearrange -- so always playing the best available
turn gives structure away. A policy that learns to hold sometimes, and beats the
planner it delegates to, is that strategy made visible.

Why this trains where `train_ppo.py` struggles: the action space is **two**, and
every action is a whole legal turn produced by `rummi/solver/` or a bundled agent.
None of the failures measured on the flat 2400-action space apply -- no mask
starvation, no cloning prerequisite, and no half-built invalid tables, because the
policy only ever acts at a turn boundary and the planner's turn is valid by
construction.

**Actor-critic over turns, bootstrapped.** The reward is terminal, so handing every
decision in an episode that one outcome is unbiased but nearly useless: ~30
decisions share one label, and the only thing learnable from it is a global
"hold less often". Situational holding needs per-decision credit, so each turn
decision is a transition `(features, choice, reward since the last decision, next
features)` and the advantage is bootstrapped through the critic. That is what lets
the outcome propagate back to the decision that caused it.

`decide` is handed the candidate turn, so the policy sees what it would be playing
-- how many tiles it sheds and how much of the table it breaks open -- not only the
board. Judging "this play exposes too much" is impossible without it.
"""

from __future__ import annotations

import argparse
import dataclasses
import pathlib
import time

import numpy as np
import torch
from torch import nn

from rummi.agents.delegating import DelegatingAgent, PlanSummary, tiles_at_least
from rummi.agents.learned.features import FEATURE_FIELDS, feature_dim, feature_scale
from rummi.env.fixed_opponent import FixedOpponentEnv
from rummi.evaluate.protocol import Suite, evaluate
from rummi.rules.config import CONFIG_BY_NAME, RewardMode, RummiConfig

EXTRA = 3
"""Plan features beyond the observation: tiles shed, sets broken, turn length."""


def plan_features(cfg: RummiConfig, summary: PlanSummary) -> np.ndarray:
    """Scaled like `features.feature_scale` does, so no input dominates the rest."""
    return np.array(
        [
            summary.tiles / cfg.rack_size,
            summary.dissolves / cfg.max_sets,
            summary.length / cfg.max_set_len,
        ],
        dtype=np.float32,
    )


class Decider(nn.Module):
    """Two logits and a value, from the board plus the candidate turn."""

    def __init__(self, cfg: RummiConfig, hidden: int = 64) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(feature_dim(cfg) + EXTRA, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.pi = nn.Linear(hidden, 2)
        self.v = nn.Linear(hidden, 1)
        # Small policy head for the same reason `learned/architecture.py` uses 0.01:
        # a fresh policy should be near 50/50, not committed to always holding.
        nn.init.orthogonal_(self.pi.weight, 0.01)
        nn.init.zeros_(self.pi.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        return self.pi(h), self.v(h).squeeze(-1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="standard", choices=sorted(CONFIG_BY_NAME))
    p.add_argument(
        "--inner", default="greedy", choices=["greedy", "rearrange", "optimal"],
        help="the planner whose turn is on offer. greedy is ~100x faster per turn; "
             "the headline claim needs optimal, which cannot batch",
    )
    p.add_argument("--opponent", default="greedy", choices=["greedy", "rearrange", "optimal"])
    p.add_argument("--envs", type=int, default=64)
    p.add_argument("--horizon", type=int, default=256)
    p.add_argument("--updates", type=int, default=60)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-games", type=int, default=0)
    p.add_argument("--out", type=pathlib.Path, default=None)
    args = p.parse_args()

    cfg = dataclasses.replace(
        CONFIG_BY_NAME[args.config], reward_mode=RewardMode.SCORE_NORMALIZED
    )
    torch.manual_seed(args.seed)
    net = Decider(cfg)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    generator = torch.Generator().manual_seed(args.seed)
    scale = feature_scale(cfg)

    env = FixedOpponentEnv(
        num_envs=args.envs, cfg=cfg, seed=args.seed, opponent=args.opponent
    )
    obs, info = env.reset()

    # The decision each env is still accruing reward for, and that reward.
    open_choice: list[tuple[np.ndarray, int] | None] = [None] * args.envs
    accrued = np.zeros(args.envs, dtype=np.float32)
    steps: list[tuple[np.ndarray, int, float, np.ndarray, float]] = []
    current: dict = {}

    def decide(o, e: int, summary: PlanSummary) -> bool:
        row = np.concatenate(
            [np.asarray(o[f])[e].reshape(-1) for f in FEATURE_FIELDS]
        ).astype(np.float32) / scale
        x = np.concatenate([row, plan_features(cfg, summary)])
        logits, _ = net(torch.as_tensor(x)[None])
        action = int(torch.multinomial(torch.softmax(logits[0], -1), 1, generator=generator))
        # A new decision closes the previous one: everything that happened in
        # between is the reward for taking it.
        if open_choice[e] is not None:
            prev_x, prev_a = open_choice[e]
            steps.append((prev_x, prev_a, float(accrued[e]), x, 0.0))
        accrued[e] = 0.0
        open_choice[e] = (x, action)
        current.setdefault("n", 0)
        current["n"] += 1
        current["held"] = current.get("held", 0) + (1 - action)
        return action == 1

    agent = DelegatingAgent(cfg, inner=args.inner, decide=decide)
    agent.reset(args.envs)

    print(
        f"config={args.config} inner={args.inner} opponent={args.opponent} "
        f"envs={args.envs} params={sum(p.numel() for p in net.parameters()):,}",
        flush=True,
    )
    started = time.perf_counter()

    for update in range(1, args.updates + 1):
        steps.clear()
        current.clear()
        finished = 0

        for _ in range(args.horizon):
            mask = np.asarray(info["action_mask"])
            actions = agent.act(obs, mask)
            obs, reward, term, trunc, info = env.step(actions)
            done = np.asarray(term) | np.asarray(trunc)
            reward = np.asarray(reward, dtype=np.float32)
            accrued += reward
            for e in np.flatnonzero(done):
                # The last decision of an episode is closed by the terminal
                # reward, with no successor to bootstrap from.
                if open_choice[e] is not None:
                    prev_x, prev_a = open_choice[e]
                    steps.append((prev_x, prev_a, float(accrued[e]), prev_x, 1.0))
                    open_choice[e] = None
                accrued[e] = 0.0
                finished += 1

        if not steps:
            print(f"update {update:>4}  no decision closed; raise --horizon", flush=True)
            continue

        x = torch.as_tensor(np.stack([s[0] for s in steps]))
        a = torch.as_tensor(np.asarray([s[1] for s in steps]))
        r = torch.as_tensor(np.asarray([s[2] for s in steps], dtype=np.float32))
        nxt = torch.as_tensor(np.stack([s[3] for s in steps]))
        terminal = torch.as_tensor(np.asarray([s[4] for s in steps], dtype=np.float32))

        logits, value = net(x)
        with torch.no_grad():
            _, next_value = net(nxt)
            target = r + args.gamma * (1.0 - terminal) * next_value
        logp = torch.log_softmax(logits, -1).gather(1, a[:, None])[:, 0]
        advantage = target - value.detach()
        # Normalised because the terminal score dominates the scale and every
        # non-terminal transition carries reward zero.
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        probs = torch.softmax(logits, -1)
        entropy = -(probs * torch.log_softmax(logits, -1)).sum(-1).mean()
        loss = -(logp * advantage).mean() + 0.5 * (value - target).pow(2).mean()
        loss = loss - args.entropy_coef * entropy

        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 0.5)
        opt.step()

        held = current.get("held", 0) / max(current.get("n", 1), 1)
        print(
            f"update {update:>4}  episodes {finished:>4}  decisions {len(steps):>6,}  "
            f"held {held:>6.1%}  terminal {r[terminal > 0].mean().item():>+7.3f}  "
            f"H {entropy.item():.3f}  {len(steps) / (time.perf_counter() - started):>6.0f} dec/s",
            flush=True,
        )
        started = time.perf_counter()

    env.close()

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"cfg": args.config, "inner": args.inner, "state": net.state_dict()}, args.out)
        print(f"wrote {args.out}")

    if args.eval_games:
        # Deterministic for scoring, and against the planner it delegates to: that
        # is the only opponent with headroom, since `inner` already beats greedy.
        def greedy_decide(o, e: int, summary: PlanSummary) -> bool:
            row = np.concatenate(
                [np.asarray(o[f])[e].reshape(-1) for f in FEATURE_FIELDS]
            ).astype(np.float32) / scale
            x = np.concatenate([row, plan_features(cfg, summary)])
            with torch.no_grad():
                logits, _ = net(torch.as_tensor(x)[None])
            return bool(logits[0].argmax().item() == 1)

        suite = Suite(
            "standard-optimal" if args.config == "standard" else "tiny",
            CONFIG_BY_NAME[args.config], opponent=args.inner,
            games=args.eval_games, seed_base=3_000, batch_size=16,
        )
        for label, dec in (
            ("learned", greedy_decide),
            ("always_play", tiles_at_least(1)),
        ):
            scored = DelegatingAgent(cfg, inner=args.inner, decide=dec)
            result = evaluate(label, suite, build_agent=lambda c, a=scored: a)
            frac = scored.held / max(scored.held + scored.played, 1)
            print(
                f"  {label:12s} win {result.win_rate:>6.1%}  score {result.mean_score:>+8.2f}  "
                f"held {frac:>6.1%}  n={result.games}"
            )


if __name__ == "__main__":
    main()
