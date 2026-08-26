"""RL over set templates: which complete set to play, when to stop, when to draw.

    python tools/train_macro.py --updates 200 --eval-games 60

The action space is `rummi/agents/macro.py`'s -- 329 sets, `END_TURN`, `DRAW` --
and every action leaves the table whole, so the half-built invalid workbench that
defeated the primitive-action learner is unreachable rather than penalised.

What that buys, measured before any learning: turns are bounded at 7 micro-actions
against the primitive policy's 71, and the hand-written orderings score **-141**
(`by_value`) and **-147** (`first_legal`) where the best cloned-then-PPO'd
primitive policy scored -230. Those two being so close is the point: *which* set to
play is worth ~6 points, so what this trains is mostly when to keep playing, when
to end the turn, and when not to start.

**PPO over turn decisions.** Each batch is reused for `--epochs` passes over
`--minibatches` minibatches with the ratio clipped, and the mask is stored so
scoring an old action against the policy that took it stays meaningful. One pass
and discard leaves most of the signal in the data; `--epochs 1` recovers the plain
policy gradient this started as.

The advantage is bootstrapped through the critic, per decision, for the reason
recorded in `train_delegate.py`: handing every decision in an episode that
episode's outcome is unbiased but teaches only a global bias, and situational play
needs per-decision credit. Decisions are per *set* here rather than per turn, so an
episode yields several times more of them.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import pathlib
import time

import numpy as np
import torch
from torch import nn

from rummi.agents.learned.features import FEATURE_FIELDS, feature_dim, feature_scale
from rummi.agents.learned.torch_net import MASKED
from rummi.agents.hybrid import (
    HybridAgent,
    hybrid_action_features,
    macro_first,
    primitives_only,
)
from rummi.agents.hybrid import n_actions as n_hybrid_actions
from rummi.agents.macro import (
    MacroAgent,
    action_features,
    by_value,
    first_legal,
    n_macros,
)
from rummi.env.fixed_opponent import FixedOpponentEnv
from rummi.evaluate.protocol import SUITE_BY_NAME, evaluate
from rummi.rules.config import CONFIG_BY_NAME, RewardMode, RummiConfig


class MacroNet(nn.Module):
    """Logits over the macro actions, and a value.

    `flat` gives every macro its own row, so nothing learned about one set carries
    to a similar one. `pointer` scores each macro against `macro.action_features`
    -- what the action *does* -- so the scoring function is shared and a per-action
    bias carries whatever is left over.
    """

    def __init__(
        self, cfg: RummiConfig, macros: int, hidden: int = 256,
        head: str = "flat", key_dim: int = 64, describe: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self.head = head
        self.trunk = nn.Sequential(
            nn.Linear(feature_dim(cfg), hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        if head == "pointer":
            desc = torch.as_tensor(
                action_features(cfg) if describe is None else describe
            )
            self.register_buffer("desc", desc)
            self.key = nn.Linear(desc.shape[1], key_dim, bias=False)
            self.query = nn.Linear(hidden, key_dim)
            self.action_bias = nn.Parameter(torch.zeros(macros))
            # Small, for the same reason the flat head uses gain 0.01: a fresh
            # policy should be near-uniform over the legal macros.
            nn.init.orthogonal_(self.query.weight, 0.01)
            nn.init.zeros_(self.query.bias)
        else:
            self.pi = nn.Linear(hidden, macros)
            nn.init.orthogonal_(self.pi.weight, 0.01)
            nn.init.zeros_(self.pi.bias)
        self.v = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        if self.head == "pointer":
            logits = self.query(h) @ self.key(self.desc).T + self.action_bias
        else:
            logits = self.pi(h)
        return logits, self.v(h).squeeze(-1)


def gather(
    net: MacroNet, cfg: RummiConfig, env, teacher, samples: int, beta: float, generator,
    hybrid: bool = False,
) -> dict:
    """States, and the macro the teacher would pick in each.

    `beta` is the chance the *teacher* drives; below 1 the student steers and the
    labels land on states the student actually reaches. The poisoned half-built
    table that made this worthless on the primitive action space cannot occur here,
    because every macro leaves the table whole.
    """
    obs, info = env.reset()
    xs: list[np.ndarray] = []
    legals: list[np.ndarray] = []
    ys: list[int] = []
    agreed = 0

    def choose(o, e: int, legal: np.ndarray) -> int:
        label = teacher(o, e, legal)
        x = np.concatenate(
            [np.asarray(o[f])[e].reshape(-1) for f in FEATURE_FIELDS]
        ).astype(np.float32) / feature_scale(cfg)
        xs.append(x)
        legals.append(legal.copy())
        ys.append(label)
        if float(torch.rand(1, generator=generator)) < beta:
            return label
        with torch.no_grad():
            logits, _ = net(torch.as_tensor(x)[None])
        logits = torch.where(
            torch.as_tensor(legal)[None], logits, torch.full_like(logits, MASKED)
        )
        return int(logits[0].argmax())

    agent = HybridAgent(cfg, choose=choose) if hybrid else MacroAgent(cfg, choose=choose)
    agent.reset(env.num_envs)
    while len(xs) < samples:
        obs, _, _, _, info = env.step(agent.act(obs, np.asarray(info["action_mask"])))

    # Agreement on whatever states these are: the number to watch when beta is 0.
    with torch.no_grad():
        x = torch.as_tensor(np.stack(xs))
        legal = torch.as_tensor(np.stack(legals))
        logits, _ = net(x)
        logits = torch.where(legal, logits, torch.full_like(logits, MASKED))
        agreed = int((logits.argmax(-1) == torch.as_tensor(np.asarray(ys))).sum())
    return {
        "x": x,
        "legal": legal,
        "y": torch.as_tensor(np.asarray(ys)),
        "agreement": agreed / len(ys),
    }


def fit(net: MacroNet, data: dict, epochs: int, lr: float, generator) -> None:
    """Cross-entropy on the masked logits, policy head only.

    The critic is deliberately left alone: fitting it through the shared trunk
    moves the policy, which `train_ppo.py` measured as catastrophic. `--value-warmup`
    is where it gets fitted, on its own.
    """
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    n = len(data["y"])
    for epoch in range(1, epochs + 1):
        order = torch.randperm(n, generator=generator)
        total = 0.0
        for start in range(0, n, 4096):
            idx = order[start : start + 4096]
            logits, _ = net(data["x"][idx])
            logits = torch.where(
                data["legal"][idx], logits, torch.full_like(logits, MASKED)
            )
            loss = nn.functional.cross_entropy(logits, data["y"][idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            opt.step()
            total += float(loss) * len(idx)
        with torch.no_grad():
            logits, _ = net(data["x"])
            logits = torch.where(data["legal"], logits, torch.full_like(logits, MASKED))
            acc = float((logits.argmax(-1) == data["y"]).float().mean())
        if epoch % max(epochs // 5, 1) == 0 or epoch == epochs:
            print(f"  clone epoch {epoch:>3}/{epochs}  nll {total / n:.4f}  "
                  f"agrees {acc:.1%}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="standard", choices=sorted(CONFIG_BY_NAME))
    p.add_argument(
        "--space", default="macro", choices=["macro", "hybrid"],
        help="hybrid adds the 2400 primitives alongside the macros, so any legal "
             "turn is expressible while a safe macro stays on offer",
    )
    p.add_argument("--opponent", default="greedy", choices=["greedy", "rearrange", "optimal"])
    p.add_argument("--envs", type=int, default=64)
    p.add_argument("--horizon", type=int, default=256)
    p.add_argument("--updates", type=int, default=200)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument(
        "--head", default="flat", choices=["flat", "pointer"],
        help="pointer scores a macro against what it does, so what is learned about "
             "one set transfers to similar ones; flat gives each its own row",
    )
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument(
        "--epochs", type=int, default=4,
        help="passes over each batch. 1 is the plain policy gradient, which uses "
             "every transition once and throws it away",
    )
    p.add_argument("--minibatches", type=int, default=4)
    p.add_argument("--clip", type=float, default=0.2, help="PPO ratio clip")
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument(
        "--clone", default=None, choices=["by_value", "first_legal"],
        help="imitate this heuristic before RL. by_value scores -143 on its own, "
             "and RL from scratch here stalls before the opening meld",
    )
    p.add_argument("--clone-states", type=int, default=100_000)
    p.add_argument("--clone-epochs", type=int, default=20)
    p.add_argument(
        "--clone-rounds", type=int, default=1,
        help="1 is plain behaviour cloning; more aggregates DAgger rounds with the "
             "student steering, which is cheap because the teacher is deterministic",
    )
    p.add_argument(
        "--kl-coef", type=float, default=0.0,
        help="anchor RL to the cloned policy. Unanchored, it walks straight back "
             "off what cloning bought -- measured, on the primitive action space",
    )
    p.add_argument(
        "--value-warmup", type=int, default=0,
        help="updates fitting the critic alone before the policy may move. Cloning "
             "trains the policy head only, so the critic starts at init",
    )
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
    hybrid = args.space == "hybrid"
    macros = n_hybrid_actions(cfg) if hybrid else n_macros(cfg)
    net = MacroNet(
        cfg, macros, args.hidden, head=args.head,
        describe=hybrid_action_features(cfg) if hybrid else None,
    )
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    env = FixedOpponentEnv(
        num_envs=args.envs, cfg=cfg, seed=args.seed, opponent=args.opponent
    )
    obs, info = env.reset()

    teachers = (
        {"by_value": macro_first(cfg), "first_legal": primitives_only(cfg)}
        if hybrid
        else {"by_value": by_value(cfg), "first_legal": first_legal}
    )
    if args.clone:
        per_round = max(args.clone_states // args.clone_rounds, 1)
        pool: dict | None = None
        for r in range(args.clone_rounds):
            # beta 1 -> 0: pure teacher first, pure student last. Aggregated, not
            # replaced, or each round forgets what the last one fixed.
            beta = 1.0 if args.clone_rounds == 1 else max(0.0, 1.0 - r / (args.clone_rounds - 1))
            fresh = gather(
                net, cfg, env, teachers[args.clone], per_round, beta, generator,
                hybrid=hybrid,
            )
            print(
                f"  round {r}  beta {beta:.2f}  {len(fresh['y']):,} states  "
                f"agreement {fresh['agreement']:>6.1%}",
                flush=True,
            )
            pool = fresh if pool is None else {
                "x": torch.cat([pool["x"], fresh["x"]]),
                "legal": torch.cat([pool["legal"], fresh["legal"]]),
                "y": torch.cat([pool["y"], fresh["y"]]),
                "agreement": fresh["agreement"],
            }
            assert pool is not None
            fit(net, pool, args.clone_epochs, args.lr, generator)
        obs, info = env.reset()
        agent_reset_needed = True
    else:
        agent_reset_needed = False

    reference = None
    if args.kl_coef:
        reference = copy.deepcopy(net).eval()
        for parameter in reference.parameters():
            parameter.requires_grad_(False)
        print(f"anchoring to the cloned policy, kl_coef={args.kl_coef}", flush=True)

    # The critic is at init after cloning, so its advantages are noise until fitted.
    value_opt = torch.optim.Adam(net.v.parameters(), lr=args.lr)

    # In the hybrid space END_TURN and DRAW are primitives at their own ids, not
    # the last two actions.
    end_action = cfg.end_turn_action if hybrid else macros - 2
    draw_action = cfg.draw_action if hybrid else macros - 1

    open_choice: list[tuple[np.ndarray, np.ndarray, int, float] | None] = [None] * args.envs
    accrued = np.zeros(args.envs, dtype=np.float32)
    steps: list[
        tuple[np.ndarray, np.ndarray, int, float, float, np.ndarray, float]
    ] = []
    tally: dict[str, int] = {}

    def features(o, e: int) -> np.ndarray:
        row = np.concatenate(
            [np.asarray(o[f])[e].reshape(-1) for f in FEATURE_FIELDS]
        ).astype(np.float32)
        return row / scale

    def choose(o, e: int, legal: np.ndarray) -> int:
        x = features(o, e)
        # Acting needs no graph, and building one per decision is pure waste.
        with torch.no_grad():
            logits, _ = net(torch.as_tensor(x)[None])
        masked = torch.where(
            torch.as_tensor(legal)[None], logits, torch.full_like(logits, MASKED)
        )
        macro = int(
            torch.multinomial(torch.softmax(masked[0], -1), 1, generator=generator)
        )
        # The log-prob under the policy that *acted*: reusing a batch means scoring
        # against this, and the stored mask is what keeps the ratio meaningful.
        behaviour = float(torch.log_softmax(masked[0], -1)[macro])
        if open_choice[e] is not None:
            prev_x, prev_legal, prev_a, prev_lp = open_choice[e]
            steps.append((prev_x, prev_legal, prev_a, prev_lp, float(accrued[e]), x, 0.0))
        accrued[e] = 0.0
        open_choice[e] = (x, legal.copy(), macro, behaviour)
        tally["n"] = tally.get("n", 0) + 1
        tally["end"] = tally.get("end", 0) + int(macro == end_action)
        tally["draw"] = tally.get("draw", 0) + int(macro == draw_action)
        return macro

    agent = HybridAgent(cfg, choose=choose) if hybrid else MacroAgent(cfg, choose=choose)
    agent.reset(args.envs)
    if agent_reset_needed:
        open_choice[:] = [None] * args.envs
    print(
        f"config={args.config} space={args.space} opponent={args.opponent} "
        f"actions={macros} head={args.head} "
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
                    prev_x, prev_legal, prev_a, prev_lp = open_choice[e]
                    steps.append(
                        (prev_x, prev_legal, prev_a, prev_lp, float(accrued[e]), prev_x, 1.0)
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
        old_logp = torch.as_tensor(np.asarray([s[3] for s in steps], dtype=np.float32))
        r = torch.as_tensor(np.asarray([s[4] for s in steps], dtype=np.float32))
        nxt = torch.as_tensor(np.stack([s[5] for s in steps]))
        terminal = torch.as_tensor(np.asarray([s[6] for s in steps], dtype=np.float32))

        warming = update <= args.value_warmup
        # Targets and advantages come from the policy that acted, once, and are held
        # fixed across the passes: recomputing them per pass chases a moving critic.
        with torch.no_grad():
            _, value_old = net(x)
            _, next_value = net(nxt)
            target = r + args.gamma * (1.0 - terminal) * next_value
            advantage = target - value_old
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        total = len(steps)
        size = max(total // args.minibatches, 1)
        entropy = torch.zeros(())
        for _ in range(1 if warming else args.epochs):
            order = torch.randperm(total, generator=generator)
            for start in range(0, total, size):
                idx = order[start : start + size]
                logits, value = net(x[idx])
                logits = torch.where(legal[idx], logits, torch.full_like(logits, MASKED))
                logp_all = torch.log_softmax(logits, -1)
                logp = logp_all.gather(1, a[idx][:, None])[:, 0]
                entropy = -(logp_all.exp() * logp_all).sum(-1).mean()
                value_loss = (value - target[idx]).pow(2).mean()

                if warming:
                    # The value head *alone*: fitting the critic through the shared
                    # trunk moves the policy, the opposite of a warmup.
                    loss = value_loss
                else:
                    ratio = (logp - old_logp[idx]).exp()
                    adv = advantage[idx]
                    clipped = torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * adv
                    loss = -torch.min(ratio * adv, clipped).mean()
                    loss = loss + 0.5 * value_loss - args.entropy_coef * entropy
                    if reference is not None:
                        with torch.no_grad():
                            ref_logits, _ = reference(x[idx])
                            ref_logits = torch.where(
                                legal[idx], ref_logits,
                                torch.full_like(ref_logits, MASKED),
                            )
                        kl = (
                            logp_all.exp()
                            * (logp_all - torch.log_softmax(ref_logits, -1))
                        ).sum(-1)
                        loss = loss + args.kl_coef * kl.mean()

                active = value_opt if warming else opt
                active.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                active.step()

        n = max(tally.get("n", 1), 1)
        print(
            f"update {update:>4}{' warm' if warming else ''}  episodes {finished:>4}  "
            f"decisions {len(steps):>6,}  "
            f"end {tally.get('end', 0) / n:>5.1%}  draw {tally.get('draw', 0) / n:>5.1%}  "
            f"terminal {r[terminal > 0].mean().item():>+7.3f}  H {entropy.item():>5.3f}  "
            f"{len(steps) / (time.perf_counter() - started):>5.0f} dec/s",
            flush=True,
        )
        started = time.perf_counter()

    env.close()

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "cfg": args.config, "hidden": args.hidden, "head": args.head,
                "state": net.state_dict(),
            },
            args.out,
        )
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
        baselines = (
            (("learned", greedy_choose), ("macro_first", macro_first(cfg)))
            if hybrid
            else (
                ("learned", greedy_choose),
                ("by_value", by_value(cfg)),
                ("first_legal", first_legal),
            )
        )
        for label, ch in baselines:
            scored = HybridAgent(cfg, choose=ch) if hybrid else MacroAgent(cfg, choose=ch)
            result = evaluate(label, suite, build_agent=lambda c, s=scored: s,
                              games=args.eval_games)
            print(
                f"  {label:12s} win {result.win_rate:>6.1%}  score {result.mean_score:>+8.2f}  "
                f"illegal {result.illegal_attempts}  n={result.games}"
            )


if __name__ == "__main__":
    main()
