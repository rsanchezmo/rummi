"""Reference PPO against a bundled opponent.

    python tools/train_ppo.py --config tiny_groups --updates 40
    python tools/train_ppo.py --config standard --opponent greedy --shaping

Deliberately one file and no framework: it exists so the numbers in the README
have something behind them, and so a submission has a working starting point
rather than a blank page. `rummi/agents/learned/` holds the reusable parts.

Three things about this game that shape the loop.

*Masking is not optional.* About 1.5% of the action space is legal at any step, so
the mask is stored in the rollout and reapplied at update time -- scoring an
action under a policy that has forgotten which actions were available gives a
meaningless ratio.

*Episodes are long and reward is terminal.* ~120 turns of ~2.7 micro-actions, and
`WIN_LOSS` pays only at the end. `--shaping` turns on the per-step terms from
SPEC.md section 7 to densify that; it makes the run incomparable to a published
score by design, which is why it is off by default.

*The opponent is not the environment.* `FixedOpponentEnv` runs the other seats
inside one `step`, so every observation is a position the learner can act in and
the reward already covers the replies.
"""

from __future__ import annotations

import argparse
import dataclasses
import pathlib
import time

import numpy as np

from rummi.rules.config import CONFIG_BY_NAME, RummiConfig


@dataclasses.dataclass(frozen=True, slots=True)
class Hyper:
    envs: int = 64
    horizon: int = 128
    epochs: int = 4
    minibatches: int = 4
    lr: float = 3e-4
    gamma: float = 0.999
    """High: a 300-step episode paying out only at the end needs the credit to
    reach the start of it."""
    gae_lambda: float = 0.95
    clip: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5


def shaped(cfg: RummiConfig) -> RummiConfig:
    """Dense-ish reward for training only. See SPEC.md section 7."""
    return dataclasses.replace(cfg, tiles_placed_bonus=0.01, rack_value_delta=0.002)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", choices=sorted(CONFIG_BY_NAME), default="tiny_groups")
    p.add_argument("--opponent", default="greedy")
    p.add_argument("--updates", type=int, default=40)
    p.add_argument("--envs", type=int, default=64)
    p.add_argument("--horizon", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shaping", action="store_true", help="turn on the per-step reward terms")
    p.add_argument("--out", type=pathlib.Path, default=None, help="save weights here")
    p.add_argument("--eval-games", type=int, default=0, help="score through the frozen protocol")
    args = p.parse_args()

    import torch

    from rummi.agents.learned.architecture import Architecture
    from rummi.agents.learned.torch_net import TorchPolicy, entropy, log_prob, sample
    from rummi.env.fixed_opponent import FixedOpponentEnv

    hyper = Hyper(envs=args.envs, horizon=args.horizon, lr=args.lr)
    base = CONFIG_BY_NAME[args.config]
    cfg = shaped(base) if args.shaping else base

    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    arch = Architecture()
    net = TorchPolicy(cfg, arch, seed=args.seed)
    opt = torch.optim.Adam(net.parameters(), lr=hyper.lr, eps=1e-5)

    env = FixedOpponentEnv(
        num_envs=hyper.envs, cfg=cfg, seed=args.seed, opponent=args.opponent
    )
    obs, info = env.reset()
    n_actions = cfg.n_actions

    print(
        f"config={args.config} opponent={args.opponent} shaping={args.shaping} "
        f"envs={hyper.envs} horizon={hyper.horizon} params="
        f"{sum(v.numel() for v in net.parameters()):,}"
    )
    started = time.perf_counter()
    # Wins come from `info["winner"]`, not from the sign of the reward. Under
    # `--shaping` a losing episode can still end on a positive reward, so
    # `reward > 0` is not a win rate -- it silently becomes one only when the
    # shaping terms are zero.
    wins_seen: list[bool] = []

    for update in range(1, args.updates + 1):
        # --- rollout ---------------------------------------------------------
        b_obs = {k: torch.zeros((hyper.horizon, hyper.envs, *v.shape[1:])) for k, v in obs.items()}
        b_mask = torch.zeros((hyper.horizon, hyper.envs, n_actions), dtype=torch.bool)
        b_act = torch.zeros((hyper.horizon, hyper.envs), dtype=torch.long)
        b_logp = torch.zeros((hyper.horizon, hyper.envs))
        b_val = torch.zeros((hyper.horizon, hyper.envs))
        b_rew = torch.zeros((hyper.horizon, hyper.envs))
        b_done = torch.zeros((hyper.horizon, hyper.envs))

        for t in range(hyper.horizon):
            obs_t = {k: torch.as_tensor(np.asarray(v)) for k, v in obs.items()}
            mask_t = torch.as_tensor(np.asarray(info["action_mask"]))
            with torch.no_grad():
                logits, value = net(obs_t, mask_t)
                action = sample(logits, generator)
                logp = log_prob(logits, action)

            for k, v in obs_t.items():
                b_obs[k][t] = v.to(b_obs[k].dtype)
            b_mask[t], b_act[t], b_logp[t], b_val[t] = mask_t, action, logp, value

            obs, reward, term, trunc, info = env.step(action.numpy())
            b_rew[t] = torch.as_tensor(np.asarray(reward, dtype=np.float32))
            b_done[t] = torch.as_tensor((term | trunc).astype(np.float32))
            finished = term | trunc
            if finished.any():
                won = info["winner"][finished] == env.learner_seat
                wins_seen.extend(won.tolist())

        with torch.no_grad():
            _, last_value = net(
                {k: torch.as_tensor(np.asarray(v)) for k, v in obs.items()},
                torch.as_tensor(np.asarray(info["action_mask"])),
            )

        # --- GAE -------------------------------------------------------------
        adv = torch.zeros_like(b_rew)
        running = torch.zeros(hyper.envs)
        for t in reversed(range(hyper.horizon)):
            nonterminal = 1.0 - b_done[t]
            next_value = last_value if t == hyper.horizon - 1 else b_val[t + 1]
            delta = b_rew[t] + hyper.gamma * next_value * nonterminal - b_val[t]
            running = delta + hyper.gamma * hyper.gae_lambda * nonterminal * running
            adv[t] = running
        ret = adv + b_val

        # --- update ----------------------------------------------------------
        flat_obs = {k: v.reshape(-1, *v.shape[2:]) for k, v in b_obs.items()}
        flat = (
            b_mask.reshape(-1, n_actions),
            b_act.reshape(-1),
            b_logp.reshape(-1),
            adv.reshape(-1),
            ret.reshape(-1),
        )
        total = hyper.horizon * hyper.envs
        size = total // hyper.minibatches
        last_stats = (0.0, 0.0, 0.0)

        for _ in range(hyper.epochs):
            order = torch.randperm(total, generator=generator)
            for start in range(0, total, size):
                idx = order[start : start + size]
                mask_b, act_b, logp_b, adv_b, ret_b = (x[idx] for x in flat)
                logits, value = net({k: v[idx] for k, v in flat_obs.items()}, mask_b)

                new_logp = log_prob(logits, act_b)
                ratio = (new_logp - logp_b).exp()
                norm_adv = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)
                policy_loss = -torch.min(
                    ratio * norm_adv,
                    ratio.clamp(1 - hyper.clip, 1 + hyper.clip) * norm_adv,
                ).mean()
                value_loss = 0.5 * (value - ret_b).pow(2).mean()
                ent = entropy(logits).mean()
                loss = policy_loss + hyper.value_coef * value_loss - hyper.entropy_coef * ent

                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), hyper.max_grad_norm)
                opt.step()
                last_stats = (
                    policy_loss.detach().item(),
                    value_loss.detach().item(),
                    ent.detach().item(),
                )

        recent = wins_seen[-200:]
        win_rate = float(np.mean(recent)) if recent else float("nan")
        steps = update * hyper.horizon * hyper.envs
        print(
            f"update {update:>4}  steps {steps:>9,}  "
            f"finished {len(wins_seen):>5}  win {win_rate:>6.1%}  "
            f"pi {last_stats[0]:>+8.4f}  v {last_stats[1]:>8.4f}  H {last_stats[2]:>6.3f}  "
            f"{steps / (time.perf_counter() - started):>7.0f} steps/s",
            flush=True,
        )

    env.close()

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"cfg": args.config, "hidden": arch.hidden, "state": net.state_dict()}, args.out)
        print(f"wrote {args.out}")

    if args.eval_games:
        from rummi.agents.learned.agent import torch_agent
        from rummi.evaluate.protocol import SUITE_BY_NAME, evaluate

        # The *unshaped* suite: shaping is a training aid, not part of a score.
        suite_name = "tiny" if args.config == "tiny_groups" else "standard-greedy"
        agent = torch_agent(net, name="learned")
        result = evaluate("learned", SUITE_BY_NAME[suite_name], build_agent=lambda c: agent,
                          games=args.eval_games)
        print(result.report())


if __name__ == "__main__":
    main()
