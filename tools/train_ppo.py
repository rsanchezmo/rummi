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

*On `standard`, `--bc-updates` is not optional.* Forming the first valid set needs
three compatible tiles PLACEd and then ASSIGNed to the same slot, out of 1855
ASSIGN variants. Measured: uniform-random play reaches `END_TURN` **zero** times in
192k steps, at any `initial_meld` including 0 -- the barrier is set formation, not
the 30-point threshold. PPO explores no better than random until it melds once, so
without a teacher it never sees a reward that distinguishes any action from any
other, and lands exactly on the score of passing every turn. Cloning `greedy`
first is what gets it over the wall.
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
    kl_coef: float = 0.0
    """Penalty on KL from the policy PPO started with.

    Zero reproduces plain PPO. It matters only after cloning: with ~1 terminal
    reward per 3,600 steps the advantages are mostly value-error noise, and
    unanchored PPO walked a cloned policy's melding rate from 3.1% back to 0.0%.
    The anchor lets it move where it has signal and holds it where it does not."""
    max_grad_norm: float = 0.5


def shaped(cfg: RummiConfig) -> RummiConfig:
    """Dense-ish reward for training only. See SPEC.md section 7."""
    return dataclasses.replace(cfg, tiles_placed_bonus=0.01, rack_value_delta=0.002)


def clone_expert(
    net,
    env,
    cfg: RummiConfig,
    teacher_name: str,
    samples: int,
    epochs: int,
    lr: float,
    seed: int,
    gamma: float = 0.999,
) -> None:
    """Behaviour cloning from a bundled agent, on that agent's own states.

    Collection and fitting are **separate phases** on purpose. `optimal` costs a
    CP-SAT solve per turn and cannot batch, so re-querying it for every minibatch
    -- which the first version of this did -- spends all its time in the solver and
    almost none in SGD. Gathering once and running many epochs over the dataset
    makes an expert teacher affordable: `greedy` is cheap either way, `optimal` is
    only viable this way.

    **Both heads are fitted, not just the policy.** The teacher's own returns are
    right there in the rollout, so the critic can start calibrated on the
    distribution PPO will begin from. Skipping this left the value head at
    initialisation: value loss 7.7 against 0.0003 for a trained one, advantages
    that were pure noise, and a policy destroyed within a handful of updates. There
    is nothing to copy from the policy head -- it is `(h, n_actions)` against
    `(h, 1)` -- but its *data* serves both.

    Features are cached rather than observations: ~570 floats against ~1025, which
    is what lets the dataset be large enough to be worth many epochs.

    The teacher drives, so the states are ones a competent player reaches. That is
    the entire point -- the learner cannot reach them itself. On `standard`,
    uniform-random play never assembles a first meld at all.
    """
    import torch

    from rummi.agents import build
    from rummi.agents.base import act_on_state
    from rummi.agents.learned.torch_net import log_prob

    teacher = build(teacher_name, cfg)
    teacher.reset(env.num_envs)
    obs, info = env.reset()

    feats: list[torch.Tensor] = []
    masks: list[np.ndarray] = []
    acts: list[np.ndarray] = []
    rewards: list[np.ndarray] = []
    dones: list[np.ndarray] = []
    gathered = 0
    started = time.perf_counter()

    while gathered < samples:
        mask = np.asarray(info["action_mask"])
        wanted = np.asarray(act_on_state(teacher, env.state, mask))
        with torch.no_grad():
            feats.append(
                net.features({k: torch.as_tensor(np.asarray(v)) for k, v in obs.items()})
            )
        masks.append(mask.copy())
        acts.append(wanted.copy())
        gathered += len(wanted)
        obs, reward, term, trunc, info = env.step(wanted)
        rewards.append(np.asarray(reward, dtype=np.float32).copy())
        dones.append((term | trunc).astype(np.float32))
        if gathered % (env.num_envs * 200) == 0:
            rate = gathered / (time.perf_counter() - started)
            print(f"  gathering {gathered:>7,}/{samples:,}  {rate:>6.0f} states/s", flush=True)

    x = torch.cat(feats)
    m = torch.as_tensor(np.concatenate(masks))
    y = torch.as_tensor(np.concatenate(acts)).long()

    # Discounted returns along each env's own timeline, bootstrapped at zero at the
    # end of collection -- the tail is a small fraction of a long rollout.
    reward_grid = np.stack(rewards)
    done_grid = np.stack(dones)
    returns = np.zeros_like(reward_grid)
    running = np.zeros(reward_grid.shape[1], dtype=np.float32)
    for t in reversed(range(len(reward_grid))):
        running = reward_grid[t] + gamma * running * (1.0 - done_grid[t])
        returns[t] = running
    g = torch.as_tensor(returns.reshape(-1))
    print(
        f"  {len(y):,} states from {teacher_name} in {time.perf_counter() - started:.0f}s"
        f"  (returns: mean {float(g.mean()):+.3f}, std {float(g.std()):.3f})"
    )

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    batch = 4096
    generator = torch.Generator().manual_seed(seed)

    for epoch in range(1, epochs + 1):
        order = torch.randperm(len(y), generator=generator)
        total_nll = total_agree = total_vloss = 0.0
        for start in range(0, len(y), batch):
            idx = order[start : start + batch]
            logits, value = net.head(x[idx], m[idx])
            policy_nll = -log_prob(logits, y[idx]).mean()
            value_mse = 0.5 * (value - g[idx]).pow(2).mean()
            loss = policy_nll + 0.5 * value_mse
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            opt.step()
            with torch.no_grad():
                total_nll += policy_nll.detach().item() * len(idx)
                total_vloss += value_mse.detach().item() * len(idx)
                total_agree += (logits.argmax(-1) == y[idx]).sum().item()
        if epoch % max(1, epochs // 10) == 0 or epoch == 1:
            print(
                f"  clone epoch {epoch:>3}/{epochs}  nll {total_nll / len(y):>7.4f}  "
                f"v {total_vloss / len(y):>7.4f}  "
                f"agrees with {teacher_name} {total_agree / len(y):>6.1%}",
                flush=True,
            )

    env.reset()


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
    p.add_argument(
        "--clone", default=None, choices=["greedy", "rearrange", "optimal"],
        help="clone this agent before PPO. Required on standard -- see the docstring.",
    )
    p.add_argument("--clone-states", type=int, default=50_000)
    p.add_argument("--clone-epochs", type=int, default=20)
    p.add_argument(
        "--hidden", default="256,256",
        help="trunk widths, comma separated. The default is small: at 570 inputs "
             "and 2400 actions, 74%% of a (256,256) net is the output projection.",
    )
    p.add_argument("--activation", default="relu", choices=["relu", "tanh"])
    p.add_argument(
        "--init-from", type=pathlib.Path, default=None,
        help="start from a saved checkpoint (skips cloning; takes its architecture)",
    )
    p.add_argument("--kl-coef", type=float, default=0.0, help="anchor PPO to its starting policy")
    p.add_argument(
        "--value-warmup", type=int, default=0,
        help="updates fitting only the critic before the policy is allowed to move. "
             "Required after cloning: BC trains the policy head alone, so the value "
             "head is still at init and its advantages are noise.",
    )
    p.add_argument(
        "--entropy-coef", type=float, default=0.01,
        help="0.0 after cloning: this game punishes imprecision, since one wrong "
             "action ends the turn in DRAW and reverts it",
    )
    p.add_argument("--out", type=pathlib.Path, default=None, help="save weights here")
    p.add_argument("--eval-games", type=int, default=0, help="score through the frozen protocol")
    args = p.parse_args()

    import torch

    from rummi.agents.learned.architecture import Architecture
    from rummi.agents.learned.torch_net import TorchPolicy, entropy, log_prob, sample
    from rummi.env.fixed_opponent import FixedOpponentEnv

    hyper = Hyper(
        envs=args.envs, horizon=args.horizon, lr=args.lr,
        entropy_coef=args.entropy_coef, kl_coef=args.kl_coef,
    )
    base = CONFIG_BY_NAME[args.config]
    cfg = shaped(base) if args.shaping else base

    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    if args.init_from:
        checkpoint = torch.load(args.init_from, weights_only=False)
        arch = Architecture(
            hidden=tuple(checkpoint["hidden"]),
            activation=checkpoint.get("activation", args.activation),
        )
        net = TorchPolicy(cfg, arch, seed=args.seed)
        net.load_state_dict(checkpoint["state"])
        print(f"resumed from {args.init_from} (hidden={arch.hidden}, {arch.activation})")
    else:
        arch = Architecture(
            hidden=tuple(int(w) for w in args.hidden.split(",")), activation=args.activation
        )
        net = TorchPolicy(cfg, arch, seed=args.seed)
    opt = torch.optim.Adam(net.parameters(), lr=hyper.lr, eps=1e-5)
    value_opt = torch.optim.Adam(net.v.parameters(), lr=hyper.lr, eps=1e-5)

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
    if args.clone:
        print(
            f"cloning {args.clone}: {args.clone_states:,} states, "
            f"{args.clone_epochs} epochs"
        )
        clone_expert(
            net, env, cfg, args.clone, args.clone_states, args.clone_epochs,
            hyper.lr, args.seed, hyper.gamma,
        )
        obs, info = env.reset()

    # Frozen snapshot of whatever PPO starts from, for the KL anchor.
    reference = None
    if hyper.kl_coef:
        import copy

        reference = copy.deepcopy(net).eval()
        for parameter in reference.parameters():
            parameter.requires_grad_(False)
        print(f"anchoring PPO to its starting policy, kl_coef={hyper.kl_coef}")

    started = time.perf_counter()
    # Wins come from `info["winner"]`, not from the sign of the reward. Under
    # `--shaping` a losing episode can still end on a positive reward, so
    # `reward > 0` is not a win rate -- it silently becomes one only when the
    # shaping terms are zero.
    wins_seen: list[bool] = []
    # The diagnostics that matter before a win rate does. Under
    # `strict_initial_meld` a seat that has not opened cannot END_TURN at all --
    # DRAW is its only way to end a turn -- so "did it ever meld" is the real
    # question on the standard config, and a flat win rate hides it completely.
    melded_frac = 0.0
    end_turn_frac = 0.0

    for update in range(1, args.updates + 1):
        # --- rollout ---------------------------------------------------------
        b_obs = {k: torch.zeros((hyper.horizon, hyper.envs, *v.shape[1:])) for k, v in obs.items()}
        b_mask = torch.zeros((hyper.horizon, hyper.envs, n_actions), dtype=torch.bool)
        b_act = torch.zeros((hyper.horizon, hyper.envs), dtype=torch.long)
        b_logp = torch.zeros((hyper.horizon, hyper.envs))
        b_val = torch.zeros((hyper.horizon, hyper.envs))
        b_rew = torch.zeros((hyper.horizon, hyper.envs))
        b_done = torch.zeros((hyper.horizon, hyper.envs))
        melded_frac = end_turn_frac = 0.0

        for t in range(hyper.horizon):
            obs_t = {k: torch.as_tensor(np.asarray(v)) for k, v in obs.items()}
            mask_t = torch.as_tensor(np.asarray(info["action_mask"]))
            with torch.no_grad():
                logits, value = net(obs_t, mask_t)
                action = sample(logits, generator)
                logp = log_prob(logits, action)

            melded_frac += float(np.asarray(obs["melded"])[:, 0].mean()) / hyper.horizon
            end_turn_frac += float(
                (action.numpy() == cfg.end_turn_action).mean()
            ) / hyper.horizon

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

        # Cloning fits the policy head only, so straight after it the critic is
        # still at initialisation -- value loss came in at 7.7 against 0.0003 for a
        # trained one -- and acting on those advantages destroys what cloning
        # built. Fit the critic first.
        #
        # `value_opt` touches the value head *only*, deliberately. The head shares
        # the trunk, so optimising the critic through the trunk moves the policy
        # too: measured, that alone took `end_turn` from 1.97% to 0.02% in a single
        # update. A warmup that perturbs the policy is not a warmup.
        warming = update <= args.value_warmup

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
                if warming:
                    loss = value_loss
                else:
                    loss = (
                        policy_loss + hyper.value_coef * value_loss - hyper.entropy_coef * ent
                    )
                if reference is not None and not warming:
                    with torch.no_grad():
                        ref_logits, _ = reference(
                            {k: v[idx] for k, v in flat_obs.items()}, mask_b
                        )
                    # Masked entries sit at MASKED in both, so they contribute ~0.
                    logp_new = torch.log_softmax(logits, dim=-1)
                    kl = (logp_new.exp() * (logp_new - torch.log_softmax(ref_logits, -1))).sum(-1)
                    loss = loss + hyper.kl_coef * kl.mean()

                active_opt = value_opt if warming else opt
                active_opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), hyper.max_grad_norm)
                active_opt.step()
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
            f"{'warm ' if warming else ''}melded {melded_frac:>6.1%}  "
            f"end_turn {end_turn_frac:>6.2%}  "
            f"pi {last_stats[0]:>+8.4f}  v {last_stats[1]:>8.4f}  H {last_stats[2]:>6.3f}  "
            f"{steps / (time.perf_counter() - started):>7.0f} steps/s",
            flush=True,
        )
        melded_frac = end_turn_frac = 0.0

    env.close()

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "cfg": args.config,
                "hidden": arch.hidden,
                "activation": arch.activation,
                "state": net.state_dict(),
            },
            args.out,
        )
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
