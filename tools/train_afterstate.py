"""Value-based learning over the macro space: score the afterstate, play the argmax.

    python tools/train_afterstate.py --updates 120 --eval-games 120

Every learned agent in this repo so far has been a policy over an action head --
flat, factored, bilinear, pointer -- trained by policy gradient or by imitation.
This one has no action head at all. A macro's afterstate is deterministic and
computable without stepping the env (`rummi/agents/learned/afterstate.py`), so the
network scores *positions* and the policy is the argmax over the positions the
legal macros lead to. That is TD-Gammon's shape, and it is available here for the
same reason it was there: the move is deterministic and only what happens
afterwards is not.

Three consequences worth stating, because they are what makes this a different
experiment and not another recipe.

**Nothing is learned per action id.** The flat head had to learn each of 713
outputs separately -- the measured reason nothing transferred between similar sets
(`docs/EXPERIMENTS.md`). Here a template the policy has never played is scored by
the position it produces, which the network has seen thousands of near-copies of.

**The target is the outcome, not an advantage.** There is no critic feeding its
own advantage estimate, which removes the noise source left standing after four
refuted explanations of the policy trainer's collapse. **Explained variance is
therefore the diagnostic**, and it is reported every update against the outcome
the rows actually led to: a value net that explains nothing cannot rank two
positions, and no behaviour aggregate would have said so.

`--td-lambda` picks the estimator, and the difference is not cosmetic. At 1 every
decision in an episode is paired with that episode's terminal reward -- unbiased,
and with an effective sample size of *episodes*, because every row of one shares
its target. At 0 the target is `r + V(next chosen afterstate)`, which is dense,
low-variance, and counts per decision; the terminal flag is the only thing
anchoring the chain, which is exactly TD-Gammon's arrangement.

That distinction is measured, not argued. On 112,416 rows over **1,199 episodes**
of `by_value` trajectories, split by episode, Monte Carlo regression reaches
**train EV +0.94 and holdout EV -0.35** -- it memorises the episode and predicts
nothing -- and weight decay buys generalisation only by flattening the net to the
mean (holdout EV +0.009 at train EV +0.019). The reason is visible in the data
rather than in the fit: every row of an episode carries the same target, so only
*between-episode* variance is explainable, and the crudest candidate predictor of
it -- the learner's own rack size -- correlates with the outcome at **+0.035**.

**One averaged gradient step per batch.** The measured disaster (-394 against +27)
was splitting one batch into chunks and stepping per chunk with a noisy
bootstrapped advantage. Every step here is a full averaged gradient over `--batch`
rows, and `--fit-steps` says how many such batches one update spends -- a value net
given one step per update would still be at its init when the run ends.

`REPARTITION` is taken unconditionally wherever it is legal, which is `by_value`'s
own rule and needs no value estimate: the macro is offered only in states where
nothing else plays. It is off by default so the headline comparison is against
`by_value`'s +29.60 on the same capability set; `--repartition` adds the CP-SAT
gap-closer and moves the reference to `by_value+repartition`'s +48.01.

**What it measured**, 200 updates at `--envs 64 --horizon 256`, `standard-greedy`
at n=240, `by_value` reproducing its published +29.60 / 85.4% inside every run as
the harness check:

| arm | scored checkpoints | note |
|---|---|---|
| `--td-lambda 1` (MC), seed 0 | +4.69 at u200; -40.5 to +3.1 earlier | held-out EV negative at every update |
| `--td-lambda 0`, seed 0 | +26.16 (u120), **+29.54** (u200) | end-turn rate holds at the teacher's 25% |
| `--td-lambda 0`, seed 1 | **+31.70** (u080), +29.97 (u120), +20.22 (u200) | |

So the estimator is the whole result: the same rollouts, the same net and the same
argmax score +4.69 under Monte Carlo and land on `by_value`'s level under a
one-step bootstrap. What the bootstrap does *not* fix is stability -- the
checkpoint sequence swings between +6 and +32 -- which is the same shape as the
macro policy trainer's, and for a visible reason: explained variance sits at 0.00
even when the agent plays at the heuristic's level, so the ranking that produces
those scores lives inside the noise floor of the outcome it is regressing.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import time

import numpy as np
import torch
from torch import nn

from rummi.agents.learned.afterstate import afterstate_batch, afterstate_dim
from rummi.agents.learned.afterstate_net import ValueNet, argmax_chooser
from rummi.agents.macro import (
    MacroAgent,
    by_value,
    extend_offset,
    first_legal,
    steal_offset,
)
from rummi.env.fixed_opponent import FixedOpponentEnv
from rummi.evaluate.protocol import SUITE_BY_NAME, evaluate
from rummi.rules.config import CONFIG_BY_NAME, RewardMode, RummiConfig

BLOCKS = ("new_set", "extend", "steal", "repart", "end", "draw")


class Replay:
    """Afterstate rows and the outcome each one led to, as a fixed-size ring.

    Preallocated because the alternative is a list of ~570-float rows growing for
    the length of a run; the ring also keeps the batch near the current policy,
    which a Monte Carlo target is only valid for.

    Under ``--td-lambda 0`` it holds the successor row as well, and the target is
    rebuilt from it at fit time rather than stored: a bootstrapped target computed
    when the transition was *collected* is as stale as the buffer is deep.
    """

    def __init__(self, capacity: int, dim: int, bootstrap: bool) -> None:
        self.x = np.zeros((capacity, dim), dtype=np.float32)
        self.y = np.zeros(capacity, dtype=np.float32)
        self.reward = np.zeros(capacity, dtype=np.float32)
        self.terminal = np.zeros(capacity, dtype=np.float32)
        self.nxt = np.zeros((capacity, dim), dtype=np.float32) if bootstrap else None
        self.write = 0
        self.filled = 0

    def add(
        self,
        row: np.ndarray,
        outcome: float,
        reward: float,
        successor: np.ndarray | None,
        terminal: bool,
    ) -> None:
        at = self.write
        self.x[at] = row
        self.y[at] = outcome
        self.reward[at] = reward
        self.terminal[at] = float(terminal)
        if self.nxt is not None and successor is not None:
            self.nxt[at] = successor
        elif self.nxt is not None:
            self.nxt[at] = 0.0
        self.write = (at + 1) % len(self.y)
        self.filled = min(self.filled + 1, len(self.y))


def explained_variance(y: np.ndarray, predicted: np.ndarray) -> float:
    """1 - Var(residual) / Var(y). Zero means the mean would have done as well."""
    spread = float(y.var())
    if spread == 0.0:
        return float("nan")
    return float(1.0 - (y - predicted).var() / spread)


def block_of(cfg: RummiConfig, agent: MacroAgent) -> np.ndarray:
    """Which block each macro id belongs to, for the per-update behaviour line."""
    out = np.zeros(agent.n_macros, dtype=np.int64)
    out[extend_offset(cfg) : steal_offset(cfg)] = 1
    out[steal_offset(cfg) :] = 2
    if agent.repartition_macro is not None:
        out[agent.repartition_macro] = 3
    out[agent.end_macro] = 4
    out[agent.draw_macro] = 5
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="standard", choices=sorted(CONFIG_BY_NAME))
    p.add_argument(
        "--opponent", default="greedy", choices=("greedy",),
        help="the value target is the outcome against *this* opponent, and the "
             "observation says nothing about who is sitting across the table -- so "
             "a mixed pool would have the net regress two different games onto one "
             "scalar. One opponent until that is designed for",
    )
    p.add_argument("--envs", type=int, default=64)
    p.add_argument("--horizon", type=int, default=256, help="env steps per update")
    p.add_argument("--updates", type=int, default=120)
    p.add_argument("--hidden", type=int, nargs="+", default=[256, 256])
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--fit-steps", type=int, default=20,
        help="averaged gradient steps per update, each over its own --batch sample",
    )
    p.add_argument("--batch", type=int, default=4096)
    p.add_argument("--buffer", type=int, default=40_000, help="rows kept, oldest dropped")
    p.add_argument(
        "--explore-eps", type=float, default=0.1,
        help="chance of a uniform legal macro instead of the argmax. Behavioural "
             "exploration, which is the lever that works here: an entropy bonus has "
             "no policy to bonus",
    )
    p.add_argument(
        "--explore-final", type=float, default=0.02,
        help="epsilon at the last update; annealed linearly from --explore-eps",
    )
    p.add_argument(
        "--teacher-warmup", type=int, default=5,
        help="updates where `by_value` drives while V fits its trajectories -- "
             "fitted evaluation of the teacher, so the first argmax ranks positions "
             "instead of noise",
    )
    p.add_argument(
        "--td-lambda", type=float, default=1.0,
        help="1 regresses the Monte Carlo return, 0 the one-step bootstrap "
             "r + V(next afterstate). Nothing in between: see the code",
    )
    p.add_argument(
        "--repartition", action="store_true",
        help="offer the CP-SAT macro where nothing else plays, and take it "
             "unconditionally. Moves the reference from by_value's +29.60 to "
             "by_value+repartition's +48.01",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-games", type=int, default=0)
    p.add_argument("--out", type=pathlib.Path, default=None)
    p.add_argument("--checkpoint-every", type=int, default=0)
    p.add_argument("--log-json", type=pathlib.Path, default=None)
    args = p.parse_args()

    if args.td_lambda not in (0.0, 1.0):
        # The forward view for an intermediate lambda mixes V at every step of the
        # episode, which means replaying whole episodes at fit time -- and the ring
        # buffer keeps transitions, not episodes.
        p.error("--td-lambda takes 0 (one-step bootstrap) or 1 (Monte Carlo)")
    bootstrap = args.td_lambda == 0.0

    cfg = dataclasses.replace(
        CONFIG_BY_NAME[args.config], reward_mode=RewardMode.SCORE_NORMALIZED
    )
    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    dim = afterstate_dim(cfg)
    net = ValueNet(dim, tuple(args.hidden))
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    replay = Replay(args.buffer, dim, bootstrap)
    teacher = by_value(cfg)

    # Per env: the transitions of the episode in progress, the decision not yet
    # closed, and the reward accrued behind it. An episode spans many updates, so
    # all three survive one -- and a Monte Carlo return needs the whole episode
    # before any of its rows can be stored at all.
    Transition = tuple[np.ndarray, float, np.ndarray | None, bool]
    episode: list[list[Transition]] = [[] for _ in range(args.envs)]
    open_row: list[np.ndarray | None] = [None] * args.envs
    accrued = np.zeros(args.envs, dtype=np.float32)
    # The env re-deals a finished episode on the *following* step and discards the
    # action supplied with it, so a decision taken there belongs to no episode.
    blocked = np.zeros(args.envs, dtype=bool)
    fresh_x: list[np.ndarray] = []
    fresh_y: list[float] = []
    epsilon = args.explore_eps
    driving_teacher = False

    def value_of(rows: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return net(torch.as_tensor(rows)).numpy()

    def record(env: int, row: np.ndarray, macro: int) -> None:
        """Close the env's previous decision with the reward that followed it.

        A transition spans the opponent's whole reply, because that is what happens
        between two of the learner's decisions -- and it is why an afterstate is
        the thing being valued rather than the state the learner moves from.
        """
        if blocked[env]:
            return
        if open_row[env] is not None:
            episode[env].append((open_row[env], float(accrued[env]), row, False))
        accrued[env] = 0.0
        open_row[env] = row
        blocks[block[macro]] += 1

    def close(env: int) -> None:
        """The episode is over: turn its transitions into rows and store them."""
        if open_row[env] is not None:
            episode[env].append((open_row[env], float(accrued[env]), None, True))
        # Backwards, so the return is the return and not an assumption that the
        # reward is terminal-only -- which is true of every preset, and not of
        # SPEC section 7's shaping terms.
        total = 0.0
        outcomes: list[float] = []
        for _, reward, _, terminal in reversed(episode[env]):
            total = reward + (0.0 if terminal else total)
            outcomes.append(total)
        outcomes.reverse()
        for (row, reward, successor, terminal), outcome in zip(
            episode[env], outcomes, strict=True
        ):
            replay.add(row, outcome, reward, successor, terminal)
            fresh_x.append(row)
            fresh_y.append(outcome)
        episode[env] = []
        open_row[env] = None
        accrued[env] = 0.0

    def pick(obs, env: int, legal: np.ndarray, explore: float) -> int:
        """One decision. `REPARTITION` short-circuits; everything else is ranked."""
        options = np.flatnonzero(legal)
        if agent.repartition_macro is not None and legal[agent.repartition_macro]:
            # Offered only where nothing else plays, so there is nothing to compare
            # it against and no row for V to fit.
            return int(agent.repartition_macro)

        if driving_teacher:
            chosen = teacher(obs, env, legal)
        elif explore > 0.0 and rng.random() < explore:
            chosen = int(rng.choice(options))
        else:
            rows = afterstate_batch(cfg, obs, env, options.tolist(), agent)
            best = int(np.argmax(value_of(rows)))
            record(env, rows[best], int(options[best]))
            return int(options[best])
        record(env, afterstate_batch(cfg, obs, env, [chosen], agent)[0], chosen)
        return chosen

    def learner_choose(obs, env: int, legal: np.ndarray) -> int:
        return pick(obs, env, legal, epsilon)

    env = FixedOpponentEnv(
        num_envs=args.envs, cfg=cfg, seed=args.seed, opponent=args.opponent
    )
    agent = MacroAgent(cfg, choose=learner_choose, repartition=args.repartition)
    agent.reset(args.envs)
    block = block_of(cfg, agent)
    blocks = np.zeros(len(BLOCKS), dtype=np.int64)

    obs, info = env.reset()
    print(
        f"config={args.config} opponent={args.opponent} envs={args.envs} "
        f"dim={dim} params={sum(q.numel() for q in net.parameters()):,} "
        f"lr={args.lr} fit_steps={args.fit_steps}x{args.batch} "
        f"target={'TD(0)' if bootstrap else 'MC'} "
        f"eps={args.explore_eps}->{args.explore_final} warmup={args.teacher_warmup}"
        + (" repartition" if args.repartition else ""),
        flush=True,
    )

    history: list[dict] = []
    started = time.perf_counter()

    def save(path: pathlib.Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "cfg": args.config,
                "hidden": list(args.hidden),
                "repartition": bool(args.repartition),
                "dim": dim,
                "state": net.state_dict(),
            },
            path,
        )
        print(f"wrote {path}", flush=True)

    for update in range(1, args.updates + 1):
        driving_teacher = update <= args.teacher_warmup
        span = max(args.updates - 1, 1)
        epsilon = 0.0 if driving_teacher else (
            args.explore_eps
            + (args.explore_final - args.explore_eps) * (update - 1) / span
        )
        blocks[:] = 0
        fresh_x.clear()
        fresh_y.clear()
        finished = 0
        melded = 0.0

        for _ in range(args.horizon):
            melded += float(np.asarray(obs["melded"])[:, 0].mean()) / args.horizon
            actions = agent.act(obs, np.asarray(info["action_mask"]))
            obs, reward, term, trunc, info = env.step(actions)
            accrued += np.asarray(reward, dtype=np.float32)
            done = np.asarray(term) | np.asarray(trunc)
            for e in np.flatnonzero(done):
                close(int(e))
                finished += 1
            blocked[:] = done

        decisions = int(blocks.sum())
        # Before the fit and on rows nobody trained on yet: what the net predicted
        # while it was acting, against what actually happened.
        if fresh_x:
            fresh = np.stack(fresh_x)
            target = np.asarray(fresh_y, dtype=np.float32)
            ev = explained_variance(target, value_of(fresh))
        else:
            ev = float("nan")

        loss_sum, fitted = 0.0, 0
        if replay.filled >= args.batch:
            for _ in range(args.fit_steps):
                idx = torch.randint(
                    replay.filled, (args.batch,), generator=generator
                ).numpy()
                x = torch.as_tensor(replay.x[idx])
                if replay.nxt is None:
                    y = torch.as_tensor(replay.y[idx])
                else:
                    # Semi-gradient TD(0), gamma 1: the target is rebuilt from the
                    # current net every time the transition is drawn, and the
                    # terminal flag is what anchors the whole chain to a real
                    # outcome.
                    with torch.no_grad():
                        y = torch.as_tensor(replay.reward[idx]) + (
                            1.0 - torch.as_tensor(replay.terminal[idx])
                        ) * net(torch.as_tensor(replay.nxt[idx]))
                # One averaged gradient over the whole sampled batch, never a step
                # per slice of it.
                opt.zero_grad(set_to_none=True)
                loss = (net(x) - y).pow(2).mean()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()
                loss_sum += float(loss.detach())
                fitted += 1

        total = max(decisions, 1)
        end_rate = int(blocks[BLOCKS.index("end")]) / total
        draw_rate = int(blocks[BLOCKS.index("draw")]) / total
        terminal = float(np.mean(fresh_y)) if fresh_y else float("nan")
        elapsed = time.perf_counter() - started
        print(
            f"update {update:>4}{' teach' if driving_teacher else ''}  "
            f"episodes {finished:>4}  rows {len(fresh_y):>6,}  "
            f"end {end_rate:>5.1%}  draw {draw_rate:>5.1%}  meld {melded:>5.1%}  "
            f"terminal {terminal:>+7.3f}  EV {ev:>+6.3f}  "
            f"loss {loss_sum / max(fitted, 1):>7.4f}  eps {epsilon:>4.2f}  "
            f"{decisions / elapsed:>5.0f} dec/s",
            flush=True,
        )
        print(
            "      "
            + "  ".join(
                f"{name} {int(blocks[i]) / total:>5.1%}" for i, name in enumerate(BLOCKS)
            ),
            flush=True,
        )
        history.append(
            {
                "update": update,
                "teacher": driving_teacher,
                "episodes": finished,
                "decisions": decisions,
                "rows": len(fresh_y),
                "end_rate": end_rate,
                "draw_rate": draw_rate,
                "melded": melded,
                "terminal": terminal,
                "explained_variance": ev,
                "loss": loss_sum / max(fitted, 1),
                "epsilon": epsilon,
                "blocks": {
                    name: int(blocks[i]) / total for i, name in enumerate(BLOCKS)
                },
                "decisions_per_second": decisions / elapsed,
            }
        )
        if args.out and args.checkpoint_every and update % args.checkpoint_every == 0:
            save(args.out.with_name(f"{args.out.stem}-u{update:03d}{args.out.suffix}"))
        started = time.perf_counter()

    env.close()
    if args.out:
        save(args.out)

    scores: list[dict] = []
    if args.eval_games:
        suite = SUITE_BY_NAME[
            "standard-greedy" if args.config == "standard" else "tiny"
        ]
        scored = MacroAgent(cfg, repartition=args.repartition)
        scored.choose = argmax_chooser(cfg, scored, value_of)
        # `by_value` reproducing its published score inside this run is the check
        # that the harness did not move under the learned row.
        arms = (
            ("learned", scored),
            ("by_value", MacroAgent(cfg, choose=by_value(cfg), repartition=args.repartition)),
            ("first_legal", MacroAgent(cfg, choose=first_legal, repartition=args.repartition)),
        )
        for label, played in arms:
            result = evaluate(
                label, suite, build_agent=lambda c, a=played: a, games=args.eval_games
            )
            print(
                f"  {label:12s} win {result.win_rate:>6.1%}  "
                f"score {result.mean_score:>+8.2f}  "
                f"illegal {result.illegal_attempts}  n={result.games}",
                flush=True,
            )
            scores.append(
                {
                    "label": label,
                    "suite": suite.name,
                    "win_rate": result.win_rate,
                    "mean_score": result.mean_score,
                    "illegal_attempts": result.illegal_attempts,
                    "games": result.games,
                }
            )

    if args.log_json:
        args.log_json.parent.mkdir(parents=True, exist_ok=True)
        args.log_json.write_text(
            json.dumps(
                {
                    "config": args.config,
                    "opponent": args.opponent,
                    "envs": args.envs,
                    "horizon": args.horizon,
                    "hidden": list(args.hidden),
                    "lr": args.lr,
                    "td_lambda": args.td_lambda,
                    "fit_steps": args.fit_steps,
                    "batch": args.batch,
                    "buffer": args.buffer,
                    "explore_eps": args.explore_eps,
                    "explore_final": args.explore_final,
                    "teacher_warmup": args.teacher_warmup,
                    "repartition": bool(args.repartition),
                    "seed": args.seed,
                    "history": history,
                    "eval": scores,
                },
                indent=2,
            )
        )
        print(f"wrote {args.log_json}", flush=True)


if __name__ == "__main__":
    main()
