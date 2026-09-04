"""Value-based learning over the macro space: score the afterstate, play the argmax.

    python tools/train_afterstate.py --td-lambda 0 --updates 120 --eval-games 120

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

**Those rows predate two fixes in this file** and were produced with the earlier
behaviour: a truncated episode was stored as terminated with a zero Monte Carlo
target, and a decision the `REPARTITION` macro answered was left out of the
per-block counters -- so `repart` read 0.0% and every other block rate, along with
`dec/s`, was a rate over the decisions the solver was not reached on. Neither
touches the estimator the table compares, and both change what a re-run collects,
so a number from this file is comparable to another run of it and not to the table
above.
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

from rummi.agents.base import Observation
from rummi.agents.learned.afterstate import afterstate_batch, afterstate_dim
from rummi.agents.learned.afterstate_net import Value, ValueNet, argmax_chooser, value_fn
from rummi.agents.macro import (
    Choose,
    MacroAgent,
    by_value,
    extend_offset,
    first_legal,
    steal_offset,
)
from rummi.env.fixed_opponent import FixedOpponentEnv
from rummi.evaluate.protocol import evaluate, suite_for
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


Transition = tuple[np.ndarray, float, "np.ndarray | None", bool]
"""`(afterstate row, reward accrued behind it, successor row, terminal)`."""

Stored = tuple[np.ndarray, float, float, "np.ndarray | None", bool]
"""A transition with the outcome it led to, in `Replay.add`'s argument order."""


class Episodes:
    """Per env: the episode in progress, the decision not yet closed, and the
    reward accrued behind it.

    An episode spans many updates, so all of it survives one -- and a Monte Carlo
    return needs the whole episode before any of its rows can be stored at all.
    A class rather than three closures because two of its rules decide what the
    net is trained on, and `tests/test_train_afterstate.py` asserts them.
    """

    def __init__(self, n_envs: int, n_blocks: int) -> None:
        self.episode: list[list[Transition]] = [[] for _ in range(n_envs)]
        self.open_row: list[np.ndarray | None] = [None] * n_envs
        self.accrued = np.zeros(n_envs, dtype=np.float32)
        # The env re-deals a finished episode on the *following* step and discards
        # the action supplied with it, so a decision taken there belongs to no
        # episode.
        self.blocked = np.zeros(n_envs, dtype=bool)
        self.blocks = np.zeros(n_blocks, dtype=np.int64)
        self.dropped = 0
        """Episodes the clock cut off, which are not stored. Reported rather than
        silent: a dropped episode is rows the update did not collect."""

    def count(self, env: int, block: int) -> None:
        """Note a decision that carries no afterstate to value."""
        if not self.blocked[env]:
            self.blocks[block] += 1

    def record(self, env: int, row: np.ndarray, block: int) -> None:
        """Close the env's previous decision with the reward that followed it.

        A transition spans the opponent's whole reply, because that is what happens
        between two of the learner's decisions -- and it is why an afterstate is
        the thing being valued rather than the state the learner moves from.
        """
        if self.blocked[env]:
            return
        standing = self.open_row[env]
        if standing is not None:
            self.episode[env].append((standing, float(self.accrued[env]), row, False))
        self.accrued[env] = 0.0
        self.open_row[env] = row
        self.blocks[block] += 1

    def close(self, env: int, *, terminal: bool) -> list[Stored]:
        """The episode is over: its transitions with the outcome each one led to.

        `terminal` is the env's own `term`, not `term | trunc`. A truncated episode
        is **dropped**: the engine pays nothing for reaching `max_turns` (SPEC.md
        section 7), so storing it as terminated would regress every row of it
        towards an outcome of zero -- and under `--td-lambda 0` its last row would
        anchor the whole chain there, since there is no successor to bootstrap from.
        The cut-off falls on whichever side happened to be behind, so it is not an
        outcome about the policy at all.
        """
        if not terminal:
            self.dropped += 1
            self._forget(env)
            return []
        standing = self.open_row[env]
        if standing is not None:
            self.episode[env].append((standing, float(self.accrued[env]), None, True))
        # Backwards, so the return is the return and not an assumption that the
        # reward is terminal-only -- which is true of every preset, and not of
        # SPEC section 7's shaping terms.
        total = 0.0
        outcomes: list[float] = []
        for _, reward, _, terminal in reversed(self.episode[env]):
            total = reward + (0.0 if terminal else total)
            outcomes.append(total)
        outcomes.reverse()
        out: list[Stored] = [
            (row, outcome, reward, successor, terminal)
            for (row, reward, successor, terminal), outcome in zip(
                self.episode[env], outcomes, strict=True
            )
        ]
        self._forget(env)
        return out

    def _forget(self, env: int) -> None:
        self.episode[env] = []
        self.open_row[env] = None
        self.accrued[env] = 0.0


class Learner:
    """The decision: rank the legal macros by V over the afterstates they reach.

    A class rather than a closure because the rollout reaches into it every update
    -- the teacher drives the warmup and epsilon anneals -- and because what it
    hands to :class:`Episodes` is what the net ends up trained on. It builds its own
    `MacroAgent`, since the chooser has to read back the macro layout it is ranking.
    """

    def __init__(
        self,
        cfg: RummiConfig,
        episodes: Episodes,
        value_of: Value,
        teacher: Choose,
        rng: np.random.Generator,
        repartition: bool,
    ) -> None:
        self.cfg = cfg
        self.episodes = episodes
        self.value_of = value_of
        self.teacher = teacher
        self.rng = rng
        self.epsilon = 0.0
        self.driving_teacher = False
        self.agent = MacroAgent(cfg, choose=self.choose, repartition=repartition)
        self.block = block_of(cfg, self.agent)

    def choose(self, obs: Observation, env: int, legal: np.ndarray) -> int:
        """One decision. `REPARTITION` short-circuits; everything else is ranked."""
        cfg, agent = self.cfg, self.agent
        options = np.flatnonzero(legal)
        if agent.repartition_macro is not None and legal[agent.repartition_macro]:
            # Offered only where nothing else plays, so there is nothing to compare
            # it against and no row for V to fit. Counted all the same: `decisions`
            # is `blocks.sum()`, so leaving it out makes every other block's rate --
            # and `dec/s` -- a rate over the decisions the solver was not reached on.
            self.episodes.count(env, int(self.block[agent.repartition_macro]))
            return int(agent.repartition_macro)

        if self.driving_teacher:
            chosen = self.teacher(obs, env, legal)
        elif self.epsilon > 0.0 and self.rng.random() < self.epsilon:
            chosen = int(self.rng.choice(options))
        else:
            rows = afterstate_batch(cfg, obs, env, options.tolist(), agent)
            best = int(np.argmax(self.value_of(rows)))
            macro = int(options[best])
            self.episodes.record(env, rows[best], int(self.block[macro]))
            return macro
        row = afterstate_batch(cfg, obs, env, [chosen], agent)[0]
        self.episodes.record(env, row, int(self.block[chosen]))
        return chosen


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
             "r + V(next afterstate). Nothing in between: see the code. Every "
             "published row here is at 0; 1 is the default because it is the "
             "estimator the experiment set out to test, and it is the one that lost",
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
    # The package's own scorer: the agent and `tools/search_afterstate.py` read the
    # net through it, so the trainer's argmax cannot be a second reading of it.
    value_of = value_fn(net)

    episodes = Episodes(args.envs, len(BLOCKS))
    fresh_x: list[np.ndarray] = []
    fresh_y: list[float] = []

    env = FixedOpponentEnv(
        num_envs=args.envs, cfg=cfg, seed=args.seed, opponent=args.opponent
    )
    learner = Learner(cfg, episodes, value_of, teacher, rng, args.repartition)
    agent = learner.agent
    agent.reset(args.envs)

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
        learner.driving_teacher = driving_teacher
        learner.epsilon = epsilon
        episodes.blocks[:] = 0
        episodes.dropped = 0
        fresh_x.clear()
        fresh_y.clear()
        finished = 0
        melded = 0.0

        for _ in range(args.horizon):
            melded += float(np.asarray(obs["melded"])[:, 0].mean()) / args.horizon
            actions = agent.act(obs, np.asarray(info["action_mask"]))
            obs, reward, term, trunc, info = env.step(actions)
            episodes.accrued += np.asarray(reward, dtype=np.float32)
            ended = np.asarray(term)
            done = ended | np.asarray(trunc)
            for e in np.flatnonzero(done):
                closed = episodes.close(int(e), terminal=bool(ended[e]))
                for row, outcome, gained, successor, at_end in closed:
                    replay.add(row, outcome, gained, successor, at_end)
                    fresh_x.append(row)
                    fresh_y.append(outcome)
                finished += 1
            episodes.blocked[:] = done

        decisions = int(episodes.blocks.sum())
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
        end_rate = int(episodes.blocks[BLOCKS.index("end")]) / total
        draw_rate = int(episodes.blocks[BLOCKS.index("draw")]) / total
        terminal = float(np.mean(fresh_y)) if fresh_y else float("nan")
        elapsed = time.perf_counter() - started
        print(
            f"update {update:>4}{' teach' if driving_teacher else ''}  "
            f"episodes {finished:>4}"
            # A cut-off episode is stored nowhere, so saying nothing would hide
            # rows the update did not collect.
            + (f" (-{episodes.dropped} cut off)" if episodes.dropped else "")
            + f"  rows {len(fresh_y):>6,}  "
            f"end {end_rate:>5.1%}  draw {draw_rate:>5.1%}  meld {melded:>5.1%}  "
            f"terminal {terminal:>+7.3f}  EV {ev:>+6.3f}  "
            f"loss {loss_sum / max(fitted, 1):>7.4f}  eps {epsilon:>4.2f}  "
            f"{decisions / elapsed:>5.0f} dec/s",
            flush=True,
        )
        print(
            "      "
            + "  ".join(
                f"{name} {int(episodes.blocks[i]) / total:>5.1%}"
                for i, name in enumerate(BLOCKS)
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
                "dropped": episodes.dropped,
                "end_rate": end_rate,
                "draw_rate": draw_rate,
                "melded": melded,
                "terminal": terminal,
                "explained_variance": ev,
                "loss": loss_sum / max(fitted, 1),
                "epsilon": epsilon,
                "blocks": {
                    name: int(episodes.blocks[i]) / total
                    for i, name in enumerate(BLOCKS)
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
        suite = suite_for(args.config)
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
