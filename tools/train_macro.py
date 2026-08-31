"""RL over macro actions: which move to make, when to stop, when to draw.

    python tools/train_macro.py --updates 200 --eval-games 60

The action space is `rummi/agents/macro.py`'s -- 329 set templates, `EXTEND` per
kind, `STEAL` per template, `END_TURN`, `DRAW`; 713 actions on `standard` -- and
every action leaves the table whole, so the half-built invalid workbench that
defeated the primitive-action learner is unreachable rather than penalised.

What that buys, measured before any learning: turns are bounded at 7 micro-actions
against the primitive policy's 71, and the hand-written orderings score **-141**
(`by_value`) and **-147** (`first_legal`) where the best cloned-then-PPO'd
primitive policy scored -230. Those two being so close is the point: *which* set to
play is worth ~6 points, so what this trains is mostly when to keep playing, when
to end the turn, and when not to start.

**One averaged gradient step per batch, computed in chunks.** `--minibatches` splits
the batch for memory only -- gradients accumulate across the chunks and a single step
follows -- because taking a step per chunk scored **-394** where one averaged step
scored **+27**. With a bootstrapped critic on a terminal reward the advantage is
noisy, and four noisy small-batch steps are worse than one averaged one.

`--epochs` reuses a batch, with the ratio clipped and the mask stored so scoring an
old action against the policy that took it stays meaningful. It defaults to 1:
reuse measured *worse* here, since reusing a noisy advantage amplifies its error.
The batch wants to be **large** instead, which is what the chunking is for -- a
whole game spans many decisions, so a short horizon can contain no terminal reward
at all and leaves both heads with nothing to fit.

The advantage is bootstrapped through the critic, per decision, for the reason
recorded in `train_delegate.py`: handing every decision in an episode that
episode's outcome is unbiased but teaches only a global bias, and situational play
needs per-decision credit. Decisions are per *set* here rather than per turn, so an
episode yields several times more of them.

**`--opponent` is a pool, and `self` is one of its members.** Against a single
fixed opponent the terminal reward saturates once the policy beats it, and the
gradient thins out; `--opponent greedy,rearrange` mixes the batch, and `self` seats
frozen copies of the learner. Frozen and *lagging* on purpose: an opponent that is
the learner itself is a target moving in step with the policy chasing it.
`--init-from` warm-starts from a run against a weaker opponent, which is the other
half of the same curriculum.

Three things make that member behave, all of them things a single recent copy gets
wrong. `--snapshot-pool` holds several past selves at once and refreshes them in
rotation, so the batch spans lags rather than one, and beating last week's self
cannot make the policy forget what beat the week before. `--snapshot-gate` promotes
only when the learner has improved against the pool since it last promoted into it,
so a regression cannot install itself as the opponent. Against *that* and not
against zero, because training pins the learner to one seat: the evaluation protocol
scores a mirrored agent at exactly +0.0 only because it rotates every deal through
every seat, and nothing here does. And the schedule counts policy
updates, not wall-clock ones, because a `--value-warmup` update moves the critic
alone.

**Every rate is reported per opponent as well as pooled**, which the fixed
round-robin seating is what makes possible. Pooled, a rising terminal reward cannot
be told apart from an opponent that got worse -- and with `self` in the pool that is
the failure mode, not a corner case. The advantage is normalised per opponent for
the same reason: the observation says nothing about who is playing, so the critic
cannot predict the value gap between facing `greedy` and facing a snapshot, and one
shared normaliser reads that gap as advantage.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import pathlib
import time

import numpy as np
import torch
from torch import nn

from rummi.agents.base import Agent
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
    Choose,
    MacroAgent,
    action_features,
    by_value,
    first_legal,
    n_macros,
)
from rummi.env.fixed_opponent import FixedOpponentEnv
from rummi.evaluate.protocol import SUITE_BY_NAME, evaluate
from rummi.rules.config import CONFIG_BY_NAME, RewardMode, RummiConfig

OPPONENTS = ("greedy", "rearrange", "optimal", "self")
"""What `--opponent` accepts, comma separated. `self` is a frozen snapshot of the
learner; the rest are bundled agents."""

HIDDEN, HEAD = 256, "flat"
"""Defaults for the architecture flags, which parse to None so that a flag passed
alongside `--init-from` can be told apart from one left alone."""


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
    obs = host(obs)
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
        obs = host(obs)

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


def restore(
    path: pathlib.Path, config: str, space: str, macros: int,
    hidden: int | None, head: str | None,
) -> tuple[dict, int, str]:
    """A checkpoint from `--out`, with the architecture *it* was saved with.

    The architecture comes from the file, never from the CLI: `--hidden` and
    `--head` describe tensors that already exist in it, so a flag that disagrees is
    a mistake to report rather than something to reconcile silently. The head's
    width is checked against today's layout for the same reason `eval_macro.py`
    checks it -- a hybrid-space or older-layout checkpoint indexes different actions
    with the same ids, and `load_state_dict` would accept the ones that happen to
    match in size.
    """
    checkpoint = torch.load(path, weights_only=True)
    saved_hidden, saved_head = int(checkpoint["hidden"]), str(checkpoint["head"])
    saved_space = str(checkpoint.get("space", "macro"))
    for flag, given, saved in (
        ("--config", config, str(checkpoint["cfg"])),
        ("--space", space, saved_space),
        ("--hidden", hidden, saved_hidden),
        ("--head", head, saved_head),
    ):
        if given is not None and given != saved:
            raise SystemExit(
                f"{path}: {flag}={given} contradicts the checkpoint's {saved!r}; "
                f"the architecture comes from the checkpoint, so pass {flag}={saved} "
                "or leave it off"
            )

    key = "action_bias" if saved_head == "pointer" else "pi.weight"
    width = int(checkpoint["state"][key].shape[0])
    if width != macros:
        raise SystemExit(
            f"{path}: {width} actions against {macros} in the current {space} layout "
            f"for '{config}' -- a checkpoint from a different action space, which "
            "would load into the wrong rows"
        )
    return checkpoint, saved_hidden, saved_head


def _by_opponent(names, tally, closed, faced, rewards) -> list[dict]:
    """What the batch says about each pool member, and nothing about the others.

    The fixed round-robin seating is what makes this readable at all: pooled over a
    mixed batch, a rising terminal reward cannot be told apart from an opponent that
    got worse -- and with `self` in the pool that is the failure mode, not a corner
    case.
    """
    rows = []
    for member, name in enumerate(names):
        mine = closed & (faced == member)
        decisions = max(int(tally[member, 0]), 1)
        rows.append(
            {
                "opponent": name,
                "decisions": int(tally[member, 0]),
                "end_rate": int(tally[member, 1]) / decisions,
                "draw_rate": int(tally[member, 2]) / decisions,
                "terminal": rewards[mine].mean().item() if bool(mine.any()) else None,
            }
        )
    return rows


def host(obs: dict) -> dict:
    """The observation as NumPy, whatever the backend underneath produced.

    Everything on the learner's side of this trainer is NumPy -- `features`,
    `MacroAgent`, the rack-shaping term -- so it converts once here rather than at
    each of the dozen places that index it. A no-op on the NumPy backend, and JAX is
    CPU-only in this env, so there is nothing to copy off a device either.
    """
    return {key: np.asarray(value) for key, value in obs.items()}


def _opponent_line(row: dict) -> str:
    """One pool member's slice of an update, for the per-opponent log line."""
    head = f"{row['opponent']}: {row['decisions']:>5,} dec end {row['end_rate']:>5.1%}"
    if row["terminal"] is None:
        return f"{head} term (none closed)"
    return f"{head} term {row['terminal']:>+7.3f}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="standard", choices=sorted(CONFIG_BY_NAME))
    p.add_argument(
        "--space", default="macro", choices=["macro", "hybrid"],
        help="hybrid adds the 2400 primitives alongside the macros, so any legal "
             "turn is expressible while a safe macro stays on offer",
    )
    p.add_argument(
        "--backend", default="numpy",
        help="simulator backend under the env. Measured a wash here -- 1362 against "
             "1332 dec/s at --envs 256 -- because this trainer scores one env per "
             "forward pass, so the policy dominates and the simulator is not what a "
             "larger batch is waiting on. `jax` is 1.6x under train_ppo.py, whose "
             "policy is batched; the flag is here for when this one's is too",
    )
    p.add_argument(
        "--opponent", default="greedy",
        help=f"one of {', '.join(OPPONENTS)}, or a comma-separated pool of them "
             "('greedy,self'), which each env draws from by round-robin over its "
             "index. self is a frozen snapshot of the learner",
    )
    p.add_argument(
        "--snapshot-every", type=int, default=25,
        help="policy updates between refreshes of a 'self' snapshot. Warmup updates "
             "do not count: the policy does not move during them",
    )
    p.add_argument(
        "--snapshot-pool", type=int, default=4,
        help="how many past selves the 'self' member holds. They are refreshed one "
             "at a time in rotation, so the pool spans lags of --snapshot-every, "
             "2x that, and so on -- which is what stops the policy chasing one "
             "recent copy of itself in a circle and forgetting older play",
    )
    p.add_argument(
        "--snapshot-gate", type=float, default=0.0,
        help="promote a snapshot only when the learner's mean terminal reward "
             "against the 'self' envs has improved by at least this much since the "
             "last promotion, so a policy that got worse cannot install itself as "
             "the opponent. The comparison is against that remembered score and not "
             "against zero: the learner is pinned to one seat here, so a mirrored "
             "match does not score +0.0 the way the rotated protocol makes it. Pass "
             "a large negative number to promote on the clock alone",
    )
    p.add_argument("--envs", type=int, default=64)
    p.add_argument("--horizon", type=int, default=256)
    p.add_argument("--updates", type=int, default=200)
    p.add_argument(
        "--hidden", type=int, default=None, help=f"trunk width (default {HIDDEN})"
    )
    p.add_argument(
        "--head", default=None, choices=["flat", "pointer"],
        help="pointer scores a macro against what it does, so what is learned about "
             "one set transfers to similar ones; flat (the default) gives each its "
             "own row",
    )
    p.add_argument(
        "--init-from", type=pathlib.Path, default=None,
        help="warm-start from a checkpoint saved by --out. Takes its architecture: "
             "--hidden and --head describe tensors already in the file, so passing "
             "one that disagrees is an error",
    )
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument(
        "--lr-decay", action="store_true",
        help="anneal the learning rate linearly to zero over --updates. The recipe "
             "reaches by_value's level by update 40 and then three seeds in five "
             "take themselves apart; the ones that survive are the ones whose "
             "entropy settles, so the suspect is step size and not exploration",
    )
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument(
        "--micro-step-cost", type=float, default=0.0,
        help="SPEC section 7: charged on every PLACE/PICK/DISSOLVE/ASSIGN and *not* "
             "on a committing action, so it is the one term that penalises dithering. "
             "A hybrid policy stalls turns to the 155-micro budget rather than "
             "ending them, and nothing else in the reward makes that expensive",
    )
    p.add_argument(
        "--rack-shaping", type=float, default=0.0,
        help="potential-based reward on shrinking the rack. The macro space does not "
             "need it -- turns end every 2-4 steps -- but a hybrid policy ends one "
             "every ~32, so a whole game outruns any practical horizon and no "
             "terminal reward reaches the batch at all",
    )
    p.add_argument(
        "--epochs", type=int, default=1,
        help="passes over each batch. Measured worse above 1: reusing a noisy "
             "bootstrapped advantage amplifies its error rather than extracting more",
    )
    p.add_argument(
        "--minibatches", type=int, default=4,
        help="chunks the batch is computed in, for memory. Gradients accumulate "
             "across them into ONE averaged step -- taking a step per chunk instead "
             "scored -394 where one averaged step scored +27",
    )
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
    p.add_argument(
        "--checkpoint-every", type=int, default=0,
        help="also save every N updates, as <out>-uNNN.pt. The curves peak around "
             "update 60-80 and degrade after, so a run's final weights are not its "
             "best ones and scoring only those measures the far side of the peak",
    )
    p.add_argument(
        "--log-json", type=pathlib.Path, default=None,
        help="per-update metrics. Its rates are per *decision*, not the per-step "
             "end_turn/melded that tools/plot_training.py expects, so its panels do "
             "not read this file",
    )
    args = p.parse_args()

    opponents = [name.strip() for name in args.opponent.split(",")]
    unknown = sorted({name for name in opponents if name not in OPPONENTS})
    if unknown:
        p.error(f"unknown opponent(s) {', '.join(unknown)}; choose from {', '.join(OPPONENTS)}")

    cfg = dataclasses.replace(
        CONFIG_BY_NAME[args.config],
        reward_mode=RewardMode.SCORE_NORMALIZED,
        micro_step_cost=args.micro_step_cost,
    )
    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    scale = feature_scale(cfg)
    hybrid = args.space == "hybrid"
    macros = n_hybrid_actions(cfg) if hybrid else n_macros(cfg)
    if args.init_from:
        checkpoint, hidden, head = restore(
            args.init_from, args.config, args.space, macros, args.hidden, args.head
        )
    else:
        checkpoint = None
        hidden = HIDDEN if args.hidden is None else args.hidden
        head = HEAD if args.head is None else args.head
    net = MacroNet(
        cfg, macros, hidden, head=head,
        describe=hybrid_action_features(cfg) if hybrid else None,
    )
    if checkpoint is not None:
        net.load_state_dict(checkpoint["state"])
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    def features(o, e: int) -> np.ndarray:
        row = np.concatenate(
            [np.asarray(o[f])[e].reshape(-1) for f in FEATURE_FIELDS]
        ).astype(np.float32)
        return row / scale

    def argmax_choose(model: MacroNet) -> Choose:
        """Deterministic: a snapshot opponent and a reported score are both meant to
        be reproducible."""
        def choose(o, e: int, legal: np.ndarray) -> int:
            with torch.no_grad():
                logits, _ = model(torch.as_tensor(features(o, e))[None])
            logits = torch.where(
                torch.as_tensor(legal)[None], logits, torch.full_like(logits, MASKED)
            )
            return int(logits[0].argmax())

        return choose

    # Copies of the weights, not references to them. An opponent that lags the
    # learner is a curriculum; one that *is* the learner is a target moving in step
    # with whatever is chasing it. Several lags at once, because a single recent
    # copy lets the policy cycle: it beats last week's self, forgets what beat the
    # week before, and goes round.
    def sample_choose(model: MacroNet) -> Choose:
        """Samples, where `argmax_choose` takes the mode.

        A snapshot is a copy of the learner's *policy*, and the learner samples. The
        mode of a mid-training policy is a different and much worse player: measured
        on seed 2, the update-25 argmax never chose `END_TURN`, stacked macros to the
        micro cap and lost every game -- a broken opponent rather than a lagging one,
        and the learner then trains against a free win. Reproducibility comes from
        the seeded generator, not from taking the mode.
        """
        def choose(o, e: int, legal: np.ndarray) -> int:
            x = features(o, e)
            with torch.no_grad():
                logits, _ = model(torch.as_tensor(x)[None])
            masked = torch.where(
                torch.as_tensor(legal)[None], logits, torch.full_like(logits, MASKED)
            )
            return int(torch.multinomial(torch.softmax(masked[0], -1), 1, generator=generator))

        return choose

    snapshots: list[MacroNet] = []
    if "self" in opponents:
        for _ in range(max(args.snapshot_pool, 1)):
            past = copy.deepcopy(net).eval()
            for parameter in past.parameters():
                parameter.requires_grad_(False)
            snapshots.append(past)
    next_snapshot = 0
    # What the learner scored against the pool as it stood when it was last
    # promoted into. The gate is read against this rather than against zero: unlike
    # the evaluation protocol, training does not rotate seats, so turn order does
    # not cancel and an even match does not sit at +0.0.
    promoted_at: float | None = None
    measuring = False

    def refresh_snapshot(all_of_them: bool = False) -> None:
        """In place, so the agents already seated in the env keep working.

        One at a time in rotation: refreshing the whole pool at once would collapse
        it to a single lag, which is the thing it exists to avoid.
        """
        nonlocal next_snapshot
        if not snapshots:
            return
        targets = snapshots if all_of_them else [snapshots[next_snapshot]]
        for target in targets:
            target.load_state_dict(net.state_dict())
        next_snapshot = (next_snapshot + 1) % len(snapshots)

    # The seating is a fixed round-robin over the env index, which is exactly what
    # makes a metric read per env mean something. Derived before the env so the
    # snapshot member can be built against it, and checked against it after.
    member_of = np.arange(args.envs) % len(opponents)
    self_member = opponents.index("self") if "self" in opponents else -1
    # Which snapshot each of *its own* envs faces. Ranked within the member's share,
    # not `env % pool`: with `greedy,self` the member only ever sees odd env indices,
    # so a plain modulo would consult half the pool and leave the rest to go stale.
    snapshot_of = np.zeros(args.envs, dtype=np.int64)
    if snapshots:
        ours = np.flatnonzero(member_of == self_member)
        snapshot_of[ours] = np.arange(len(ours)) % len(snapshots)

    def opponent_member(name: str) -> str | Agent:
        """`self` is **one** member of the pool whatever --snapshot-pool says.

        Spreading the snapshots over their own pool slots would silently re-weight
        the batch -- `greedy,self` with four snapshots would leave greedy a fifth of
        the envs rather than half. So the member dispatches per env instead, and the
        shares stay where the caller put them.
        """
        if name != "self":
            return name
        choosers = [sample_choose(past) for past in snapshots]

        def choose(o, e: int, legal: np.ndarray) -> int:
            return choosers[snapshot_of[e]](o, e, legal)

        return HybridAgent(cfg, choose=choose) if hybrid else MacroAgent(cfg, choose=choose)

    env = FixedOpponentEnv(
        num_envs=args.envs, cfg=cfg, seed=args.seed, backend=args.backend,
        opponent=[opponent_member(name) for name in opponents],
    )
    assert np.array_equal(member_of, env.pool_index), "the seating is not what was assumed"
    obs, info = env.reset()
    obs = host(obs)

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
        obs = host(obs)
        agent_reset_needed = True
        # The self-play opponents start where the learner does, so it is the cloned
        # policy they face at update 1, not the random init they were taken of.
        refresh_snapshot(all_of_them=True)
    else:
        agent_reset_needed = False

    reference = None
    if args.kl_coef:
        reference = copy.deepcopy(net).eval()
        for parameter in reference.parameters():
            parameter.requires_grad_(False)
        print(f"anchoring to the cloned policy, kl_coef={args.kl_coef}", flush=True)

    def save(path: pathlib.Path) -> None:
        """`eval_macro.py` rebuilds the architecture from these, never from a flag."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "cfg": args.config, "space": args.space, "hidden": hidden, "head": head,
                "state": net.state_dict(),
            },
            path,
        )
        print(f"wrote {path}", flush=True)

    # The critic is at init after cloning, so its advantages are noise until fitted.
    value_opt = torch.optim.Adam(net.v.parameters(), lr=args.lr)

    # In the hybrid space END_TURN and DRAW are primitives at their own ids, not
    # the last two actions.
    end_action = cfg.end_turn_action if hybrid else macros - 2
    draw_action = cfg.draw_action if hybrid else macros - 1

    open_choice: list[tuple[np.ndarray, np.ndarray, int, float] | None] = [None] * args.envs
    accrued = np.zeros(args.envs, dtype=np.float32)
    steps: list[
        tuple[np.ndarray, np.ndarray, int, float, float, np.ndarray, float, int]
    ] = []
    # [decisions, END_TURN, DRAW] per pool member. Pooled across a mixed batch these
    # three cannot separate "the learner improved" from "its opponent got worse",
    # which is the whole reason the seating is a fixed split.
    tally = np.zeros((len(opponents), 3), dtype=np.int64)
    history: list[dict] = []

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
            steps.append(
                (prev_x, prev_legal, prev_a, prev_lp, float(accrued[e]), x, 0.0, e)
            )
        accrued[e] = 0.0
        open_choice[e] = (x, legal.copy(), macro, behaviour)
        tally[member_of[e]] += (1, int(macro == end_action), int(macro == draw_action))
        return macro

    agent = HybridAgent(cfg, choose=choose) if hybrid else MacroAgent(cfg, choose=choose)
    agent.reset(args.envs)
    if agent_reset_needed:
        open_choice[:] = [None] * args.envs
    print(
        f"config={args.config} space={args.space} opponent={args.opponent} "
        f"actions={macros} head={head} micro_cost={args.micro_step_cost} "
        f"params={sum(q.numel() for q in net.parameters()):,}"
        + (f" snapshot_every={args.snapshot_every} pool={len(snapshots)}"
           f" gate={args.snapshot_gate:+.2f}" if snapshots else "")
        + (f" init_from={args.init_from}" if args.init_from else ""),
        flush=True,
    )
    started = time.perf_counter()

    for update in range(1, args.updates + 1):
        if args.lr_decay:
            # Both optimisers: the warmup fits the critic through `value_opt`, and a
            # critic still taking full-size steps late in a run is its own problem.
            scaled = args.lr * (1.0 - (update - 1) / args.updates)
            for optimiser in (opt, value_opt):
                for group in optimiser.param_groups:
                    group["lr"] = scaled
        steps.clear()
        tally[:] = 0
        finished = 0

        for _ in range(args.horizon):
            mask = np.asarray(info["action_mask"])
            actions = agent.act(obs, mask)
            rack_before = np.asarray(obs["rack"]).sum(-1).astype(np.float32)
            obs, reward, term, trunc, info = env.step(actions)
            obs = host(obs)
            reward = np.asarray(reward, dtype=np.float32)
            done = np.asarray(term) | np.asarray(trunc)
            if args.rack_shaping:
                # Phi = -rack_size, F = gamma*Phi(s') - Phi(s). Policy-invariant
                # (Ng, Harada & Russell 1999), so it adds signal without moving the
                # optimum -- and it self-corrects, because DRAW reverts the turn and
                # hands the tiles back. Zero across an episode boundary, where the
                # next observation is a fresh deal with a full rack.
                rack_after = np.asarray(obs["rack"]).sum(-1).astype(np.float32)
                potential = rack_before - args.gamma * rack_after
                reward = reward + args.rack_shaping * np.where(done, 0.0, potential)
            accrued += reward
            for e in np.flatnonzero(done):
                if open_choice[e] is not None:
                    prev_x, prev_legal, prev_a, prev_lp = open_choice[e]
                    steps.append(
                        (prev_x, prev_legal, prev_a, prev_lp, float(accrued[e]),
                         prev_x, 1.0, e)
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
        faced = torch.as_tensor(member_of[np.asarray([s[7] for s in steps])])

        warming = update <= args.value_warmup
        # Targets and advantages come from the policy that acted, once, and are held
        # fixed across the passes: recomputing them per pass chases a moving critic.
        with torch.no_grad():
            _, value_old = net(x)
            _, next_value = net(nxt)
            target = r + args.gamma * (1.0 - terminal) * next_value
            advantage = target - value_old
            # Normalised per opponent, not over the pooled batch. The observation
            # says nothing about who is sitting across the table -- opponents are
            # merged into `unseen` by design -- so the critic cannot predict the
            # value gap between facing greedy and facing a snapshot, and one shared
            # normaliser reads that gap as advantage on whatever action was taken.
            for member in range(len(opponents)):
                mine = faced == member
                count = int(mine.sum())
                if count == 0:
                    continue
                block = advantage[mine]
                spread = block.std() if count > 1 else torch.zeros(())
                advantage[mine] = (block - block.mean()) / (spread + 1e-8)

        total = len(steps)
        size = max(total // args.minibatches, 1)
        # Accumulated over the chunks, not left holding the last one: the chunking is
        # a memory device, and a quarter of the batch is not what H means.
        entropy_sum, entropy_n = 0.0, 0
        for _ in range(1 if warming else args.epochs):
            order = torch.randperm(total, generator=generator)
            active = value_opt if warming else opt
            active.zero_grad(set_to_none=True)
            chunks = max((total + size - 1) // size, 1)
            for start in range(0, total, size):
                idx = order[start : start + size]
                logits, value = net(x[idx])
                logits = torch.where(legal[idx], logits, torch.full_like(logits, MASKED))
                logp_all = torch.log_softmax(logits, -1)
                logp = logp_all.gather(1, a[idx][:, None])[:, 0]
                entropy = -(logp_all.exp() * logp_all).sum(-1).mean()
                entropy_sum += float(entropy.detach()) * len(idx)
                entropy_n += len(idx)
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

                # Scaled so the accumulated gradient is the full-batch mean, then
                # one step per pass: the chunking is a memory device, not a schedule.
                (loss / chunks).backward()

            nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            active.step()

        mean_entropy = entropy_sum / max(entropy_n, 1)
        closed = terminal > 0
        totals = tally.sum(0)
        n = max(int(totals[0]), 1)
        end_rate = int(totals[1]) / n
        draw_rate = int(totals[2]) / n
        terminal_mean = r[closed].mean().item() if bool(closed.any()) else float("nan")

        by_opponent = _by_opponent(opponents, tally, closed, faced, r)

        # The schedule counts *policy* updates: a warmup update fits the critic
        # alone, so the policy a snapshot would copy has not moved. The first one
        # refreshes too -- the pool is still the untrained init, and the policy moves
        # furthest in exactly the phase a fixed period would leave it there for.
        policy_updates = max(update - args.value_warmup, 0)
        due = bool(snapshots) and (
            policy_updates == 1 or (policy_updates > 0 and policy_updates % args.snapshot_every == 0)
        )
        beat_self = by_opponent[self_member]["terminal"] if self_member >= 0 else None
        if measuring and beat_self is not None:
            # The update right after a promotion, so this is what the pool the
            # learner now faces is worth to it. Everything later is read against it.
            promoted_at, measuring = beat_self, False
        held = (
            due
            and beat_self is not None
            and promoted_at is not None
            and beat_self - promoted_at < args.snapshot_gate
        )
        refreshed = due and not held
        if refreshed:
            refresh_snapshot()
            measuring = True
        print(
            f"update {update:>4}{' warm' if warming else ''}{' snap' if refreshed else ''}"
            f"{' held' if held else ''}  "
            f"episodes {finished:>4}  decisions {len(steps):>6,}  "
            f"end {end_rate:>5.1%}  draw {draw_rate:>5.1%}  "
            f"terminal {terminal_mean:>+7.3f}  H {mean_entropy:>5.3f}  "
            f"{len(steps) / (time.perf_counter() - started):>5.0f} dec/s",
            flush=True,
        )
        if len(opponents) > 1:
            print("      " + "   ".join(_opponent_line(row) for row in by_opponent), flush=True)
        if args.out and args.checkpoint_every and update % args.checkpoint_every == 0:
            save(args.out.with_name(f"{args.out.stem}-u{update:03d}{args.out.suffix}"))
        history.append(
            {
                "update": update,
                "episodes": finished,
                "decisions": len(steps),
                "end_rate": end_rate,
                "draw_rate": draw_rate,
                "terminal": terminal_mean,
                "entropy": mean_entropy,
                "warmup": bool(warming),
                "snapshot_refreshed": bool(refreshed),
                "snapshot_held": bool(held),
                "promoted_at": promoted_at,
                "by_opponent": by_opponent,
            }
        )
        started = time.perf_counter()

    env.close()

    if args.out:
        save(args.out)

    scores: list[dict] = []
    if args.eval_games:
        suite = SUITE_BY_NAME[
            "standard-greedy" if args.config == "standard" else "tiny"
        ]
        learned = argmax_choose(net)
        baselines = (
            (("learned", learned), ("macro_first", macro_first(cfg)))
            if hybrid
            else (
                ("learned", learned),
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
            scores.append(
                {
                    "label": label, "suite": suite.name, "win_rate": result.win_rate,
                    "mean_score": result.mean_score,
                    "illegal_attempts": result.illegal_attempts, "games": result.games,
                }
            )

    if args.log_json:
        args.log_json.parent.mkdir(parents=True, exist_ok=True)
        args.log_json.write_text(
            json.dumps(
                {
                    "config": args.config,
                    "space": args.space,
                    "opponent": args.opponent,
                    "snapshot_every": args.snapshot_every if snapshots else None,
                    "snapshot_pool": len(snapshots) or None,
                    "snapshot_gate": args.snapshot_gate if snapshots else None,
                    "init_from": str(args.init_from) if args.init_from else None,
                    "hidden": hidden,
                    "head": head,
                    "clone": args.clone,
                    "seed": args.seed,
                    "micro_step_cost": args.micro_step_cost,
                    "rack_shaping": args.rack_shaping,
                    "lr_decay": bool(args.lr_decay),
                    "history": history,
                    "eval": scores,
                },
                indent=2,
            )
        )
        print(f"wrote {args.log_json}")


if __name__ == "__main__":
    main()
