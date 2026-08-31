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
half of the same curriculum. `--init-from-macro` is the *cross-space* one:
`--space hybrid` from scratch has never trained, because a near-uniform policy over
2400 primitives lifts tiles onto the workbench and out of reach of every macro, so
a hybrid run can start from a macro-space checkpoint instead and spend its
exploration on the block that space did not have.

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
from rummi.agents.learned.history import HistoryMacroAgent, OpponentHistory, history_dim
from rummi.agents.learned.torch_net import MASKED
from rummi.agents.hybrid import (
    HybridAgent,
    hybrid_action_features,
    macro_first,
    macro_to_hybrid_actions,
    primitives_only,
)
from rummi.agents.hybrid import n_actions as n_hybrid_actions
from rummi.agents.macro import (
    Choose,
    MacroAgent,
    action_features,
    by_value,
    extend_offset,
    first_legal,
    n_macros,
    steal_offset,
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

BLOCKS = ("new_set", "extend", "steal", "end", "draw", "prim")
"""What a decision is counted as. `prim` exists only in the hybrid space, where a
policy can spend a whole horizon inside the 2400 primitives: pooled end and draw
rates cannot say where the rest of the mass went, and whether the macro block is
reached at all is the question that space exists to answer."""


class MacroNet(nn.Module):
    """Logits over the macro actions, and a value.

    `flat` gives every macro its own row, so nothing learned about one set carries
    to a similar one. `pointer` scores each macro against `macro.action_features`
    -- what the action *does* -- so the scoring function is shared and a per-action
    bias carries whatever is left over.

    `memory="lstm"` puts an LSTM cell between the trunk and the heads, stepped once
    per decision, so the policy can *learn* what to carry across steps where the
    engineered history block hands it a fixed summary. The heads read the trunk's
    output and the cell's side by side rather than the cell's alone: the snapshot
    path stays intact and memory is strictly additive, which is what makes an arm
    without it a control.
    """

    def __init__(
        self, cfg: RummiConfig, macros: int, hidden: int = 256,
        head: str = "flat", key_dim: int = 64, describe: np.ndarray | None = None,
        extra: int = 0, memory: str = "none", memory_dim: int = 128,
    ) -> None:
        super().__init__()
        self.head = head
        self.memory = memory
        self.memory_dim = memory_dim if memory == "lstm" else 0
        # `extra` widens the input alone -- the opponent-history block is appended
        # to the observation features, so at 0 the trunk is the one it always was.
        self.trunk = nn.Sequential(
            nn.Linear(feature_dim(cfg) + extra, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.cell = nn.LSTMCell(hidden, self.memory_dim) if memory == "lstm" else None
        width = hidden + self.memory_dim
        if head == "pointer":
            desc = torch.as_tensor(
                action_features(cfg) if describe is None else describe
            )
            self.register_buffer("desc", desc)
            self.key = nn.Linear(desc.shape[1], key_dim, bias=False)
            self.query = nn.Linear(width, key_dim)
            self.action_bias = nn.Parameter(torch.zeros(macros))
            # Small, for the same reason the flat head uses gain 0.01: a fresh
            # policy should be near-uniform over the legal macros.
            nn.init.orthogonal_(self.query.weight, 0.01)
            nn.init.zeros_(self.query.bias)
        else:
            self.pi = nn.Linear(width, macros)
            nn.init.orthogonal_(self.pi.weight, 0.01)
            nn.init.zeros_(self.pi.bias)
        self.v = nn.Linear(width, 1)

    def heads(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Logits and value from the representation the heads read -- the trunk's
        output, with the cell's concatenated when there is one."""
        if self.head == "pointer":
            logits = self.query(h) @ self.key(self.desc).T + self.action_bias
        else:
            logits = self.pi(h)
        return logits, self.v(h).squeeze(-1)

    def forward(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ):
        h = self.trunk(x)
        if self.cell is None:
            return self.heads(h)
        hx, cx = self.cell(h, state)
        logits, v = self.heads(torch.cat([h, hx], -1))
        return logits, v, (hx, cx)


def replay_features(
    net: MacroNet,
    x: torch.Tensor,
    pre_h: torch.Tensor,
    pre_c: torch.Tensor,
    terminal: torch.Tensor,
    sequences: list[list[int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The heads' input for every decision, with the cell run in decision order.

    `sequences` lists each env's decision indices into `x`, oldest first. The cell
    starts from the first decision's *stored* act-time state and steps forward, so
    every later decision's state is recomputed on the graph -- which is what lets
    the gradient reach a write through the decisions that read it. Truncated-BPTT-1
    (replaying every step from its stored state) cannot: information that is
    useless now and useful later gets no gradient for being stored at all, and
    "useful later" is the only thing a memory is for.

    A step's `terminal` flag zeroes the state after it, exactly as the rollout
    zeroes the live state on `done`, so a sequence spanning a re-deal cannot carry
    one episode's opponent into the next.

    Returns the `(N, width)` representation plus the final `(h, c)` per sequence,
    in `sequences` order -- the state a bootstrap value for the decision *after*
    the batch has to start from.
    """
    assert net.cell is not None
    trunk = net.trunk(x)
    h = torch.stack([pre_h[s[0]] for s in sequences])
    c = torch.stack([pre_c[s[0]] for s in sequences])
    put_index: list[torch.Tensor] = []
    put_value: list[torch.Tensor] = []
    t = 0
    while True:
        rows = np.flatnonzero([len(s) > t for s in sequences])
        if rows.size == 0:
            break
        idx = torch.as_tensor(np.asarray([sequences[r][t] for r in rows]))
        live = torch.as_tensor(rows)
        hj, cj = net.cell(trunk[idx], (h[live], c[live]))
        put_index.append(idx)
        put_value.append(hj)
        keep = 1.0 - terminal[idx][:, None]
        # Clone-then-assign rather than writing in place: the previous state was
        # saved by the cell's backward, and mutating it would corrupt the graph.
        h, c = h.clone(), c.clone()
        h[live] = hj * keep
        c[live] = cj * keep
        t += 1
    out = trunk.new_zeros(x.shape[0], net.memory_dim)
    out[torch.cat(put_index)] = torch.cat(put_value)
    return torch.cat([trunk, out], -1), h, c


def gather(
    net: MacroNet, env, teacher, samples: int, beta: float, generator,
    encode, make_agent,
) -> dict:
    """States, and the macro the teacher would pick in each.

    `beta` is the chance the *teacher* drives; below 1 the student steers and the
    labels land on states the student actually reaches. The poisoned half-built
    table that made this worthless on the primitive action space cannot occur here,
    because every macro leaves the table whole.

    `encode` is the one state-encoding path the whole trainer shares, so a cloned
    net is fitted on exactly the rows RL will feed it -- history block included.
    """
    obs, info = env.reset()
    obs = host(obs)
    xs: list[np.ndarray] = []
    legals: list[np.ndarray] = []
    ys: list[int] = []
    agreed = 0

    def choose(o, e: int, legal: np.ndarray) -> int:
        label = teacher(o, e, legal)
        x = encode(o, e)
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

    agent = make_agent(choose)
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
    hidden: int | None, head: str | None, history: bool = False,
    memory: str = "none", memory_dim: int = 0,
) -> tuple[dict, int, str]:
    """A checkpoint from `--out`, in `space` and with the architecture *it* was
    saved with.

    The architecture comes from the file, never from the CLI: `--hidden` and
    `--head` describe tensors that already exist in it, so a flag that disagrees is
    a mistake to report rather than something to reconcile silently. The head's
    width is checked against today's layout for the same reason `eval_macro.py`
    checks it -- an older-layout checkpoint indexes different actions with the same
    ids, and `load_state_dict` would accept the ones that happen to match in size.

    `space` is what the caller intends to load *into*, so the two warm starts state
    it differently: `--init-from` passes the run's own space and a cross-space file
    is refused, while `--init-from-macro` passes `macro` and hands the result to
    `transfer_from_macro`.
    """
    checkpoint = torch.load(path, weights_only=True)
    saved_hidden, saved_head = int(checkpoint["hidden"]), str(checkpoint["head"])
    saved_space = str(checkpoint.get("space", "macro"))
    if space != saved_space:
        raise SystemExit(
            f"{path}: a {saved_space!r}-space checkpoint contradicts space={space!r}. "
            "--init-from resumes within one space; a macro-space checkpoint seeds a "
            "hybrid run through --init-from-macro, which transfers only the tensors "
            "the two spaces share"
        )
    for flag, given, saved in (
        ("--config", config, str(checkpoint["cfg"])),
        ("--hidden", hidden, saved_hidden),
        ("--head", head, saved_head),
        # The history block widens the trunk's input, so this is architecture too --
        # and a bool has no "left alone" value, hence the explicit comparison.
        ("--history", history, bool(checkpoint.get("history", False))),
        # The cell is architecture the same way: tensors that exist or do not.
        ("--memory", memory, str(checkpoint.get("memory", "none"))),
        ("--memory-dim", memory_dim, int(checkpoint.get("memory_dim", 0))),
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
            f"for '{config}' -- a checkpoint from an older layout, which would load "
            "into the wrong rows"
        )
    return checkpoint, saved_hidden, saved_head


def transfer_from_macro(
    net: MacroNet, state: dict[str, torch.Tensor], cfg: RummiConfig
) -> None:
    """Macro-space weights into a fresh hybrid-space `net`, in place.

    The two spaces differ in their action list and in nothing else, so the trunk,
    the pointer's query and the critic are the same tensors on both sides and copy
    across whole. The action-shaped tensors -- `action_bias`, or `pi` under the flat
    head -- have an exact row map, `macro_to_hybrid_actions`, and the rows the macro
    space has nothing to say about keep the fresh init: zero for a bias, gain 0.01
    for a `pi` row, both of which mean *near-uniform*.

    The pointer's `key` reads the action description, and `hybrid_action_features`
    lays the macro columns out first and unchanged, so those columns of `key.weight`
    carry over and the four new PLACE/PICK/DISSOLVE/ASSIGN flags keep their init.
    `END_TURN` and `DRAW` are the one place the description itself moves: they share
    one column of the macro table and have an `ActionKind` flag each here, so that
    column's weights are copied to both. That is what makes the transfer *exact* --
    on any observation the warm-started net's logit over each of the 713 macro-space
    actions equals the source net's, so their whole ranking survives. What cannot
    survive is the softmax: 2398 further actions now share the denominator, so every
    probability is scaled down by the mass the primitives take.
    """
    where = torch.as_tensor(macro_to_hybrid_actions(cfg))
    out = dict(net.state_dict())
    action_shaped = ("action_bias", "key.weight", "pi.weight", "pi.bias")
    for name, tensor in state.items():
        # `desc` is a buffer holding the *other* space's action table.
        if name != "desc" and name not in action_shaped:
            out[name] = tensor

    if net.head == "pointer":
        bias = out["action_bias"].clone()
        bias[where] = state["action_bias"]
        out["action_bias"] = bias
        key = out["key.weight"].clone()
        macro, hybrid = action_features(cfg), hybrid_action_features(cfg)
        key[:, : macro.shape[1]] = state["key.weight"]
        shared = np.flatnonzero(macro[n_macros(cfg) - 1])
        assert len(shared) == 1, "END_TURN and DRAW are one column of the macro table"
        for action in (cfg.end_turn_action, cfg.draw_action):
            flag = np.flatnonzero(hybrid[action])
            assert len(flag) == 1, "a committing primitive describes itself by one flag"
            key[:, flag[0]] = state["key.weight"][:, shared[0]]
        out["key.weight"] = key
    else:
        for name in ("pi.weight", "pi.bias"):
            rows = out[name].clone()
            rows[where] = state[name]
            out[name] = rows
    net.load_state_dict(out)


def action_blocks(cfg: RummiConfig, hybrid: bool) -> np.ndarray:
    """`BLOCKS` index of every action id in the space being trained.

    Derived from the layout functions rather than restated, so it cannot drift from
    the head the net is built with.
    """
    total = n_hybrid_actions(cfg) if hybrid else n_macros(cfg)
    offset = cfg.n_actions if hybrid else 0
    n_macro = total - offset - (0 if hybrid else 2)
    out = np.full(total, BLOCKS.index("prim"), dtype=np.int64)
    macro = np.arange(n_macro)
    out[offset : offset + n_macro] = np.where(
        macro < extend_offset(cfg), 0, np.where(macro < steal_offset(cfg), 1, 2)
    )
    out[cfg.end_turn_action if hybrid else n_macro] = BLOCKS.index("end")
    out[cfg.draw_action if hybrid else n_macro + 1] = BLOCKS.index("draw")
    return out


def _block_line(names: tuple[str, ...], blocks: np.ndarray) -> str:
    """Where an update's decisions went, as shares of the decisions counted."""
    n = max(int(blocks.sum()), 1)
    return "      " + "  ".join(
        f"{name} {int(blocks[i]) / n:>5.1%}" for i, name in enumerate(names)
    )


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
        "--history", action="store_true",
        help="append the opponent-history block to the observation features. The "
             "observation is one state and the net is feedforward, so what the "
             "opponent declined to play is unreachable without it -- and against a "
             "deterministic opponent a declined lay-off is a hard constraint on its "
             "rack, not a hint. Two seats only. It buys nothing measured: 3 seeds "
             "each way score +29.9 without it and land inside that range with it, "
             "once matched on entropy. And they have to be matched, because the 111 "
             "extra inputs delay the entropy collapse by ~20 updates -- H 1.15-1.73 "
             "at update 40 against 0.44-0.67 -- so the recipe's usual scoring point "
             "reads this arm mid-convergence rather than at its peak",
    )
    p.add_argument(
        "--history-decay", type=float, default=0.8,
        help="per-turn decay on the 'declined while the table would have taken it' "
             "counter. Evidence about a rack goes stale as the opponent draws",
    )
    p.add_argument(
        "--memory", default="none", choices=["none", "lstm"],
        help="an LSTM cell between the trunk and the heads, stepped once per "
             "decision and zeroed on a re-deal. Where --history hands the net a "
             "fixed summary of the opponent's turns, this asks it to *learn* what "
             "is worth carrying -- the other of the two candidate mechanisms for "
             "the information a snapshot observation cannot hold. The update "
             "replays each env's decisions in order from their stored initial "
             "state, so the gradient reaches a write through the later decisions "
             "that read it: full BPTT over the batch, not the one-step truncation "
             "that cannot learn to store anything. Like --history it buys nothing "
             "measured against greedy: converged seeds land at the top of the "
             "control's range but inside it, sampled ties too, and zeroing the "
             "trained cell state flips 0.0%% of 9,625 argmax decisions -- the "
             "policy converges memoryless",
    )
    p.add_argument(
        "--memory-dim", type=int, default=128,
        help="width of the --memory cell's state; ignored without one",
    )
    p.add_argument(
        "--init-from", type=pathlib.Path, default=None,
        help="warm-start from a checkpoint saved by --out, in this run's own space. "
             "Takes its architecture: --hidden and --head describe tensors already "
             "in the file, so passing one that disagrees is an error",
    )
    p.add_argument(
        "--init-from-macro", type=pathlib.Path, default=None,
        help="seed a --space hybrid run from a macro-space checkpoint. The trunk, "
             "the critic and every macro's weights transfer exactly -- the hybrid "
             "action table embeds the macro one -- and the 2400 primitives start "
             "near-uniform, so the run begins knowing how to play a macro and spends "
             "its exploration on the block it does not know",
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
        help="per-update metrics. The end, draw and block rates are per *decision*, "
             "not the per-step end_turn tools/plot_training.py expects, so its panels "
             "do not read this file",
    )
    args = p.parse_args()

    opponents = [name.strip() for name in args.opponent.split(",")]
    unknown = sorted({name for name in opponents if name not in OPPONENTS})
    if unknown:
        p.error(f"unknown opponent(s) {', '.join(unknown)}; choose from {', '.join(OPPONENTS)}")
    if args.history and "self" in opponents:
        # The tracker belongs to the learner's seat, and a snapshot sits on the
        # other one. Feeding it the learner's block would hand it the wrong
        # opponent's history -- silently, since the widths match.
        p.error("--history has no tracker for a 'self' opponent; drop one of them")
    if args.history and args.space == "hybrid":
        # `make_agent` seats a plain HybridAgent, so nothing would ever update the
        # tracker while `extra` widens the net to read it -- a block of stale zeros,
        # silently. Refusing is honest until the hybrid agent carries one.
        p.error("--history is wired for the macro agent only; the hybrid agent has no tracker")
    recurrent = args.memory != "none"
    if recurrent and "self" in opponents:
        # Unlike the tracker this is designable -- a snapshot would carry its own
        # per-env state, cleared on the same re-deals -- but nothing drives one yet.
        p.error("--memory has no per-env state for a 'self' snapshot yet; drop one of them")
    if recurrent and (args.clone or args.kl_coef):
        # `gather`/`fit` score states one row at a time with no state threaded
        # through, and the KL reference would need the same replay the update does.
        p.error("--memory supports neither --clone nor --kl-coef; the recipe here uses neither")
    if recurrent and args.eval_games:
        # `argmax_choose` closes over one state per env with nothing resetting it
        # between the protocol's seat rotations; eval_macro.py builds a fresh state
        # per rotation, which is what makes its score mean something.
        p.error("--memory checkpoints are scored with tools/eval_macro.py; drop --eval-games")

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
    if args.init_from and args.init_from_macro:
        p.error("--init-from and --init-from-macro are two warm starts; pass one")
    if args.init_from_macro and not hybrid:
        p.error("--init-from-macro seeds a hybrid run; within the macro space that is "
                "--init-from")
    if args.init_from:
        checkpoint, hidden, head = restore(
            args.init_from, args.config, args.space, macros, args.hidden, args.head,
            args.history, args.memory, args.memory_dim if recurrent else 0,
        )
    elif args.init_from_macro:
        checkpoint, hidden, head = restore(
            args.init_from_macro, args.config, "macro", n_macros(cfg),
            args.hidden, args.head,
        )
    else:
        checkpoint = None
        hidden = HIDDEN if args.hidden is None else args.hidden
        head = HEAD if args.head is None else args.head
    tracker = OpponentHistory(cfg, decay=args.history_decay) if args.history else None
    net = MacroNet(
        cfg, macros, hidden, head=head,
        describe=hybrid_action_features(cfg) if hybrid else None,
        extra=history_dim(cfg) if tracker is not None else 0,
        memory=args.memory, memory_dim=args.memory_dim,
    )
    if args.init_from_macro:
        transfer_from_macro(net, checkpoint["state"], cfg)
    elif checkpoint is not None:
        net.load_state_dict(checkpoint["state"])
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    def features(o, e: int) -> np.ndarray:
        row = np.concatenate(
            [np.asarray(o[f])[e].reshape(-1) for f in FEATURE_FIELDS]
        ).astype(np.float32)
        row = row / scale
        if tracker is None:
            return row
        return np.concatenate([row, tracker.row(e)])

    def make_agent(chooser: Choose) -> Agent:
        """The one place an agent is built, so the tracker can never be left out
        of a path that feeds the net a history block."""
        if hybrid:
            return HybridAgent(cfg, choose=chooser)
        if tracker is None:
            return MacroAgent(cfg, choose=chooser)
        return HistoryMacroAgent(cfg, tracker, choose=chooser)

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
                net, env, teachers[args.clone], per_round, beta, generator,
                features, make_agent,
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
                # The block widens the trunk's input and the evaluator has to run
                # the same tracker, so the flag travels with the weights: a net
                # scored with the block zeroed is not the net that was trained.
                "history": bool(args.history), "history_decay": args.history_decay,
                # Same contract: the evaluator has to step the same cell.
                "memory": args.memory, "memory_dim": net.memory_dim,
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
    block_of = action_blocks(cfg, hybrid)
    block_names = BLOCKS if hybrid else BLOCKS[:-1]

    # The live cell state per env, and each decision's copy of it from before the
    # cell stepped. The copies are what the update replays from: the first decision
    # of an env's batch starts there, and every later state is recomputed on the
    # graph so BPTT can reach it.
    mem = (
        (torch.zeros(args.envs, net.memory_dim), torch.zeros(args.envs, net.memory_dim))
        if recurrent else None
    )
    Pre = tuple[np.ndarray, np.ndarray] | None
    open_choice: list[tuple[np.ndarray, np.ndarray, int, float, Pre] | None] = (
        [None] * args.envs
    )
    accrued = np.zeros(args.envs, dtype=np.float32)
    steps: list[
        tuple[np.ndarray, np.ndarray, int, float, Pre, float, np.ndarray, float, int]
    ] = []
    # [decisions, END_TURN, DRAW] per pool member. Pooled across a mixed batch these
    # three cannot separate "the learner improved" from "its opponent got worse",
    # which is the whole reason the seating is a fixed split.
    tally = np.zeros((len(opponents), 3), dtype=np.int64)
    # Pooled, unlike `tally`: this asks what kind of move the policy makes, which is
    # a property of the policy and not of who it is sitting across from.
    blocks = np.zeros(len(BLOCKS), dtype=np.int64)
    history: list[dict] = []

    def choose(o, e: int, legal: np.ndarray) -> int:
        x = features(o, e)
        # Acting needs no graph, and building one per decision is pure waste.
        with torch.no_grad():
            if mem is None:
                pre: Pre = None
                logits, _ = net(torch.as_tensor(x)[None])
            else:
                pre = (mem[0][e].numpy().copy(), mem[1][e].numpy().copy())
                logits, _, after = net(
                    torch.as_tensor(x)[None], (mem[0][e : e + 1], mem[1][e : e + 1])
                )
                mem[0][e], mem[1][e] = after[0][0], after[1][0]
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
            prev_x, prev_legal, prev_a, prev_lp, prev_pre = open_choice[e]
            steps.append(
                (prev_x, prev_legal, prev_a, prev_lp, prev_pre,
                 float(accrued[e]), x, 0.0, e)
            )
        accrued[e] = 0.0
        open_choice[e] = (x, legal.copy(), macro, behaviour, pre)
        tally[member_of[e]] += (1, int(macro == end_action), int(macro == draw_action))
        blocks[block_of[macro]] += 1
        return macro

    agent = make_agent(choose)
    agent.reset(args.envs)
    if agent_reset_needed:
        open_choice[:] = [None] * args.envs
    print(
        f"config={args.config} space={args.space} opponent={args.opponent} "
        f"actions={macros} head={head} micro_cost={args.micro_step_cost} "
        f"params={sum(q.numel() for q in net.parameters()):,}"
        + (f" history={history_dim(cfg)}d decay={args.history_decay}" if tracker else "")
        + (f" memory={args.memory}:{net.memory_dim}d" if recurrent else "")
        + (f" snapshot_every={args.snapshot_every} pool={len(snapshots)}"
           f" gate={args.snapshot_gate:+.2f}" if snapshots else "")
        + (f" init_from={args.init_from}" if args.init_from else "")
        + (f" init_from_macro={args.init_from_macro}" if args.init_from_macro else ""),
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
        blocks[:] = 0
        finished = 0
        melded = 0.0

        for _ in range(args.horizon):
            # Seat 0 of the rotated flags is the learner, since this is its turn.
            # A standing flag, not an event: it says how much of the batch has
            # opened, and it falls only when an episode ends and re-deals.
            melded += float(np.asarray(obs["melded"])[:, 0].mean()) / args.horizon
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
            if tracker is not None:
                # Envs are recycled, so nothing the tracker holds may cross a
                # re-deal. The env re-deals on the *next* step, which is what the
                # tracker's own one-observation blind spot covers.
                tracker.clear(done)
            if mem is not None and done.any():
                # The cell state is per episode the way the tracker is: the next
                # decision on this env is a fresh deal against a fresh rack.
                finished_rows = torch.as_tensor(done)
                mem[0][finished_rows] = 0.0
                mem[1][finished_rows] = 0.0
            for e in np.flatnonzero(done):
                if open_choice[e] is not None:
                    prev_x, prev_legal, prev_a, prev_lp, prev_pre = open_choice[e]
                    steps.append(
                        (prev_x, prev_legal, prev_a, prev_lp, prev_pre,
                         float(accrued[e]), prev_x, 1.0, e)
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
        r = torch.as_tensor(np.asarray([s[5] for s in steps], dtype=np.float32))
        nxt = torch.as_tensor(np.stack([s[6] for s in steps]))
        terminal = torch.as_tensor(np.asarray([s[7] for s in steps], dtype=np.float32))
        faced = torch.as_tensor(member_of[np.asarray([s[8] for s in steps])])
        if recurrent:
            pre_h = torch.as_tensor(np.stack([s[4][0] for s in steps]))
            pre_c = torch.as_tensor(np.stack([s[4][1] for s in steps]))
            # Each env's decisions, oldest first -- `steps` is appended in play
            # order, so per-env order survives the interleaving.
            by_env: dict[int, list[int]] = {}
            for i, s in enumerate(steps):
                by_env.setdefault(s[8], []).append(i)
            sequences = list(by_env.values())

        warming = update <= args.value_warmup
        # Targets and advantages come from the policy that acted, once, and are held
        # fixed across the passes: recomputing them per pass chases a moving critic.
        with torch.no_grad():
            if recurrent:
                rep, fin_h, fin_c = replay_features(
                    net, x, pre_h, pre_c, terminal, sequences
                )
                _, value_old = net.heads(rep)
                # A non-terminal step's successor is the next decision in its env's
                # sequence; the last one per env bootstraps through one more cell
                # step from the replay's final state. A terminal step's successor
                # is nulled by (1 - terminal) either way.
                last = torch.as_tensor([s[-1] for s in sequences])
                tail_trunk = net.trunk(nxt[last])
                tail_h, _ = net.cell(tail_trunk, (fin_h, fin_c))
                _, tail_value = net.heads(torch.cat([tail_trunk, tail_h], -1))
                next_value = torch.zeros_like(value_old)
                for j, s in enumerate(sequences):
                    run = torch.as_tensor(s)
                    next_value[run[:-1]] = value_old[run[1:]]
                    next_value[run[-1]] = tail_value[j]
            else:
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
            active = value_opt if warming else opt
            active.zero_grad(set_to_none=True)
            if recurrent:
                # Chunked by env, not by decision: a sequence sliced across chunks
                # would replay from states no chunk computes. Losses are summed and
                # scaled by the whole batch, because env-sized chunks are unequal
                # and a per-chunk mean would weight the small ones up. Still one
                # averaged step per pass, exactly as below.
                order = torch.randperm(len(sequences), generator=generator)
                for group in np.array_split(
                    order.numpy(), min(args.minibatches, len(sequences))
                ):
                    if group.size == 0:
                        continue
                    picked = [sequences[int(j)] for j in group]
                    flat = torch.as_tensor(
                        np.concatenate([np.asarray(s) for s in picked])
                    )
                    local: list[list[int]] = []
                    offset = 0
                    for s in picked:
                        local.append(list(range(offset, offset + len(s))))
                        offset += len(s)
                    rep, _, _ = replay_features(
                        net, x[flat], pre_h[flat], pre_c[flat], terminal[flat], local
                    )
                    logits, value = net.heads(rep)
                    logits = torch.where(
                        legal[flat], logits, torch.full_like(logits, MASKED)
                    )
                    logp_all = torch.log_softmax(logits, -1)
                    logp = logp_all.gather(1, a[flat][:, None])[:, 0]
                    each = -(logp_all.exp() * logp_all).sum(-1)
                    entropy_sum += float(each.detach().sum())
                    entropy_n += len(flat)
                    value_loss = (value - target[flat]).pow(2).sum()
                    if warming:
                        loss = value_loss
                    else:
                        ratio = (logp - old_logp[flat]).exp()
                        adv = advantage[flat]
                        clipped = torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * adv
                        loss = -torch.min(ratio * adv, clipped).sum()
                        loss = loss + 0.5 * value_loss - args.entropy_coef * each.sum()
                    (loss / total).backward()
                nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                active.step()
                continue
            order = torch.randperm(total, generator=generator)
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
            f"end {end_rate:>5.1%}  draw {draw_rate:>5.1%}  meld {melded:>5.1%}  "
            f"terminal {terminal_mean:>+7.3f}  H {mean_entropy:>5.3f}  "
            f"{len(steps) / (time.perf_counter() - started):>5.0f} dec/s",
            flush=True,
        )
        print(_block_line(block_names, blocks), flush=True)
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
                "melded": melded,
                "blocks": {
                    name: int(blocks[i]) / max(int(blocks.sum()), 1)
                    for i, name in enumerate(block_names)
                },
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
            # Only the learned one reads the tracker, and a baseline sharing it
            # would drive the same per-env memory from a different game.
            scored = (
                make_agent(ch) if label == "learned"
                else HybridAgent(cfg, choose=ch) if hybrid
                else MacroAgent(cfg, choose=ch)
            )
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
                    "init_from_macro": (
                        str(args.init_from_macro) if args.init_from_macro else None
                    ),
                    "hidden": hidden,
                    "head": head,
                    "opponent_history": bool(args.history),
                    "opponent_history_decay": args.history_decay if args.history else None,
                    "memory": args.memory,
                    "memory_dim": net.memory_dim,
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
