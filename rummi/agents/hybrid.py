"""Primitives and macros in one action space: expressive *and* learnable.

The two spaces fail in opposite ways, both measured on `standard`. Primitives are
fully expressive -- `DISSOLVE`/`PICK`/`PLACE`/`ASSIGN` can reach any legal turn,
multi-set repartitions included, which is what `optimal`'s own plans are made of --
but a policy trained on them from scratch scores **-442.3**, exactly the random
floor, because `PLACE` moves a tile out of the rack and `END_TURN` needs the
workbench empty, so a policy that lifts tiles forming no valid set has killed the
turn and revert is the only exit. Macros cannot reach a repartition -- `optimal` can
still play in 47.7% of the states where a macro agent draws, and 100% of those plays
dissolve a set -- but they score **+25** because every action leaves the table
whole.

So: offer both -- the primitives for the turns macros cannot express, and a macro
where one fits.

**A macro consumes what is held.** Feasibility is judged against `rack + workbench`,
and every tile on the workbench must be laid down by the macro itself. That second
half is what makes offering one mid-turn safe at all: `to_actions.plan` balances the
board against exactly what the hand plays, so a held tile the target does not want
has nowhere to go, and the turn would stay uncommittable behind it. Per family:

- a **template** lays its tiles, one joker standing in for at most one it cannot
  cover, so it is offered when the workbench is within what it lays;
- an **EXTEND** lays exactly one tile, so it is offered only when that tile *is* the
  whole workbench;
- a **STEAL** takes one tile off the table and lays the rest, so it is offered when
  the workbench is within the template net of the stolen tile.

The rack supplies whatever the workbench does not, and `plan` emits a `PLACE` only
for those: a held tile is already in hand and needs its `ASSIGN` alone.

**What that buys, and what it does not.** Measured under a uniform-random policy over
the legal hybrid actions: the workbench is dirty in **94.1%** of decisions and a macro
is on offer in **14.7%**, against 4.9% under the clean-workbench rule this replaced.
The rate is the shape of a macro, not a gap in the rule -- 90.1% with nothing held,
44.5% holding one tile, 7.7% holding two, ~0% past four -- because **one macro plays
one set**, so it can only absorb a workbench that fits inside one. Lift four
unrelated tiles and only primitives and `DRAW` remain. Every macro that *is* offered
clears the workbench, and 67.4% of the time leaves the table whole; the rest is a set
an earlier `PICK` broke in the middle, which no single macro repairs. One hole is left
open on purpose: an expansion longer than the turn's remaining micro budget is still
offered, and is abandoned for `DRAW` at its first masked action like any stale plan,
because measuring its length means expanding all 711 macros at every decision.

**It is not yet enough to train.** `END_TURN` is legal in 0.7% of decisions, barely
moved, because an empty workbench is only half of that condition and the opening-meld
threshold is the other. 30 updates of PPO leave `end` at 0.0% and the terminal reward
at ~-1.04, where the clean-workbench rule left them. The hatch is open wherever a set
fits around the held tiles; finding it is now an exploration problem rather than an
expressiveness one.
"""

from __future__ import annotations

import numpy as np

from rummi.agents.base import Observation
from rummi.agents.macro import Choose, MacroAgent, _n_choices, action_features, n_macros
from rummi.rules.config import RummiConfig

_FEATURES: dict[RummiConfig, np.ndarray] = {}


def n_actions(cfg: RummiConfig) -> int:
    """Primitives, then the tile-playing macros. `END_TURN` and `DRAW` are already
    primitives, so the macro block stops short of them."""
    return cfg.n_actions + _n_choices(cfg)


def macro_to_hybrid_actions(cfg: RummiConfig) -> np.ndarray:
    """`(n_macros,)` -- where each macro-space action sits in the hybrid space.

    The tile-playing macros keep their order, after the primitive block; `END_TURN`
    and `DRAW` are the last two macros there and primitives at their own ids here.
    One map rather than one per caller, because both directions of it are used --
    a macro ranking reads a hybrid mask through it, a warm start moves macro-space
    weights into hybrid-space rows -- and two spellings would drift.
    """
    n_macro = _n_choices(cfg)
    out = np.empty(n_macros(cfg), dtype=np.int64)
    out[:n_macro] = cfg.n_actions + np.arange(n_macro)
    out[n_macro] = cfg.end_turn_action
    out[n_macro + 1] = cfg.draw_action
    return out


def hybrid_action_features(cfg: RummiConfig) -> np.ndarray:
    """`(n_actions, d)` description of every action, primitives included.

    The macro rows are `macro.action_features`; the primitive rows describe
    themselves the same way, so a pointer head can score a `PLACE(R7)` against the
    same tile columns a template uses. Without that the two blocks would be scored
    on unrelated grounds.
    """
    from rummi.rules.actions import ActionKind, decode_batch
    from rummi.rules.encoding import tables

    cached = _FEATURES.get(cfg)
    if cached is not None:
        return cached

    macro = action_features(cfg)
    width = macro.shape[1]
    # One column per ActionKind, so END_TURN and DRAW cannot share a block with
    # PLACE and PICK -- which they would under any narrower encoding.
    out = np.zeros((n_actions(cfg), width + len(ActionKind)), dtype=np.float32)
    out[cfg.n_actions :, :width] = macro[: _n_choices(cfg)]

    values = tables(cfg).value
    scale = float(cfg.n_numbers * cfg.max_set_len)
    decoded = decode_batch(cfg, np.arange(cfg.n_actions))
    for action in range(cfg.n_actions):
        row = out[action]
        for block, flag in enumerate(
            (decoded.is_place, decoded.is_pick, decoded.is_dissolve,
             decoded.is_assign, decoded.is_end_turn, decoded.is_draw)
        ):
            if flag[action]:
                row[width + block] = 1.0
        # A primitive plays at most one tile, and only PLACE and ASSIGN name one.
        if decoded.is_place[action] or decoded.is_assign[action]:
            kind = int(decoded.kind[action])
            row[kind] = 1.0
            row[cfg.n_kinds] = values[kind] / scale
            row[cfg.n_kinds + 1] = 1.0 / cfg.max_set_len
        slot = int(decoded.slot[action])
        if slot >= 0:
            row[cfg.n_kinds + 3] = slot / cfg.max_sets

    _FEATURES[cfg] = out
    return out


class HybridAgent:
    """Actions are the `cfg.n_actions` primitives, then the tile-playing macros."""

    name = "hybrid"

    needs_mask_per_action = True
    """The primitives are offered straight out of the mask, so unlike `MacroAgent`
    this one cannot be handed a stale or unchecked view: it would choose from it."""

    def __init__(self, cfg: RummiConfig, choose: Choose | None = None) -> None:
        self.cfg = cfg
        self.macro = MacroAgent(cfg)
        self.macro_offset = cfg.n_actions
        self.n_macro = _n_choices(cfg)
        self.n_actions = n_actions(cfg)
        self.choose = choose if choose is not None else macro_first(cfg)
        self._queues: dict[int, list[int]] = {}

    def reset(self, n_envs: int) -> None:
        self._queues = {}
        self.macro.reset(n_envs)

    def legal(self, obs: Observation, env: int, mask: np.ndarray) -> np.ndarray:
        out = np.zeros(self.n_actions, dtype=bool)
        out[: self.macro_offset] = mask[env]
        # The workbench goes to `legal_macros`, which offers a macro only if its
        # expansion lays every held tile down. That is what keeps a way out of a
        # half-built turn on offer instead of only before one starts.
        # `can_end` is dead here: the slice stops below the END_TURN macro, because
        # in this space ending the turn is the primitive at its own id.
        out[self.macro_offset :] = self.macro.legal_macros(
            obs, env, held=np.asarray(obs["workbench"])[env], can_end=False
        )[: self.n_macro]
        return out

    def act(
        self, obs: Observation, mask: np.ndarray, active: np.ndarray | None = None
    ) -> np.ndarray:
        cfg = self.cfg
        n = mask.shape[0]
        out = np.full(n, cfg.draw_action, dtype=np.int64)

        for env in range(n):
            if active is not None and not active[env]:
                continue
            queue = self._queues.setdefault(env, [])
            if not queue:
                action = self.choose(obs, env, self.legal(obs, env, mask))
                if action < self.macro_offset:
                    # A primitive is one step; it may well leave the turn half built,
                    # which is the point of having it.
                    out[env] = action if mask[env, action] else cfg.draw_action
                    continue
                queue = self._queues[env] = self.macro.expand(
                    obs,
                    env,
                    action - self.macro_offset,
                    held=np.asarray(obs["workbench"])[env],
                )
            if not queue:
                continue
            step = queue.pop(0)
            if not mask[env, step]:
                queue.clear()
                continue
            out[env] = step
        return out


def macro_first(cfg: RummiConfig) -> Choose:
    """Play the best macro when one is on offer, else the first legal primitive.

    A floor to beat, and a check the macro block is reachable at all: a hybrid
    policy that never picks a macro has thrown away the reason for having one. The
    ranking is `macro.by_value`, so this scores exactly as the macro-only agent does
    wherever a macro is available.
    """
    from rummi.agents.macro import by_value

    rank = by_value(cfg)
    where = macro_to_hybrid_actions(cfg)
    n_macro = _n_choices(cfg)

    def choose(obs: Observation, env: int, legal: np.ndarray) -> int:
        # `by_value` ranks over a macro-space mask, so the hybrid one is read
        # through the map and its answer mapped back. DRAW is never masked.
        padded = legal[where].copy()
        padded[n_macro + 1] = True
        return int(where[rank(obs, env, padded)])

    return choose


def primitives_only(cfg: RummiConfig) -> Choose:
    """Ignore the macro block entirely, for measuring what it is worth."""

    def choose(obs: Observation, env: int, legal: np.ndarray) -> int:
        return int(np.flatnonzero(legal[: cfg.n_actions])[0])

    return choose
