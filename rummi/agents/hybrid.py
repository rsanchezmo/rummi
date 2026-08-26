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

**As written this does not work, and the reason is the rule below.** Measured under
an untrained policy: the workbench is dirty in **94.5%** of decisions, `END_TURN` is
legal in **0.5%** (against ~9% in the macro space), and **a macro is on offer in
4.9%**. So the escape hatch is shut exactly when it is needed, training collapses to
stalling (`end` 0.0%, terminal ~-1.03, entropy falling *into* the stalling policy),
and no reward term helps: a bigger batch, `--rack-shaping` and `--micro-step-cost`
were each tried and cannot make an illegal action attractive.

**The fix is to let a macro consume the held tiles** -- judge `playable` against
`rack + workbench` with the workbench required to be used, and teach
`to_actions.plan` to account for tiles already held rather than only for what leaves
the rack. Until then the primitive trap is fully intact, and the claim that the
half-built workbench is "avoidable" is only true of a policy that never takes a
primitive.

**Macros require a clean state.** `to_actions.plan` balances the board against the
tiles played from the rack, so tiles already sitting on the workbench are
unaccounted for and it would refuse the plan. Once a primitive has dirtied the
state, only primitives are on offer until the turn is whole again -- which is also
the honest shape of the trade the agent is making.
"""

from __future__ import annotations

import numpy as np

from rummi.agents.base import Observation
from rummi.agents.macro import Choose, MacroAgent, _n_choices, action_features
from rummi.rules.config import RummiConfig

_FEATURES: dict[RummiConfig, np.ndarray] = {}


def n_actions(cfg: RummiConfig) -> int:
    """Primitives, then the tile-playing macros. `END_TURN` and `DRAW` are already
    primitives, so the macro block stops short of them."""
    return cfg.n_actions + _n_choices(cfg)


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
        # A macro's expansion accounts for the board and the rack only, so it is
        # offered exactly when nothing is held mid-turn.
        if int(np.asarray(obs["workbench"])[env].sum()) == 0:
            out[self.macro_offset :] = self.macro.legal_macros(obs, env, mask)[
                : self.n_macro
            ]
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
                    obs, env, action - self.macro_offset
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
    from rummi.agents.macro import by_value, n_macros

    rank = by_value(cfg)
    n_macro = _n_choices(cfg)
    total = n_macros(cfg)

    def choose(obs: Observation, env: int, legal: np.ndarray) -> int:
        # `by_value` expects the macro-space mask, whose last two entries are
        # END_TURN and DRAW; here those live in the primitive block.
        padded = np.zeros(total, dtype=bool)
        padded[:n_macro] = legal[cfg.n_actions :]
        padded[n_macro] = bool(legal[cfg.end_turn_action])
        padded[n_macro + 1] = True
        macro = rank(obs, env, padded)
        if macro < n_macro:
            return cfg.n_actions + macro
        return cfg.end_turn_action if macro == n_macro else cfg.draw_action

    return choose


def primitives_only(cfg: RummiConfig) -> Choose:
    """Ignore the macro block entirely, for measuring what it is worth."""

    def choose(obs: Observation, env: int, legal: np.ndarray) -> int:
        return int(np.flatnonzero(legal[: cfg.n_actions])[0])

    return choose
