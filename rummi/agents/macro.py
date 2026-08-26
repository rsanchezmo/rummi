"""One decision per *set*, not per tile: play a complete set, end the turn, or draw.

This exists because of what the primitive action space does to a learner, measured
on the standard config: `PLACE` moves a tile rack -> workbench and `END_TURN`
requires the workbench empty, so a policy that lifts tiles forming no valid set has
killed the turn -- revert is the only exit, and no teacher can label its way out.
A cloned-then-PPO'd policy spent a median of 7 and a mean of 17 micro-actions per
turn doing exactly that, against greedy's median of 1, and reached a committable
table in 0.7% of its steps against greedy's 14.4%.

Every action here leaves the table **whole**, so those states are unreachable
rather than merely discouraged. Decisions happen whenever the workbench is empty,
which is once per set played rather than once per turn, so multi-set turns stay
expressible and `END_TURN` is a real choice.

**No rules change.** The 2400-action space, `SPEC.md`, the golden fixtures and
`PROTOCOL_VERSION` are all untouched: an agent may offer itself macro actions and
expand them into primitives, exactly as `PlanningAgent` does with a solver's plan.
`rummi/solver/to_actions.plan` is that expansion, and it needs no solve -- what
makes `optimal` expensive is CP-SAT *choosing* a target, and here the policy
chooses it.

The templates are the tractable fragment of a whole-turn action space. Enumerating
achievable *turns* is the NP-hard partition problem, but enumerating the sets that
exist at all is a fixed list -- 264 runs and 65 groups on the standard config --
and asking which of them the rack can cover is one matrix comparison.

Jokers are not substituted: a template is playable only when the rack holds its
exact tiles. Standing in a joker for a missing tile means the set that lands on the
table differs from the template, which the expansion would have to represent, so
it is a later extension rather than a silent approximation.
"""

from __future__ import annotations

from collections.abc import Callable
from itertools import combinations

import numpy as np

from rummi.agents.base import Observation, has_melded, table, turn_starting
from rummi.rules.config import RummiConfig
from rummi.rules.encoding import kind_of

_TEMPLATES: dict[RummiConfig, np.ndarray] = {}


def set_templates(cfg: RummiConfig) -> np.ndarray:
    """`(T, n_kinds)` counts, one row per set that could ever be legal.

    NumPy and built outside any trace, per the backend trap in `CLAUDE.md`: a
    lookup table first populated inside a JAX trace caches tracers.
    """
    cached = _TEMPLATES.get(cfg)
    if cached is not None:
        return cached

    rows: list[list[int]] = []
    for colour in range(cfg.n_colors):
        for length in range(cfg.min_set, cfg.max_set_len + 1):
            for start in range(1, cfg.n_numbers - length + 2):
                rows.append([kind_of(cfg, colour, start + i) for i in range(length)])
    for number in range(1, cfg.n_numbers + 1):
        for size in range(cfg.min_set, cfg.n_colors + 1):
            for colours in combinations(range(cfg.n_colors), size):
                rows.append([kind_of(cfg, c, number) for c in colours])

    out = np.zeros((len(rows), cfg.n_kinds), dtype=np.int16)
    for i, kinds in enumerate(rows):
        for k in kinds:
            out[i, k] += 1
    _TEMPLATES[cfg] = out
    return out


def template_points(cfg: RummiConfig) -> np.ndarray:
    """`(T,)` face value of each template, for the opening-meld threshold."""
    numbers = np.array(
        [
            (k % cfg.n_numbers) + 1 if k != cfg.joker_kind else 0
            for k in range(cfg.n_kinds)
        ],
        dtype=np.int32,
    )
    return (set_templates(cfg) * numbers).sum(-1).astype(np.int32)


def playable(cfg: RummiConfig, rack: np.ndarray) -> np.ndarray:
    """`(B, T)` -- which templates a rack holds outright.

    One comparison over the whole template table rather than a search, which is
    what keeps this out of the NP-hard partitioning problem.
    """
    covered = np.asarray(rack)[:, None, :] >= set_templates(cfg)[None]
    return np.asarray(covered.all(-1), dtype=bool)


Choose = Callable[[Observation, int, np.ndarray], int]
"""`(obs, env, legal) -> macro index`, where `legal` is the `(n_macros,)` mask."""


class MacroAgent:
    """Actions are `T` set templates, then `END_TURN`, then `DRAW`."""

    name = "macro"

    def __init__(self, cfg: RummiConfig, choose: Choose | None = None) -> None:
        self.cfg = cfg
        self.templates = set_templates(cfg)
        self.n_macros = len(self.templates) + 2
        self.end_macro = len(self.templates)
        self.draw_macro = len(self.templates) + 1
        self.choose = choose if choose is not None else first_legal
        self._queues: dict[int, list[int]] = {}

    def reset(self, n_envs: int) -> None:
        self._queues = {}

    def legal_macros(self, obs: Observation, env: int, mask: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        out = np.zeros(self.n_macros, dtype=bool)
        board = table(obs)[env]
        if (board.max(-1) < 0).any():  # a free slot to put the set in
            out[: self.end_macro] = playable(cfg, obs["rack"][env][None])[0]
        # Pre-meld the threshold is on the whole turn, so a single template that
        # cannot reach it is still worth playing alongside another.
        out[self.end_macro] = bool(mask[env, cfg.end_turn_action])
        out[self.draw_macro] = True  # DRAW is never masked, by design
        return out

    def expand(self, obs: Observation, env: int, macro: int) -> list[int]:
        """Micro-actions for one macro. Empty means DRAW."""
        from rummi.solver.to_actions import plan, slot_contents

        if macro == self.draw_macro:
            return []
        if macro == self.end_macro:
            return [self.cfg.end_turn_action]

        board = table(obs)[env]
        played = self.templates[macro].astype(np.int64)
        kinds = tuple(sorted(np.repeat(np.arange(self.cfg.n_kinds), played).tolist()))
        target = [c for c in slot_contents(board) if c] + [kinds]
        return plan(self.cfg, board, target, played)

    def act(
        self, obs: Observation, mask: np.ndarray, active: np.ndarray | None = None
    ) -> np.ndarray:
        cfg = self.cfg
        n = mask.shape[0]
        out = np.full(n, cfg.draw_action, dtype=np.int64)
        fresh = turn_starting(obs)

        for env in range(n):
            if active is not None and not active[env]:
                continue
            if fresh[env]:
                self._queues[env] = []
            queue = self._queues.setdefault(env, [])
            if not queue:
                # An empty queue is a clean decision point: the table is whole and
                # nothing is held, because every macro finishes what it starts.
                macro = self.choose(obs, env, self.legal_macros(obs, env, mask))
                queue = self._queues[env] = self.expand(obs, env, macro)
            if not queue:
                continue
            action = queue.pop(0)
            # A stale expansion is abandoned rather than forced: DRAW reverts the
            # turn cleanly and is always legal.
            if not mask[env, action]:
                queue.clear()
                continue
            out[env] = action
        return out


def first_legal(obs: Observation, env: int, legal: np.ndarray) -> int:
    """Play the first playable set, else end the turn, else draw.

    A deterministic stand-in for a learned policy, and a floor to beat: it never
    chooses *which* set, which is the decision the whole action space exists for.
    """
    return int(np.flatnonzero(legal)[0])


def by_value(cfg: RummiConfig) -> Choose:
    """Highest-scoring set first before the opening meld, most tiles after.

    The floor that `first_legal` should be measured against, and the reason it is
    worth having: playing the cheapest legal set before melding is how a policy
    reverts every turn without ever reaching the threshold.
    """
    points = template_points(cfg)
    tiles = set_templates(cfg).sum(-1).astype(np.int32)
    end = len(set_templates(cfg))

    def choose(obs: Observation, env: int, legal: np.ndarray) -> int:
        options = np.flatnonzero(legal[:end])
        if options.size:
            rank = points if not bool(has_melded(obs)[env]) else tiles
            return int(options[np.argmax(rank[options])])
        return int(np.flatnonzero(legal)[0])

    return choose


def melded_only(cfg: RummiConfig) -> Choose:
    """`first_legal`, but it ends the turn as soon as ending is legal.

    Separates "can it build sets" from "does it know when to stop", which the
    diagnostics say are different failures.
    """
    agent_end = len(set_templates(cfg))

    def choose(obs: Observation, env: int, legal: np.ndarray) -> int:
        if legal[agent_end] and bool(has_melded(obs)[env]):
            return agent_end
        return int(np.flatnonzero(legal)[0])

    return choose
