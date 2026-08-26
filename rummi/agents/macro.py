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
which is once per set played rather than once per turn: `END_TURN` is a choice
alongside playing another set, and a turn may hold several. `expand` drops the
`END_TURN` that `to_actions.plan` appends for exactly that reason.

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
from rummi.rules.encoding import kind_of, tables

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
    """`(T,)` face value of each template, for the opening-meld threshold.

    Values come from `rules.encoding`, which owns them, rather than being derived
    from the kind index here -- that is the sort of definitional arithmetic the
    three backends are kept from restating.
    """
    return (set_templates(cfg) * tables(cfg).value).sum(-1).astype(np.int32)


def shortfall(cfg: RummiConfig, rack: np.ndarray) -> np.ndarray:
    """`(B, T)` tiles each template needs that the rack does not hold.

    One comparison over the whole template table rather than a search, which is
    what keeps this out of the NP-hard partitioning problem.
    """
    missing = set_templates(cfg)[None] - np.asarray(rack)[:, None, :]
    # Templates never contain the joker, so its own column contributes nothing.
    return np.maximum(missing, 0).sum(-1)


def playable(cfg: RummiConfig, rack: np.ndarray) -> np.ndarray:
    """`(B, T)` -- which templates the rack can lay down, jokers included.

    A joker stands in for **one** missing tile. Two would be legal Rummikub and
    are refused here, because with two gaps the pairing of jokers to gaps stops
    being determined and `expand` would have to choose one.
    """
    rack = np.asarray(rack)
    short = shortfall(cfg, rack)
    jokers = rack[:, cfg.joker_kind][:, None]
    return np.asarray((short == 0) | ((short == 1) & (jokers >= 1)), dtype=bool)


def extensions(cfg: RummiConfig, contents: tuple[int, ...], rack: np.ndarray) -> list[int]:
    """Up to two kinds in `rack` that legally extend this set, by end; -1 for none.

    Laying a tile onto a set already on the table is **84% of what greedy does** --
    measured over its ASSIGNs -- so a macro space without it cannot express the
    commonest move in the game, whatever its policy.

    A set with a legal tile added is still a legal set, so this preserves the
    invariant the whole action space rests on: the table stays whole.
    """
    if not contents or cfg.joker_kind in contents or len(contents) >= cfg.max_set_len:
        # A joker's role in a set is ambiguous, so sets holding one are left alone,
        # for the same reason templates do not substitute them.
        return [-1, -1]

    colours = {k // cfg.n_numbers for k in contents}
    numbers = sorted(k % cfg.n_numbers + 1 for k in contents)
    out: list[int] = []
    if len(colours) == 1:
        colour = next(iter(colours))
        for number in (numbers[0] - 1, numbers[-1] + 1):
            k = kind_of(cfg, colour, number) if 1 <= number <= cfg.n_numbers else -1
            out.append(k if k >= 0 and rack[k] > 0 else -1)
    elif len(set(numbers)) == 1 and len(colours) == len(contents):
        missing = [c for c in range(cfg.n_colors) if c not in colours]
        for colour in missing[:2]:
            k = kind_of(cfg, colour, numbers[0])
            out.append(k if rack[k] > 0 else -1)
    return [*out, -1, -1][:2]


Choose = Callable[[Observation, int, np.ndarray], int]
"""`(obs, env, legal) -> macro index`, where `legal` is the `(n_macros,)` mask."""


class MacroAgent:
    """Actions are `T` set templates, then `END_TURN`, then `DRAW`."""

    name = "macro"

    def __init__(self, cfg: RummiConfig, choose: Choose | None = None) -> None:
        self.cfg = cfg
        self.templates = set_templates(cfg)
        # templates, then EXTEND(slot, end), then END_TURN, then DRAW.
        self.extend_offset = extend_offset(cfg)
        self.n_extend = 2 * cfg.max_sets
        self.end_macro = _n_choices(cfg)
        self.draw_macro = self.end_macro + 1
        self.n_macros = n_macros(cfg)
        self.choose = choose if choose is not None else first_legal
        self._queues: dict[int, list[int]] = {}

    def reset(self, n_envs: int) -> None:
        self._queues = {}

    def legal_macros(self, obs: Observation, env: int, mask: np.ndarray) -> np.ndarray:
        from rummi.solver.to_actions import slot_contents

        cfg = self.cfg
        out = np.zeros(self.n_macros, dtype=bool)
        board = table(obs)[env]
        rack = np.asarray(obs["rack"][env])
        if (board.max(-1) < 0).any():  # a free slot to put a new set in
            out[: self.extend_offset] = playable(cfg, rack[None])[0]

        # The table is untouchable until the opening meld, exactly as the env's own
        # mask has it -- so laying off is illegal there, not merely unwise.
        if bool(has_melded(obs)[env]) or not cfg.strict_initial_meld:
            for slot, contents in enumerate(slot_contents(board)):
                for end, kind in enumerate(extensions(cfg, contents, rack)):
                    out[self.extend_offset + slot * 2 + end] = kind >= 0

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

        cfg = self.cfg
        board = table(obs)[env]
        current = slot_contents(board)

        if macro >= self.extend_offset:
            slot, end = divmod(macro - self.extend_offset, 2)
            kind = extensions(cfg, current[slot], np.asarray(obs["rack"][env]))[end]
            played = np.zeros(cfg.n_kinds, dtype=np.int64)
            played[kind] = 1
            target = [
                tuple(sorted((*c, kind))) if i == slot else c
                for i, c in enumerate(current)
                if c
            ]
        else:
            rack = np.asarray(obs["rack"][env])
            template = self.templates[macro].astype(np.int64)
            # Whatever the rack cannot cover is stood in for by a joker, so the set
            # that lands holds the joker and the rack loses it instead of the tile.
            played = np.minimum(template, rack).astype(np.int64)
            gap = np.maximum(template - rack, 0)
            laid = template - gap
            if gap.any():
                played[cfg.joker_kind] += int(gap.sum())
                laid[cfg.joker_kind] += int(gap.sum())
            kinds = tuple(sorted(np.repeat(np.arange(cfg.n_kinds), laid).tolist()))
            target = [c for c in current if c] + [kinds]
        actions = plan(cfg, board, target, played)
        # `plan` commits the turn, because it exists for a solver that decides a
        # whole turn at once. Here the turn is not over: the set is complete, so the
        # table is whole and ending is *legal*, but whether to end it or play
        # another set is the next decision. Leaving this in caps every turn at one
        # set, which before the opening meld means finding 30 points in a single one.
        if actions and actions[-1] == self.cfg.end_turn_action:
            actions.pop()
        return actions

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


def extend_offset(cfg: RummiConfig) -> int:
    """Where `EXTEND(slot, end)` starts; below it, templates lay down new sets."""
    return len(set_templates(cfg))


def _n_choices(cfg: RummiConfig) -> int:
    """Index of the `END_TURN` macro: everything below it plays tiles."""
    return extend_offset(cfg) + 2 * cfg.max_sets


def n_macros(cfg: RummiConfig) -> int:
    """Size of the macro action space. Derive the layout from here, never restate
    it: a caller that recomputed this built a head of the wrong width."""
    return _n_choices(cfg) + 2


def first_legal(obs: Observation, env: int, legal: np.ndarray) -> int:
    """Play the first playable set, else end the turn, else draw.

    A deterministic stand-in for a learned policy, and the floor to beat. It never
    chooses *which* set, and measured against `by_value` that is worth surprisingly
    little -- -147.4 against -141.5 -- because a turn may play several sets, so a
    cheap first pick is recoverable.
    """
    return int(np.flatnonzero(legal)[0])


def by_value(cfg: RummiConfig) -> Choose:
    """Highest-scoring set first before the opening meld, most tiles after.

    Worth about 6 points of mean score over `first_legal`, measured. The ordering
    matters most before the opening meld, where the threshold is on the whole turn:
    a dear set reaches 30 in fewer plays.
    """
    n_templates = len(set_templates(cfg))
    end = _n_choices(cfg)
    # A lay-off plays exactly one tile, so it ranks below any new set on tiles shed
    # and is only preferred when nothing else is available.
    points = np.concatenate([template_points(cfg), np.zeros(end - n_templates, np.int32)])
    tiles = np.concatenate([
        set_templates(cfg).sum(-1).astype(np.int32),
        np.ones(end - n_templates, np.int32),
    ])

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
    agent_end = _n_choices(cfg)

    def choose(obs: Observation, env: int, legal: np.ndarray) -> int:
        if legal[agent_end] and bool(has_melded(obs)[env]):
            return agent_end
        return int(np.flatnonzero(legal)[0])

    return choose
