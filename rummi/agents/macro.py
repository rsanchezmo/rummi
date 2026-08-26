"""One decision per *move*, not per tile: play a set, lay off, steal, end, or draw.

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

A joker stands in for at most **one** missing tile of a template. Two would be
legal Rummikub and are refused, because with two gaps the pairing of jokers to gaps
stops being determined and `expand` would have to choose.

Once a joker is *on the table* nothing has to name its role: `EXTEND` asks
`greedy_agent.appendable`, which grows the slot row and validates it through the
env's own `evaluate_slots`, so a set holding a joker takes tiles and the rack's own
joker lays off wherever there is room -- together every state where
`tools/diagnose_stuck.py` found `greedy` playing and this space only drawing. `STEAL`
still refuses joker-holding sets, because it decides what a set can spare from the
real numbers alone and a joker leaves that undetermined: `(R1,R2,R3,*)` can spare
`R3`, which the arithmetic calls a middle tile, because the joker slides down to
cover it. Answering that through `evaluate_slots` too is a different move to
describe -- a steal also has to name the donor -- so it is left as it is.
"""

from __future__ import annotations

from collections.abc import Callable
from itertools import combinations

import numpy as np

from rummi.agents.base import Observation, has_melded, table, turn_starting
from rummi.agents.greedy_agent import appendable
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


def removals(cfg: RummiConfig, contents: tuple[int, ...]) -> list[int]:
    """Kinds that can leave this set with what remains still a legal set.

    The inverse of `appendable`, and the whole of what `rearrange` does: steal one
    tile. A run gives up either end while it stays at least `min_set` long; a group
    gives up any member while it does. Middle tiles of a run are refused -- removing
    one splits the set in two, which needs a second free slot and is a different
    move.
    """
    if not contents or cfg.joker_kind in contents or len(contents) - 1 < cfg.min_set:
        return []

    colours = {k // cfg.n_numbers for k in contents}
    numbers = sorted(k % cfg.n_numbers + 1 for k in contents)
    if len(colours) == 1:
        colour = next(iter(colours))
        return [kind_of(cfg, colour, numbers[0]), kind_of(cfg, colour, numbers[-1])]
    if len(set(numbers)) == 1 and len(colours) == len(contents):
        return list(contents)
    return []


_FEATURES: dict[RummiConfig, np.ndarray] = {}

ACTION_FEATURE_DIM = 4
"""Beyond the per-kind counts: points, size, tiles taken off the table, and which
block the action belongs to is one-hot on top of that."""


def action_features(cfg: RummiConfig) -> np.ndarray:
    """`(n_macros, d)` description of what each macro *does*, as data.

    A flat head has to learn action 147 from its index alone, so nothing it learns
    about one set transfers to a similar one. Scoring an action against its own
    description shares that instead -- the same argument as the bilinear head in
    `learned/architecture.py`, and a much better fit here, because a macro has a
    real description and an `ASSIGN` id does not.

    `EXTEND` rows carry only their block and slot: which tile they add is
    state-dependent, so it cannot live in a static table. Those 2*max_sets rows
    lean on the per-action bias instead.
    """
    cached = _FEATURES.get(cfg)
    if cached is not None:
        return cached

    templates = set_templates(cfg)
    points = template_points(cfg)
    n_kinds, n_sets = cfg.n_kinds, len(templates)
    scale = float(cfg.n_numbers * cfg.max_set_len)
    out = np.zeros((n_macros(cfg), n_kinds + ACTION_FEATURE_DIM + 4), dtype=np.float32)

    def rows(offset: int, block: int, from_table: float) -> None:
        for t in range(n_sets):
            row = out[offset + t]
            row[:n_kinds] = templates[t]
            row[n_kinds] = points[t] / scale
            row[n_kinds + 1] = templates[t].sum() / cfg.max_set_len
            row[n_kinds + 2] = from_table
            row[n_kinds + ACTION_FEATURE_DIM + block] = 1.0

    rows(0, 0, 0.0)
    rows(steal_offset(cfg), 2, 1.0)
    values = tables(cfg).value
    for kind in range(n_kinds):
        row = out[extend_offset(cfg) + kind]
        row[kind] = 1.0  # exactly which tile this lays off
        row[n_kinds] = values[kind] / scale
        row[n_kinds + 1] = 1.0 / cfg.max_set_len  # a lay-off plays one tile
        row[n_kinds + ACTION_FEATURE_DIM + 1] = 1.0
    for macro in (_n_choices(cfg), _n_choices(cfg) + 1):
        out[macro, n_kinds + ACTION_FEATURE_DIM + 3] = 1.0

    _FEATURES[cfg] = out
    return out


Choose = Callable[[Observation, int, np.ndarray], int]
"""`(obs, env, legal) -> macro index`, where `legal` is the `(n_macros,)` mask."""


class MacroAgent:
    """Actions are `T` set templates, then `END_TURN`, then `DRAW`."""

    name = "macro"

    def __init__(self, cfg: RummiConfig, choose: Choose | None = None) -> None:
        self.cfg = cfg
        self.templates = set_templates(cfg)
        # templates, then EXTEND(kind), then STEAL(template), then END_TURN, then DRAW.
        self.extend_offset = extend_offset(cfg)
        self.n_extend = cfg.n_kinds
        self.steal_offset = steal_offset(cfg)
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
            # Indexed by tile, so a kind is legal when *any* slot takes it; `expand`
            # reads the same matrix to pick which one.
            out[self.extend_offset : self.steal_offset] = appendable(cfg, board, rack).any(0)

            # Stealing dissolves the donor and rebuilds it beside the new set, so it
            # needs one free slot on top of the donor's own.
            if (board.max(-1) < 0).any():
                stealable = np.zeros(cfg.n_kinds, dtype=bool)
                for contents in slot_contents(board):
                    for kind in removals(cfg, contents):
                        stealable[kind] = True
                gap = np.maximum(self.templates - rack, 0)
                short = gap.sum(-1)
                # `argmax` is the missing kind only where exactly one is missing.
                out[self.steal_offset : self.end_macro] = (short == 1) & stealable[
                    gap.argmax(-1)
                ]

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

        if macro >= self.steal_offset:
            rack = np.asarray(obs["rack"][env])
            template = self.templates[macro - self.steal_offset].astype(np.int64)
            gap = np.maximum(template - rack, 0)
            kind = int(gap.argmax())
            donor = next(
                slot for slot, c in enumerate(current) if c and kind in removals(cfg, c)
            )
            left = list(current[donor])
            left.remove(kind)
            whole = tuple(sorted(np.repeat(np.arange(cfg.n_kinds), template).tolist()))
            played = np.minimum(template, rack).astype(np.int64)
            target = [
                tuple(sorted(left)) if slot == donor else c
                for slot, c in enumerate(current)
                if c
            ] + [whole]
        elif macro >= self.extend_offset:
            kind = macro - self.extend_offset
            rack = np.asarray(obs["rack"][env])
            # Indexed by tile rather than by slot, so the receiving set is whichever
            # takes it. Choosing between two sets that both accept the same tile is
            # given up deliberately: it makes the action describable to a policy --
            # its tile is known -- where a slot index says nothing about what is
            # played. Same matrix `legal_macros` gated on, so the two cannot disagree
            # about which slot that is.
            slot = int(np.flatnonzero(appendable(cfg, board, rack)[:, kind])[0])
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
    """Where `EXTEND(kind)` starts; below it, templates lay down new sets.

    One entry per kind, the joker included. Laying a tile onto a set already on the
    table is **84% of what greedy does** -- measured over its ASSIGNs -- so this block
    is the commonest move in the game and what a macro space without it cannot say.
    """
    return len(set_templates(cfg))


def steal_offset(cfg: RummiConfig) -> int:
    """Where `STEAL(template)` starts: the same templates, but one of the tiles is
    taken off the table instead of out of the rack."""
    return extend_offset(cfg) + cfg.n_kinds


def _n_choices(cfg: RummiConfig) -> int:
    """Index of the `END_TURN` macro: everything below it plays tiles."""
    return steal_offset(cfg) + len(set_templates(cfg))


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
    end = _n_choices(cfg)
    n_extend = steal_offset(cfg) - extend_offset(cfg)
    set_points = template_points(cfg)
    set_tiles = set_templates(cfg).sum(-1).astype(np.int32)
    # A lay-off is worth the value of the tile it sheds. `tables().value` scores the
    # joker 0, because its face value is positional -- what it costs to keep is
    # `joker_penalty`, which is what shedding it saves, exactly as greedy ranks it.
    layoff_points = tables(cfg).value.astype(np.int32)[:n_extend].copy()
    layoff_points[cfg.joker_kind] = cfg.joker_penalty
    # A lay-off plays exactly one tile, and a steal sheds one fewer than the same
    # set played outright, because one of its tiles comes off the table.
    points = np.concatenate([set_points, layoff_points, set_points])
    tiles = np.concatenate([
        set_tiles, np.ones(n_extend, np.int32), np.maximum(set_tiles - 1, 1)
    ])
    assert len(points) == end and len(tiles) == end

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
