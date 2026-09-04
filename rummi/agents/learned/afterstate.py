"""The state a macro leaves behind, computed without stepping the env.

Every tile-playing macro in :mod:`rummi.agents.macro` is deterministic given the
observation -- it lays known tiles into known slots and leaves the table whole --
so the position it reaches can be *scored* before it is chosen. That is what a
value-based learner needs and what the primitive action space could never give:
one network over positions, the policy being the argmax over the afterstates the
legal macros lead to, which is TD-Gammon's shape rather than an action head's.

Nothing about the rules is restated. The slot verdicts come from the env's own
:func:`~rummi.env.numpy.sets.summarize`, the columns are indexed by the ``F_*``
names in :mod:`rummi.rules.observation`, and the scaling is
:func:`~rummi.agents.learned.features.feature_scale`. What *is* mirrored is which
tiles a macro plays (:meth:`MacroAgent.expand`); where they land is not mirrored at
all but taken from :func:`rummi.solver.to_actions.allocate_slots`, the one place
that decides it. Being a mirror it can still drift, so `tests/test_afterstate.py`
plays real games and holds each prediction against the observation the env actually
reports at the next decision.

Two macros are deliberately not reconstructed.

``END_TURN`` and ``DRAW`` carry the *current* state plus their action flag. Their
successor is the opponent's reply and a fresh deal from the pool, neither of which
is a function of this state, and ``DRAW`` also reverts the whole turn (SPEC.md
section 4). Modelling the revert would be modelling one branch of a stochastic
successor exactly and the other not at all; the flag says which commitment is
being valued and the outcomes say what it is worth.

``REPARTITION`` is never scored, and needs no flag either. It is offered only
where nothing else plays, so taking it whenever it is legal is the whole of its
policy -- what ``by_value`` does with it, and ``by_value+repartition`` measures at
``optimal`` tier on that rule alone. A CP-SAT solve is not a position a value head
has to rank.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from rummi.agents.base import Observation, table
from rummi.agents.greedy_agent import appendable
from rummi.agents.learned.features import FEATURE_FIELDS, feature_dim, feature_scale
from rummi.agents.macro import MacroAgent, laid_tiles, removals
from rummi.env.numpy.sets import summarize
from rummi.rules.config import RummiConfig
from rummi.rules.encoding import EMPTY, tables
from rummi.rules.observation import (
    F_COLOR,
    F_EXTENDABLE,
    F_GROUP_VALID,
    F_HI,
    F_IS_NEW,
    F_JOKERS,
    F_LEN,
    F_LO,
    F_RUN_VALID,
    F_VALUE,
    MELD_PROGRESS,
    MELD_REMAINING,
    MICRO_COUNT,
    N_SCALARS,
    POOL_SIZE,
    SLOT_FEATURES,
)
from rummi.solver.to_actions import Content, allocate_slots, slot_contents

ACTION_KINDS = 3
"""Width of the action-kind one-hot appended to every afterstate row."""

KIND_PLAY = 0
"""Tiles went down and the turn continues, so the learner still has to move."""
KIND_END = 1
KIND_DRAW = 2

_SCALE: dict[RummiConfig, np.ndarray] = {}


def afterstate_dim(cfg: RummiConfig) -> int:
    """Width of one afterstate row: the observation's features plus the flag."""
    return feature_dim(cfg) + ACTION_KINDS


def _scale(cfg: RummiConfig) -> np.ndarray:
    """:func:`feature_scale`, cached: it is read once per decision per env."""
    cached = _SCALE.get(cfg)
    if cached is None:
        cached = _SCALE[cfg] = feature_scale(cfg)
    return cached


def _donors(cfg: RummiConfig, current: Sequence[Content]) -> dict[int, int]:
    """Kind -> the slot a ``STEAL`` takes it from.

    The lowest slot that can spare it, which is the one :meth:`MacroAgent.expand`
    picks: it scans the slots in index order.
    """
    out: dict[int, int] = {}
    for slot, content in enumerate(current):
        for kind in removals(cfg, content):
            out.setdefault(kind, slot)
    return out


def _resolve(
    cfg: RummiConfig,
    agent: MacroAgent,
    macro: int,
    current: Sequence[Content],
    rack: np.ndarray,
    appends: np.ndarray,
    donors: dict[int, int],
) -> tuple[np.ndarray, list[Content]]:
    """``(played, target sets)`` for one tile-playing macro.

    The hand is the rack alone: a decision happens only where the workbench is
    empty, because every macro finishes what it starts.
    """
    occupied = [content for content in current if content]

    if macro >= agent.steal_offset:
        template = agent.templates[macro - agent.steal_offset].astype(np.int64)
        gap = np.maximum(template - rack, 0)
        kind = int(gap.argmax())
        donor = donors[kind]
        left = list(current[donor])
        left.remove(kind)
        whole = tuple(sorted(np.repeat(np.arange(cfg.n_kinds), template).tolist()))
        # The stolen tile comes off the table, so the hand supplies the rest.
        played = np.minimum(template, rack)
        target = [
            tuple(sorted(left)) if slot == donor else content
            for slot, content in enumerate(current)
            if content
        ]
        return played, [*target, whole]

    if macro >= agent.extend_offset:
        kind = macro - agent.extend_offset
        slot = int(np.flatnonzero(appends[:, kind])[0])
        played = np.zeros(cfg.n_kinds, dtype=np.int64)
        played[kind] = 1
        return played, [
            tuple(sorted((*content, kind))) if i == slot else content
            for i, content in enumerate(current)
            if content
        ]

    template = agent.templates[macro].astype(np.int64)
    # A gap the rack cannot cover is stood in for by a joker, so what lands is the
    # laid set and not the template.
    laid = laid_tiles(cfg, template, rack)
    kinds = tuple(sorted(np.repeat(np.arange(cfg.n_kinds), laid).tolist()))
    return laid, [*occupied, kinds]


def _lay_out(
    cfg: RummiConfig,
    board: np.ndarray,
    current: Sequence[Content],
    target: Sequence[Content],
    n_place: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """The table the expansion reaches, the slots it creates, and what it spends.

    The allocation is read from :func:`~rummi.solver.to_actions.allocate_slots`
    rather than restated, because the observation reports slots by position and a
    disagreement about which slot a set lands in is a silent, shape-clean mismatch
    in every ``slot_features`` column.
    """
    alloc = allocate_slots(list(current), target)

    out = np.array(board, dtype=np.int16, copy=True)
    out[list(alloc.dissolve)] = EMPTY
    created = np.zeros(cfg.max_sets, dtype=bool)
    for slot, content in alloc.keep.items():
        out[slot] = _row(cfg, content)
    for content, slot in zip(alloc.fresh, alloc.free, strict=False):
        out[slot] = _row(cfg, content)
        # Only a slot that was empty counts as new, exactly as the engine's own
        # ASSIGN records it -- a set morphed in place keeps whatever it was.
        created[slot] = True

    # `micro_cost` excludes the trailing END_TURN, which is what `expand` drops
    # from every macro.
    return out, created, alloc.micro_cost(n_place)


def _row(cfg: RummiConfig, content: Content) -> np.ndarray:
    """One slot row: kinds ascending, ``EMPTY`` last, which is what the engine
    sorts a slot into after every ``ASSIGN``."""
    row = np.full(cfg.max_set_len, EMPTY, dtype=np.int16)
    row[: len(content)] = content
    return row


def afterstate_obs(
    cfg: RummiConfig,
    obs: Observation,
    env: int,
    macros: Sequence[int],
    agent: MacroAgent,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Per macro, the observation the env would report next, and its action kind.

    Batched on axis 0 over ``macros`` so the slot kernel runs once for the whole
    decision: :func:`summarize` over ``(n, S, L)`` costs barely more than over one
    table, and a decision offers tens of macros.

    Unscaled and per field, because that is what a mismatch is diagnosable in --
    :func:`afterstate_batch` is the flat, scaled view of the same thing.
    """
    if agent.repartition_macro is not None and agent.repartition_macro in macros:
        raise ValueError(
            "REPARTITION has no afterstate here: it is offered only where nothing "
            "else plays, so a chooser takes it unconditionally"
        )
    board = np.asarray(table(obs)[env]).astype(np.int16)
    rack = np.asarray(obs["rack"][env]).astype(np.int64)
    scalars = np.asarray(obs["scalars"][env]).astype(np.int64)
    current = slot_contents(board)

    n = len(macros)
    tables_out = np.repeat(board[None], n, axis=0)
    new_out = np.repeat(
        np.asarray(obs["slot_features"][env, :, F_IS_NEW]).astype(bool)[None], n, axis=0
    )
    racks = np.repeat(rack[None], n, axis=0)
    placed = np.repeat(
        np.asarray(obs["placed_this_turn"][env]).astype(np.int64)[None], n, axis=0
    )
    micro = np.full(n, scalars[MICRO_COUNT], dtype=np.int64)
    kinds = np.full(n, KIND_PLAY, dtype=np.int64)

    playing = [
        macro for macro in macros if macro not in (agent.end_macro, agent.draw_macro)
    ]
    # Both are per decision rather than per macro, and both are what `legal_macros`
    # itself gated on -- so the two cannot disagree about which slot is involved.
    appends = (
        appendable(cfg, board, rack)
        if any(agent.extend_offset <= macro < agent.steal_offset for macro in playing)
        else np.zeros((cfg.max_sets, cfg.n_kinds), dtype=bool)
    )
    donors = (
        _donors(cfg, current)
        if any(macro >= agent.steal_offset for macro in playing)
        else {}
    )

    for i, macro in enumerate(macros):
        if macro == agent.draw_macro:
            kinds[i] = KIND_DRAW
            continue
        if macro == agent.end_macro:
            kinds[i] = KIND_END
            continue
        played, target = _resolve(cfg, agent, macro, current, rack, appends, donors)
        tables_out[i], created, spent = _lay_out(cfg, board, current, target, int(played.sum()))
        new_out[i] |= created
        racks[i] -= played
        placed[i] += played
        micro[i] += spent

    summary = summarize(cfg, tables_out)
    stats, ev = summary.stats, summary.ev
    slot_features = np.zeros((n, cfg.max_sets, SLOT_FEATURES), dtype=np.int32)
    slot_features[..., F_LEN] = stats.n
    slot_features[..., F_RUN_VALID] = ev.run_valid
    slot_features[..., F_GROUP_VALID] = ev.group_valid
    slot_features[..., F_EXTENDABLE] = ev.is_extendable
    slot_features[..., F_COLOR] = stats.color
    slot_features[..., F_LO] = stats.lo
    slot_features[..., F_HI] = stats.hi
    slot_features[..., F_JOKERS] = stats.n_jokers
    slot_features[..., F_VALUE] = ev.value
    slot_features[..., F_IS_NEW] = new_out

    melded = np.repeat(np.asarray(obs["melded"][env]).astype(np.int8)[None], n, axis=0)
    # `masks.meld_value`, which cannot be called without a `BatchState`: under the
    # official rule the opening meld is the sets created this turn, and where the
    # table may be rearranged only the face value of what left the rack counts.
    if cfg.strict_initial_meld:
        progress = (ev.value * new_out).sum(-1).astype(np.int64)
    else:
        progress = placed @ tables(cfg).value.astype(np.int64)
    remaining = np.where(
        melded[:, 0].astype(bool), 0, np.maximum(0, cfg.initial_meld - progress)
    )

    rack_sizes = np.repeat(
        np.asarray(obs["rack_sizes"][env]).astype(np.int64)[None], n, axis=0
    )
    # Only the acting seat plays, and it is seat 0 of the rotated fields.
    rack_sizes[:, 0] = racks.sum(-1)

    # Written by name rather than stacked in order: the scalar block is four
    # unrelated quantities and only `rules.observation` says which is which.
    out_scalars = np.zeros((n, N_SCALARS), dtype=np.int64)
    out_scalars[:, POOL_SIZE] = scalars[POOL_SIZE]
    out_scalars[:, MELD_PROGRESS] = progress
    out_scalars[:, MELD_REMAINING] = remaining
    out_scalars[:, MICRO_COUNT] = micro

    return {
        "rack": racks,
        # Every macro lays down what it lifts, so a decision point is always clean.
        "workbench": np.zeros_like(racks),
        "placed_this_turn": placed,
        "unseen": np.repeat(np.asarray(obs["unseen"][env]).astype(np.int64)[None], n, axis=0),
        "slot_features": slot_features,
        "rack_sizes": rack_sizes,
        "melded": melded,
        "scalars": out_scalars,
        "table_sets": tables_out,
    }, kinds


def afterstate_view(fields: dict[str, np.ndarray], index: int) -> Observation:
    """Row ``index`` of :func:`afterstate_obs` as a one-env observation.

    Every field the encoder produces is there and the leading axis is kept, so an
    agent cannot tell this from what the env would hand it at that decision --
    which is what makes lookahead a recursion rather than a second model of the
    rules: :meth:`MacroAgent.legal_macros` runs on it, and its macros have
    afterstates of their own.
    """
    return {name: value[index : index + 1] for name, value in fields.items()}


def afterstate_rows(
    cfg: RummiConfig, fields: dict[str, np.ndarray], kinds: np.ndarray
) -> np.ndarray:
    """:func:`afterstate_obs`'s output, flattened and scaled as the network reads it.

    Split from :func:`afterstate_batch` so a caller that already holds the fields --
    a search reading them back as observations -- is not made to recompute them.
    """
    n = len(kinds)
    flat = np.concatenate(
        [fields[name].reshape(n, -1) for name in FEATURE_FIELDS], axis=-1
    ).astype(np.float32)
    flat /= _scale(cfg)
    out = np.zeros((n, afterstate_dim(cfg)), dtype=np.float32)
    out[:, : flat.shape[1]] = flat
    out[np.arange(n), flat.shape[1] + kinds] = 1.0
    return out


def afterstate_batch(
    cfg: RummiConfig,
    obs: Observation,
    env: int,
    macros: Sequence[int],
    agent: MacroAgent,
) -> np.ndarray:
    """``(len(macros), afterstate_dim)`` rows, scaled as the network reads them."""
    fields, kinds = afterstate_obs(cfg, obs, env, macros, agent)
    return afterstate_rows(cfg, fields, kinds)


def afterstate_features(
    cfg: RummiConfig,
    obs: Observation,
    env: int,
    macro: int,
    agent: MacroAgent,
) -> np.ndarray:
    """``(afterstate_dim,)`` for one macro. The batch of one, so there is one path."""
    return afterstate_batch(cfg, obs, env, [macro], agent)[0]
