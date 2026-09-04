"""Turn a target table into the micro-actions that reach it.

This is where the solver meets the action space, and it matters for more than the
optimal policy: if every table CP-SAT can propose is reachable through
``PLACE``/``PICK``/``DISSOLVE``/``ASSIGN``, then the micro-action decomposition is
*complete* -- no legal turn is expressible in the rules but not in the MDP. The
tests assert exactly that.

**A standing set is morphed rather than rebuilt.** Turn length is the one cost the
solver's own objective cannot see, and the commonest move in the game is a lay-off:
84.2% of ``greedy``'s ASSIGNs land on a slot that already holds tiles. Dissolving
the receiving set and rebuilding it renders one tile as seven actions where the
mask allows three, so a target that is a *superset* of a standing slot is reached
by ``ASSIGN``ing the difference onto it and a *subset* by ``PICK``ing the surplus
off. Only a target that is neither -- or a subset so much smaller that picking
costs more than rebuilding -- dissolves.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from rummi.rules.actions import encode_assign, encode_dissolve, encode_pick, encode_place
from rummi.rules.config import RummiConfig
from rummi.rules.encoding import counts_to_kinds

Content = tuple[int, ...]


def slot_contents(table: np.ndarray) -> list[Content]:
    """``(S, L)`` table to one sorted tuple per slot; empty slots give ``()``."""
    return [tuple(sorted(int(k) for k in row if k >= 0)) for row in table]


@dataclass(frozen=True, slots=True)
class Allocation:
    """Which slot each target set ends up in, and how each one gets there.

    Separate from the actions because
    :func:`rummi.agents.learned.afterstate._lay_out` needs the table this reaches
    without emitting one, and the observation reports slots by position -- so the
    decision lives here once and both readers take it from the same place.
    """

    current: tuple[Content, ...]
    keep: dict[int, Content]
    """Standing slot -> the target it becomes, in place."""
    dissolve: tuple[int, ...]
    fresh: tuple[Content, ...]
    """Targets no standing slot serves, in the order ``free`` receives them."""
    free: tuple[int, ...]
    """Slots a fresh set may be built in, lowest first: the mask offers only the
    lowest empty one, so a set must be finished before the next can be started."""

    def removals(self, slot: int) -> list[int]:
        """Kinds to ``PICK`` off ``slot``, ascending."""
        return sorted((Counter(self.current[slot]) - Counter(self.keep[slot])).elements())

    def additions(self, slot: int) -> list[int]:
        """Kinds to ``ASSIGN`` onto ``slot``, ascending."""
        return sorted((Counter(self.keep[slot]) - Counter(self.current[slot])).elements())

    def micro_cost(self, n_place: int) -> int:
        """Actions this allocation spends, the trailing ``END_TURN`` excluded."""
        moved = sum(
            len(self.removals(slot)) + len(self.additions(slot)) for slot in self.keep
        )
        return (
            len(self.dissolve)
            + n_place
            + moved
            + sum(len(content) for content in self.fresh)
        )


def _saving(content: Content, target: Content) -> int | None:
    """Actions saved by morphing ``content`` into ``target``, or ``None`` if none are.

    Rebuilding costs a ``DISSOLVE`` plus an ``ASSIGN`` per tile of the target;
    morphing costs one action per tile of the difference, which has to run in a
    single direction for the mask to allow it -- a slot that needs both a removal
    and an addition is a different set, not a longer or a shorter one.
    """
    add = Counter(target) - Counter(content)
    drop = Counter(content) - Counter(target)
    if add and drop:
        return None
    saved = 1 + len(target) - sum(add.values()) - sum(drop.values())
    return saved if saved > 0 else None


def _morph(
    current: list[Content], wanted: Counter[Content], taken: set[int]
) -> list[tuple[int, Content]]:
    """Standing slots paired with the target they become, most saved first.

    Greedy over the pairs rather than optimal over the assignment: the ranking is
    what decides which of two slots serves a target, and the alternative is a
    matching problem solved once per macro expansion for the sake of a tile or two.
    """
    options: list[tuple[int, int, Content]] = []
    for slot, content in enumerate(current):
        if not content or slot in taken:
            continue
        for target, count in wanted.items():
            if count <= 0:
                continue
            saved = _saving(content, target)
            if saved is not None:
                options.append((-saved, slot, target))
    options.sort()
    left = Counter(wanted)
    used = set(taken)
    out: list[tuple[int, Content]] = []
    for _, slot, target in options:
        if slot in used or left[target] <= 0:
            continue
        used.add(slot)
        left[target] -= 1
        out.append((slot, target))
    return out


def allocate_slots(current: list[Content], target_sets) -> Allocation:
    """Match the standing slots to ``target_sets``: keep, morph, or dissolve.

    Exact matches are claimed first, so a set the target wants exactly where it
    already stands can never be spent on a longer one that would then have to be
    built somewhere else.
    """
    wanted: Counter[Content] = Counter(tuple(sorted(s)) for s in target_sets)
    keep: dict[int, Content] = {}
    for slot, content in enumerate(current):
        if content and wanted[content] > 0:
            wanted[content] -= 1
            keep[slot] = content
    for slot, target in _morph(current, wanted, set(keep)):
        wanted[target] -= 1
        keep[slot] = target

    dissolve = [slot for slot, content in enumerate(current) if content and slot not in keep]
    free = sorted([slot for slot, content in enumerate(current) if not content] + dissolve)
    return Allocation(
        current=tuple(current),
        keep=keep,
        dissolve=tuple(dissolve),
        fresh=tuple(wanted.elements()),
        free=tuple(free),
    )


def plan(
    cfg: RummiConfig,
    table: np.ndarray,
    target_sets,
    played: np.ndarray,
    held: np.ndarray | None = None,
) -> list[int]:
    """Micro-actions taking ``table`` to ``target_sets`` while playing ``played``.

    ``played`` counts tiles still in the rack, so each one costs a ``PLACE`` and
    then an ``ASSIGN``. ``held`` counts tiles already on the workbench: they are
    lifted already, so they are consumed by the target but never placed again.
    Every removal precedes every ``PLACE`` and every ``PLACE`` every ``ASSIGN``,
    which is why a held tile needs no separate treatment beyond being left out of
    the placing.

    Raises if the request is inconsistent -- a target that does not account for
    every tile in play is a solver bug, and silently emitting a partial plan
    would surface it much later as an illegal action. That check is also what
    guarantees a held tile the target does not want is refused rather than
    stranded on the workbench.
    """
    current = slot_contents(table)
    alloc = allocate_slots(current, target_sets)

    # Whatever comes off the table, plus whatever leaves the rack, plus whatever is
    # already in hand must be exactly what the table takes back.
    freed: Counter[int] = Counter()
    for slot in alloc.dissolve:
        freed.update(current[slot])
    for slot in alloc.keep:
        freed.update(alloc.removals(slot))
    freed.update(int(k) for k in counts_to_kinds(np.maximum(0, played)))
    if held is not None:
        freed.update(int(k) for k in counts_to_kinds(np.maximum(0, held)))
    needed: Counter[int] = Counter()
    for content in alloc.fresh:
        needed.update(content)
    for slot in alloc.keep:
        needed.update(alloc.additions(slot))
    if freed != needed:
        raise ValueError(
            f"plan does not balance: dissolved+picked+played+held={dict(freed)} but "
            f"the table needs {dict(needed)}"
        )

    if len(alloc.fresh) > len(alloc.free):
        raise ValueError(
            f"{len(alloc.fresh)} new sets do not fit in {len(alloc.free)} free slots"
        )

    actions = [encode_dissolve(cfg, slot) for slot in alloc.dissolve]
    for slot in sorted(alloc.keep):
        # The engine re-sorts a slot after every removal, so each position is read
        # off what is still standing there rather than off the original row.
        left = list(alloc.current[slot])
        for kind in alloc.removals(slot):
            actions.append(encode_pick(cfg, slot, left.index(kind)))
            left.remove(kind)
    actions += [
        encode_place(cfg, int(kind)) for kind in counts_to_kinds(np.maximum(0, played))
    ]
    for slot in sorted(alloc.keep):
        actions += [encode_assign(cfg, kind, slot) for kind in alloc.additions(slot)]
    # One set at a time, lowest free slot first: the mask only ever offers the
    # lowest empty slot, so a set must be started before the next one can be.
    for content, slot in zip(alloc.fresh, alloc.free, strict=False):
        actions += [encode_assign(cfg, kind, slot) for kind in content]
    actions.append(cfg.end_turn_action)
    return actions
