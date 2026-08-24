"""Turn a target table into the micro-actions that reach it.

This is where the solver meets the action space, and it matters for more than the
optimal policy: if every table CP-SAT can propose is reachable through
``PLACE``/``DISSOLVE``/``ASSIGN``, then the micro-action decomposition is
*complete* -- no legal turn is expressible in the rules but not in the MDP. The
tests assert exactly that.

Sets already on the table that also appear in the target are left alone. That is
not cosmetic: every untouched set is two fewer actions, and turn length is the
one cost the solver's own objective cannot see.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from rummi.core.actions import encode_assign, encode_dissolve, encode_place
from rummi.core.config import RummiConfig
from rummi.core.encoding import counts_to_kinds

Content = tuple[int, ...]


def slot_contents(table: np.ndarray) -> list[Content]:
    """``(S, L)`` table to one sorted tuple per slot; empty slots give ``()``."""
    return [tuple(sorted(int(k) for k in row if k >= 0)) for row in table]


def plan(
    cfg: RummiConfig,
    table: np.ndarray,
    target_sets,
    played: np.ndarray,
) -> list[int]:
    """Micro-actions taking ``table`` to ``target_sets`` while playing ``played``.

    Raises if the request is inconsistent -- a target that does not account for
    every tile in play is a solver bug, and silently emitting a partial plan
    would surface it much later as an illegal action.
    """
    current = slot_contents(table)
    wanted = Counter(tuple(sorted(s)) for s in target_sets)

    keep: set[int] = set()
    for slot, content in enumerate(current):
        if content and wanted[content] > 0:
            wanted[content] -= 1
            keep.add(slot)

    dissolve = [
        slot for slot, content in enumerate(current) if content and slot not in keep
    ]
    new_sets = list(wanted.elements())

    # Whatever comes off the table plus whatever leaves the rack must be exactly
    # what the new sets consume.
    freed: Counter[int] = Counter()
    for slot in dissolve:
        freed.update(current[slot])
    freed.update(int(k) for k in counts_to_kinds(np.maximum(0, played)))
    needed: Counter[int] = Counter()
    for content in new_sets:
        needed.update(content)
    if freed != needed:
        raise ValueError(
            f"plan does not balance: dissolved+played={dict(freed)} but new sets "
            f"need {dict(needed)}"
        )

    empty = [slot for slot, content in enumerate(current) if not content]
    empty.extend(dissolve)
    empty.sort()
    if len(new_sets) > len(empty):
        raise ValueError(f"{len(new_sets)} new sets do not fit in {len(empty)} free slots")

    actions = [encode_dissolve(cfg, slot) for slot in dissolve]
    actions += [
        encode_place(cfg, int(kind)) for kind in counts_to_kinds(np.maximum(0, played))
    ]
    # One set at a time, lowest free slot first: the mask only ever offers the
    # lowest empty slot, so a set must be started before the next one can be.
    for content, slot in zip(new_sets, empty):
        actions += [encode_assign(cfg, kind, slot) for kind in content]
    actions.append(cfg.end_turn_action)
    return actions
