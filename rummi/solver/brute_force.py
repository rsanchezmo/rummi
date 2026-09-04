"""Exhaustive oracles, derived straight from the rules rather than from the
simulator's arithmetic.

These are deliberately slow and obvious: they exist so the vectorised kernels in
``rummi.env.numpy`` and the fast solvers in ``rummi.solver`` can be checked against a
definition nobody has optimised. Only tractable on the reduced configs.
"""

from __future__ import annotations

from collections import Counter
from functools import cache
from itertools import combinations

from rummi.rules.config import RummiConfig
from rummi.rules.encoding import kind_of

Content = tuple[int, ...]
"""A slot's tiles as a sorted tuple of kind ids."""


def _base_sets(cfg: RummiConfig) -> list[list[tuple[int, int]]]:
    """Every legal joker-free set, as a list of ``(kind, face_value)`` positions."""
    out: list[list[tuple[int, int]]] = []

    for color in range(cfg.n_colors):
        for length in range(cfg.min_set, min(cfg.max_set_len, cfg.n_numbers) + 1):
            for start in range(1, cfg.n_numbers - length + 2):
                out.append(
                    [(kind_of(cfg, color, num), num) for num in range(start, start + length)]
                )

    if cfg.group_possible:
        for number in range(1, cfg.n_numbers + 1):
            for size in range(cfg.min_set, min(cfg.max_set_len, cfg.n_colors) + 1):
                for colors in combinations(range(cfg.n_colors), size):
                    out.append([(kind_of(cfg, c, number), number) for c in colors])

    return out


@cache
def valid_sets(cfg: RummiConfig, max_jokers: int) -> dict[Content, int]:
    """Map every legal slot content to its best-case face-value total.

    A joker inherits the value of the position it occupies, so substituting
    jokers into a base set leaves the total unchanged; distinct base sets can
    collapse to the same content, in which case the player declares the reading
    that suits them, hence the ``max``.
    """
    out: dict[Content, int] = {}
    for positions in _base_sets(cfg):
        total = sum(value for _, value in positions)
        for n_jokers in range(min(max_jokers, len(positions)) + 1):
            for slots in combinations(range(len(positions)), n_jokers):
                kinds = [
                    cfg.joker_kind if i in slots else kind
                    for i, (kind, _) in enumerate(positions)
                ]
                key = tuple(sorted(kinds))
                out[key] = max(out.get(key, 0), total)
    return out


def _jokers(cfg: RummiConfig, max_jokers: int | None) -> int:
    """The config's own joker count unless the caller overrides it.

    One convention for the whole module: a default of 2 is the standard deck's
    count hard-coded, and it made this half of the oracle call a set legal on a
    config with no jokers while ``partitionable`` called it illegal.
    """
    return cfg.n_jokers if max_jokers is None else max_jokers


def is_valid(cfg: RummiConfig, content, max_jokers: int | None = None) -> bool:
    return tuple(sorted(content)) in valid_sets(cfg, _jokers(cfg, max_jokers))


def value(cfg: RummiConfig, content, max_jokers: int | None = None) -> int:
    return valid_sets(cfg, _jokers(cfg, max_jokers)).get(tuple(sorted(content)), 0)


def is_extendable(cfg: RummiConfig, content, max_jokers: int | None = None) -> bool:
    """Is ``content`` a sub-multiset of some legal set?"""
    have = Counter(content)
    if not have:
        return True
    for candidate in valid_sets(cfg, _jokers(cfg, max_jokers)):
        want = Counter(candidate)
        if all(want[k] >= v for k, v in have.items()):
            return True
    return False


@cache
def _sets_by_lowest(cfg: RummiConfig, max_jokers: int) -> tuple[tuple[Content, ...], ...]:
    """Legal contents indexed by their lowest kind, for the partition search.

    A tuple rather than a dict so it can be an ``lru_cache`` key.
    """
    buckets: list[list[Content]] = [[] for _ in range(cfg.n_kinds)]
    for content in valid_sets(cfg, max_jokers):
        buckets[content[0]].append(content)
    return tuple(tuple(b) for b in buckets)


def partitionable(cfg: RummiConfig, counts, max_jokers: int | None = None) -> bool:
    """Can this multiset of tiles be split into legal sets with nothing left over?

    The question the whole design turns on: unrestricted rearrangement means a
    table is legal exactly when its tiles admit such a partition. Solved here by
    naive recursion so it can check the fast solvers.
    """
    by_lowest = _sets_by_lowest(cfg, _jokers(cfg, max_jokers))
    return _partition(tuple(int(c) for c in counts), by_lowest)


@cache
def _partition(counts: tuple[int, ...], by_lowest) -> bool:
    lowest = next((k for k, n in enumerate(counts) if n), None)
    if lowest is None:
        return True
    # Every tile must be used, so whichever kind is lowest has to sit in some set,
    # and that set's own lowest kind is therefore this one.
    for content in by_lowest[lowest]:
        remaining = list(counts)
        for kind in content:
            remaining[kind] -= 1
            if remaining[kind] < 0:
                break
        else:
            if _partition(tuple(remaining), by_lowest):
                return True
    return False


def best_turn(cfg: RummiConfig, rack, table_counts) -> int:
    """Most rack tiles that can legally be played in one turn, ignoring melding.

    Exponential in the rack -- only for the reduced configs.
    """
    import numpy as np
    from itertools import product

    rack = np.asarray(rack, dtype=np.int64)
    table_counts = np.asarray(table_counts, dtype=np.int64)
    best = 0
    for combo in product(*(range(int(n) + 1) for n in rack)):
        played = int(sum(combo))
        if played <= best:
            continue
        if partitionable(cfg, tuple(table_counts + np.asarray(combo, dtype=np.int64))):
            best = played
    return best
