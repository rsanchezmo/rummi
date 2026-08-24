"""Enumeration of every legal set a config admits, joker-free.

This is the shared vocabulary of the solvers and the greedy baseline: the ILP
uses one usage variable per candidate, and the greedy policy scores candidates
directly. Jokers are not enumerated -- they are handled by substitution, since a
joker inherits the value of the position it fills and therefore never changes a
candidate's total.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations

import numpy as np

from rummi.rules.config import RummiConfig
from rummi.rules.encoding import EMPTY, kind_of


@dataclass(frozen=True, slots=True)
class Candidates:
    cfg: RummiConfig
    kinds: np.ndarray
    """``(n, max_set_len)`` the tiles of each candidate, ``EMPTY`` padded."""
    counts: np.ndarray
    """``(n, K)`` required count of each kind."""
    length: np.ndarray
    value: np.ndarray
    is_run: np.ndarray

    def __len__(self) -> int:
        return int(self.length.shape[0])


@lru_cache(maxsize=None)
def candidates(cfg: RummiConfig) -> Candidates:
    rows: list[list[int]] = []
    values: list[int] = []
    runs: list[bool] = []

    max_run = min(cfg.max_set_len, cfg.n_numbers)
    for color in range(cfg.n_colors):
        for length in range(cfg.min_set, max_run + 1):
            for start in range(1, cfg.n_numbers - length + 2):
                numbers = range(start, start + length)
                rows.append([kind_of(cfg, color, n) for n in numbers])
                values.append(sum(numbers))
                runs.append(True)

    if cfg.group_possible:
        max_group = min(cfg.max_set_len, cfg.n_colors)
        for number in range(1, cfg.n_numbers + 1):
            for size in range(cfg.min_set, max_group + 1):
                for colors in combinations(range(cfg.n_colors), size):
                    rows.append([kind_of(cfg, c, number) for c in colors])
                    values.append(size * number)
                    runs.append(False)

    n = len(rows)
    kinds = np.full((n, cfg.max_set_len), EMPTY, dtype=np.int16)
    counts = np.zeros((n, cfg.n_kinds), dtype=np.int16)
    length = np.zeros(n, dtype=np.int16)
    for i, row in enumerate(rows):
        kinds[i, : len(row)] = row
        counts[i, row] = 1
        length[i] = len(row)

    return Candidates(
        cfg=cfg,
        kinds=kinds,
        counts=counts,
        length=length,
        value=np.asarray(values, dtype=np.int32),
        is_run=np.asarray(runs, dtype=bool),
    )
