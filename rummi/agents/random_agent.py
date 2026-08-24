"""Random baselines.

Be careful reading their scores. A flat uniform draw over ~2400 actions almost
never stumbles onto a legal 30-point opening meld -- four times in ten million
fuzz steps -- so on the standard config random play is *byte-identical to passing
every turn*. It is a sanity check that the plumbing works, not a floor to beat.

Weighting the action families produces games that at least commit turns, which is
what makes it useful as a fuzz driver.
"""

from __future__ import annotations

import numpy as np

from rummi.agents.base import Observation
from rummi.rules.config import RummiConfig

FAMILY_WEIGHTS = {
    "place": 1.0,
    "pick": 0.6,
    "dissolve": 0.3,
    "assign": 3.0,
    "end_turn": 8.0,
    "draw": 0.4,
}


def action_weights(cfg: RummiConfig, weights: dict[str, float] | None = None) -> np.ndarray:
    """``(A,)`` sampling weight per action id."""
    w = {**FAMILY_WEIGHTS, **(weights or {})}
    out = np.empty(cfg.n_actions, dtype=np.float64)
    out[cfg.place_offset : cfg.pick_offset] = w["place"]
    out[cfg.pick_offset : cfg.dissolve_offset] = w["pick"]
    out[cfg.dissolve_offset : cfg.assign_offset] = w["dissolve"]
    out[cfg.assign_offset : cfg.end_turn_action] = w["assign"]
    out[cfg.end_turn_action] = w["end_turn"]
    out[cfg.draw_action] = w["draw"]
    return out


class RandomAgent:
    """Uniform over legal actions."""

    name = "random"

    def __init__(self, cfg: RummiConfig, seed: int = 0) -> None:
        self.cfg = cfg
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def reset(self, n_envs: int) -> None:
        # Re-seeded, so a suite scores the same however many times it is run.
        self.rng = np.random.default_rng(self.seed)

    def act(
        self, obs: Observation, mask: np.ndarray, active: np.ndarray | None = None
    ) -> np.ndarray:
        return np.argmax(np.where(mask, self.rng.random(mask.shape), -1.0), axis=-1)


class WeightedRandomAgent(RandomAgent):
    """Random, but biased towards finishing what it starts."""

    name = "weighted-random"

    def __init__(
        self, cfg: RummiConfig, seed: int = 0, weights: dict[str, float] | None = None
    ) -> None:
        super().__init__(cfg, seed)
        self.weights = action_weights(cfg, weights)

    def act(
        self, obs: Observation, mask: np.ndarray, active: np.ndarray | None = None
    ) -> np.ndarray:
        """Sampling by ``argmax`` of weighted exponential draws avoids a per-env
        normalise-and-search, so the whole batch is one vectorised pass."""
        scores = self.rng.exponential(size=mask.shape) / self.weights
        return np.argmax(np.where(mask, -scores, -np.inf), axis=-1)
