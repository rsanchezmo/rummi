"""Uniform-ish random play over the legal mask.

A flat uniform draw over ~2400 actions almost never stumbles onto a legal
``END_TURN``, so games would degenerate into an unbroken run of draws and leave
most of the engine unexercised. Weighting the families instead keeps the policy
trivially simple while producing games that actually commit turns, which is what
makes it useful as a fuzz driver and as a baseline floor.
"""

from __future__ import annotations

import numpy as np

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


class RandomPolicy:
    def __init__(
        self,
        cfg: RummiConfig,
        seed: int = 0,
        weights: dict[str, float] | None = None,
    ) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.weights = action_weights(cfg, weights)

    def act(self, mask: np.ndarray) -> np.ndarray:
        """``(B,)`` action ids sampled from the legal actions of each env.

        Sampling by ``argmax`` of exponential draws scaled by the weights avoids a
        per-env normalise-and-search, so the whole batch is one vectorised pass.
        """
        scores = self.rng.exponential(size=mask.shape) / self.weights
        return np.argmax(np.where(mask, -scores, -np.inf), axis=-1)
