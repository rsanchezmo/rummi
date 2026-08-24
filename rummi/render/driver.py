"""One facade over both renderers, with the cost controls attached.

The env and the replay CLI both go through this, so throttling and env-slicing
are decided once. ``RenderMode.NONE`` short-circuits before touching the state at
all, which is what makes rendering free when it is switched off.
"""

from __future__ import annotations

import time
from enum import Enum

import numpy as np

from rummi.rules.config import RummiConfig
from rummi.env.numpy.state import BatchState
from rummi.render.view_model import GameView, view


class RenderMode(str, Enum):
    NONE = "none"
    ANSI = "ansi"
    """Live in-place terminal view."""
    HUMAN = "human"
    """Live pygame window."""
    RGB_ARRAY = "rgb_array"
    """Offscreen frame as ``(H, W, 3)`` uint8, for video capture."""


class Renderer:
    def __init__(
        self,
        cfg: RummiConfig,
        mode: RenderMode | str = RenderMode.NONE,
        env_index: int = 0,
        fps: float | None = None,
        every: int = 1,
        tile_size: tuple[int, int] = (30, 40),
    ) -> None:
        self.cfg = cfg
        self.mode = RenderMode(mode)
        self.env_index = env_index
        self.fps = fps
        self.every = max(1, every)
        self.tile_size = tile_size
        self._calls = 0
        self._last_drawn = 0.0
        self._terminal = None
        self._window = None

    # --- lazily built so an unused mode costs nothing -------------------------
    @property
    def terminal(self):
        if self._terminal is None:
            from rummi.render.text import TerminalView

            self._terminal = TerminalView()
        return self._terminal

    def window(self, headless: bool):
        if self._window is None:
            from rummi.render.pygame_view import PygameView

            self._window = PygameView(
                self.cfg, headless=headless, tile_w=self.tile_size[0], tile_h=self.tile_size[1]
            )
        return self._window

    def _due(self) -> bool:
        """Throttle drawing so a fast rollout does not try to draw every step."""
        self._calls += 1
        if self._calls % self.every:
            return False
        if self.fps:
            now = time.perf_counter()
            if now - self._last_drawn < 1.0 / self.fps:
                return False
            self._last_drawn = now
        return True

    def snapshot(self, state: BatchState, mask: np.ndarray | None = None) -> GameView:
        return view(state, self.env_index, mask)

    def frame(self, state: BatchState, mask: np.ndarray | None = None):
        """Draw unconditionally. This is what ``env.render()`` calls: an explicit
        request must always produce a frame, whatever the throttle says."""
        if self.mode is RenderMode.NONE:
            return None
        snap = self.snapshot(state, mask)
        if self.mode is RenderMode.RGB_ARRAY:
            return self.window(headless=True).rgb_array(snap)
        if self.mode is RenderMode.ANSI:
            self.terminal.render(snap)
            return None
        self.window(headless=False).render(snap)
        return None

    def render(self, state: BatchState, mask: np.ndarray | None = None):
        """Draw if the throttle allows. This is what the step loop calls, so it
        must stay bounded however fast the simulator runs."""
        if self.mode is RenderMode.NONE or not self._due():
            return None
        return self.frame(state, mask)

    def close(self) -> None:
        if self._terminal is not None:
            self._terminal.close()
            self._terminal = None
        if self._window is not None:
            self._window.close()
            self._window = None
