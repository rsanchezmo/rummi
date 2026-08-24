"""Render one game state through both views and compose them side by side.

The point of the image is that both panels come from a *single* ``GameView``:
the terminal view and the pygame window are two renderers over one shared view
model, not two implementations that happen to agree.

The terminal panel is rasterised from the real ANSI output -- the same escape
sequences a terminal would receive -- rather than redrawn to look like a
terminal, so what the image shows is what you get.

    python tools/make_screenshot.py --out docs/render.png
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import pygame

from rummi.env.numpy.deal import reset
from rummi.env.numpy.engine import step
from rummi.env.numpy.masks import legal_actions
from rummi.render.pygame_view import PygameView
from rummi.render.text import Palette, frame
from rummi.render.view_model import GameView, view
from rummi.rules.config import STANDARD

TERM_BG = (18, 20, 24)
TERM_FG = (208, 212, 220)
PAGE_BG = (12, 14, 17)
LABEL = (150, 158, 170)

SGR = re.compile(r"\x1b\[([0-9;]*)m")


def _monospace(size: int):
    path = pygame.font.match_font("menlo,dejavusansmono,consolas,couriernew,monospace")
    return pygame.font.Font(path, size) if path else pygame.font.SysFont("monospace", size)


def _spans(line: str):
    """Split one ANSI line into ``(text, colour, bold, dim)`` runs."""
    out, pos = [], 0
    colour, bold, dim = TERM_FG, False, False
    for m in SGR.finditer(line):
        if m.start() > pos:
            out.append((line[pos : m.start()], colour, bold, dim))
        for code in (m.group(1) or "0").split(";"):
            pass
        params = [int(p) for p in (m.group(1) or "0").split(";") if p != ""]
        i = 0
        while i < len(params):
            p = params[i]
            if p == 0:
                colour, bold, dim = TERM_FG, False, False
            elif p == 1:
                bold = True
            elif p == 2:
                dim = True
            elif p == 38 and params[i + 1 : i + 2] == [2]:
                colour = tuple(params[i + 2 : i + 5])
                i += 4
            i += 1
        pos = m.end()
    if pos < len(line):
        out.append((line[pos:], colour, bold, dim))
    return out


def render_terminal(text: str, size: int = 15, pad: int = 14):
    """Rasterise ANSI text exactly as a truecolour terminal would show it."""
    pygame.font.init()
    font = _monospace(size)
    bold = _monospace(size)
    bold.set_bold(True)
    cw, ch = font.size("M")[0], font.get_linesize()

    lines = text.split("\n")
    width = pad * 2 + cw * max(len(SGR.sub("", ln)) for ln in lines)
    surface = pygame.Surface((width, pad * 2 + ch * len(lines)))
    surface.fill(TERM_BG)

    for row, line in enumerate(lines):
        x = pad
        for chunk, colour, is_bold, is_dim in _spans(line):
            if is_dim:
                colour = tuple(int(c * 0.55) for c in colour)
            glyph = (bold if is_bold else font).render(chunk, True, colour)
            surface.blit(glyph, (x, pad + row * ch))
            x += cw * len(chunk)
    return surface


def compose(left, right, gap: int = 26, pad: int = 26, label_size: int = 17):
    pygame.font.init()
    font = _monospace(label_size)
    lh = font.get_linesize() + 10

    height = pad * 2 + lh + max(left.get_height(), right.get_height())
    width = pad * 2 + left.get_width() + gap + right.get_width()
    page = pygame.Surface((width, height))
    page.fill(PAGE_BG)

    tallest = max(left.get_height(), right.get_height())
    for surf, x, caption in (
        (left, pad, "render_mode='ansi'  -- live in the terminal"),
        (right, pad + left.get_width() + gap, "render_mode='human'  -- pygame window"),
    ):
        page.blit(font.render(caption, True, LABEL), (x, pad))
        page.blit(surf, (x, pad + lh + (tallest - surf.get_height()) // 2))
    return page


def interesting_state(cfg, seed: int, turns: int) -> GameView:
    """Drive a game to a frame worth looking at.

    Ends mid-turn with a set taken apart, so the image shows the workbench, a
    just-touched slot, and a table that is temporarily illegal -- the states that
    only exist because a turn spans several steps.
    """
    from rummi.policies.greedy_policy import GreedyPolicy

    policy = GreedyPolicy(cfg)
    state = reset(cfg, 1, seed=seed)
    for _ in range(turns):
        mask = legal_actions(state)
        step(state, policy.act(state, mask), mask)

    mask = legal_actions(state)
    dissolve = next(a for a in range(cfg.dissolve_offset, cfg.assign_offset) if mask[0, a])
    step(state, np.array([dissolve]), mask)
    for _ in range(2):
        mask = legal_actions(state)
        assign = next((a for a in range(cfg.assign_offset, cfg.end_turn_action) if mask[0, a]), None)
        if assign is None:
            break
        step(state, np.array([assign]), mask)
    return view(state, 0, legal_actions(state))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("docs/render.png"))
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--turns", type=int, default=150)
    # The window reserves a fixed grid so dirty-rect tracking has stable rects.
    # For a still image, size it to the table actually on screen.
    p.add_argument("--capacity", type=int, default=14)
    args = p.parse_args()

    pygame.init()
    snapshot = interesting_state(STANDARD, args.seed, args.turns)

    window = PygameView(STANDARD, headless=True, capacity=args.capacity)
    window.draw(snapshot)
    right = window._surface.copy()
    window.close()

    left = render_terminal(frame(snapshot, Palette(True)))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    page = compose(left, right)
    pygame.image.save(page, str(args.out))
    print(f"wrote {args.out}  {page.get_size()}")


if __name__ == "__main__":
    main()
