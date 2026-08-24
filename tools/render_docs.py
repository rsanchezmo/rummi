"""Render a game through both views, composed side by side.

The point of the figure is that both panels come from a *single* ``GameView``:
the terminal view and the pygame window are two renderers over one shared view
model, not two implementations that happen to agree.

The terminal panel is rasterised from the real ANSI output -- the same escape
sequences a terminal would receive -- rather than redrawn to look like a
terminal, so what the image shows is what you get.

The animation samples one frame per *committed turn*, matching the library's own
``RenderOn.TURN``: a turn is the meaningful unit of play, so every frame is a
real board change rather than a single tile moving.

    python tools/render_docs.py --format gif --out docs/render.gif
    python tools/render_docs.py --format png --out docs/render.png
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


TERMINAL_CAPTION = "render_mode='ansi'  -- live in the terminal"
WINDOW_CAPTION = "render_mode='human'  -- pygame window"


def panelize(surface, caption: str | None = None, pad: int = 18, label_size: int = 17):
    """Wrap one panel with its own background, optionally captioned.

    The README labels the two animations in its table header, so the GIFs are
    generated without captions -- burning the same words into the image would
    only cost pixels.
    """
    pygame.font.init()
    lh = 0
    font = None
    if caption:
        font = _monospace(label_size)
        lh = font.get_linesize() + 10
    page = pygame.Surface((surface.get_width() + pad * 2, surface.get_height() + pad * 2 + lh))
    page.fill(PAGE_BG)
    if caption:
        page.blit(font.render(caption, True, LABEL), (pad, pad))
    page.blit(surface, (pad, pad + lh))
    return page


def compose(left, right, gap: int = 26, pad: int = 26, label_size: int = 17):
    pygame.font.init()
    font = _monospace(label_size)
    lh = font.get_linesize() + 10

    height = pad * 2 + lh + max(left.get_height(), right.get_height())
    width = pad * 2 + left.get_width() + gap + right.get_width()
    page = pygame.Surface((width, height))
    page.fill(PAGE_BG)

    # Top-anchored, not centred: across an animation the table grows, and
    # centring would slide both panels up and down every frame.
    for surf, x, caption in (
        (left, pad, TERMINAL_CAPTION),
        (right, pad + left.get_width() + gap, WINDOW_CAPTION),
    ):
        page.blit(font.render(caption, True, LABEL), (x, pad))
        page.blit(surf, (x, pad + lh))
    return page


def game_frames(cfg, seed: int, policy_name: str, max_turns: int):
    """Yield one ``GameView`` per committed turn, plus the opening position."""
    from rummi.bench.fuzz import make_policy

    policy = make_policy(cfg, policy_name, 0)
    state = reset(cfg, 1, seed=seed)
    yield view(state, 0, legal_actions(state))

    last_turn = 0
    for _ in range(60_000):
        mask = legal_actions(state)
        step(state, policy(state, mask), mask)
        turn = int(state.turn_count[0])
        if turn != last_turn:
            last_turn = turn
            yield view(state, 0, legal_actions(state))
            if turn >= max_turns:
                return
        if state.done.all():
            yield view(state, 0, legal_actions(state))
            return


def to_pil(surface):
    from PIL import Image

    return Image.frombytes("RGB", surface.get_size(), pygame.image.tobytes(surface, "RGB"))


def on_fixed_canvas(surfaces: list) -> list:
    """Pad every frame to one shared size.

    The terminal panel is only as wide as its longest line, so its width changes
    as sets are laid and taken apart. A GIF has a single canvas, so frames of
    differing sizes would be cropped or shifted; padding keeps the layout still.
    """
    width = max(s.get_width() for s in surfaces)
    height = max(s.get_height() for s in surfaces)
    if all(s.get_size() == (width, height) for s in surfaces):
        return surfaces
    out = []
    for surf in surfaces:
        canvas = pygame.Surface((width, height))
        canvas.fill(PAGE_BG)
        canvas.blit(surf, (0, 0))
        out.append(canvas)
    return out


def write_gif(frames, out: Path, scale: float, ms: int, hold: int) -> None:
    """Encode with a single shared palette.

    Per-frame palettes would shift colours as the table changes, which on a
    board of coloured tiles reads as flicker. One palette built from a
    mid-game frame keeps every tile the same colour throughout.
    """
    from PIL import Image

    images = [to_pil(f) for f in on_fixed_canvas(list(frames))]
    if scale != 1.0:
        size = (int(images[0].width * scale), int(images[0].height * scale))
        images = [im.resize(size, Image.LANCZOS) for im in images]

    palette_source = images[len(images) // 2]
    palette = palette_source.quantize(colors=128, method=Image.MEDIANCUT)
    quantized = [im.quantize(palette=palette, dither=Image.Dither.NONE) for im in images]

    durations = [ms] * len(quantized)
    durations[0] = durations[-1] = hold
    quantized[0].save(
        out,
        save_all=True,
        append_images=quantized[1:],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=1,
    )


def interesting_state(cfg, seed: int, turns: int) -> GameView:
    """Drive a game to a frame worth looking at.

    Ends mid-turn with a set taken apart, so the image shows the workbench, a
    just-touched slot, and a table that is temporarily illegal -- the states that
    only exist because a turn spans several steps.
    """
    from rummi.bench.fuzz import make_policy

    policy = make_policy(cfg, "greedy", 0)
    state = reset(cfg, 1, seed=seed)
    for _ in range(turns):
        mask = legal_actions(state)
        step(state, policy(state, mask), mask)

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


def panel(window, snapshot: GameView, font_size: int):
    """One composed figure for a single game state, both views side by side."""
    window.draw(snapshot)
    return compose(render_terminal(frame(snapshot, Palette(True)), size=font_size),
                   window._surface.copy())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--format", choices=["png", "gif"], default="gif")
    p.add_argument(
        "--out", type=Path, default=Path("docs/render"),
        help="gif: stem, written as <stem>-terminal.gif and <stem>-window.gif; png: full path",
    )
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--policy", default="optimal", help="who plays; optimal ends by winning")
    p.add_argument("--turns", type=int, default=150, help="png: which turn to capture")
    p.add_argument("--max-turns", type=int, default=90, help="gif: cap on frames")
    p.add_argument("--tile", type=int, nargs=2, default=(26, 36))
    p.add_argument("--font", type=int, default=15)
    # Full resolution by default. Downscaling is available for trimming file
    # size, but it costs legibility unevenly: terminal text survives it, while a
    # tile's numeral blurs into its cream face and is then flattened by GIF
    # quantisation -- hence a separate knob per panel.
    p.add_argument("--scale", type=float, default=1.0, help="terminal panel")
    p.add_argument("--window-scale", type=float, default=1.0, help="pygame panel")
    p.add_argument("--ms", type=int, default=680, help="gif: milliseconds per frame")
    p.add_argument("--hold", type=int, default=2600, help="gif: hold on first and last")
    args = p.parse_args()

    pygame.init()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    window = PygameView(STANDARD, headless=True, tile_w=args.tile[0], tile_h=args.tile[1])

    if args.format == "png":
        page = panel(window, interesting_state(STANDARD, args.seed, args.turns), args.font)
        pygame.image.save(page, str(args.out))
        print(f"wrote {args.out}  {page.get_size()}")
    else:
        # Both animations are built from one pass over the game, so frame N of
        # each is the same turn. Identical frame counts and durations are what
        # keep two separately-looping GIFs from drifting apart in a browser.
        terminal_pages, window_pages = [], []
        for snapshot in game_frames(STANDARD, args.seed, args.policy, args.max_turns):
            terminal_pages.append(
                panelize(render_terminal(frame(snapshot, Palette(True)), size=args.font))
            )
            window.draw(snapshot)
            window_pages.append(panelize(window._surface.copy()))

        for pages, suffix, scale in (
            (terminal_pages, "terminal", args.scale),
            (window_pages, "window", args.window_scale),
        ):
            out = args.out.with_name(f"{args.out.name}-{suffix}.gif")
            write_gif(pages, out, scale, args.ms, args.hold)
            print(f"wrote {out}  {len(pages)} frames  {out.stat().st_size / 1e6:.1f} MB")
    window.close()


if __name__ == "__main__":
    main()
