"""Pre-rasterised tile faces.

Every tile face is drawn once into a single surface at startup, so a frame is
pure blitting: no text rasterisation, no shape drawing, no per-frame allocation.
That is what keeps the live window off the critical path of a fast rollout.

Builds without a display (``SDL_VIDEODRIVER=dummy``), which is what makes it
testable and lets ``rgb_array`` work headless.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from rummi.core.config import RummiConfig
from rummi.core.encoding import tables
from rummi.render.text import COLOR_RGB, JOKER_RGB

TILE_FACE = (238, 232, 216)
TILE_EDGE = (206, 198, 180)
NEW_EDGE = (96, 196, 120)
TOUCH_EDGE = (226, 178, 72)
BACKGROUND = (26, 28, 33)
PANEL = (36, 39, 46)
INVALID_EDGE = (222, 92, 92)
TEXT = (222, 226, 234)
TEXT_DIM = (138, 146, 160)


class Variant(IntEnum):
    NORMAL = 0
    NEW = 1
    """Placed this turn -- drawn with a highlight edge."""
    GHOST = 2
    """Dimmed, for tiles still loose in the workbench."""


@dataclass(frozen=True, slots=True)
class Atlas:
    cfg: RummiConfig
    surface: object
    """One ``pygame.Surface`` holding every face, kind-major by variant."""
    tile_w: int
    tile_h: int

    def rect(self, kind: int, variant: Variant = Variant.NORMAL):
        import pygame

        return pygame.Rect(kind * self.tile_w, int(variant) * self.tile_h, self.tile_w, self.tile_h)


def build(cfg: RummiConfig, tile_w: int = 30, tile_h: int = 40) -> Atlas:
    import pygame

    if not pygame.font.get_init():
        pygame.font.init()
    font = pygame.font.Font(None, max(16, int(tile_h * 0.55)))

    n_variants = len(Variant)
    surface = pygame.Surface((cfg.n_kinds * tile_w, n_variants * tile_h), pygame.SRCALPHA)
    surface.fill((0, 0, 0, 0))

    t = tables(cfg)
    for variant in Variant:
        for kind in range(cfg.n_kinds):
            is_joker = kind == cfg.joker_kind
            rgb = JOKER_RGB if is_joker else COLOR_RGB[int(t.color[kind]) % len(COLOR_RGB)]
            label = "J" if is_joker else str(int(t.number[kind]))

            face = pygame.Surface((tile_w, tile_h), pygame.SRCALPHA)
            body = pygame.Rect(1, 1, tile_w - 2, tile_h - 2)
            # Ghosting dims the face only; the glyph stays at full strength so a
            # tile in hand is still readable.
            alpha = 175 if variant is Variant.GHOST else 255
            pygame.draw.rect(face, (*TILE_FACE, alpha), body, border_radius=4)
            edge = NEW_EDGE if variant is Variant.NEW else TILE_EDGE
            pygame.draw.rect(face, (*edge, alpha), body, width=2, border_radius=4)

            glyph = font.render(label, True, rgb)
            face.blit(glyph, glyph.get_rect(center=body.center))
            surface.blit(face, (kind * tile_w, int(variant) * tile_h))

    return Atlas(cfg=cfg, surface=surface, tile_w=tile_w, tile_h=tile_h)
