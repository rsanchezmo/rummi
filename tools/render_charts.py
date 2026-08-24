"""Generate the README's charts as SVG, from committed measurement data.

SVG rather than a rendered image because GitHub inlines it, it scales, and it
diffs as text. That rules out a hover layer -- GitHub strips scripts from SVG --
so the interaction budget is spent on direct labels instead, and the README's
tables stand as the table view the accessibility pass requires.

Colours come from the reference data-viz palette and were checked with its
validator in both modes rather than by eye; see docs/charts/README.md. Light-mode
aqua, yellow and magenta sit under 3:1 on the light surface, which obliges
visible labels -- hence every series is both direct-labelled and in the legend.

    python tools/render_charts.py
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

# --- palette (validated; see module docstring) --------------------------------
SERIES_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
SERIES_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300"]
ORDINAL_LIGHT = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
ORDINAL_DARK = ["#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4"]

INK = ("#0b0b0b", "#ffffff")
INK_2 = ("#52514e", "#c3c2b7")
MUTED = ("#898781", "#898781")
GRID = ("#e1e0d9", "#2c2c2a")
AXIS = ("#c3c2b7", "#383835")

FONT = 'ui-sans-serif,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif'
MONO = 'ui-monospace,SFMono-Regular,Menlo,Consolas,monospace'


# Calibrated against the rendered stack: 17 characters of bold 12px measured
# 118px, i.e. 0.58em per character. Good enough to size a margin and to refuse
# to emit a chart whose labels would run off the edge.
CHAR_EM = 0.58


def text_width(text: str, size: float, bold: bool = False) -> float:
    return len(text) * size * (CHAR_EM + (0.02 if bold else 0.0))


def esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def theme_css(pairs: dict[str, tuple[str, str]]) -> str:
    """Light values on :root, dark ones behind prefers-color-scheme.

    GitHub renders these through <img>, which still resolves the media query, so
    one file serves both themes. The light values are the fallback, which is the
    right way round: an <img> with no colour-scheme context renders light.
    """
    light = "\n".join(f"    --{k}: {v[0]};" for k, v in pairs.items())
    dark = "\n".join(f"      --{k}: {v[1]};" for k, v in pairs.items())
    return (
        "  :root {\n" + light + "\n  }\n"
        "  @media (prefers-color-scheme: dark) {\n    :root {\n" + dark + "\n    }\n  }\n"
    )


@dataclass
class Svg:
    width: int
    height: int
    parts: list[str]

    def add(self, markup: str) -> None:
        self.parts.append(markup)

    def text(self, x, y, s, *, size=12, fill="var(--muted)", anchor="start",
             weight=400, mono=False, baseline="middle") -> None:
        family = MONO if mono else FONT
        self.add(
            f'<text x="{x:.1f}" y="{y:.1f}" font-family=\'{family}\' font-size="{size}" '
            f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}" '
            f'dominant-baseline="{baseline}">{esc(s)}</text>'
        )

    def render(self, css: str, title: str) -> str:
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.width} {self.height}" '
            f'width="{self.width}" height="{self.height}" role="img" '
            f'aria-label="{esc(title)}">\n'
            f"<style>\n{css}</style>\n" + "\n".join(self.parts) + "\n</svg>\n"
        )


# --- chart 1: throughput ------------------------------------------------------
# Six of the eight measured backends. jax-step and jax-fused are dropped because
# all three JAX variants land within a few percent of each other -- plotting them
# would spend three of six colours saying one thing.
THROUGHPUT_SERIES = [
    "numpy",
    "torch-cpu",
    "torch-cpu+compile",
    "torch-mps",
    "torch-mps+compile",
    "jax-scan",
]
LABELLED = {"numpy", "torch-cpu+compile", "torch-mps+compile", "jax-scan"}
"""Direct-labelled at the right end. Capped at four so the labels do not collide;
the other two are identified by the legend."""


def nice_ceiling(value: float) -> float:
    step = 10 ** math.floor(math.log10(value))
    for mult in (1, 1.5, 2, 2.5, 3, 4, 5, 8, 10):
        if step * mult >= value:
            return step * mult
    return step * 10


def throughput_chart(data: dict, width=900, height=440) -> str:
    rows = [r for r in data["rows"] if r["backend"] in THROUGHPUT_SERIES]
    batches = sorted({r["batch_size"] for r in rows})
    by_backend = {name: {} for name in THROUGHPUT_SERIES}
    for r in rows:
        by_backend[r["backend"]][r["batch_size"]] = r["env_steps_per_sec"]

    # Sized to the longest direct label rather than guessed, so the labels
    # cannot silently run off the right edge when a backend name changes.
    label_w = max(text_width(n, 11.5, bold=True) for n in LABELLED)
    left, top, bottom = 64, 54, 84
    right = int(10 + label_w + 14)
    plot_w, plot_h = width - left - right, height - top - bottom
    y_max = nice_ceiling(max(r["env_steps_per_sec"] for r in rows))

    def px(b: int) -> float:
        lo, hi = math.log2(batches[0]), math.log2(batches[-1])
        return left + plot_w * (math.log2(b) - lo) / (hi - lo)

    def py(v: float) -> float:
        return top + plot_h * (1 - v / y_max)

    svg = Svg(width, height, [])
    svg.add(f'<rect width="{width}" height="{height}" fill="var(--surface)"/>')
    svg.text(left, 22, "Simulator throughput", size=15, fill="var(--ink)", weight=600)
    svg.text(
        left, 40,
        f"env-steps per second, {data['config']} config · higher is better · {data['machine']}",
        size=11.5,
    )

    # Recessive grid: horizontal only, so the eye compares heights not positions.
    for i in range(6):
        v = y_max * i / 5
        y = py(v)
        svg.add(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" '
            f'stroke="var(--grid)" stroke-width="1"/>'
        )
        svg.text(left - 10, y, f"{v / 1000:.0f}k" if v else "0", size=11, anchor="end")

    svg.add(
        f'<line x1="{left}" y1="{py(0):.1f}" x2="{left + plot_w}" y2="{py(0):.1f}" '
        f'stroke="var(--axis)" stroke-width="1.5"/>'
    )
    for b in batches:
        svg.text(px(b), py(0) + 16, f"{b:,}", size=11, anchor="middle", mono=True)
    svg.text(left + plot_w / 2, py(0) + 36, "parallel environments", size=11.5, anchor="middle")

    for slot, name in enumerate(THROUGHPUT_SERIES):
        points = [(px(b), py(by_backend[name][b])) for b in batches if b in by_backend[name]]
        if not points:
            continue
        colour = f"var(--s{slot + 1})"
        path = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f} {y:.1f}" for i, (x, y) in enumerate(points))
        svg.add(f'<path d="{path}" fill="none" stroke="{colour}" stroke-width="2" '
                f'stroke-linejoin="round" stroke-linecap="round"/>')
        for x, y in points:
            # A 2px surface ring keeps overlapping markers separable.
            svg.add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{colour}" '
                    f'stroke="var(--surface)" stroke-width="2"/>')
        if name in LABELLED:
            x, y = points[-1]
            if x + 10 + text_width(name, 11.5, bold=True) > width:
                raise ValueError(f"direct label {name!r} would overflow the chart")
            svg.text(x + 10, y, name, size=11.5, fill=colour, weight=600)

    # Legend: always present for >= 2 series, so identity is never colour alone.
    lx, ly = left, height - 18
    for slot, name in enumerate(THROUGHPUT_SERIES):
        svg.add(f'<rect x="{lx}" y="{ly - 5}" width="10" height="10" rx="2" fill="var(--s{slot + 1})"/>')
        svg.text(lx + 15, ly, name, size=11, fill="var(--ink2)")
        lx += 22 + 6.4 * len(name)
    return svg.render(
        theme_css(
            {
                "surface": ("#fcfcfb", "#1a1a19"),
                "ink": INK, "ink2": INK_2, "muted": MUTED, "grid": GRID, "axis": AXIS,
                **{f"s{i + 1}": (SERIES_LIGHT[i], SERIES_DARK[i]) for i in range(6)},
            }
        ),
        "Simulator throughput in env-steps per second, by backend and batch size",
    )


# --- chart 2: the agent ladder ------------------------------------------------
def agents_chart(data: dict, width=900, height=364) -> str:
    """Two panels over one shared list of agents.

    Win rate alone hides the more interesting half: `greedy` stalls out in nearly
    every game and `optimal` in none, and that is the same fact the win rate is
    reporting, seen from the other side.

    An ordinal ramp rather than categorical hues, because the agents are ordered
    by strength -- categorical colour would assert they are merely different.
    """
    agents = data["agents"]
    rows = len(agents)
    # `value_w` reserves room for a value printed beside a full-width bar; without
    # it a 100% bar pushes its own label off the edge.
    left, gap, top, bottom, value_w = 136, 44, 96, 46, 52
    panel_w = (width - left - gap - value_w) / 2
    row_h = (height - top - bottom) / rows
    bar_h = min(22, row_h - 12)

    svg = Svg(width, height, [])
    svg.add(f'<rect width="{width}" height="{height}" fill="var(--surface)"/>')
    svg.text(24, 22, "Agent strength", size=15, fill="var(--ink)", weight=600)
    svg.text(
        24, 40,
        f"{data['suite']} suite · {data['games']} mirrored games each · opponent: {data['opponent']}",
        size=11.5,
    )

    panels = [
        ("win rate vs greedy", "win_rate", 1.0, lambda v: f"{v:.0%}"),
        ("games ending in stalemate", "stalemate_rate", 1.0, lambda v: f"{v:.0%}"),
    ]
    for panel, (title, key, scale, fmt) in enumerate(panels):
        x0 = left + panel * (panel_w + gap)
        # Below the subtitle, not beside it -- these two used to overlap.
        svg.text(x0, top - 22, title, size=11.5, fill="var(--ink2)", weight=600)
        for i, agent in enumerate(agents):
            y = top + i * row_h + row_h / 2
            if panel == 0:
                svg.text(left - 14, y, agent["name"], size=12, anchor="end",
                         fill="var(--ink)", mono=True)
            value = agent[key]
            w = max(2.0, panel_w * value / scale)
            colour = f"var(--o{i + 1})"
            svg.add(
                f'<rect x="{x0}" y="{y - bar_h / 2:.1f}" width="{w:.1f}" height="{bar_h:.1f}" '
                f'rx="4" fill="{colour}"/>'
            )
            # Value inside the bar when it fits, outside when it does not -- and
            # always in ink, never the bar's colour.
            inside = w > 52
            svg.text(
                x0 + w - 8 if inside else x0 + w + 8, y, fmt(value), size=11.5,
                anchor="end" if inside else "start",
                fill="var(--on-bar)" if inside else "var(--ink2)", mono=True, weight=600,
            )
        svg.add(
            f'<line x1="{x0}" y1="{top - 4}" x2="{x0}" y2="{top + rows * row_h:.1f}" '
            f'stroke="var(--axis)" stroke-width="1.5"/>'
        )

    svg.text(
        24, height - 16,
        "greedy is the opponent, so it scores exactly 50% — the mirroring makes that exact, not approximate",
        size=11,
    )
    return svg.render(
        theme_css(
            {
                "surface": ("#fcfcfb", "#1a1a19"),
                "ink": INK, "ink2": INK_2, "muted": MUTED, "axis": AXIS,
                # Ink on the darkest ordinal steps, which are dark in light mode
                # and light in dark mode.
                "on-bar": ("#ffffff", "#0b0b0b"),
                **{f"o{i + 1}": (ORDINAL_LIGHT[i], ORDINAL_DARK[i]) for i in range(5)},
            }
        ),
        "Agent win rate and stalemate rate, weakest to strongest",
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("docs/data"))
    p.add_argument("--out", type=Path, default=Path("docs/charts"))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    for name, builder in (("throughput", throughput_chart), ("agents", agents_chart)):
        source = args.data / f"{'backends' if name == 'throughput' else 'agents'}.json"
        if not source.exists():
            print(f"skipping {name}: {source} not found")
            continue
        out = args.out / f"{name}.svg"
        out.write_text(builder(json.loads(source.read_text())))
        print(f"wrote {out}  {out.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
