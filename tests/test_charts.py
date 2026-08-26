"""The README charts: geometry, themes, and agreement with the data.

Eyeballing a chart catches proportions but not overflow -- a label that runs two
pixels past the viewBox looks fine in a thumbnail and is clipped in the browser.
These assertions cover what looking cannot.
"""

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs" / "data"

pytest.importorskip("numpy")
import sys

sys.path.insert(0, str(ROOT / "tools"))
import render_charts as rc

VIEWBOX = re.compile(r'viewBox="0 0 ([\d.]+) ([\d.]+)"')
RECT = re.compile(r'<rect x="([-\d.]+)" y="([-\d.]+)" width="([\d.]+)" height="([\d.]+)"')
TEXT = re.compile(r'<text x="([-\d.]+)" y="([-\d.]+)"[^>]*font-size="([\d.]+)"[^>]*'
                  r'text-anchor="(\w+)"[^>]*>([^<]*)</text>')
CIRCLE = re.compile(r'<circle cx="([-\d.]+)" cy="([-\d.]+)" r="([\d.]+)"')


@pytest.fixture(scope="module")
def charts():
    backends = json.loads((DATA / "backends.json").read_text())
    agents = json.loads((DATA / "agents.json").read_text())
    return {
        "throughput": rc.throughput_chart(backends),
        "agents": rc.agents_chart(agents),
    }


@pytest.mark.parametrize("name", ["throughput", "agents"])
def test_nothing_escapes_the_viewbox(charts, name):
    svg = charts[name]
    w, h = (float(v) for v in VIEWBOX.search(svg).groups())

    for x, y, rw, rh in RECT.findall(svg):
        assert float(x) + float(rw) <= w + 0.5, f"{name}: a rect runs past the right edge"
        assert float(y) + float(rh) <= h + 0.5, f"{name}: a rect runs past the bottom"

    for cx, _cy, r in CIRCLE.findall(svg):
        assert float(cx) + float(r) <= w + 0.5, f"{name}: a marker runs past the right edge"

    for x, y, size, anchor, content in TEXT.findall(svg):
        x, size = float(x), float(size)
        span = rc.text_width(content, size)
        right = x + span if anchor == "start" else x + span / 2 if anchor == "middle" else x
        left = x if anchor == "start" else x - span / 2 if anchor == "middle" else x - span
        assert left >= -0.5, f"{name}: {content!r} runs past the left edge"
        assert right <= w + 0.5, f"{name}: {content!r} runs past the right edge ({right:.0f} > {w:.0f})"
        assert 0 <= float(y) <= h, f"{name}: {content!r} sits outside vertically"


@pytest.mark.parametrize("name", ["throughput", "agents"])
def test_both_themes_are_defined(charts, name):
    """The chart renders on GitHub in whichever theme the reader uses, and a
    colour defined only for one of them shows as text on its own background."""
    svg = charts[name]
    assert "prefers-color-scheme: dark" in svg
    light_vars = set(re.findall(r"--([\w-]+):", svg.split("@media")[0]))
    dark_vars = set(re.findall(r"--([\w-]+):", svg.split("@media")[1]))
    assert light_vars == dark_vars, f"{name}: {light_vars ^ dark_vars} defined in only one theme"
    assert light_vars, "no theme variables at all"


def test_throughput_plots_the_measured_numbers(charts):
    """Guards against the chart and the data drifting apart."""
    data = json.loads((DATA / "backends.json").read_text())
    plotted = {r["backend"] for r in data["rows"] if r["backend"] in rc.THROUGHPUT_SERIES}
    assert plotted == set(rc.THROUGHPUT_SERIES), "a plotted series is missing from the data"
    assert rc.LABELLED <= set(rc.THROUGHPUT_SERIES)
    assert len(rc.LABELLED) <= 4, "direct labels are capped at four to avoid collisions"


def test_agent_chart_covers_every_bundled_agent():
    from rummi.agents import REGISTRY

    data = json.loads((DATA / "agents.json").read_text())
    assert {a["name"] for a in data["agents"]} == set(REGISTRY)
    wins = [a["win_rate"] for a in data["agents"]]
    assert wins == sorted(wins), "the ladder must be plotted weakest to strongest"


def test_series_colours_come_from_the_validated_palette(charts):
    """Colours are validated by the palette script, so the chart must not invent
    any of its own."""
    allowed = set(rc.SERIES_LIGHT + rc.SERIES_DARK + rc.ORDINAL_LIGHT + rc.ORDINAL_DARK)
    allowed |= {rc.INK[i] for i in (0, 1)} | {rc.INK_2[i] for i in (0, 1)}
    allowed |= {rc.MUTED[i] for i in (0, 1)} | {rc.GRID[i] for i in (0, 1)}
    allowed |= {rc.AXIS[i] for i in (0, 1)} | {"#fcfcfb", "#1a1a19", "#ffffff", "#0b0b0b"}
    for svg in charts.values():
        for hexcode in set(re.findall(r"#[0-9a-fA-F]{6}", svg)):
            assert hexcode.lower() in allowed, f"{hexcode} is not in the validated palette"


def test_every_published_capture_names_the_current_protocol():
    """A version bump without a re-capture leaves the published numbers claiming a
    protocol they were not produced under, which is the one way these files can
    lie. Covers the experimental captures too -- they are scored through the same
    frozen suites, so they go stale the same way. Regenerate with the commands in
    CLAUDE.md."""
    from rummi.evaluate.protocol import PROTOCOL_VERSION

    captures = sorted([*DATA.glob("agents*.json"), *DATA.glob("experiments*.json")])
    assert captures, "no agent captures found"
    for path in captures:
        payload = json.loads(path.read_text())
        assert payload["protocol"] == PROTOCOL_VERSION, (
            f"{path.name} was captured under {payload['protocol']}, "
            f"protocol is now {PROTOCOL_VERSION}"
        )


def test_a_capture_exists_for_every_suite_the_readme_publishes():
    """The seat-count table in README.md reads these three files."""
    for name in ("agents.json", "agents-standard-3p.json", "agents-standard-4p.json"):
        payload = json.loads((DATA / name).read_text())
        assert payload["agents"], name
        wins = [a["win_rate"] for a in payload["agents"]]
        assert wins == sorted(wins), f"{name}: the ladder is out of order"
