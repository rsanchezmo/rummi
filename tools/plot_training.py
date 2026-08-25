"""Training curves from `--log-json` output.

matplotlib rather than the hand-written SVG in `render_charts.py`: that module
exists because the README's two figures need to be theme-aware inside a GitHub
`<img>`, which matplotlib cannot do. A training plot has no such constraint, and
laying out axes by hand is how the first attempt at this ended up with a clipped
legend and a title on top of its own subtitle.

    python tools/plot_training.py --out docs/training.png
"""

from __future__ import annotations

import argparse
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# The same validated hues render_charts.py uses, so the figures agree.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]

PANELS = [
    ("win_rate", "Win rate vs greedy", 0.50, "greedy = 50%"),
    ("end_turn", "Turns completed", 0.36, "greedy = 36%"),
    ("melded", "Has opened", 0.92, "greedy = 92%"),
]
"""One panel per metric, never two scales on one axis. The reference line is what
makes each panel readable: without it "6%" does not obviously mean "far away"."""


def smooth(values: list[float], window: int = 9) -> list[float]:
    """Trailing mean. Win rate is a share of ~5 finished episodes per update, so
    raw it is mostly quantisation noise."""
    out = []
    for i in range(len(values)):
        chunk = [v for v in values[max(0, i - window + 1) : i + 1] if v == v]
        out.append(sum(chunk) / len(chunk) if chunk else float("nan"))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=pathlib.Path, default=pathlib.Path("docs/data/training.json"))
    p.add_argument("--out", type=pathlib.Path, default=pathlib.Path("docs/training.png"))
    p.add_argument("--dpi", type=int, default=140)
    args = p.parse_args()

    data = json.loads(args.data.read_text())
    runs = data["runs"]

    fig, axes = plt.subplots(len(PANELS), 1, figsize=(9, 3.1 * len(PANELS)), sharex=True)
    fig.suptitle(
        f"PPO after cloning greedy — {data['config']} config",
        fontsize=13, fontweight="bold", y=0.985,
    )

    for ax, (key, title, reference, note) in zip(axes, PANELS, strict=True):
        for i, run in enumerate(runs):
            steps = [h["steps"] / 1e6 for h in run["history"]]
            ax.plot(
                steps, smooth([h[key] for h in run["history"]]),
                color=SERIES[i], linewidth=2, label=run["label"], solid_capstyle="round",
            )
        ax.axhline(reference, color="#898781", linewidth=1, linestyle="--", zorder=0)
        # Left-anchored: the legend sits top-right, and the two collided there.
        ax.annotate(
            note, xy=(0.0, reference), xycoords=("axes fraction", "data"),
            xytext=(6, 4), textcoords="offset points",
            ha="left", va="bottom", fontsize=9, color="#898781",
        )
        ax.set_title(title, fontsize=11, fontweight="bold", loc="left")
        ax.set_ylim(0, max(reference * 1.15, 0.05))
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        ax.grid(axis="y", color="#e1e0d9", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    axes[-1].set_xlabel("env steps (millions)")
    # One legend for the figure, below it: per-axes legends had nowhere to sit that
    # did not overlap either the reference line's label or the curves themselves.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, frameon=False, fontsize=10, ncols=len(runs),
        loc="lower center", bbox_to_anchor=(0.5, 0.0),
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.975))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi)
    print(f"wrote {args.out}  ({args.out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
