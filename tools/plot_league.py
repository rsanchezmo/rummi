"""League curves from `train_macro.py --log-json`.

    python tools/plot_league.py runs/*.json --out league.png

`plot_training.py` plots the primitive trainer, whose panels are per-step rates
against a fixed opponent. A macro run with `--opponent greedy,self` has a second
thing to show and it is the one that matters: the learner's score is split *per
opponent*, so "am I improving" and "did my opponent get worse" stop being the same
line. The palette and the trailing mean come from that module rather than being
restated here.

Every line is one seed. Colour is the arm, never the seed -- the question these
figures answer is whether self-play changes anything, and a run's identity is
which arm it belongs to.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from plot_training import SERIES, smooth

MUTED, GRID = "#898781", "#e1e0d9"

ARM_COLOR = {"greedy": SERIES[0], "greedy,self": SERIES[1]}
"""Two arms, two hues, taken in slot order. Seeds share their arm's colour."""


def series(run: dict, opponent: str, key: str) -> list[float]:
    """One metric for one pool member, `nan` where no episode closed that update."""
    out = []
    for row in run["history"]:
        found = next(
            (r for r in row.get("by_opponent", []) if r["opponent"] == opponent), None
        )
        value = found[key] if found else None
        out.append(float("nan") if value is None else float(value))
    return out


def arm_of(run: dict) -> str:
    return str(run["opponent"])


def events(run: dict, kind: str) -> list[int]:
    """Update numbers where a snapshot was promoted, or refused promotion."""
    return [row["update"] for row in run["history"] if row.get(kind)]


def style(ax, title: str, ylabel: str | None = None) -> None:
    ax.set_title(title, fontsize=11, fontweight="bold", loc="left")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, color=MUTED)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("logs", nargs="+", type=pathlib.Path, help="--log-json files")
    p.add_argument("--out", type=pathlib.Path, default=pathlib.Path("league.png"))
    p.add_argument("--dpi", type=int, default=140)
    p.add_argument(
        "--smooth", type=int, default=15,
        help="trailing-mean window. A single update closes ~30 episodes, so the raw "
             "terminal reward is mostly sampling noise",
    )
    args = p.parse_args()

    runs = [json.loads(path.read_text()) for path in args.logs]
    runs.sort(key=lambda r: (arm_of(r), r["seed"]))
    league = [r for r in runs if "self" in arm_of(r).split(",")]

    fig, axes = plt.subplots(3, 1, figsize=(9, 9.3), sharex=True)
    fig.suptitle(
        "Self-play A/B on the macro action space — standard config",
        fontsize=13, fontweight="bold", y=0.985,
    )

    # 1. The comparison. Both arms play greedy, so this is the one metric that means
    #    the same thing in each -- the self arm's own envs are a different question.
    for run in runs:
        axes[0].plot(
            [h["update"] for h in run["history"]],
            smooth(series(run, "greedy", "terminal"), args.smooth),
            color=ARM_COLOR[arm_of(run)], linewidth=2, solid_capstyle="round",
        )
    style(axes[0], "Terminal reward against greedy — one line per seed", "reward")

    # 2. The league. Against its own past selves a healthy run sits near zero: the
    #    snapshots are copies of the learner, so a run pulling far above them is
    #    beating a broken opponent, not a lagging one.
    for run in league:
        updates = [h["update"] for h in run["history"]]
        values = smooth(series(run, "self", "terminal"), args.smooth)
        axes[1].plot(
            updates, values, color=ARM_COLOR[arm_of(run)], linewidth=2,
            solid_capstyle="round",
        )
        at = dict(zip(updates, values, strict=True))
        for kind, marker in (("snapshot_refreshed", "^"), ("snapshot_held", "x")):
            marks = [(u, at[u]) for u in events(run, kind) if u in at]
            if marks:
                axes[1].plot(
                    *zip(*marks, strict=True), linestyle="none", marker=marker,
                    markersize=6, color=ARM_COLOR[arm_of(run)],
                )
    axes[1].axhline(0.0, color=MUTED, linewidth=1, linestyle="--", zorder=0)
    style(axes[1], "The league: terminal reward against the snapshot pool", "reward")
    axes[1].legend(
        handles=[
            Line2D([], [], linestyle="none", marker="^", color=MUTED, label="promoted"),
            Line2D([], [], linestyle="none", marker="x", color=MUTED, label="held back"),
        ],
        frameon=False, fontsize=9, loc="upper left",
    )

    # 3. Entropy collapsing is how a run stops exploring, and the reason to expect
    #    self-play to differ at all: a matched opponent should keep it up.
    for run in runs:
        axes[2].plot(
            [h["update"] for h in run["history"]],
            smooth([h["entropy"] for h in run["history"]], args.smooth),
            color=ARM_COLOR[arm_of(run)], linewidth=2, solid_capstyle="round",
        )
    style(axes[2], "Policy entropy", "nats")
    axes[2].set_xlabel("update")

    # One legend for the figure: the arms are the only identity in it, and three
    # per-axes copies of a two-entry legend is noise.
    fig.legend(
        handles=[
            Line2D([], [], color=ARM_COLOR[arm], linewidth=2, label=f"--opponent {arm}")
            for arm in sorted({arm_of(r) for r in runs})
        ],
        frameon=False, fontsize=10, ncols=2, loc="lower center", bbox_to_anchor=(0.5, 0.0),
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.975))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi)
    print(f"wrote {args.out}  ({args.out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
