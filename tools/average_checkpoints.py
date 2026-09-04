"""Plain weight averaging over a run's own checkpoint series -- the SWA recipe.

    python tools/average_checkpoints.py --series checkpoints/afterstate-sweep-s0 \
        --updates 300 600 20 --out checkpoints/afterstate-sweep-s0-swa.pt

That command, run for `s0`, `s1` and `s2`, reproduces the three shipped
`afterstate-sweep-s{n}-swa.pt` **tensor for tensor at 0.0 max absolute difference**
-- so the artefact three published rows rest on has a recipe in the repo again
rather than in a scratch script. `docs/EXPERIMENTS.md`' sixteen u300-u600
checkpoints are `range(300, 601, 20)`: both ends inclusive.

Two details are load-bearing for that exactness, and neither is obvious.

**The accumulation is float32**, not float64. Summing sixteen float32 tensors in
float64 and rounding once is the *more* accurate mean and it does not reproduce the
shipped files -- it lands 1.2e-7 to 3.6e-7 away. `--dtype float64` is offered for a
longer series, where the accumulation error is worth more than agreeing with an
existing artefact, but it is not the default.

**The order is ascending update**, because float32 addition is not associative. One
pass in `--updates` order, then a single divide.

Nothing here knows what a value net is. Every key beside `state` is carried through
unchanged and must *agree* across the inputs, which is what supplies `dim`,
`hidden`, `repartition` and `cfg` to
:func:`~rummi.agents.learned.afterstate_net.load_value_net` without this tool
naming any of them -- and what refuses to average two runs of different shape.
Provenance is deliberately not written into the file: the point of the recipe is
that its output can be compared against an artefact produced before it existed.

Averaging is only sound where the parameters are the whole model. These nets are
plain MLPs with no normalisation layers, which is why `docs/EXPERIMENTS.md` says
"plain weight averaging" and why there are no running statistics to re-estimate.
"""

from __future__ import annotations

import argparse
import pathlib
from collections.abc import Sequence
from typing import Any

import torch

STATE = "state"
"""The checkpoint key holding the tensors to average; everything else is metadata."""


def series_paths(prefix: pathlib.Path, lo: int, hi: int, every: int) -> list[pathlib.Path]:
    """`<prefix>-uNNN.pt` for each update in `range(lo, hi + 1, every)`.

    The naming is `tools/train_afterstate.py --out`'s own, and the range is
    inclusive at both ends because that is what "u300-u600" reads as -- sixteen
    files at `--every 20`, which is the count `docs/EXPERIMENTS.md` states.
    """
    if every <= 0:
        raise ValueError(f"--updates step must be positive, got {every}")
    if hi < lo:
        raise ValueError(f"--updates range is empty: {lo} to {hi}")
    return [
        prefix.with_name(f"{prefix.name}-u{update:03d}.pt")
        for update in range(lo, hi + 1, every)
    ]


def average_states(
    states: Sequence[dict[str, torch.Tensor]], dtype: torch.dtype = torch.float32
) -> dict[str, torch.Tensor]:
    """The elementwise mean of `states`, each tensor back in its own dtype.

    Refused rather than broadcast where the shapes or the key sets disagree: two
    nets of different width average to something that loads cleanly into neither.
    """
    if not states:
        raise ValueError("nothing to average")
    first = states[0]
    for i, state in enumerate(states[1:], start=1):
        if state.keys() != first.keys():
            missing = sorted(set(first) ^ set(state))
            raise ValueError(f"checkpoint {i} has a different parameter set: {missing}")
        for name, tensor in state.items():
            if tensor.shape != first[name].shape:
                raise ValueError(
                    f"checkpoint {i} has {name} of shape {tuple(tensor.shape)}, "
                    f"not {tuple(first[name].shape)}"
                )

    total = {name: tensor.to(dtype).clone() for name, tensor in first.items()}
    for state in states[1:]:
        for name in total:
            total[name] += state[name].to(dtype)
    return {
        name: (tensor / len(states)).to(first[name].dtype)
        for name, tensor in total.items()
    }


def merge_metadata(checkpoints: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Every key beside `state`, carried through, and refused where they disagree.

    A disagreement is the averaging being asked for across two different runs --
    two configs, two widths, one with `REPARTITION` and one without -- and the
    result would carry one run's metadata over the other's weights.
    """
    first = checkpoints[0]
    keys = [key for key in first if key != STATE]
    for i, checkpoint in enumerate(checkpoints[1:], start=1):
        for key in keys:
            if key not in checkpoint:
                raise ValueError(f"checkpoint {i} is missing {key!r}")
            if checkpoint[key] != first[key]:
                raise ValueError(
                    f"checkpoint {i} has {key}={checkpoint[key]!r}, "
                    f"not {first[key]!r}; these are different runs"
                )
    return {key: first[key] for key in keys}


def average_checkpoints(
    paths: Sequence[pathlib.Path], dtype: torch.dtype = torch.float32
) -> dict[str, Any]:
    """One checkpoint out of many: their shared metadata and their mean weights."""
    loaded = []
    for path in paths:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if STATE not in checkpoint:
            raise ValueError(f"{path} has no {STATE!r} to average")
        loaded.append(checkpoint)
    merged = merge_metadata(loaded)
    merged[STATE] = average_states([c[STATE] for c in loaded], dtype)
    return merged


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "checkpoints", type=pathlib.Path, nargs="*",
        help="the checkpoints to average, in the order they are summed",
    )
    p.add_argument(
        "--series", type=pathlib.Path, default=None,
        help="a `train_afterstate --out` prefix, expanded with --updates into "
             "<prefix>-uNNN.pt. The alternative to listing the paths",
    )
    p.add_argument(
        "--updates", type=int, nargs=3, metavar=("LO", "HI", "EVERY"),
        default=(300, 600, 20),
        help="which updates of --series to average, both ends inclusive",
    )
    p.add_argument(
        "--dtype", choices=("float32", "float64"), default="float32",
        help="accumulation precision. float32 is the default because it is what "
             "reproduces the shipped SWA checkpoints; float64 is the more accurate "
             "mean and does not",
    )
    p.add_argument("--out", type=pathlib.Path, required=True)
    args = p.parse_args()

    if bool(args.checkpoints) == (args.series is not None):
        p.error("pass either the checkpoint paths or --series, not both and not neither")
    paths = (
        list(args.checkpoints)
        if args.checkpoints
        else series_paths(args.series, *args.updates)
    )
    missing = [path for path in paths if not path.exists()]
    if missing:
        p.error(f"{len(missing)} of {len(paths)} checkpoints are missing: {missing[:3]}")

    merged = average_checkpoints(paths, getattr(torch, args.dtype))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, args.out)
    meta = {key: value for key, value in merged.items() if key != STATE}
    print(
        f"averaged {len(paths)} checkpoints in {args.dtype}\n"
        f"  from {paths[0].name} to {paths[-1].name}\n"
        f"  carried {meta}\n"
        f"wrote {args.out}"
    )


if __name__ == "__main__":
    main()
