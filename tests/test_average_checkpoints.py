"""The SWA recipe: a mean of state dicts, and the two ways it must refuse.

`tools/average_checkpoints.py` exists because the averaging that produced the
shipped `afterstate-sweep-s{n}-swa.pt` was never committed -- three published rows
rested on an artefact with no recipe. The tests that can run anywhere are on
synthetic checkpoints; the one that closes the gap needs the artefacts, which are
gitignored, so it skips where they are absent and asserts bit-identity where they
are not.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import average_checkpoints as tool


def _write(path: Path, value: float, *, width: int = 3, **meta: object) -> Path:
    torch.save(
        {
            "cfg": "standard",
            "hidden": [width],
            "dim": width,
            "repartition": False,
            **meta,
            "state": {
                "net.0.weight": torch.full((width, width), value),
                "net.0.bias": torch.arange(width, dtype=torch.float32) * value,
            },
        },
        path,
    )
    return path


def test_the_mean_is_the_mean_and_the_metadata_comes_through(tmp_path: Path) -> None:
    paths = [_write(tmp_path / "a.pt", 1.0), _write(tmp_path / "b.pt", 3.0)]
    merged = tool.average_checkpoints(paths)

    assert set(merged) == {"cfg", "hidden", "dim", "repartition", "state"}
    assert (merged["cfg"], merged["hidden"], merged["dim"]) == ("standard", [3], 3)
    assert merged["repartition"] is False
    torch.testing.assert_close(
        merged["state"]["net.0.weight"], torch.full((3, 3), 2.0)
    )
    torch.testing.assert_close(
        merged["state"]["net.0.bias"], torch.tensor([0.0, 2.0, 4.0])
    )
    # Averaged in float32 by default, and handed back in the tensors' own dtype --
    # a loader building the net from `hidden` gets what it expects.
    assert merged["state"]["net.0.weight"].dtype == torch.float32


def test_averaging_one_checkpoint_is_that_checkpoint(tmp_path: Path) -> None:
    """The batch of one, so a single-element series is not a special case."""
    merged = tool.average_checkpoints([_write(tmp_path / "a.pt", 1.5)])
    torch.testing.assert_close(
        merged["state"]["net.0.weight"], torch.full((3, 3), 1.5)
    )


def test_two_runs_of_different_shape_are_refused(tmp_path: Path) -> None:
    paths = [
        _write(tmp_path / "a.pt", 1.0, width=3),
        _write(tmp_path / "b.pt", 1.0, width=4),
    ]
    # The metadata carries the width, so the disagreement is caught before the
    # tensors are touched -- but either check alone has to be enough.
    with pytest.raises(ValueError, match="different runs"):
        tool.average_checkpoints(paths)
    with pytest.raises(ValueError, match="shape"):
        tool.average_states(
            [
                torch.load(path, map_location="cpu", weights_only=True)["state"]
                for path in paths
            ]
        )


def test_a_different_parameter_set_is_refused() -> None:
    with pytest.raises(ValueError, match="parameter set"):
        tool.average_states(
            [{"a": torch.zeros(2)}, {"a": torch.zeros(2), "b": torch.zeros(2)}]
        )


def test_averaging_nothing_says_so() -> None:
    with pytest.raises(ValueError, match="nothing to average"):
        tool.average_states([])


def test_disagreeing_metadata_is_refused_by_name(tmp_path: Path) -> None:
    """One run with `REPARTITION` and one without average to neither run's net."""
    paths = [
        _write(tmp_path / "a.pt", 1.0),
        _write(tmp_path / "b.pt", 1.0, repartition=True),
    ]
    with pytest.raises(ValueError, match="repartition"):
        tool.average_checkpoints(paths)


def test_the_series_is_inclusive_at_both_ends() -> None:
    """`docs/EXPERIMENTS.md` says sixteen checkpoints for u300-u600 every 20."""
    paths = tool.series_paths(Path("checkpoints/sweep-s0"), 300, 600, 20)
    assert len(paths) == 16
    assert paths[0].name == "sweep-s0-u300.pt"
    assert paths[-1].name == "sweep-s0-u600.pt"

    with pytest.raises(ValueError, match="step must be positive"):
        tool.series_paths(Path("x"), 300, 600, 0)
    with pytest.raises(ValueError, match="range is empty"):
        tool.series_paths(Path("x"), 600, 300, 20)


SEEDS = (0, 1, 2)
CHECKPOINTS = ROOT / "checkpoints"


@pytest.mark.parametrize("seed", SEEDS)
def test_the_recipe_reproduces_the_shipped_swa_checkpoint(seed: int) -> None:
    """The point of the tool: the artefact three published rows rest on, re-derived.

    Bit-identity rather than a tolerance, because that is what the float32
    accumulation buys and it is the only reading that proves the recipe is *the*
    one, not merely a close one.
    """
    shipped = CHECKPOINTS / f"afterstate-sweep-s{seed}-swa.pt"
    parts = tool.series_paths(CHECKPOINTS / f"afterstate-sweep-s{seed}", 300, 600, 20)
    if not shipped.exists() or not all(path.exists() for path in parts):
        pytest.skip("checkpoints/ is gitignored; this runs where the artefacts are")

    merged = tool.average_checkpoints(parts)
    want = torch.load(shipped, map_location="cpu", weights_only=True)
    assert {k: v for k, v in merged.items() if k != "state"} == {
        k: v for k, v in want.items() if k != "state"
    }
    for name, tensor in want["state"].items():
        assert torch.equal(merged["state"][name], tensor), name
