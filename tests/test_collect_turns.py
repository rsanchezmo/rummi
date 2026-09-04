"""The collector writes two populations, so it has to refuse when it has one.

`tools/collect_turns.py` records committed turns and answered gate states out of
one rollout, and the file it writes holds both under their own key prefixes. A run
that reaches its turn target without ever seeing a stuck state used to reach
`TurnStart.stack([])` and die inside `np.concatenate`, after the whole collection
had been paid for and with nothing said about which half was missing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("ortools")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import collect_turns as tool


def test_a_run_with_no_gate_states_says_which_half_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "turns.npz"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "collect_turns.py",
            "--config", "tiny",
            "--envs", "4",
            "--target", "20",
            "--stuck-target", "0",
            "--max-steps", "400",
            "--out", str(out),
        ],
    )
    with pytest.raises(SystemExit, match="gate states"):
        tool.main()
    assert not out.exists(), "a refused run must not leave half a dataset behind"
