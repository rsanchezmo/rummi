"""The suite arm has to be scored on deals from the config it built its arms for.

`tools/eval_primitive_turn.py` builds every arm on `--config` and then scores them
against a protocol suite. The suite used to be picked by *name*, and the names do
not line up with the configs: `--config tiny` selected the suite called `tiny`,
which deals `tiny_groups`, and `--config standard_3p` selected it too -- 53-kind
agents scored on a 13-kind two-seat board, with nothing downstream to catch it
because `evaluate` hands `build_agent` the suite's config and these arms are built
already.

The resolution itself is now :func:`rummi.evaluate.protocol.suite_for`, so what is
left to test here is that this tool asks it *before* it reads a checkpoint -- an
unscorable config should cost nothing to find out about.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("ortools")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import eval_primitive_turn as tool

from rummi.evaluate.protocol import suite_for
from rummi.rules.config import CONFIG_BY_NAME


def test_the_standard_config_still_scores_on_the_published_suite() -> None:
    """The suite every number in the tool's docstring was measured against."""
    assert suite_for(CONFIG_BY_NAME["standard"]).name == "standard-greedy"
    assert suite_for(CONFIG_BY_NAME["tiny_groups"]).name == "tiny"


def test_a_config_with_no_suite_is_refused_before_anything_is_loaded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Refused at the flags, so it costs nothing and names the real problem."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_primitive_turn.py",
            "--config", "tiny",
            "--games", "1",
            "--data", "no-such-collection.npz",
            "--init", "no-such-checkpoint.pt",
        ],
    )
    with pytest.raises(SystemExit):
        tool.main()
