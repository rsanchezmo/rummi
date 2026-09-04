"""The stuck-state diagnostic has to keep running against the agent it measures.

Every number this tool prints is a count taken inside `by_value`'s own `choose`,
and its greedy columns are read from `greedy_agent`'s private helpers on purpose --
"what would greedy do here" is only answerable by asking greedy. That coupling is
the thing worth a test: a rename inside the agent breaks the tool at import, and a
diagnostic nobody runs until the next investigation is a diagnostic that is broken
when it is needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("gymnasium")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import diagnose_stuck as tool


def test_the_sweep_counts_the_states_it_claims_to() -> None:
    counts = tool.diagnose("tiny_groups", envs=4, steps=120, seed=0)
    assert counts["decisions"] > 0, "no decision was reached, so nothing was counted"
    # Every decision picks exactly one macro block, which is what makes the
    # `chose_*` columns a partition rather than five independent tallies.
    chosen = sum(
        counts[name]
        for name in ("chose_set", "chose_extend", "chose_steal", "chose_end", "chose_draw")
    )
    assert chosen == counts["decisions"]
    # The greedy columns are only reached from a stuck state, and the whole point of
    # the sweep is that they are reachable.
    assert counts["stuck"] > 0
    assert counts["stuck_greedy_acts"] <= counts["stuck"]
    assert counts["stuck_newset_gap"] <= counts["stuck"]
