"""The game-structure tool: the mirror is exact, and the file it writes round-trips.

Four deals is not a measurement, but the mirror is not a statistic either -- an agent
played against itself must score exactly 50%, and if it does not, the swap is broken
rather than noisy.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

pytest.importorskip("numpy")
pytest.importorskip("ortools")

import numpy as np

import game_structure as gs
from rummi.rules.config import CONFIG_BY_NAME, STANDARD

DEALS = 4


@pytest.fixture(scope="module")
def mirror() -> dict:
    seeds = [np.random.SeedSequence([12_000, i]) for i in range(DEALS)]
    forward = gs.play(STANDARD, ("frugal", "frugal"), seeds)
    reverse = gs.play(STANDARD, ("frugal", "frugal"), seeds)
    return gs.summarise("frugal", "frugal", DEALS, forward, reverse)


def test_an_agent_against_itself_scores_exactly_even(mirror):
    assert mirror["a_win_rate"] == 0.5
    assert mirror["a_wins_both_seats"] == 0
    assert mirror["b_wins_both_seats"] == 0
    assert mirror["seat_decides"] == DEALS


def test_every_deal_finishes_with_a_winner_holding_nothing(mirror):
    assert mirror["stalemate_rate"] == 0.0
    assert mirror["games"] == 2 * DEALS
    assert mirror["loser_tiles"]["mean"] > 0
    assert sum(mirror["plays_per_seat"]) > 0, "greedy-or-better must reach END_TURN"


def test_the_json_round_trips_and_merges_by_pairing(mirror, tmp_path):
    path = tmp_path / "game_structure.json"
    path.write_text(json.dumps(gs.merge(path, "standard", mirror), indent=1))
    other = {**mirror, "a": "greedy"}
    path.write_text(json.dumps(gs.merge(path, "standard", other), indent=1))

    payload = json.loads(path.read_text())
    assert payload["config"] == "standard"
    assert sorted(payload["pairings"]) == ["frugal-vs-frugal", "greedy-vs-frugal"]
    assert payload["pairings"]["frugal-vs-frugal"] == mirror


def test_a_config_with_more_than_two_seats_is_refused(monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["game_structure", "--config", "standard_3p", "--deals", "1"]
    )
    assert CONFIG_BY_NAME["standard_3p"].n_players == 3
    with pytest.raises(SystemExit, match="two-seat only"):
        gs.main()
