"""The config's derivations, and the one way it is easy to misuse.

Everything downstream takes its shapes from a `RummiConfig`, so a capacity that
comes out wrong is not a small bug -- it is a table that cannot represent a legal
position.
"""

from dataclasses import fields, replace

import pytest

from rummi.rules.config import (
    CONFIG_BY_NAME,
    STANDARD,
    STANDARD_3P,
    STANDARD_4P,
    RummiConfig,
)


@pytest.mark.parametrize("preset,seats", [(STANDARD_3P, 3), (STANDARD_4P, 4)])
def test_the_seat_count_presets_match_a_fresh_construction(preset: RummiConfig, seats: int):
    """`replace` carries resolved capacities over, so a preset built that way is
    only correct while no capacity is derived from the field being replaced. This
    is what licenses defining them with `replace` at all."""
    fresh = RummiConfig(n_players=seats)
    differing = {
        f.name for f in fields(RummiConfig)
        if getattr(preset, f.name) != getattr(fresh, f.name)
    }
    assert not differing, f"replace() diverged from a fresh build on {differing}"


def test_replace_does_carry_stale_capacities_for_other_fields():
    """The flip side, pinned so the licence above is not mistaken for a general
    one: change a field a capacity *is* derived from and the old value rides
    along."""
    assert replace(STANDARD, n_numbers=5).max_set_len == STANDARD.max_set_len
    assert RummiConfig(n_numbers=5).max_set_len == 5


@pytest.mark.parametrize("name", sorted(CONFIG_BY_NAME))
def test_every_preset_resolves_its_capacities(name: str):
    """`-1` means "derive me" and must never survive construction, or it would
    reach the simulator as a negative array dimension."""
    cfg = CONFIG_BY_NAME[name]
    for field in ("max_sets", "max_set_len", "max_micro_per_turn", "joker_penalty"):
        assert getattr(cfg, field) > 0, f"{name}.{field} was left unresolved"


@pytest.mark.parametrize("name", sorted(CONFIG_BY_NAME))
def test_every_preset_can_deal_every_seat_a_full_rack(name: str):
    cfg = CONFIG_BY_NAME[name]
    assert cfg.rack_size * cfg.n_players <= cfg.n_tiles


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"n_players": 1}, "n_players must be >= 2"),
        ({"min_set": 1}, "min_set must be >= 2"),
        ({"min_set": 20}, "min_set > n_numbers"),
        ({"max_set_len": 2}, "max_set_len < min_set"),
        ({"n_players": 8}, "not enough tiles"),
        ({"n_jokers": -1}, "n_jokers must be >= 0"),
    ],
)
def test_an_impossible_variant_is_rejected_at_construction(kwargs: dict, message: str):
    with pytest.raises(ValueError, match=message):
        RummiConfig(**kwargs)
