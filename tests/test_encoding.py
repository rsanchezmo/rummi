import numpy as np
import pytest

from rummi.rules.config import STANDARD, TINY, TINY_GROUPS, RummiConfig
from rummi.rules.encoding import (
    EMPTY,
    kind_name,
    kind_of,
    kinds_to_counts,
    counts_to_kinds,
    tables,
)

ALL_CONFIGS = [STANDARD, TINY, TINY_GROUPS]


@pytest.mark.parametrize("cfg", ALL_CONFIGS)
def test_kind_roundtrip_covers_every_numbered_kind(cfg: RummiConfig):
    t = tables(cfg)
    seen = set()
    for color in range(cfg.n_colors):
        for number in range(1, cfg.n_numbers + 1):
            k = kind_of(cfg, color, number)
            assert 0 <= k < cfg.n_numbered_kinds
            assert (t.color[k], t.number[k]) == (color, number)
            seen.add(k)
    assert seen == set(range(cfg.n_numbered_kinds))


@pytest.mark.parametrize("cfg", ALL_CONFIGS)
def test_runs_are_contiguous_in_kind_space(cfg: RummiConfig):
    # A run is consecutive numbers in one colour, so the encoding must make it a
    # contiguous kind range; masks and set checks rely on this.
    for color in range(cfg.n_colors):
        base = kind_of(cfg, color, 1)
        for number in range(1, cfg.n_numbers + 1):
            assert kind_of(cfg, color, number) == base + number - 1


@pytest.mark.parametrize("cfg", ALL_CONFIGS)
def test_table_shapes_and_totals(cfg: RummiConfig):
    t = tables(cfg)
    for arr in (t.color, t.number, t.value, t.is_joker, t.copies):
        assert arr.shape == (cfg.n_kinds,)
    assert t.total_copies == cfg.n_tiles
    assert t.is_joker.sum() == 1
    assert t.is_joker[cfg.joker_kind]
    assert t.copies[cfg.joker_kind] == cfg.n_jokers
    assert t.color[cfg.joker_kind] == -1
    assert t.value[cfg.joker_kind] == 0
    np.testing.assert_array_equal(t.value[: cfg.n_numbered_kinds], t.number[: cfg.n_numbered_kinds])


@pytest.mark.parametrize("cfg", ALL_CONFIGS)
def test_tables_are_memoised(cfg: RummiConfig):
    assert tables(cfg) is tables(cfg)


@pytest.mark.parametrize("cfg", ALL_CONFIGS)
def test_action_layout_is_a_contiguous_partition(cfg: RummiConfig):
    blocks = [
        (cfg.place_offset, cfg.n_kinds),
        (cfg.pick_offset, cfg.max_sets * cfg.max_set_len),
        (cfg.dissolve_offset, cfg.max_sets),
        (cfg.assign_offset, cfg.n_kinds * cfg.max_sets),
        (cfg.end_turn_action, 1),
        (cfg.draw_action, 1),
    ]
    cursor = 0
    for offset, size in blocks:
        assert offset == cursor
        cursor += size
    assert cursor == cfg.n_actions


def test_kind_names():
    assert kind_name(STANDARD, EMPTY) == "."
    assert kind_name(STANDARD, STANDARD.joker_kind) == "*"
    assert kind_name(STANDARD, kind_of(STANDARD, 0, 1)) == "R1"
    assert kind_name(STANDARD, kind_of(STANDARD, 3, 13)) == "K13"


def test_counts_kinds_roundtrip():
    counts = np.array([2, 0, 1, 3, 0], dtype=np.int16)
    cfg = RummiConfig(n_colors=2, n_numbers=2, n_copies=3, n_jokers=1, rack_size=2, min_set=2)
    assert cfg.n_kinds == 5
    np.testing.assert_array_equal(kinds_to_counts(cfg, counts_to_kinds(counts)), counts)


def test_bad_config_is_rejected():
    with pytest.raises(ValueError, match="min_set > n_numbers"):
        RummiConfig(n_numbers=2, min_set=3)
    with pytest.raises(ValueError, match="n_players"):
        RummiConfig(n_players=1)
    with pytest.raises(ValueError, match="deal every player a full rack"):
        RummiConfig(n_players=4, rack_size=30, n_numbers=5, n_colors=2, n_copies=1, n_jokers=0)


def test_bad_kind_lookup_is_rejected():
    with pytest.raises(ValueError, match="color"):
        kind_of(STANDARD, 4, 1)
    with pytest.raises(ValueError, match="number"):
        kind_of(STANDARD, 0, 0)
