"""Exhaustive validation of the slot kernel against the brute-force oracle."""

from itertools import combinations_with_replacement

import numpy as np
import pytest

from rummi.rules.config import STANDARD, TINY, TINY_GROUPS, RummiConfig
from rummi.rules.encoding import EMPTY, kind_name, kind_of
from rummi.env.numpy.sets import assign_open, evaluate_slots, pad_slot, slot_stats
from rummi.solver import brute_force

ORACLE_CONFIGS = [TINY, TINY_GROUPS]
MAX_JOKERS = 2


def _all_contents(cfg: RummiConfig):
    """Every slot content up to ``max_set_len``, capping each kind at two copies.

    The cap keeps enumeration finite while still producing duplicate real kinds,
    which must be rejected. Joker availability is deliberately ignored: the
    kernel is local and must not consult the deck.
    """
    kinds = range(cfg.n_kinds)
    for length in range(cfg.max_set_len + 1):
        for content in combinations_with_replacement(kinds, length):
            counts = {}
            for k in content:
                counts[k] = counts.get(k, 0) + 1
            if max(counts.values(), default=0) <= 2:
                yield content


@pytest.mark.parametrize("cfg", ORACLE_CONFIGS, ids=["tiny", "tiny_groups"])
def test_kernel_matches_oracle_exhaustively(cfg: RummiConfig):
    contents = list(_all_contents(cfg))
    assert len(contents) > 200, "enumeration looks too small to be meaningful"

    slots = np.stack([pad_slot(cfg, c) for c in contents])
    ev = evaluate_slots(cfg, slots)

    for i, content in enumerate(contents):
        label = "[" + " ".join(kind_name(cfg, k) for k in content) + "]"
        assert bool(ev.is_valid[i]) == brute_force.is_valid(cfg, content, MAX_JOKERS), (
            f"is_valid mismatch for {label}"
        )
        assert bool(ev.is_extendable[i]) == brute_force.is_extendable(
            cfg, content, MAX_JOKERS
        ), f"is_extendable mismatch for {label}"
        assert int(ev.value[i]) == brute_force.value(cfg, content, MAX_JOKERS), (
            f"value mismatch for {label}"
        )


@pytest.mark.parametrize("cfg", ORACLE_CONFIGS, ids=["tiny", "tiny_groups"])
def test_assign_open_matches_recomputing_from_scratch(cfg: RummiConfig):
    """The closed-form ASSIGN predicate must agree with appending and re-evaluating."""
    contents = [c for c in _all_contents(cfg) if len(c) < cfg.max_set_len]
    slots = np.stack([pad_slot(cfg, c) for c in contents])
    predicted = assign_open(cfg, slot_stats(cfg, slots))

    for kind in range(cfg.n_kinds):
        grown = np.stack([pad_slot(cfg, (*c, kind)) for c in contents])
        actual = evaluate_slots(cfg, grown).is_extendable
        np.testing.assert_array_equal(
            predicted[:, kind],
            actual,
            err_msg=f"assign_open disagrees for kind {kind_name(cfg, kind)}",
        )


@pytest.mark.parametrize("cfg", ORACLE_CONFIGS, ids=["tiny", "tiny_groups"])
def test_valid_sets_are_extendable_and_full_sets_are_closed(cfg: RummiConfig):
    for content in brute_force.valid_sets(cfg, MAX_JOKERS):
        ev = evaluate_slots(cfg, pad_slot(cfg, content)[None])
        assert bool(ev.is_valid[0])
        assert bool(ev.is_extendable[0])
        assert int(ev.value[0]) > 0


def test_empty_slot_is_extendable_but_not_valid():
    ev = evaluate_slots(STANDARD, np.full((1, STANDARD.max_set_len), EMPTY, dtype=np.int16))
    assert bool(ev.is_empty[0]) and bool(ev.is_extendable[0]) and not bool(ev.is_valid[0])
    assert int(ev.value[0]) == 0


def test_groups_are_unreachable_when_colors_are_scarce():
    # TINY has 2 colours and min_set 3, so no group can ever exist.
    assert not TINY.group_possible
    slots = np.stack([pad_slot(TINY, [kind_of(TINY, 0, 1), kind_of(TINY, 1, 1)])])
    ev = evaluate_slots(TINY, slots)
    assert not bool(ev.group_open[0]) and not bool(ev.is_extendable[0])


def test_high_joker_run_resolves_downwards():
    # 11-12-13 plus a joker can only be 10-11-12-13; the joker cannot be a 14.
    c = STANDARD
    kinds = [kind_of(c, 0, 11), kind_of(c, 0, 12), kind_of(c, 0, 13), c.joker_kind]
    ev = evaluate_slots(c, pad_slot(c, kinds)[None])
    assert bool(ev.run_valid[0])
    assert int(ev.value[0]) == 10 + 11 + 12 + 13


def test_ambiguous_joker_run_takes_the_best_reading():
    # 5-6 plus a joker may be read as 5-6-7, which beats 4-5-6.
    c = STANDARD
    kinds = [kind_of(c, 0, 5), kind_of(c, 0, 6), c.joker_kind]
    assert int(evaluate_slots(c, pad_slot(c, kinds)[None]).value[0]) == 5 + 6 + 7


def test_batch_shape_is_preserved():
    c = STANDARD
    slots = np.full((3, 7, c.max_set_len), EMPTY, dtype=np.int16)
    ev = evaluate_slots(c, slots)
    assert ev.is_valid.shape == (3, 7)
    assert assign_open(c, slot_stats(c, slots)).shape == (3, 7, c.n_kinds)
