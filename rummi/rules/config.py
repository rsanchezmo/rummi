"""Static configuration for a Rummikub variant.

Every shape in the simulator is derived from a :class:`RummiConfig`, so reduced
variants (fewer colours/numbers/copies) are first-class: they make brute-force
oracles and fast RL smoke tests tractable.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class RewardMode(str, Enum):
    """How terminal reward is credited to each seat."""

    WIN_LOSS = "win_loss"
    SCORE = "score"
    SCORE_NORMALIZED = "score_normalized"


@dataclass(frozen=True, slots=True)
class RummiConfig:
    # --- deck ---------------------------------------------------------------
    n_colors: int = 4
    n_numbers: int = 13
    n_copies: int = 2
    n_jokers: int = 2

    # --- table ---------------------------------------------------------------
    n_players: int = 2
    rack_size: int = 14
    initial_meld: int = 30
    min_set: int = 3
    strict_initial_meld: bool = True

    # --- capacities (None => derived, see __post_init__) ---------------------
    max_sets: int | None = None
    max_set_len: int | None = None
    max_micro_per_turn: int | None = None
    max_turns: int = 500

    # --- scoring / reward ----------------------------------------------------
    reward_mode: RewardMode = RewardMode.WIN_LOSS
    joker_penalty: int | None = None
    tiles_placed_bonus: float = 0.0
    rack_value_delta: float = 0.0
    micro_step_cost: float = 0.0

    def __post_init__(self) -> None:
        if self.n_colors < 1 or self.n_numbers < 1 or self.n_copies < 1:
            raise ValueError("n_colors, n_numbers and n_copies must all be >= 1")
        if self.n_jokers < 0:
            raise ValueError("n_jokers must be >= 0")
        if self.min_set < 2:
            raise ValueError("min_set must be >= 2")
        if self.n_players < 2:
            raise ValueError("n_players must be >= 2")
        if self.min_set > self.n_numbers:
            raise ValueError("min_set > n_numbers makes runs impossible")

        if self.max_set_len is None:
            object.__setattr__(self, "max_set_len", max(self.n_numbers, self.n_colors))
        if self.max_sets is None:
            # Enough slots that no legal table can ever be unrepresentable.
            object.__setattr__(self, "max_sets", max(1, self.n_tiles // self.min_set))
        if self.max_micro_per_turn is None:
            # Worst-case complete rearrangement: dissolve every slot, re-assign every
            # table tile, and play out the whole rack.
            object.__setattr__(
                self,
                "max_micro_per_turn",
                self.max_sets + self.n_tiles + self.rack_size,
            )
        if self.joker_penalty is None:
            # 30 for the standard 1-13 deck; scales down for reduced variants.
            object.__setattr__(self, "joker_penalty", 2 * self.n_numbers + 4)

        if self.max_set_len < self.min_set:
            raise ValueError("max_set_len < min_set makes every set invalid")
        if self.rack_size * self.n_players > self.n_tiles:
            raise ValueError("not enough tiles to deal every player a full rack")

    # --- derived sizes -------------------------------------------------------
    @property
    def n_numbered_kinds(self) -> int:
        return self.n_colors * self.n_numbers

    @property
    def n_kinds(self) -> int:
        """Distinct tile kinds, K. The last kind is the joker."""
        return self.n_numbered_kinds + 1

    @property
    def joker_kind(self) -> int:
        return self.n_numbered_kinds

    @property
    def n_tiles(self) -> int:
        return self.n_numbered_kinds * self.n_copies + self.n_jokers

    @property
    def group_possible(self) -> bool:
        """Groups need at least ``min_set`` distinct colours to exist at all."""
        return self.n_colors >= self.min_set

    @property
    def max_group_len(self) -> int:
        return self.n_colors

    # --- flat action layout --------------------------------------------------
    # PLACE | PICK | DISSOLVE | ASSIGN | END_TURN | DRAW
    @property
    def place_offset(self) -> int:
        return 0

    @property
    def pick_offset(self) -> int:
        return self.n_kinds

    @property
    def dissolve_offset(self) -> int:
        return self.pick_offset + self.max_sets * self.max_set_len

    @property
    def assign_offset(self) -> int:
        return self.dissolve_offset + self.max_sets

    @property
    def end_turn_action(self) -> int:
        return self.assign_offset + self.n_kinds * self.max_sets

    @property
    def draw_action(self) -> int:
        return self.end_turn_action + 1

    @property
    def n_actions(self) -> int:
        return self.draw_action + 1


TINY = RummiConfig(
    n_colors=2,
    n_numbers=5,
    n_copies=1,
    n_jokers=0,
    n_players=2,
    rack_size=4,
    initial_meld=6,
    max_sets=4,
    max_turns=60,
)
"""Small variant used by brute-force oracles and smoke tests."""

STANDARD = RummiConfig()

TINY_GROUPS = RummiConfig(
    n_colors=3,
    n_numbers=4,
    n_copies=1,
    n_jokers=1,
    n_players=2,
    rack_size=4,
    initial_meld=6,
    max_sets=4,
    max_turns=60,
)
"""Small variant that *can* form groups (n_colors >= min_set), unlike :data:`TINY`."""
