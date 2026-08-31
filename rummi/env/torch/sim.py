"""Torch port of state, dealing, masks and the step function. See SPEC.md.

The one substantive departure from the NumPy reference is *how* effects are
applied. NumPy selects the envs that chose each action family with
``flatnonzero`` and updates just those rows; on an accelerator that read forces a
host synchronisation every step and serialises the whole pipeline. Here every
family is applied to the entire batch as a masked update, so a step is a fixed
sequence of shape-static kernels with no sync and nothing data-dependent -- which
is also what makes it safe to ``torch.compile``.

Two consequences worth knowing:

* Slots are re-sorted every step rather than only the slot that changed. Sorting
  is idempotent on an already-sorted slot, so this is correct, and unconditional
  work is cheaper than a branch here.
* Canonical slot *order* is likewise recomputed every step and then selected for
  with ``where``, using a radix of packed integer keys rather than a lexsort.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np
import torch

from rummi.rules.config import RewardMode, RummiConfig
from rummi.rules.encoding import EMPTY, tables
from rummi.env.torch.kernel import (
    SlotStats,
    SlotSummary,
    assign_codes,
    lookup,
    summarize,
)

NO_WINNER = -1
HISTORY_LEN = 8


@dataclass(slots=True)
class TorchState:
    cfg: RummiConfig
    racks: torch.Tensor
    table_sets: torch.Tensor
    workbench: torch.Tensor
    placed_rack: torch.Tensor
    slot_new: torch.Tensor
    table_snapshot: torch.Tensor
    pool: torch.Tensor
    deck_order: torch.Tensor
    draw_ptr: torch.Tensor
    melded: torch.Tensor
    current: torch.Tensor
    micro_count: torch.Tensor
    turn_count: torch.Tensor
    consecutive_draws: torch.Tensor
    last_action: torch.Tensor
    action_history: torch.Tensor
    winner: torch.Tensor
    done: torch.Tensor
    truncated: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(self.racks.shape[0])

    @property
    def device(self) -> torch.device:
        return self.racks.device

    @property
    def pool_size(self) -> torch.Tensor:
        return self.cfg.n_tiles - self.draw_ptr

    def clone(self) -> TorchState:
        return TorchState(
            cfg=self.cfg,
            **{f.name: getattr(self, f.name).clone() for f in fields(self) if f.name != "cfg"},
        )

    def table_counts(self) -> torch.Tensor:
        return counts_of(self.cfg, self.table_sets)

    def rack_values(self) -> torch.Tensor:
        return self.racks @ lookup(self.cfg, self.device).offload

    def digest(self) -> str:
        """Must match ``rummi.env.numpy.state.BatchState.digest`` bit for bit."""
        import hashlib

        h = hashlib.sha256()
        for name in (
            "racks", "table_sets", "workbench", "placed_rack", "slot_new",
            "table_snapshot", "pool", "deck_order", "draw_ptr", "melded",
            "current", "micro_count", "turn_count", "consecutive_draws",
            "winner", "done", "truncated",
        ):
            arr = getattr(self, name).detach().cpu().numpy()
            h.update(name.encode())
            h.update(str(arr.shape).encode())
            h.update(np.ascontiguousarray(arr).astype(np.int64).tobytes())
        return h.hexdigest()

    def check_invariants(self) -> None:
        cfg = self.cfg
        total = self.racks.sum(1) + self.table_counts() + self.workbench + self.pool
        expected = lookup(cfg, self.device).copies
        if not torch.equal(total, expected.expand_as(total)):
            bad = (total != expected).nonzero()[0]
            raise AssertionError(
                f"tile conservation violated in env {int(bad[0])} for kind {int(bad[1])}"
            )


def counts_of(cfg: RummiConfig, kinds: torch.Tensor) -> torch.Tensor:
    """``(B, K)`` counts from a ``(B, ...)`` array of kind ids with EMPTY padding."""
    b = kinds.shape[0]
    flat = kinds.reshape(b, -1).to(torch.int64)
    occupied = flat >= 0
    out = torch.zeros((b, cfg.n_kinds), dtype=torch.int64, device=kinds.device)
    return out.scatter_add_(1, flat.clamp(min=0), occupied.to(torch.int64))


def allocate(cfg: RummiConfig, batch_size: int, device: torch.device) -> TorchState:
    b, p, k = batch_size, cfg.n_players, cfg.n_kinds
    s, ell = cfg.max_sets, cfg.max_set_len
    i64 = {"dtype": torch.int64, "device": device}
    return TorchState(
        cfg=cfg,
        racks=torch.zeros((b, p, k), **i64),
        table_sets=torch.full((b, s, ell), EMPTY, **i64),
        workbench=torch.zeros((b, k), **i64),
        placed_rack=torch.zeros((b, k), **i64),
        slot_new=torch.zeros((b, s), dtype=torch.bool, device=device),
        table_snapshot=torch.full((b, s, ell), EMPTY, **i64),
        pool=torch.zeros((b, k), **i64),
        deck_order=torch.zeros((b, cfg.n_tiles), **i64),
        draw_ptr=torch.zeros(b, **i64),
        melded=torch.zeros((b, p), dtype=torch.bool, device=device),
        current=torch.zeros(b, **i64),
        micro_count=torch.zeros(b, **i64),
        turn_count=torch.zeros(b, **i64),
        consecutive_draws=torch.zeros(b, **i64),
        last_action=torch.full((b,), -1, **i64),
        action_history=torch.full((b, HISTORY_LEN), -1, **i64),
        winner=torch.full((b,), NO_WINNER, **i64),
        done=torch.zeros(b, dtype=torch.bool, device=device),
        truncated=torch.zeros(b, dtype=torch.bool, device=device),
    )


# --- dealing ------------------------------------------------------------------
def reset(
    cfg: RummiConfig, batch_size: int, seed: int = 0, device: torch.device | None = None
) -> TorchState:
    device = device or torch.device("cpu")
    state = allocate(cfg, batch_size, device)
    reset_envs(state, torch.arange(batch_size, device=device), _deck_orders(cfg, seed, batch_size, device))
    return state


def _deck_orders(cfg: RummiConfig, seed: int, count: int, device) -> torch.Tensor:
    """``(count, n_tiles)`` shuffled decks.

    Deliberately built with NumPy's ``SeedSequence``/``Generator``: the shuffle is
    part of the *contract*, not the implementation, and a torch RNG would produce
    different decks and break conformance with the golden trajectories.
    """
    base = np.repeat(np.arange(cfg.n_kinds), tables(cfg).copies)
    orders = np.stack(
        [np.random.default_rng(s).permutation(base) for s in np.random.SeedSequence(seed).spawn(count)]
    )
    return torch.as_tensor(orders, dtype=torch.int64, device=device)


def derived_deck_orders(cfg: RummiConfig, base: int, step: int, envs, device) -> torch.Tensor:
    base_deck = np.repeat(np.arange(cfg.n_kinds), tables(cfg).copies)
    orders = np.stack(
        [
            np.random.default_rng(np.random.SeedSequence([base, step, int(e)])).permutation(base_deck)
            for e in envs
        ]
    )
    return torch.as_tensor(orders, dtype=torch.int64, device=device)


def reset_envs(state: TorchState, which: torch.Tensor, orders: torch.Tensor) -> None:
    """Re-deal the given envs in place; ``orders`` is parallel to ``which``."""
    cfg = state.cfg
    which = which.to(torch.int64)
    state.deck_order[which] = orders

    dealt = orders[:, : cfg.n_players * cfg.rack_size].reshape(-1, cfg.n_players, cfg.rack_size)
    racks = torch.zeros(
        (which.shape[0], cfg.n_players, cfg.n_kinds), dtype=torch.int64, device=orders.device
    )
    racks.scatter_add_(2, dealt, torch.ones_like(dealt))
    state.racks[which] = racks

    state.draw_ptr[which] = cfg.n_players * cfg.rack_size
    state.pool[which] = lookup(cfg, state.device).copies - racks.sum(1)

    state.table_sets[which] = EMPTY
    state.table_snapshot[which] = EMPTY
    state.workbench[which] = 0
    state.placed_rack[which] = 0
    state.slot_new[which] = False
    state.melded[which] = False
    state.current[which] = 0
    state.micro_count[which] = 0
    state.turn_count[which] = 0
    state.consecutive_draws[which] = 0
    state.last_action[which] = -1
    state.action_history[which] = -1
    state.winner[which] = NO_WINNER
    state.done[which] = False
    state.truncated[which] = False


# --- masks --------------------------------------------------------------------
def current_rack(state: TorchState) -> torch.Tensor:
    return state.racks.gather(
        1, state.current.view(-1, 1, 1).expand(-1, 1, state.cfg.n_kinds)
    ).squeeze(1)


def meld_value(state: TorchState, slot_value: torch.Tensor) -> torch.Tensor:
    if state.cfg.strict_initial_meld:
        return (slot_value * state.slot_new).sum(-1)
    return state.placed_rack @ lookup(state.cfg, state.device).value


def legal_actions(state: TorchState, summary: SlotSummary | None = None) -> torch.Tensor:
    """``summary`` is this table's :func:`~rummi.env.torch.kernel.summarize`, passed
    in by a caller that is also encoding an observation from the same state."""
    cfg = state.cfg
    b = state.batch_size
    if summary is None:
        summary = summarize(cfg, state.table_sets)
    stats, ev = summary.stats, summary.ev

    rack = current_rack(state)
    has_melded = state.melded.gather(1, state.current.view(-1, 1)).squeeze(1)
    lengths = stats.n
    occupied_slot = lengths > 0

    may_touch = has_melded if cfg.strict_initial_meld else torch.ones_like(has_melded)
    mine = may_touch.unsqueeze(-1) | state.slot_new

    empty_slot = ~occupied_slot
    # Only the lowest empty slot is offered, collapsing the S! equivalent choices.
    first_empty_idx = torch.argmax(empty_slot.to(torch.int8), dim=-1)
    is_first_empty = torch.zeros_like(empty_slot)
    is_first_empty.scatter_(1, first_empty_idx.view(-1, 1), True)
    is_first_empty &= empty_slot.any(-1, keepdim=True)

    touchable = occupied_slot & mine
    place = rack > 0
    pick = (state.table_sets >= 0) & touchable.unsqueeze(-1)
    dissolve = touchable

    table_whole = (ev.is_valid | ev.is_empty).all(-1)
    played_something = state.placed_rack.sum(-1) > 0
    meld_ok = has_melded | (meld_value(state, ev.value) >= cfg.initial_meld)
    end_turn = (state.workbench.sum(-1) == 0) & table_whole & played_something & meld_ok
    playable = (state.micro_count < cfg.max_micro_per_turn) & ~state.done
    # Everything ASSIGN asks of the slot rather than of the tile.
    slot_ok = (lengths < cfg.max_set_len) & (touchable | is_first_empty) & playable.unsqueeze(-1)

    mask = torch.zeros((b, cfg.n_actions), dtype=torch.bool, device=state.device)
    gate = playable.unsqueeze(-1)
    mask[:, cfg.place_offset : cfg.pick_offset] = place & gate
    mask[:, cfg.pick_offset : cfg.dissolve_offset] = pick.reshape(b, -1) & gate
    mask[:, cfg.dissolve_offset : cfg.assign_offset] = dissolve & gate
    mask[:, cfg.assign_offset : cfg.end_turn_action] = _assign_block(cfg, state, stats, slot_ok)
    mask[:, cfg.end_turn_action] = end_turn & playable
    mask[:, cfg.draw_action] = True
    return mask


def _assign_block(
    cfg: RummiConfig, state: TorchState, stats: SlotStats, slot_ok: torch.Tensor
) -> torch.Tensor:
    """``(B, S*K)`` the ASSIGN block, built kind-major as the action ids are.

    :class:`~rummi.env.torch.kernel.AssignCode` is a (colour x number) product, so
    the block comes out in the order the ids want. Building the ``(S, K)`` form
    first would mean transposing S*K booleans per env to say the same thing.
    """
    codes = assign_codes(cfg, stats)
    color, number = codes.color.transpose(1, 2), codes.number.transpose(1, 2)
    grid = (color.unsqueeze(2) & number.unsqueeze(1)) != 0
    block = torch.cat([grid.flatten(1, 2), codes.joker.unsqueeze(1)], dim=1)
    # The tile has to be in hand and the slot has to accept it. Both are applied to
    # the finished block rather than folded into the narrower code: selecting a code
    # by `slot_ok` inside this graph makes Inductor's MPS backend fail codegen for
    # the `slot_ok` buffer, and MPS + `compile` is the benchmark's headline figure.
    # Compiled, the extra pass fuses away anyway.
    block &= (state.workbench > 0).unsqueeze(-1)
    block &= slot_ok.unsqueeze(1)
    return block.reshape(state.batch_size, -1)


# --- engine -------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class StepResult:
    rewards: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor


def _sort_slots(cfg: RummiConfig, table: torch.Tensor) -> torch.Tensor:
    """Canonical order *within* each slot: kinds ascending, EMPTY last."""
    key = torch.where(table >= 0, table, torch.full_like(table, cfg.n_kinds)).sort(-1).values
    return torch.where(key == cfg.n_kinds, torch.full_like(key, EMPTY), key)


def _pack_keys(cfg: RummiConfig, table: torch.Tensor) -> list[torch.Tensor]:
    """Pack each slot's contents into as few int64 sort keys as fit.

    A lexicographic sort of slots by their L columns would be L stable sorts.
    Packing base-(K+1) digits into 63-bit words cuts that to two for the standard
    config, which matters because this runs on every step.
    """
    radix = cfg.n_kinds + 1
    per_word = 1
    while radix ** (per_word + 1) < 2**62:
        per_word += 1
    digits = torch.where(table >= 0, table, torch.full_like(table, cfg.n_kinds))
    keys = []
    for start in range(0, cfg.max_set_len, per_word):
        chunk = digits[..., start : start + per_word]
        acc = torch.zeros(chunk.shape[:-1], dtype=torch.int64, device=table.device)
        for i in range(chunk.shape[-1]):
            acc = acc * radix + chunk[..., i]
        keys.append(acc)
    return keys


def _sort_slot_order(cfg: RummiConfig, table: torch.Tensor) -> torch.Tensor:
    """Canonical order *of* the slots, so one table has one representation.

    Applied only when a turn commits: slot indices must be stable while a player
    is mid-turn, or a multi-step plan would aim at moving targets.
    """
    keys = _pack_keys(cfg, table)
    order = torch.arange(cfg.max_sets, device=table.device).expand(table.shape[0], -1).contiguous()
    # Least significant key first, each sort stable, gives a lexicographic order.
    for key in reversed(keys):
        picked = key.gather(1, order)
        order = order.gather(1, picked.sort(dim=-1, stable=True).indices)
    return table.gather(1, order.unsqueeze(-1).expand_as(table))


def check_actions(
    state: TorchState,
    actions: torch.Tensor,
    mask: torch.Tensor,
    active: torch.Tensor | None = None,
) -> None:
    """Raise if a live env chose a masked-out action.

    Its own function because it reads one boolean off the device: inside a
    compiled step that branch splits the graph in two, so the compiled backend
    calls this itself and hands the step no mask.
    """
    actions = actions.to(mask.device, torch.int64)
    live = ~state.done if active is None else (~state.done & active)
    bad = ~mask.gather(1, actions.view(-1, 1)).squeeze(1) & live
    if bool(bad.any()):
        env = int(torch.argmax(bad.to(torch.int8)))
        raise ValueError(f"illegal action {int(actions[env])} in env {env}")


def step(
    state: TorchState,
    actions: torch.Tensor,
    mask: torch.Tensor | None = None,
    active: torch.Tensor | None = None,
) -> StepResult:
    """Apply one action per env, mutating ``state`` in place."""
    cfg = state.cfg
    b = state.batch_size
    dev = state.device
    actions = actions.to(dev, torch.int64)

    live = ~state.done
    if active is not None:
        live = live & active
    if mask is not None:
        check_actions(state, actions, mask, active)

    is_place = actions < cfg.pick_offset
    is_pick = (actions >= cfg.pick_offset) & (actions < cfg.dissolve_offset)
    is_dissolve = (actions >= cfg.dissolve_offset) & (actions < cfg.assign_offset)
    is_assign = (actions >= cfg.assign_offset) & (actions < cfg.end_turn_action)
    is_end = actions == cfg.end_turn_action
    is_draw = actions == cfg.draw_action

    pick_rel = actions - cfg.pick_offset
    assign_rel = actions - cfg.assign_offset
    kind = torch.where(is_place, actions, torch.where(is_assign, assign_rel // cfg.max_sets, 0))
    slot = torch.where(
        is_pick,
        pick_rel // cfg.max_set_len,
        torch.where(
            is_dissolve,
            actions - cfg.dissolve_offset,
            torch.where(is_assign, assign_rel % cfg.max_sets, 0),
        ),
    )
    pos = torch.where(is_pick, pick_rel % cfg.max_set_len, 0)

    rows = torch.arange(b, device=dev)
    rewards = torch.zeros((b, cfg.n_players), dtype=torch.float32, device=dev)
    state.last_action = torch.where(live, actions, state.last_action)
    state.action_history = torch.where(
        live.unsqueeze(-1),
        torch.cat([state.action_history[:, 1:], actions.view(-1, 1)], dim=1),
        state.action_history,
    )

    def onehot(idx: torch.Tensor, size: int, gate: torch.Tensor) -> torch.Tensor:
        out = torch.zeros((b, size), dtype=torch.int64, device=dev)
        out.scatter_(1, idx.view(-1, 1), gate.to(torch.int64).view(-1, 1))
        return out

    # --- PLACE: rack -> workbench
    hit = is_place & live
    moved = onehot(kind, cfg.n_kinds, hit)
    state.racks[rows, state.current] -= moved
    state.workbench += moved
    state.placed_rack += moved

    # --- PICK: one tile of a set -> workbench
    hit = is_pick & live
    flat = state.table_sets.reshape(b, -1)
    at = (slot * cfg.max_set_len + pos).view(-1, 1)
    taken = flat.gather(1, at).squeeze(1)
    state.workbench += onehot(taken.clamp(min=0), cfg.n_kinds, hit & (taken >= 0))
    flat.scatter_(1, at, torch.where(hit, torch.full_like(taken, EMPTY), taken).view(-1, 1))

    # --- DISSOLVE: whole set -> workbench
    hit = is_dissolve & live
    row = state.table_sets.gather(
        1, slot.view(-1, 1, 1).expand(-1, 1, cfg.max_set_len)
    ).squeeze(1)
    keep = (row >= 0) & hit.unsqueeze(-1)
    state.workbench.scatter_add_(1, row.clamp(min=0), keep.to(torch.int64))
    state.table_sets.scatter_(
        1,
        slot.view(-1, 1, 1).expand(-1, 1, cfg.max_set_len),
        torch.where(hit.unsqueeze(-1), torch.full_like(row, EMPTY), row).unsqueeze(1),
    )

    # --- ASSIGN: workbench -> set slot
    hit = is_assign & live
    row = state.table_sets.gather(
        1, slot.view(-1, 1, 1).expand(-1, 1, cfg.max_set_len)
    ).squeeze(1)
    was_empty = (row < 0).all(-1)
    free = torch.argmax((row < 0).to(torch.int8), dim=-1)
    placed = row.scatter(1, free.view(-1, 1), kind.view(-1, 1))
    state.table_sets.scatter_(
        1,
        slot.view(-1, 1, 1).expand(-1, 1, cfg.max_set_len),
        torch.where(hit.unsqueeze(-1), placed, row).unsqueeze(1),
    )
    state.workbench -= onehot(kind, cfg.n_kinds, hit)
    state.slot_new |= onehot(slot, cfg.max_sets, hit & was_empty).to(torch.bool)

    # Sorting is idempotent on an already-sorted slot, so doing it unconditionally
    # is both correct and cheaper than branching on which slot changed.
    state.table_sets = _sort_slots(cfg, state.table_sets)

    mid_turn = (is_place | is_pick | is_dissolve | is_assign) & live
    state.micro_count += mid_turn.to(torch.int64)
    if cfg.micro_step_cost:
        rewards[rows, state.current] -= cfg.micro_step_cost * mid_turn.to(torch.float32)

    # --- END_TURN
    hit = is_end & live
    if cfg.tiles_placed_bonus:
        rewards[rows, state.current] += (
            cfg.tiles_placed_bonus * (state.placed_rack.sum(-1) * hit).to(torch.float32)
        )
    if cfg.rack_value_delta:
        shed = (state.placed_rack @ lookup(cfg, dev).value) * hit
        rewards[rows, state.current] += cfg.rack_value_delta * shed.to(torch.float32)
    state.melded |= onehot(state.current, cfg.n_players, hit).to(torch.bool)
    state.table_sets = torch.where(
        hit.view(-1, 1, 1), _sort_slot_order(cfg, state.table_sets), state.table_sets
    )
    state.consecutive_draws = torch.where(hit, torch.zeros_like(state.consecutive_draws), state.consecutive_draws)

    # --- DRAW: revert the turn, take a tile, pass
    hit = is_draw & live
    state.table_sets = torch.where(hit.view(-1, 1, 1), state.table_snapshot, state.table_sets)
    state.racks[rows, state.current] += state.placed_rack * hit.unsqueeze(-1)

    can_draw = hit & (state.draw_ptr < cfg.n_tiles)
    drawn = state.deck_order.gather(1, state.draw_ptr.clamp(max=cfg.n_tiles - 1).view(-1, 1)).squeeze(1)
    got = onehot(drawn, cfg.n_kinds, can_draw)
    state.racks[rows, state.current] += got
    state.pool -= got
    state.draw_ptr += can_draw.to(torch.int64)
    state.consecutive_draws += hit.to(torch.int64)

    # --- begin turn, for whichever envs committed
    committed = (is_end | is_draw) & live
    keep = (~committed).unsqueeze(-1)
    state.workbench = state.workbench * keep
    state.placed_rack = state.placed_rack * keep
    state.slot_new = state.slot_new & keep
    state.micro_count = torch.where(committed, torch.zeros_like(state.micro_count), state.micro_count)
    state.turn_count += committed.to(torch.int64)
    state.table_snapshot = torch.where(committed.view(-1, 1, 1), state.table_sets, state.table_snapshot)
    state.current = torch.where(committed, (state.current + 1) % cfg.n_players, state.current)

    _resolve_terminal(state, committed, rewards)
    return StepResult(
        rewards=rewards,
        terminated=state.done & ~state.truncated,
        truncated=state.truncated.clone(),
    )


def _resolve_terminal(state: TorchState, committed: torch.Tensor, rewards: torch.Tensor) -> None:
    cfg = state.cfg
    actor = (state.current - 1) % cfg.n_players
    rack_totals = state.racks.sum(-1)
    emptied = committed & (rack_totals.gather(1, actor.view(-1, 1)).squeeze(1) == 0)

    stalled = committed & (state.pool_size == 0) & (state.consecutive_draws >= cfg.n_players)
    over_time = committed & (state.turn_count >= cfg.max_turns)

    values = state.rack_values()
    lowest = values.argmin(-1)
    state.winner = torch.where(emptied, actor, torch.where(stalled, lowest, state.winner))

    finishing = emptied | stalled | over_time
    newly_done = finishing & ~state.done
    state.done |= finishing
    state.truncated |= over_time & ~emptied & ~stalled

    paying = newly_done & ~state.truncated
    is_winner = torch.zeros_like(state.melded)
    is_winner.scatter_(1, state.winner.clamp(min=0).view(-1, 1), True)
    if cfg.reward_mode is RewardMode.WIN_LOSS:
        terminal = torch.where(
            is_winner,
            torch.ones_like(values, dtype=torch.float32),
            torch.full_like(values, -1.0 / (cfg.n_players - 1), dtype=torch.float32),
        )
    else:
        paid = torch.where(is_winner, torch.zeros_like(values), values).to(torch.float32)
        terminal = -paid
        terminal.scatter_(1, state.winner.clamp(min=0).view(-1, 1), paid.sum(-1, keepdim=True))
        if cfg.reward_mode is RewardMode.SCORE_NORMALIZED:
            terminal = terminal / max(1, cfg.rack_size * max(cfg.n_numbers, cfg.joker_penalty))
    rewards += terminal * paying.unsqueeze(-1)
