# Rummi simulator — normative spec

The contract a backend must satisfy. The NumPy code in `rummi/core/` is the
reference implementation; the torch and JAX ports are written independently
against *this document* rather than against a shared abstraction layer, so the
benchmark compares implementations rather than one implementation's plumbing.

Conformance is mechanical: replay `tests/golden/*.json` and reproduce every
digest. See `tests/test_golden.py`.

## 1. Encoding

`K = n_colors * n_numbers + 1` tile *kinds*. A kind is an identity, not an
instance — `n_copies` physical tiles share a kind.

```
kind = color * n_numbers + (number - 1)     0 <= kind < n_colors * n_numbers
kind = n_colors * n_numbers                 the joker
-1                                          EMPTY, an unoccupied slot position
```

Colour-major is load-bearing: a run occupies consecutive kind ids within one
colour.

## 2. State

All arrays carry a leading batch dim `B`. `S = max_sets`, `L = max_set_len`,
`P = n_players`.

| field | shape | meaning |
|---|---|---|
| `racks` | `(B, P, K)` | per-seat kind counts |
| `table_sets` | `(B, S, L)` | kind ids, `EMPTY` padded, **sorted ascending within each slot** |
| `workbench` | `(B, K)` | tiles held loose mid-turn |
| `placed_rack` | `(B, K)` | tiles that left the acting rack this turn |
| `slot_new` | `(B, S)` | slot created during the current turn |
| `table_snapshot` | `(B, S, L)` | turn-start copy of `table_sets` |
| `pool` | `(B, K)` | counts still in the bag |
| `deck_order` | `(B, n_tiles)` | the env's shuffled deck, fixed at reset |
| `draw_ptr` | `(B,)` | index of the next tile to be drawn |
| `melded` | `(B, P)` | seat has made its opening meld |
| `current` | `(B,)` | seat to act |
| `micro_count` | `(B,)` | actions taken this turn |
| `turn_count` | `(B,)` | turns committed |
| `consecutive_draws` | `(B,)` | turns ending in DRAW, in a row |
| `winner`, `done`, `truncated` | `(B,)` | outcome |
| `last_action`, `action_history` | `(B,)`, `(B, 8)` | presentation only, excluded from the digest |

**Invariant, at every step:** for each kind `k`,
`sum_p racks[k] + table[k] + workbench[k] + pool[k] == copies[k]`,
where `copies[k]` is `n_copies` for numbered kinds and `n_jokers` for the joker.

**No randomness in `step`.** Each env's deck is permuted once at reset;
drawing advances `draw_ptr`. A step is therefore a pure function of state and
action, per-env streams are independent, and a batched rollout is bit-identical
to the same envs run one at a time. Ports need no PRNG in their hot loop.

## 3. Set validity

For a slot with `n` occupied positions, `j` jokers and `r = n - j` real tiles:

**Group** — valid iff `n >= min_set`, `n <= n_colors`, all real tiles share a
number, and no real kind repeats.

**Run** — valid iff `n >= min_set`, `n <= n_numbers`, all real tiles share a
colour, no real kind repeats, and a length-`n` window fits inside
`[1, n_numbers]` covering every real number:

```
max(1, hi - n + 1) <= min(lo, n_numbers - n + 1)
```

with `lo`/`hi` the least/greatest real number, and `lo = n_numbers`, `hi = 1`
when `r == 0` (the loosest values, so an all-joker slot is judged only on
length).

**Extendable** — could still become valid by adding tiles. Same shape tests
without the length or window conditions: a partial run can always be completed
inside `[1, n_numbers]`. Empty slots are extendable.

**Value** — jokers take the value of the position they fill; where the reading is
ambiguous the best-case one is used, matching the rule that the player declares
it. A run takes `n * win_hi + n(n-1)/2`; a group takes `n * number`
(`n * n_numbers` if it is all jokers).

## 4. Actions

Six contiguous blocks; `A = K + S*L + S + K*S + 2`.

| block | size | id | effect |
|---|---|---|---|
| `PLACE(kind)` | `K` | `kind` | rack → workbench |
| `PICK(slot, pos)` | `S*L` | `K + slot*L + pos` | one tile of a set → workbench |
| `DISSOLVE(slot)` | `S` | `... + slot` | whole set → workbench |
| `ASSIGN(kind, slot)` | `K*S` | `... + kind*S + slot` | workbench → set slot |
| `END_TURN` | 1 | | commit |
| `DRAW` | 1 | | revert the turn, draw, pass |

One action per `step`. A turn is a *sequence* ending in `END_TURN` or `DRAW`.
The table may be in pieces mid-turn; it must be whole only at `END_TURN`.

## 5. Legality

Let `has_melded = melded[current]` and `len[s]` be slot occupancy.

- `PLACE(k)` — `rack[k] > 0`
- `PICK(s, p)` — `table_sets[s, p] >= 0` and `s` is touchable
- `DISSOLVE(s)` — `len[s] > 0` and `s` is touchable
- `ASSIGN(k, s)` — `workbench[k] > 0`, `len[s] < L`, adding `k` leaves `s`
  extendable, and `s` is touchable or is the lowest empty slot
- `END_TURN` — `workbench` empty, every non-empty slot valid,
  `placed_rack.sum() >= 1`, and the meld condition
- `DRAW` — **always legal**, finished envs included

*Touchable* is `len[s] > 0 and (may_touch_table or slot_new[s])`, where
`may_touch_table` is `has_melded` under `strict_initial_meld` and `True`
otherwise. This expresses the official rule that a player may not disturb the
table before opening.

Everything except `DRAW` is additionally gated on
`micro_count < max_micro_per_turn` and `not done`.

Two canonicalisations suppress redundant actions: only the **lowest** empty slot
is ever offered to `ASSIGN`, and tiles are kept sorted within a slot.

`DRAW` being unconditional matters three ways: the MDP cannot deadlock mid-turn,
no mask row is ever all-zero (a policy sampling one would produce NaNs), and
choosing to draw when a play was available is legal Rummikub.

## 6. Effects

- **PLACE** `racks[cur,k] -= 1; workbench[k] += 1; placed_rack[k] += 1`
- **PICK** move `table_sets[s,p]` to the workbench, re-sort slot `s`
- **DISSOLVE** move all of slot `s` to the workbench, clear it
- **ASSIGN** `workbench[k] -= 1`, write `k` into slot `s`, re-sort; set
  `slot_new[s]` if `s` was empty
- **END_TURN** set `melded[cur]`; sort slots into canonical order
  (lexicographic by contents, empty last); `consecutive_draws = 0`; begin turn
- **DRAW** restore `table_sets` from `table_snapshot`; return `placed_rack` to
  the rack; draw `deck_order[draw_ptr]` if any remain; `consecutive_draws += 1`;
  begin turn

*Begin turn* = clear `workbench`, `placed_rack`, `slot_new`, `micro_count`;
`turn_count += 1`; snapshot the table; `current = (current + 1) % P`.

Slot *identity* is stable within a turn — slots are reordered only at
`END_TURN`, so a player's multi-step plan is not aimed at moving targets.

## 7. Meld, termination, reward

**Opening meld.** Under `strict_initial_meld`, the value is the sum over
`slot_new` slots; because the table is untouchable pre-meld those slots are
exactly the tiles played, so jokers resolve from their sets. Otherwise it is the
face value of `placed_rack`, jokers at zero. Either way it must reach
`initial_meld`.

**Termination.** A seat emptying its rack after `END_TURN` wins. Otherwise, pool
empty and `consecutive_draws >= P` ends the game on lowest rack value. Reaching
`max_turns` truncates.

**Reward** is `(B, P)`. `WIN_LOSS` pays the winner `+1` and each loser
`-1/(P-1)`, so it is zero-sum. `SCORE` has each loser pay their rack value
(joker at `joker_penalty`) and the winner collect the sum;
`SCORE_NORMALIZED` divides by `rack_size * max(n_numbers, joker_penalty)`.
Truncation pays nothing — it is an artificial cutoff, not a result.

## 8. Port notes

- The NumPy reference mutates buffers to avoid per-step allocation. Arithmetic is
  index-and-mask only, so the JAX port is the same expressions written
  functionally with `.at[]`.
- Duplicate-real-kind detection sorts reals to the front of each slot; `torch.sort`
  and `jnp.sort` substitute directly. No popcount is required anywhere.
- `counts_of` uses an offset-`bincount` scatter; use `scatter_add` / `bincount`
  equivalents rather than a Python loop over the batch.
- `max_sets` is the dominant throughput knob — it drives `A` and the `(B, S, K)`
  ASSIGN predicate. Measured: `S=16` → 133k, `S=24` → 91k, `S=35` → 63k
  env-steps/s in NumPy. The default is the provable bound `n_tiles // min_set`;
  real games peak near 20 occupied slots.
