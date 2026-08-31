# Rummi simulator — normative spec

The contract a backend must satisfy. The NumPy code in `rummi/env/numpy/` is the
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

**Shaping** is three optional per-step terms, all `0.0` by default and all
credited to the seat that acted. They are added to the terminal reward above
rather than replacing it, and unlike it they are *not* zero-sum.

| term | when | amount |
|---|---|---|
| `micro_step_cost` | every `PLACE`, `PICK`, `DISSOLVE`, `ASSIGN` — not on a committing action | `-micro_step_cost` |
| `tiles_placed_bonus` | `END_TURN` | `tiles_placed_bonus * placed_rack.sum()` |
| `rack_value_delta` | `END_TURN` | `rack_value_delta * face_value(placed_rack)`, jokers at zero |

Shaping accrues as it is earned, so a truncated episode keeps whatever it already
paid — only the *terminal* reward is withheld. Every bundled config and every
evaluation suite leaves all three at `0.0`, so they are outside
`PROTOCOL_VERSION`: turning one on makes a run incomparable to a published score
by intent, not by accident.

## 8. Observation

What an agent sees, and the only thing it sees. Every field has a leading batch
dimension `B`.

| field | shape | dtype | meaning |
|---|---|---|---|
| `rack` | `(B, K)` | int16 | the **acting** seat's rack, by kind |
| `table_sets` | `(B, S, L)` | int16 | the table as kind ids, `EMPTY` (`-1`) padded |
| `slot_features` | `(B, S, 10)` | int32 | per-slot summary, columns below |
| `workbench` | `(B, K)` | int16 | tiles lifted this turn and not yet assigned |
| `placed_this_turn` | `(B, K)` | int16 | tiles moved from the rack this turn |
| `unseen` | `(B, K)` | int16 | `copies - rack - table - workbench` |
| `rack_sizes` | `(B, P)` | int16 | tile count per seat, seat-rotated |
| `melded` | `(B, P)` | int8 | whether each seat has opened, seat-rotated |
| `scalars` | `(B, 4)` | int32 | `[pool_size, meld_progress, meld_remaining, micro_count]` |

`slot_features` columns, in order: `n`, `run_valid`, `group_valid`,
`is_extendable`, `color`, `lo`, `hi`, `n_jokers`, `value`, `is_new` — the first
eight and `value` from section 3's kernel, `is_new` from `slot_new`.

`meld_progress` is the value credited towards the opening meld (section 7);
`meld_remaining` is `0` once the seat has opened, else
`max(0, initial_meld - meld_progress)`.

Two properties are load-bearing, and a port that breaks either is wrong even if
every shape matches.

**Seat-relative.** Per-seat fields (`rack_sizes`, `melded`) are rotated so index
`0` is the acting seat and index `i` is seat `(current + i) mod P`. `rack` is the
acting seat's outright. This is what lets one policy play every seat without
learning `P` conventions, and it is why a bundled agent can be dropped into any
seat of a match.

**Information-correct.** `unseen` merges the pool with the opponents' racks into
one vector, because those are exactly the tiles whose location the acting seat
cannot know. It is derived by subtraction — total copies less what the actor can
locate — so it cannot drift from the state. Individual opponent racks are never
exposed; only their sizes, via `rack_sizes`.

That merge is the integrity property the benchmark rests on: an agent handed the
raw state could read a specific opponent's hand, and every published score would
be meaningless. The CP-SAT agent plays from this observation alone, which is what
makes "the observation is sufficient to play optimally" a tested claim rather
than a hope.

## 9. Conformance status

| backend | state | peak env-steps/s | vs NumPy | at batch |
|---|---|---:|---:|---:|
| NumPy (`rummi/env/numpy/`) | reference | 195k | 1.0x | 16,384 |
| torch CPU | conformant | 227k | 1.2x | 16,384 |
| torch CPU + `compile` | conformant | 286k | 1.5x | 16,384 |
| torch MPS | conformant | 432k | 2.2x | 16,384 |
| torch MPS + `compile` | conformant | **809k** | **4.2x** | 16,384 |
| JAX CPU (`lax.scan`) | conformant | 346k | 1.8x | 1,024 |

Standard config, `A=2400`, best of three, action choice held to the cheapest
possible so the figure measures the simulator. Data in `docs/data/backends.json`;
reproduce with `python -m rummi.bench.bench_backends --compile --json …`.

**How this is measured matters more than the numbers.** The state advances while
it is being timed and the table fills as it goes, so the work per step is not
constant across a run. Every timed repeat therefore builds a fresh state and
advances it to the same point before the clock starts, with compilation warmed
outside the timed region. Reusing one state across repeats measures a later phase
of the game each time; an earlier version of this table did exactly that and
overstated the MPS figure by roughly a factor of two. Dynamo is reset before each
compiled cell for the same kind of reason: its recompile limit is per code object
and shared across the sweep, and past it `compile` falls back to eager without
saying so.

Read the shape rather than the top row:

* **NumPy saturates at 256 envs** and is flat from a thousand on: the reference
  runs out of one core's bandwidth before it runs out of batch.
* **The GPU needs a batch.** MPS is six times slower than NumPy at 64 envs and
  only overtakes it past a thousand.
* **Fusion multiplies the GPU, not the CPU** — 1.8x on MPS at 4,096, 1.3x on
  torch CPU at 16,384. It removes per-kernel overhead, and the CPU has less.
* **Both frameworks agree on CPU** (1.5x, 1.8x), which is decent evidence that the
  headroom over a vectorised NumPy reference is real.
* **The NumPy row does less work**: it builds the ASSIGN block only at the kinds a
  workbench holds, which a shape-static backend cannot. See section 10.
* **JAX leads at small batch** (1.4x at 64 envs) and is flattest thereafter.
  `lax.scan` buys almost nothing over a fused step.
* JAX is CPU-only here — no production Metal backend — so it cannot be compared
  against MPS on equal hardware. Re-measure on CUDA.

## 10. Port notes

- The NumPy reference mutates buffers to avoid per-step allocation. Arithmetic is
  index-and-mask only, so the JAX port is the same expressions written
  functionally with `.at[]`.
- A slot's whole summary comes from one gather of a packed per-kind code and two
  reductions over it, a bitwise OR and a sum (`rules/encoding.SlotCode`). The
  fields — colours present, numbers present, count of real tiles — are padded so a
  slot's sum can never carry out of its own field, which is what makes `sum == or`
  exactly "no duplicate": no sort, no popcount. Keep the packing under 31 bits, as
  the builder asserts: JAX is 32-bit unless x64 is enabled globally.
- Reading a bit index out of those masks is the exponent of a float, via `frexp`.
  MPS has no `frexp`, so the torch port counts thresholds instead — same value,
  one small pass over a static, tiny axis.
- The ASSIGN predicate factors: a numbered kind *is* a (colour, number) pair, and
  every term constrains only one of the two. So the halves are `(S, C)` and
  `(S, N)`, only their product is `(S, K)`, and the product can be written
  straight into the kind-major action ids without ever building the `(S, K)` form.
  Inductor's MPS backend fails codegen if the per-slot factor is selected into a
  code before the product, so the torch port applies it to the finished block.
- A tile must be in hand for ASSIGN to be legal, and a workbench holds ~6 kinds of
  53, so most of that block is computed to produce zeros. The NumPy reference takes
  the product only at the `(env, kind)` pairs held and leaves the rest of the
  zero-initialised mask alone -- 1.8x on `legal_actions`. **A port should not copy
  this**: the pair count depends on the data, and neither `jit` nor `compile` can
  have a shape that does. The dense form is the portable one, and the two are held
  equal by `test_the_mask_matches_the_dense_predicate`.
- `counts_of` uses an offset-`bincount` scatter; use `scatter_add` / `bincount`
  equivalents rather than a Python loop over the batch.
- Effects should be applied as masked whole-batch updates, not by selecting the
  envs that chose each family. NumPy uses `flatnonzero` selection, which is
  faster there; on an accelerator that read is a host synchronisation every step.
  The torch port applies all six families to the whole batch under `where`, which
  is what makes it shape-static and `compile`-able.
- Because of that, slots are re-sorted every step rather than only where they
  changed (sorting is idempotent), and canonical slot order is recomputed every
  step and selected for with `where`.
- Canonical slot order needs a lexicographic sort of `S` rows by `L` columns.
  Packing base-`(K+1)` digits into 63-bit words reduces that from `L` stable
  sorts to two for the standard config. `torch.sort` needs
  `sort(dim=-1, stable=True)` -- `stable` is keyword-only.
- Deck shuffling must use NumPy's `SeedSequence`/`Generator` in every backend: the permutation is part of the contract, and a torch RNG would produce
  different decks and fail conformance.
- Passing the mask to `step` enables action validation, which reads a bool off
  the device once per step. Measured cost is small (8.3x -> 8.6x with it off), so
  leave it on unless profiling says otherwise.
- JAX specifics: `cfg` is a static argument (it is a frozen dataclass, so it
  hashes), the state is a `NamedTuple` pytree with `cfg` deliberately excluded,
  and action validation lives outside the jitted step because checking the mask
  reads a device boolean. Constant lookup tables must be held as **NumPy**
  arrays: an `lru_cache` first populated inside a trace would cache tracers and
  leak them into every later call.
- `jnp.lexsort` exists, so the JAX port expresses canonical slot order directly;
  the torch port needed the packed-integer-key workaround instead.
- `max_sets` is the dominant throughput knob — it drives `A` and the `(B, S, K)`
  ASSIGN predicate. Measured: `S=16` → 317k, `S=24` → 226k, `S=35` → 164k
  env-steps/s in NumPy. The default is the provable bound `n_tiles // min_set`;
  real games peak near 20 occupied slots.
