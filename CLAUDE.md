# Working on rummi

Project-specific notes. The things here are the ones that are expensive to
rediscover or easy to get wrong — everything else is in `README.md` and
`SPEC.md`.

## Setup

`uv` venv at `.venv`. Extras: `dev,env,render,solver` for everything, plus
`torch`/`jax` for the other backends and `docs` (pillow) to regenerate figures.

```bash
source .venv/bin/activate
pytest                                                   # 183 tests, ~50s
python -m rummi.bench.fuzz --policy greedy --games 500    # invariant fuzzing
python -m rummi.evaluate.run --agent greedy               # agent strength
python -m rummi.bench.bench_backends --compile            # throughput
```

Git identity is set **repo-locally** to the personal address, and the remote is
the `github-personal` SSH alias. Don't "fix" either to the global config.

## The constraint that governs everything

**There are three implementations of the same rules.** `rummi/env/numpy/` is the
reference; `rummi/env/torch/` and `rummi/env/jax/` are written independently
against `SPEC.md`, deliberately *not* against a shared abstraction, so a
benchmark compares implementations rather than the cost of a common layer.

So: **any change to the rules is three changes plus a spec edit.** Touching only
NumPy leaves the repo in a state where `tests/test_backends.py` fails and the
benchmark is meaningless. If you are about to change masks, effects, termination
or reward, plan for all three up front.

`rummi/rules/` is backend-free (config, tile encoding, action layout). Anything
definitional belongs there so the three cannot drift on it.

## Golden fixtures are a contract

`tests/golden/*.json` hold seeded action sequences and state digests. Every
backend must reproduce them exactly.

**Never regenerate them to make a test pass.** A changed digest means the rules
changed — that is either a bug, or a decision that needs `PROTOCOL_VERSION` and
the published scores revisited. Regenerate only when you intended the rules to
change, and say so in the commit.

## Random play cannot test this game

In 10M fuzz steps, uniform-random play assembled a legal 30-point opening meld
**four times**. On the standard config it is byte-identical to passing every
turn. So random play never reaches `END_TURN`, melding, winning, or scoring.

Any test or fuzz run that needs those paths must use `--policy greedy` or
better. This has already caused one false "the engine is broken" diagnosis.

## Invariants worth re-reading before editing the engine

- **Tile conservation**: `racks + table + workbench + pool == copies`, every step.
  `state.check_invariants()` asserts it; the fuzzer calls it on every step.
- **`DRAW` is never masked** — not when the micro-budget is spent, not when the
  env is done. It keeps the MDP deadlock-free and stops any mask row being
  all-zero, which would hand a policy a NaN.
- **The table may be in pieces mid-turn.** It must be whole only at a turn
  boundary (`micro_count == 0`). Asserting validity mid-turn is wrong and was a
  real bug in the first fuzzer.
- **Slot identity is stable within a turn**; slots are re-sorted into canonical
  order only at `END_TURN`, or a multi-step plan would aim at moving targets.
- **No NP-hard work in the hot loop.** Deciding whether a tile multiset
  partitions into valid sets is the whole difficulty of Rummikub. The action
  space exists to keep that out of `step`; CP-SAT solves it once per turn in
  `rummi/solver/`, never per step.

## Agents see the observation, never the state

`rummi/agents/base.py` is the contract: an agent gets the observation and the
mask. The observation merges the pool and opponents' racks into one `unseen`
vector precisely so an agent cannot read an opponent's hand.

The reference agents obey this too — including the CP-SAT one, which is what
makes "the observation is sufficient to play optimally" a tested claim rather
than a hope. Don't shortcut a new agent by handing it `BatchState`.

## Backend traps already paid for

- **JAX**: constant lookup tables must be **NumPy** arrays. An `lru_cache` first
  populated inside a trace caches tracers and leaks them into every later call.
  `cfg` is a static argument (frozen dataclass, hashable). Action validation
  cannot live inside the jitted step — reading a device boolean breaks the trace.
- **torch**: apply effects as masked whole-batch updates, never NumPy's
  per-family `flatnonzero` selection — that read is a host sync every step and
  costs the entire `torch.compile` speedup. `sort` needs
  `sort(dim=-1, stable=True)`; `stable` is keyword-only.
- **Every backend** must shuffle decks with NumPy's `SeedSequence`/`Generator`.
  The permutation is part of the contract; a framework-native RNG deals
  different tiles and fails conformance.
- `rummi/env/torch/` does not shadow the global `torch` (absolute imports), but
  verify after any move.

## Evaluation protocol is frozen

`rummi/evaluate/protocol.py` pins configs, opponents, game counts and per-game
seeds. Every deal is played twice with seats swapped, which is why an agent
mirrored against itself scores *exactly* 50.0% and +0.0 — if that stops being
exact, the mirroring is broken, not noisy.

Editing a suite invalidates every score published against `PROTOCOL_VERSION`.
Bump the version if you must change one.

## Figures

`docs/render.gif` and `docs/render.png` are generated, and regeneration is
byte-identical so refreshing them does not churn the diff:

```bash
python tools/render_docs.py --format gif --out docs/render.gif
```

`.gitignore` excludes `*.png` except under `docs/`.

## Style

Comments explain *why*, and describe the code as it is — never narrate the
change or compare to a previous version. Match the density of the file you are
in; most modules carry a short docstring explaining the one non-obvious decision
they embody, and little else.
