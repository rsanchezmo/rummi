# Working on rummi

Project-specific notes. The things here are the ones that are expensive to
rediscover or easy to get wrong — everything else is in `README.md` and
`SPEC.md`.

## Setup

`uv` venv at `.venv`. Extras: `dev,env,render,solver` for everything, plus
`torch`/`jax` for the other backends and `docs` (pillow) to regenerate figures.

```bash
source .venv/bin/activate
pytest -n auto                                           # 382 tests, ~19s (80s serial)
python -m rummi.bench.fuzz --policy greedy --games 500    # invariant fuzzing
python -m rummi.evaluate.run --agent greedy               # agent strength
python -m rummi.bench.bench_backends --compile            # throughput
ruff check rummi tests tools && mypy rummi                # what CI's lint job runs
python -m rummi.bench.bench_env                           # env throughput per backend
```

`mypy` needs a **3.12+** interpreter: modern numpy stubs use PEP 695 `type`
statements, which mypy refuses to parse when it is running on anything older --
and the error names numpy, not your Python. CI's lint job pins 3.12 for this.

`mypy rummi` is clean, but `pyproject.toml` exempts `rummi/render/pygame_view.py`
and `rummi/env/torch/sim.py`. That is a ratchet with a reason written next to it,
not a blanket pass -- read it before adding a third.

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

## Reward shaping is outside the fixtures

`tiles_placed_bonus`, `rack_value_delta` and `micro_step_cost` are `0.0` in every
preset, so **every golden fixture leaves all nine branches that implement them
untouched** -- three terms in each of three backends. They are specified in
`SPEC.md` section 7 and covered by `test_each_shaping_term_matches_the_reference_and_actually_fires`,
which turns on one term at a time and asserts the total actually moved. A
conformance test that could pass with the term doing nothing is worth nothing.

Those tests need **greedy**, not random, for the reason in the next section.

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

## What belongs in a Gymnasium id

`rummi/env/__init__.py` registers `Rummi-2p-v0`, `Rummi-3p-v0`, `Rummi-4p-v0`.
The rule for adding another: **an id is for something that changes the
observation or action space.** Seat count does, so it is in the id. Opponent
choice does not, so it is a constructor argument -- crossing seats with the five
bundled agents would be fifteen ids that differ in nothing a policy can see.

Only `vector_entry_point` is registered. `gym.make` failing is the intended
behaviour: there is no single-env implementation, and wrapping a batch of one
would hide that.

`RummiVectorEnv(backend=...)` drives any of the three through `rummi/env/api.py`,
and the observation stays in that backend's array type. Two things are host-side
on purpose: the `info` telemetry (`current_player`, `winner` -- `(N,)` vectors a
caller reads on the host anyway, ~1% even on MPS) and the `(N,)` done flags
next-step autoreset needs. I measured the autoreset read at **+1.0%** on MPS at
B=4096, so device-side autoreset would buy nothing; do not build it on a hunch.

`FixedOpponentEnv` and rendering are NumPy-only and refuse other backends at
construction -- the first because `agents.base` reads a `BatchState`, the second
because the renderer does.

**Do not add a tensor-returning mode to this env.** Gymnasium already ships
`wrappers.vector.NumpyToTorch`, `JaxToTorch` and `JaxToNumpy`, which convert
through `from_dlpack` -- a same-device hand-off is a view, not a copy. Non-numpy
vector envs are sanctioned; the pattern is that the env is native in its
backend's array type and the official wrapper converts at the boundary. They are
lazily imported behind `array_api_compat`, which is why they do not show up in
`dir(gymnasium.wrappers)`.

`FixedOpponentEnv` is a **subclass, not a wrapper**, and that is not a style
choice. An opponent's whole turn must run inside one `step`, but the base env
re-deals a finished episode on the *following* step -- driving it from outside
would start a fresh deal partway through and the learner would never see the
terminal observation. Owning the autoreset boundary is the whole difference,
which is why `step` is decomposed into `_check_actions` / `_autoreset` /
`_advance` for it to recombine.

## Why the mask, when DRAW is always legal

A fair question, because the mask is the most expensive thing in a step: on the
standard config `legal_actions` costs **1.6x** the rest of the env put together
(145k env-steps/s with it, 385k without, same actions in both arms).

It stays because **35 of 2400 actions are legal at a typical step -- 1.47%**. An
agent choosing without the mask picks an illegal action ~98.5% of the time, and
`DRAW` is not a benign fallback: per SPEC.md section 4 it *reverts the turn*,
draws and passes. So "illegal falls back to DRAW" is an env where a learner
reverts and passes almost every action, which is byte-identical to the
random-play trap two sections down -- and it destroys a half-built turn rather
than merely wasting a step. Worse for learning, the fallback teaches the policy
`DRAW`'s value for every illegal action, so it can never learn that an action is
illegal *now* and legal next turn.

It is also why the benchmark **disqualifies** an illegal action instead of
substituting one silently: a fallback with no cost is something to farm.

`RummiVectorEnv(action_mask=False, validate_actions=False)` is there for the
callers that genuinely do not need it -- a throughput benchmark, or an agent that
plans a whole turn and only wants the mask once per turn. It is refused unless
validation is dropped too, and `FixedOpponentEnv` refuses it outright, since its
opponents choose from it.

## One way to write an agent

`rummi/agents/` is it — there is no parallel "policies" concept any more, and the
bundled agents are not privileged. `rummi/agents/base.py` holds the contract: an
agent gets the observation and the mask. The observation merges the pool and opponents' racks into one `unseen`
vector precisely so an agent cannot read an opponent's hand.

The bundled agents obey this too — including the CP-SAT one, which is what makes
"the observation is sufficient to play optimally" a tested claim rather than a
hope. Don't shortcut a new agent by handing it `BatchState`; if you hold a state
and need an action, go through `agents.base.act_on_state`, which is the single
bridge and encodes the observation for you.

The strength ladder is deliberate: `greedy` never rearranges, `rearrange` steals
exactly one tile, `optimal` repartitions the table. It exists so a submission has
rungs to place itself between — `optimal` beating `greedy` 100-0 told newcomers
nothing.

## The reference learned agent

`rummi/agents/learned/` — under `agents/` on purpose. A parallel "policies"
concept existed here once and was removed; a network is not a second kind of
agent, it is an agent with weights.

**It is not in `REGISTRY`.** The bundled ladder is `random` through `optimal`, and
a rung carrying weights has to earn its place by landing between `rearrange` (85%)
and `optimal` (100%) first. Promoting it is cheap when it does — `evaluate` takes
`build_agent`, so nothing in the package has to change to score one.

The split, and why:

- `features.py` — concatenation **order** and the per-position **divisor**, as
  data. Two frameworks reading one scale vector cannot drift on it. `table_sets`
  is deliberately excluded: raw kind ids are not ordinal, and `slot_features`
  already pins a run's contents exactly.
- `architecture.py` — `init_params` returns plain NumPy, so both networks are
  built from the *same* weights and parity is testable rather than hoped for.
  The orthogonal init corrects the sign of `diag(R)`; without it `qr` may return
  either valid factorisation and a seed stops being reproducible across LAPACKs.
- `torch_net.py`, `jax_net.py` — the two forward passes.
- `agent.py` — the `Agent` adapter. Deterministic by default: a score has to be
  reproducible.

**`MASKED` is `-1e8`, not `-inf`.** With `-inf` an illegal action's probability is
exactly zero and the entropy term computes `0 * -inf` = NaN, which ends a run. At
-1e8 the probability underflows to zero and the product stays zero.

`tools/train_ppo.py` is the training loop, `--log-json` its metrics, and
`tools/plot_training.py` the curves. **matplotlib, not `render_charts.py`**: that
module hand-writes SVG because the README's two figures have to be theme-aware
inside a GitHub `<img>`, which matplotlib cannot do. A training plot carries no
such constraint, and hand-laying axes produced a clipped legend sitting on top of
its own title before this was moved over.

The three panels are the diagnosis, and each is one scale -- never two on an axis.
`melded` sits near greedy's 92% while `end_turn` is flat at ~1% against its 36%:
**opening is solved, finishing a turn is not, and that is why winning is not.**

`tools/train_ppo.py` is the training loop. The mask is **stored in the rollout**
and reapplied at update time: scoring an old action under a policy that has
forgotten which actions were legal gives a meaningless ratio. `--shaping` turns on
the SPEC section 7 terms and makes the run incomparable to a published score by
design, which is why the eval at the end always uses the unshaped suite.

## Backend traps already paid for

- **JAX**: constant lookup tables must be **NumPy** arrays. An `lru_cache` first
  populated inside a trace caches tracers and leaks them into every later call.
  `cfg` is a static argument (frozen dataclass, hashable). Action validation
  cannot live inside the jitted step — reading a device boolean breaks the trace.
- **torch**: apply effects as masked whole-batch updates, never NumPy's
  per-family `flatnonzero` selection — that read is a host sync every step and
  costs the entire `torch.compile` speedup. `sort` needs
  `sort(dim=-1, stable=True)`; `stable` is keyword-only.
- **torch on MPS** has no `frexp`, so the kernel reads a bit index by counting
  thresholds there while NumPy and JAX read a float exponent. Inductor also fails
  codegen — an internal dtype error, not a message about your code — if
  `legal_actions` selects an ASSIGN code by `slot_ok` before the product, so the
  torch port applies that factor to the finished block instead. Both are about
  `torch.compile` on MPS, the benchmark's headline figure, so run
  `tests/test_torch_backend.py` after touching that kernel.
- **Every backend** must shuffle decks with NumPy's `SeedSequence`/`Generator`.
  The permutation is part of the contract; a framework-native RNG deals
  different tiles and fails conformance.
- `rummi/env/torch/` does not shadow the global `torch` (absolute imports), but
  verify after any move.

## Evaluation protocol is frozen

`rummi/evaluate/protocol.py` pins configs, opponents, game counts and per-game
seeds. Every deal is played once per seat, the agent under test rotating through
all of them, which is why an agent mirrored against itself scores *exactly*
`1 / n_players` and +0.0 — if that stops being exact, the rotation is broken, not
noisy. At two seats this is the old swap; past two it is the only thing that
cancels turn order.

There is one suite per registered Gymnasium id, which is the point of `2.0`: an
id you can train on but not score against is a dead end for a submission.

Editing a suite invalidates every score published against `PROTOCOL_VERSION`.
Bump the version if you must change one -- and then **re-capture**, or the
committed numbers claim a protocol they were not produced under
(`test_every_published_capture_names_the_current_protocol` catches that):

```bash
python tools/capture_agents.py --suite standard-greedy --games 60 --out docs/data/agents.json
python tools/capture_agents.py --suite standard-3p --games 55 --out docs/data/agents-standard-3p.json
python tools/capture_agents.py --suite standard-4p --games 55 --out docs/data/agents-standard-4p.json
python tools/render_charts.py
```

`docs/charts/agents.svg` plots win and stalemate rate only, so a change to
`mean_final_rack` alone leaves it byte-identical -- the figure not moving is not
evidence the data did not.

## The render stack

`rummi/render/board.py` owns **all** the geometry, and both pygame paths -- the
read-only renderer and interactive play -- lay out from it. The rectangle a set is
drawn in is the rectangle a click resolves against; there is no second layout.
It touches no surface and no display, which is why a whole hand can be played in
`tests/test_play.py` with no window.

One flag, `interactive`, separates the two uses. Off, this is what an env renders:
telemetry line, action log, slot numbers and shape tags — you are reading a rollout.
On, it is the play window: pills for whose turn and what is left, a bar for the
opening meld, a score under each set and nothing else, buttons in the margin beside
the rack. Do not "unify" them; the difference is the point.

Things that look like slack but are not:

- **The table area is sized from the config, not from what games do.** A set with
  no rectangle cannot be clicked, and that was a real bug. `rows_needed` bounds it
  by sweeping uniform tables — uniform widths are the worst case for greedy
  wrapping. Shrinking it to a typical game brings the bug back.
- **Cards are never narrower than `min_set` tiles**, so a two-tile set mid-
  rearrangement still has room for its caption.
- **The rack has two tiers whether or not the hand needs both**, as a real one
  does, and that is also what keeps a forty-tile hand standing at full width
  instead of fanned into slivers.
- **Tray tiles fan (overlap) rather than truncate** when they do not fit, and the
  workbench widens past its base rect when even that is not enough. Truncating
  hides tiles, and a hidden tile cannot be picked up.

Frames are painted whole. Dirty-rect tracking used to force a fixed grid of
rectangles, which is what made a row per set sized for a 13-tile run; the
throttle in `driver.py` is the cost control that actually matters.

The play UI's invariant is that **the mask decides what is expressible**: a
gesture either maps to an action `legal_actions` allows or to nothing, and a test
fires random clicks across a whole game asserting exactly that. A press takes a
tile and a release drops it, which is what makes clicking work for free -- but a
release near a press that *took* is a click, not a drop, or picking a tile up
would put it straight back.

`board.storage_order` translates a click on a displayed tile back to the position
`PICK` indexes. Do not collapse `SlotView.shown` into `SlotView.tiles` to save it.

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
