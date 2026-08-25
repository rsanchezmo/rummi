# Contributing

## Adding an agent

Agents live in `rummi/agents/`. Implement `Agent` — or subclass `PlanningAgent`
if you decide a whole turn at once and want the plan-caching handled for you —
then register it in `rummi/agents/__init__.py`.

```python
from rummi.agents.base import Observation, PlanningAgent

class MyAgent(PlanningAgent):
    name = "mine"

    def plan(self, obs: Observation, env: int) -> list[int]:
        return [...]                      # action ids for one turn, or []
```

Then:

```bash
python -m rummi.evaluate.run --agent mine --suites tiny
```

Requirements, all enforced by the test suite rather than by review:

- **Read only the observation.** Never take a `BatchState`. The observation
  deliberately merges the pool and opponents' racks into one `unseen` vector, so
  an agent that reads the state can see hands it should not, and its score means
  nothing.
- **Never return a masked-out action.** The mask exactly describes what the rules
  permit and is never all-zero, so an illegal action is a bug. It disqualifies
  the run rather than costing reward, because scoring it would invite tuning
  against the penalty.
- **Honour `active`.** In a match each agent is told which envs are its own. An
  agent that caches per-env plans and ignores `active` will consume the plan of
  an env another agent is playing. This has already broken once.

Include a benchmark result in the PR — `--suites tiny` at minimum, and
`standard-greedy` if it is not too slow.

## Changing the rules

**Every rules change is three changes plus a spec edit.** `rummi/env/numpy/` is
the reference and `rummi/env/torch/`, `rummi/env/jax/` are independent
implementations of the same contract. Touching one leaves
`tests/test_backends.py` failing and the benchmark meaningless.

Anything definitional — configuration, tile encoding, the action layout — belongs
in `rummi/rules/`, which is backend-free, so the three implementations cannot
drift on it.

`tests/golden/*.json` are a contract, not a snapshot. **Never regenerate them to
make a test pass.** A changed digest means the rules changed, which is either a
bug or a decision that needs `PROTOCOL_VERSION` and the published scores
revisited.

## Changing the evaluation protocol

`rummi/evaluate/protocol.py` is frozen and versioned. Editing a suite invalidates
every score published against the current `PROTOCOL_VERSION`, so bump it.

The rotation is self-checking: an agent played against itself must score
*exactly* `1 / n_players` and +0.0. If that stops being exact, it is broken, not
noisy.

## Testing

```bash
pytest                                                   # everything, ~50s
python -m rummi.bench.fuzz --policy greedy --games 500    # invariant fuzzing
python -m rummi.bench.bench_backends --compile            # throughput
```

Two things that will waste your time if you do not know them:

- **Random play cannot test this game.** In ten million fuzz steps it assembled a
  legal opening meld four times; on the standard config it is byte-identical to
  passing every turn. Any test needing `END_TURN`, melding or winning must use
  `--policy greedy` or better. This already caused one false "the engine is
  broken" diagnosis.
- **The table is allowed to be in pieces mid-turn.** It must be whole only at a
  turn boundary (`micro_count == 0`). Asserting validity mid-turn is wrong, and
  was a real bug in the first fuzzer.

`CLAUDE.md` collects the rest of the traps — JAX tracer leaks, torch host syncs,
deck-shuffling determinism — and is worth reading before touching a backend.
