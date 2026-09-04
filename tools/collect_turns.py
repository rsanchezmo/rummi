"""`frugal`'s turns as primitive action sequences, and the gate states inside them.

    python tools/collect_turns.py --target 50000 --out checkpoints/turns-s0.npz

The teacher is `frugal` -- `by_value` plus the stuck-state CP-SAT `REPARTITION`,
the same optimal-tier agent `tools/train_repartition.py` clones -- and what is
recorded is what it *plays*: the observation at the turn boundary, and every
primitive action from there to `END_TURN`.

**A turn it declines is recorded too**, as the one-step sequence `DRAW`. Nothing
in the mask says a turn is going nowhere -- `PLACE` is legal whenever a tile is in
the rack -- so declining is the one decision the student can only learn from a
label, and the boundaries the teacher drew from are where that label lives. Only a
*clean* decline is recorded: a `DRAW` the teacher reached after already lifting
tiles reverts a turn it was trying to play, and labelling that boundary "do not"
would teach the opposite of what happened.

Teacher-driven, and deliberately not DAgger. `oracle_actions`' docstring has the
measurement: with the student steering, 76% of the labels come back `DRAW`, because
a plan made for one state is stale in the next. `MacroAgent` is not a
`PlanningAgent` -- it re-decides whenever an expansion runs out -- so with the
teacher driving its own games every action it takes is one it would take again from
that state, and the sequence is consistent end to end.

Two populations come out of one rollout, because the experiment needs both.

**Turns** are the imitation set and the length breakdown: every committed turn,
tagged with whether it used the `REPARTITION` macro, so an ordinary turn and a
stuck-state one can be scored apart, plus every clean decline tagged `turn_drawn`.
A rate over the played population has to exclude the declines, which is what the
flag is for.

**Gate states** are the arm-A comparison set: every state the repartition gate fires
on and CP-SAT answers, in the mid-turn form it fires in -- the gate can fire after a
template has already played, so the turn boundary is the wrong state to compare on.
States the gate fired on and CP-SAT declined are counted rather than stored, exactly
as `tools/collect_repartitions.py` counts them and for the same reason: they carry
no answer to compare against, and an offline "how often does the net play" that
included them would be measuring the position.

Rows are split by **game**, never by state: two turns of one deal share a rack and
most of a table, so a state-level split scores memorisation.
"""

from __future__ import annotations

import argparse
import dataclasses
import pathlib
import time

import numpy as np

from rummi.agents.base import Observation, turn_starting
from rummi.agents.learned.turn_sim import TurnStart
from rummi.agents.macro import MacroAgent, by_value
from rummi.env.fixed_opponent import FixedOpponentEnv
from rummi.rules.config import CONFIG_BY_NAME, RewardMode, RummiConfig

FIELDS = tuple(f.name for f in dataclasses.fields(TurnStart))


def save_starts(prefix: str, starts: TurnStart) -> dict[str, np.ndarray]:
    """A `TurnStart` as `np.savez` keyword arguments, one array per field."""
    return {f"{prefix}_{name}": getattr(starts, name) for name in FIELDS}


def load_starts(data, prefix: str) -> TurnStart:
    return TurnStart(**{name: data[f"{prefix}_{name}"] for name in FIELDS})


def pad(plans: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
    """`(sequences, lengths)`, `-1` past the end of each."""
    lengths = np.asarray([len(p) for p in plans], dtype=np.int16)
    width = int(lengths.max()) if len(lengths) else 1
    out = np.full((len(plans), width), -1, dtype=np.int16)
    for i, plan in enumerate(plans):
        out[i, : len(plan)] = plan
    return out, lengths


class RecordingAgent(MacroAgent):
    """`frugal`, keeping every turn it commits and every gate state it answers.

    The turn is captured from what `act` actually emits rather than from the
    expansions it queues, because a turn is several macros and the boundary between
    them is not a boundary of anything the primitive space can see. `_repartition`
    is wrapped only to keep the plan its own gate already computed -- the solve is
    what decided the macro is legal, and solving again is neither free nor
    guaranteed to answer the same under a wall-clock limit.
    """

    def __init__(self, cfg: RummiConfig) -> None:
        super().__init__(cfg, choose=by_value(cfg), repartition=True)
        self.turns: list[tuple[TurnStart, list[int], bool, int, bool]] = []
        self.gates: list[tuple[TurnStart, list[int], int]] = []
        self.game = np.zeros(1, dtype=np.int64)
        self.fired = 0
        self.declined = 0
        self.drawn = 0
        self.abandoned = 0
        self._open: dict[int, tuple[TurnStart, list[int]]] = {}
        self._used_solver: dict[int, bool] = {}
        self._pending: dict[int, tuple[TurnStart, list[int]]] = {}
        inner = self.choose

        def choose(obs: Observation, env: int, legal: np.ndarray) -> int:
            macro = inner(obs, env, legal)
            if macro == self.repartition_macro:
                self._used_solver[env] = True
                held = self._pending.pop(env, None)
                if held is not None:
                    self.gates.append((held[0], held[1], int(self.game[env])))
            return macro

        self.choose = choose

    def _repartition(self, obs: Observation, env: int) -> list[int]:
        self.fired += 1
        actions = super()._repartition(obs, env)
        if not actions:
            self.declined += 1
            self._pending.pop(env, None)
            return actions
        # The gate fires mid-turn as readily as at a boundary, so the state the
        # comparison runs from is this one and not the turn's own start.
        self._pending[env] = (
            TurnStart.from_obs(obs, env),
            [*actions, self.cfg.end_turn_action],
        )
        return actions

    def act(
        self, obs: Observation, mask: np.ndarray, active: np.ndarray | None = None
    ) -> np.ndarray:
        fresh = turn_starting(obs)
        out = super().act(obs, mask, active)
        for env in range(mask.shape[0]):
            if active is not None and not active[env]:
                continue
            if fresh[env]:
                self._open[env] = (TurnStart.from_obs(obs, env), [])
                self._used_solver[env] = False
            entry = self._open.get(env)
            if entry is None:
                continue
            action = int(out[env])
            if action == self.cfg.draw_action:
                self.drawn += 1
                if entry[1]:
                    self.abandoned += 1
                else:
                    self.turns.append(
                        (entry[0], [action], self._used_solver[env], int(self.game[env]), True)
                    )
                self._open.pop(env, None)
                self._pending.pop(env, None)
                continue
            entry[1].append(action)
            if action == self.cfg.end_turn_action:
                self.turns.append(
                    (entry[0], entry[1], self._used_solver[env], int(self.game[env]), False)
                )
                self._open.pop(env, None)
        return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="standard", choices=sorted(CONFIG_BY_NAME))
    p.add_argument("--envs", type=int, default=32)
    p.add_argument("--target", type=int, default=50_000, help="committed turns to collect")
    p.add_argument(
        "--stuck-target", type=int, default=10_000,
        help="answered gate states to collect; the run continues past --target "
             "until both are met, because the stuck states are the rarer half and "
             "the length breakdown needs them in their own bucket",
    )
    p.add_argument("--max-steps", type=int, default=2_000_000)
    p.add_argument("--holdout", type=float, default=0.15, help="fraction of *games* held out")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=pathlib.Path, default=pathlib.Path("checkpoints/turns.npz"))
    args = p.parse_args()

    # The reward is never read here; normalising it keeps the rollout identical to
    # the one every other tool in this directory drives.
    cfg = dataclasses.replace(
        CONFIG_BY_NAME[args.config], reward_mode=RewardMode.SCORE_NORMALIZED
    )
    env = FixedOpponentEnv(num_envs=args.envs, cfg=cfg, seed=args.seed, opponent="greedy")
    agent = RecordingAgent(cfg)
    agent.reset(args.envs)
    episode = np.zeros(args.envs, dtype=np.int64)
    agent.game = np.arange(args.envs, dtype=np.int64) * 1_000_000

    obs, info = env.reset()
    started = time.perf_counter()
    steps = 0
    while steps < args.max_steps:
        played = sum(1 for row in agent.turns if not row[4])
        if played >= args.target and len(agent.gates) >= args.stuck_target:
            break
        actions = agent.act(obs, np.asarray(info["action_mask"]))
        obs, _, term, trunc, info = env.step(actions)
        done = np.asarray(term) | np.asarray(trunc)
        episode[done] += 1
        agent.game = np.arange(args.envs, dtype=np.int64) * 1_000_000 + episode
        steps += 1
        if steps % 5000 == 0:
            rate = played / max(time.perf_counter() - started, 1e-9)
            print(
                f"step {steps:>8,}  turns {played:>7,}  "
                f"stuck {len(agent.gates):>6,}  drawn {agent.drawn:>7,}  "
                f"gate {agent.fired:>7,} ({1 - agent.declined / max(agent.fired, 1):>5.1%} answered)  "
                f"{rate:>6.1f} turns/s",
                flush=True,
            )
    elapsed = time.perf_counter() - started
    env.close()

    if not agent.turns:
        raise SystemExit("collected nothing")

    turn_starts = TurnStart.stack([row[0] for row in agent.turns])
    turn_plan, turn_len = pad([row[1] for row in agent.turns])
    turn_repart = np.asarray([row[2] for row in agent.turns], dtype=bool)
    turn_game = np.asarray([row[3] for row in agent.turns], dtype=np.int64)
    turn_drawn = np.asarray([row[4] for row in agent.turns], dtype=bool)

    gate_starts = TurnStart.stack([row[0] for row in agent.gates])
    gate_plan, gate_len = pad([row[1] for row in agent.gates])
    gate_game = np.asarray([row[2] for row in agent.gates], dtype=np.int64)
    # What the solver's answer sheds, read off the plan: a `PLACE` is the only
    # action that moves a tile out of the rack, and it is the lowest block.
    gate_tiles = np.asarray(
        [sum(1 for a in plan if a < cfg.pick_offset) for _, plan, _ in agent.gates],
        dtype=np.int16,
    )

    rng = np.random.default_rng(args.seed)
    unique = np.unique(np.concatenate([turn_game, gate_game]))
    held = set(rng.permutation(unique)[: max(1, round(args.holdout * len(unique)))].tolist())
    turn_holdout = np.asarray([g in held for g in turn_game], dtype=bool)
    gate_holdout = np.asarray([g in held for g in gate_game], dtype=bool)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        config=args.config,
        **save_starts("turn", turn_starts),
        turn_plan=turn_plan,
        turn_len=turn_len,
        turn_repart=turn_repart,
        turn_game=turn_game,
        turn_drawn=turn_drawn,
        turn_holdout=turn_holdout,
        **save_starts("gate", gate_starts),
        gate_plan=gate_plan,
        gate_len=gate_len,
        gate_tiles=gate_tiles,
        gate_game=gate_game,
        gate_holdout=gate_holdout,
        fired=agent.fired,
        declined=agent.declined,
        drawn=agent.drawn,
        abandoned=agent.abandoned,
        steps=steps,
        seconds=elapsed,
    )

    played = turn_len[~turn_drawn].astype(np.int64)
    print(
        f"\ncollected {int((~turn_drawn).sum()):,} committed turns, "
        f"{int(turn_drawn.sum()):,} clean declines and {len(gate_len):,} answered "
        f"gate states in {elapsed / 60:.1f} min over {len(unique):,} games\n"
        f"  turns drawn            {agent.drawn:,} "
        f"({agent.drawn / max(agent.drawn + int((~turn_drawn).sum()), 1):.1%} of turns), "
        f"of which {agent.abandoned:,} reverted a started turn and are not labelled\n"
        f"  used REPARTITION       {turn_repart[~turn_drawn].mean():.1%} of committed turns\n"
        f"  gate fired             {agent.fired:,}, CP-SAT answered "
        f"{1 - agent.declined / max(agent.fired, 1):.1%}\n"
        f"  turn length            mean {played.mean():.1f}  median "
        f"{np.median(played):.0f}  max {played.max()}\n"
        f"    ordinary             mean {played[~turn_repart[~turn_drawn]].mean():.1f}  median "
        f"{np.median(played[~turn_repart[~turn_drawn]]):.0f}\n"
        f"    with a repartition   mean {played[turn_repart[~turn_drawn]].mean():.1f}  median "
        f"{np.median(played[turn_repart[~turn_drawn]]):.0f}\n"
        f"  gate answer length     mean {gate_len.mean():.1f}  median "
        f"{np.median(gate_len):.0f}  max {gate_len.max()}\n"
        f"  gate answer tiles      mean {gate_tiles.mean():.2f}\n"
        f"  holdout {turn_holdout.mean():.1%} of turns, {gate_holdout.mean():.1%} of gate states\n"
        f"wrote {args.out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
