"""Why `macro/by_value` can only draw, measured per config.

    python tools/diagnose_stuck.py

At every clean decision point of `macro/by_value` against the suite opponent, this
counts what the macro space refused and what `greedy` -- primitives, no
rearranging -- would have done in the same state. It exists because the `tiny`
regression was first *attributed* to slot scarcity (4 table slots against 35, and
the two slot-consuming macros gate on a free slot) and the gate turned out never
to fire on either config: the tile-feasible-but-slot-blocked columns are the
refutation, kept so the number stays reproducible.

What the same sweep did find is in the greedy columns: every stuck state where
greedy still plays is an append the macro space cannot express -- a lay-off onto a
joker-holding set (`extensions` refuses those because the joker's role is
ambiguous) or laying the rack's own joker off. The plain-append column is the
consistency check that the classification is exhaustive; it should be zero,
because a plain single-tile append is exactly what `EXTEND` offers.

Greedy's own feasibility helpers are imported rather than reimplemented: the
question is what *greedy* would do, and a reimplementation would measure a copy.
"""

from __future__ import annotations

import argparse

import numpy as np

from rummi.agents.base import Observation, has_melded, table
from rummi.agents.greedy_agent import _appendable, _best_new_set, plan_turn
from rummi.agents.macro import MacroAgent, by_value, playable, removals, set_templates
from rummi.env.fixed_opponent import FixedOpponentEnv
from rummi.rules.config import CONFIG_BY_NAME
from rummi.solver.to_actions import slot_contents


def diagnose(cfg_name: str, envs: int, steps: int, seed: int) -> dict[str, int]:
    cfg = CONFIG_BY_NAME[cfg_name]
    templates = set_templates(cfg)
    agent = MacroAgent(cfg)
    base = by_value(cfg)

    counts = dict.fromkeys(
        [
            "decisions", "no_free_slot",
            "newset_feasible", "newset_slot_blocked",
            "steal_feasible", "steal_slot_blocked",
            "stuck", "stuck_greedy_acts",
            "stuck_append_joker_set", "stuck_append_joker_tile", "stuck_append_plain",
            "stuck_newset_gap",
            "chose_set", "chose_extend", "chose_steal", "chose_end", "chose_draw",
        ],
        0,
    )

    def choose(obs: Observation, env: int, legal: np.ndarray) -> int:
        board = table(obs)[env]
        rack = np.asarray(obs["rack"][env])
        slot_free = bool((board.max(-1) < 0).any())
        can_new_set = bool(playable(cfg, rack[None])[0].any())
        melded = bool(has_melded(obs)[env]) or not cfg.strict_initial_meld

        can_steal = False
        if melded:
            stealable = np.zeros(cfg.n_kinds, dtype=bool)
            for contents in slot_contents(board):
                for kind in removals(cfg, contents):
                    stealable[kind] = True
            if stealable.any():
                gap = np.maximum(templates - rack, 0)
                short = gap.sum(-1)
                can_steal = bool(((short == 1) & stealable[gap.argmax(-1)]).any())

        counts["decisions"] += 1
        counts["no_free_slot"] += not slot_free
        counts["newset_feasible"] += can_new_set
        counts["newset_slot_blocked"] += can_new_set and not slot_free
        counts["steal_feasible"] += can_steal
        counts["steal_slot_blocked"] += can_steal and not slot_free

        # END_TURN excluded: "nothing left but end or draw" is a finished turn,
        # not a blocked one.
        stuck = not legal[: agent.end_macro].any()
        counts["stuck"] += stuck
        if stuck:
            acts = bool(plan_turn(cfg, rack, board, melded))
            counts["stuck_greedy_acts"] += acts
            if acts and melded:
                allowed = _appendable(cfg, board, rack)
                joker_slots = (board == cfg.joker_kind).any(-1)
                plain = allowed[~joker_slots].copy()
                joker_tile = bool(plain[:, cfg.joker_kind].any())
                plain[:, cfg.joker_kind] = False
                onto_joker = bool(allowed[joker_slots].any())
                counts["stuck_append_plain"] += bool(plain.any())
                counts["stuck_append_joker_tile"] += joker_tile and not plain.any()
                counts["stuck_append_joker_set"] += (
                    onto_joker and not joker_tile and not plain.any()
                )
                counts["stuck_newset_gap"] += (
                    not can_new_set and _best_new_set(cfg, rack, by_value=False) is not None
                )

        macro = base(obs, env, legal)
        if macro < agent.extend_offset:
            counts["chose_set"] += 1
        elif macro < agent.steal_offset:
            counts["chose_extend"] += 1
        elif macro < agent.end_macro:
            counts["chose_steal"] += 1
        elif macro == agent.end_macro:
            counts["chose_end"] += 1
        else:
            counts["chose_draw"] += 1
        return macro

    agent.choose = choose
    env = FixedOpponentEnv(num_envs=envs, cfg=cfg, seed=seed, opponent="greedy")
    obs, info = env.reset()
    agent.reset(envs)
    for _ in range(steps):
        obs, _, _, _, info = env.step(agent.act(obs, np.asarray(info["action_mask"])))
    env.close()
    return counts


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--configs", nargs="+", default=["tiny_groups", "standard"], choices=sorted(CONFIG_BY_NAME))
    p.add_argument("--envs", type=int, default=32)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    for name in args.configs:
        c = diagnose(name, args.envs, args.steps, args.seed)
        n = max(c["decisions"], 1)
        stuck = max(c["stuck"], 1)
        cfg = CONFIG_BY_NAME[name]
        print(f"\n{name}  (max_sets={cfg.max_sets})  decisions={c['decisions']:,}")
        print(f"  no free slot                  {c['no_free_slot'] / n:>6.1%}")
        print(f"  new set tile-feasible         {c['newset_feasible'] / n:>6.1%}"
              f"   blocked only by slot {c['newset_slot_blocked'] / n:>6.1%}")
        print(f"  steal tile-feasible           {c['steal_feasible'] / n:>6.1%}"
              f"   blocked only by slot {c['steal_slot_blocked'] / n:>6.1%}")
        print(f"  stuck (only DRAW available)   {c['stuck'] / n:>6.1%} of decisions")
        print(f"  stuck but greedy would act    {c['stuck_greedy_acts'] / stuck:>6.1%} of stuck states")
        print(f"    append onto a joker set     {c['stuck_append_joker_set'] / stuck:>6.1%}")
        print(f"    lays off the rack's joker   {c['stuck_append_joker_tile'] / stuck:>6.1%}")
        print(f"    plain append (must be 0)    {c['stuck_append_plain'] / stuck:>6.1%}")
        print(f"    new set beyond playable()   {c['stuck_newset_gap'] / stuck:>6.1%}")
        chosen = {k[6:]: v for k, v in c.items() if k.startswith("chose_")}
        total = max(sum(chosen.values()), 1)
        print("  chosen: " + "  ".join(f"{k} {v / total:.1%}" for k, v in chosen.items()))


if __name__ == "__main__":
    main()
