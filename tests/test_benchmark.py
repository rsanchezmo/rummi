"""The benchmark's own guarantees: fairness, determinism, and integrity.

These matter more than the usual unit test. A benchmark whose scores drift, or
that quietly favours a seat, produces numbers people will compare across
machines and months, and a subtle bias here is much worse than a crash.
"""

import numpy as np
import pytest

from rummi.agents.base import Agent, has_melded, table, turn_starting
from rummi.agents import REGISTRY, GreedyAgent, build
from rummi.evaluate import protocol
from rummi.evaluate.protocol import (
    PROTOCOL_VERSION,
    SUITES,
    SUITE_BY_NAME,
    Suite,
    evaluate,
)
from rummi.rules.config import STANDARD, STANDARD_3P, STANDARD_4P, TINY_GROUPS
from rummi.env.numpy.deal import reset
from rummi.env.numpy.masks import legal_actions
from rummi.env.numpy.engine import step
from rummi.env.observation import encode

TINY = SUITE_BY_NAME["tiny"]


def test_the_set_templates_are_every_set_that_could_be_legal():
    """Counted rather than trusted: 264 runs and 65 groups on the standard config.
    A template list that silently lost a shape would cap what any macro policy can
    ever play, and nothing else would fail."""
    from rummi.agents.macro import n_macros, set_templates, template_points

    templates = set_templates(STANDARD)
    assert templates.shape == (329, STANDARD.n_kinds)
    # 329 new sets, one lay-off per kind, 329 steals, END_TURN and DRAW. The lay-off
    # block already spans every kind including the joker, so what it may play is a
    # question of feasibility and never of layout -- a head trained on one width has
    # to stay comparable.
    assert n_macros(STANDARD) == 713
    # Every template is a legal set: no duplicate kind, and within the length bounds.
    assert (templates <= 1).all()
    sizes = templates.sum(-1)
    assert sizes.min() >= STANDARD.min_set
    assert sizes.max() <= STANDARD.max_set_len
    assert template_points(STANDARD).min() >= 3
    # Cached, and the cache must not hand back a mutable view that a caller edits.
    assert set_templates(STANDARD) is templates


@pytest.mark.parametrize("cfg", [TINY_GROUPS, STANDARD_4P], ids=["tiny", "4p"])
def test_a_macro_agent_only_ever_proposes_legal_actions(cfg):
    """The whole point of the macro space: every action leaves the table whole, so
    no expansion can strand tiles on the workbench. Tile conservation is the check
    that would catch a bad expansion."""
    from rummi.agents.base import act_on_state
    from rummi.agents.macro import MacroAgent, by_value, first_legal

    for choose in (first_legal, by_value(cfg)):
        agent = MacroAgent(cfg, choose=choose)
        agent.reset(8)
        state = reset(cfg, 8, seed=2)
        for _ in range(120):
            mask = legal_actions(state)
            action = act_on_state(agent, state, mask)
            assert mask[np.arange(8), action].all(), "proposed a masked-out action"
            step(state, action, mask)
            state.check_invariants()


def test_the_hybrid_space_contains_the_macro_agent_exactly():
    """The hybrid space is a strict superset, and that has to be exact: if
    `macro_first` did not reproduce the macro agent action for action, a difference
    in training could come from the wrapper rather than from the added primitives.

    It also pins the rule that makes the two blocks safe to mix -- a macro's
    expansion balances the board against tiles played from the rack, so it is
    offered only when nothing is held mid-turn."""
    from rummi.agents.base import act_on_state
    from rummi.agents.hybrid import HybridAgent, macro_first
    from rummi.agents.macro import MacroAgent, by_value

    cfg = STANDARD
    hybrid = HybridAgent(cfg, choose=macro_first(cfg))
    macro = MacroAgent(cfg, choose=by_value(cfg))
    hybrid.reset(12)
    macro.reset(12)
    left, right = reset(cfg, 12, seed=5), reset(cfg, 12, seed=5)
    dirty_seen = 0
    for _ in range(200):
        l_mask, r_mask = legal_actions(left), legal_actions(right)
        # Once each: `act` pops the agent's queued plan, so calling it twice would
        # advance the macro agent an extra step per iteration.
        l_act = act_on_state(hybrid, left, l_mask)
        r_act = act_on_state(macro, right, r_mask)
        np.testing.assert_array_equal(l_act, r_act)

        obs = encode(left)
        for env in range(12):
            offered = hybrid.legal(obs, env, l_mask)[hybrid.macro_offset :].any()
            if np.asarray(obs["workbench"])[env].sum() > 0:
                dirty_seen += 1
                assert not offered, "offered a macro with tiles held mid-turn"
        step(left, l_act, l_mask)
        step(right, r_act, r_mask)
    assert dirty_seen > 0, "never saw a mid-turn state, so the rule was not exercised"


def test_a_tile_may_only_be_appended_to_a_set_that_stays_valid():
    """`EXTEND`'s feasibility test, shared with `greedy`'s own planner. It grows the
    slot row and validates it through `evaluate_slots` rather than deciding by
    colour/number arithmetic, and the joker is the entire reason: a set holding one
    still takes tiles, because the run window accepts whichever reading keeps the set
    legal, and the rack's own joker lays off onto anything with room. Arithmetic over
    the real tiles can express neither, and that gap was **28.5%** of the states where
    this space could only draw while `greedy` played."""
    from rummi.agents.greedy_agent import appendable
    from rummi.env.numpy.sets import pad_slot
    from rummi.rules.encoding import kind_of

    cfg = STANDARD
    joker = cfg.joker_kind
    full_rack = np.ones(cfg.n_kinds, dtype=np.int16)

    def accepts(tiles, rack=full_rack) -> list[int]:
        board = np.full((cfg.max_sets, cfg.max_set_len), -1, dtype=np.int16)
        board[0] = pad_slot(cfg, tiles)
        allowed = appendable(cfg, board, rack)
        assert not allowed[1:].any(), "an empty slot is not a set to append to"
        return sorted(int(k) for k in np.flatnonzero(allowed[0]))

    red = [kind_of(cfg, 0, n) for n in range(1, cfg.n_numbers + 1)]
    sevens = [kind_of(cfg, c, 7) for c in range(cfg.n_colors)]

    # A run holding a joker takes both its real ends: the joker reads as R3 to take
    # R4, and slides to R4 to take R3.
    assert accepts([red[0], red[1], joker]) == sorted([red[2], red[3], joker])
    # A group holding one takes every colour it is missing.
    assert accepts([sevens[0], sevens[1], joker]) == sorted([sevens[2], sevens[3], joker])
    assert accepts(red[:3]) == sorted([red[3], joker]), "a plain run takes the rack's joker"

    only_joker = np.zeros(cfg.n_kinds, dtype=np.int16)
    only_joker[joker] = 1
    assert accepts(red[:3], only_joker) == [joker]
    assert accepts([red[0], red[1], joker], only_joker) == [joker]
    # Every append needs the tile in hand -- the mask would refuse it otherwise.
    assert accepts(red[:3], np.zeros(cfg.n_kinds, dtype=np.int16)) == []
    # A set at max_set_len has nowhere to put one, whatever the rack holds.
    assert accepts(red) == []
    assert accepts(sevens) == []


def test_a_tile_may_only_leave_a_set_that_survives_losing_it():
    """`removals` is the whole of what `rearrange` does, and getting it wrong would
    put an invalid set on the table. A run gives up either end while it stays
    `min_set` long; a group any member; a set holding a joker nothing, because the
    joker's role is ambiguous. Middle tiles of a run are refused -- taking one splits
    the set, which needs a second free slot and is a different move."""
    from rummi.agents.macro import removals
    from rummi.rules.encoding import kind_of

    cfg = STANDARD
    run5 = tuple(kind_of(cfg, 0, n) for n in range(1, 6))
    run3 = tuple(kind_of(cfg, 0, n) for n in range(1, 4))
    group4 = tuple(kind_of(cfg, c, 7) for c in range(4))

    assert removals(cfg, run5) == [run5[0], run5[-1]]
    assert removals(cfg, run3) == [], "a run of min_set has nothing to spare"
    assert sorted(removals(cfg, group4)) == sorted(group4)
    assert removals(cfg, group4[:3]) == [], "a group of min_set has nothing to spare"
    assert removals(cfg, (run3[0], run3[1], cfg.joker_kind)) == []
    assert removals(cfg, ()) == []


def test_every_macro_capability_is_actually_reached():
    """Each block of the action space has to be *used*, not merely offered. The
    END_TURN macro was unreachable for a whole commit while looking fine, so every
    block added since is pinned by being chosen at least once in real play."""
    from rummi.agents.base import act_on_state
    from rummi.agents.macro import MacroAgent, by_value

    cfg = STANDARD
    agent = MacroAgent(cfg)
    used = {"set": 0, "extend": 0, "steal": 0, "end": 0, "draw": 0}
    base = by_value(cfg)

    def choose(obs, env, legal):
        macro = base(obs, env, legal)
        if macro < agent.extend_offset:
            used["set"] += 1
        elif macro < agent.steal_offset:
            used["extend"] += 1
        elif macro < agent.end_macro:
            used["steal"] += 1
        elif macro == agent.end_macro:
            used["end"] += 1
        else:
            used["draw"] += 1
        return macro

    agent.choose = choose
    agent.reset(24)
    state = reset(cfg, 24, seed=5)
    for _ in range(250):
        mask = legal_actions(state)
        action = act_on_state(agent, state, mask)
        assert mask[np.arange(24), action].all()
        step(state, action, mask)
        state.check_invariants()

    assert all(count > 0 for count in used.values()), used


def test_a_joker_is_laid_off_in_real_play_and_the_table_stays_whole():
    """Being offered a joker lay-off is not the claim; playing one is. `greedy` acted
    in 28.5% of the states where this space could only draw, and *all* of it was
    jokers -- appends onto a joker-holding set (27.7%) and the rack's own joker
    (0.9%) -- so both have to be chosen in real play, and their expansions have to
    leave the table whole at every turn boundary, which is the invariant the macro
    space rests on."""
    from rummi.agents.base import act_on_state
    from rummi.agents.greedy_agent import appendable
    from rummi.agents.macro import MacroAgent, by_value
    from rummi.env.numpy.sets import evaluate_slots

    cfg = STANDARD
    agent = MacroAgent(cfg)
    base = by_value(cfg)
    onto_joker_set = laid_own_joker = 0

    def choose(obs, env, legal):
        nonlocal onto_joker_set, laid_own_joker
        macro = base(obs, env, legal)
        if agent.extend_offset <= macro < agent.steal_offset:
            kind = macro - agent.extend_offset
            board = table(obs)[env]
            allowed = appendable(cfg, board, np.asarray(obs["rack"][env]))
            # The receiving slot `expand` will pick, read from the same matrix.
            slot = int(np.flatnonzero(allowed[:, kind])[0])
            onto_joker_set += bool((board[slot] == cfg.joker_kind).any())
            laid_own_joker += kind == cfg.joker_kind
        return macro

    agent.choose = choose
    agent.reset(24)
    state = reset(cfg, 24, seed=5)
    for _ in range(300):
        mask = legal_actions(state)
        action = act_on_state(agent, state, mask)
        assert mask[np.arange(24), action].all(), "proposed a masked-out action"
        step(state, action, mask)
        state.check_invariants()
        verdict = evaluate_slots(cfg, state.table_sets)
        whole = (verdict.is_valid | verdict.is_empty).all(-1)
        assert whole[state.micro_count == 0].all(), "left a broken set at a turn boundary"

    assert onto_joker_set > 0, "never appended onto a joker-holding set"
    assert laid_own_joker > 0, "never laid off the rack's own joker"


def test_touching_the_table_is_illegal_until_the_opening_meld():
    """The table is untouchable before the opening meld, exactly as the env's own
    mask has it (`may_touch_table = has_melded if strict_initial_meld`). This covers
    both macros that touch it, EXTEND and STEAL: one offered there would be a macro
    whose expansion the mask then refuses, which surfaces as a silently abandoned
    turn rather than as an error."""
    from rummi.agents.base import act_on_state, has_melded
    from rummi.agents.macro import MacroAgent, by_value

    cfg = STANDARD
    assert cfg.strict_initial_meld, "this test is about the strict rule"
    agent = MacroAgent(cfg, choose=by_value(cfg))
    agent.reset(16)
    state = reset(cfg, 16, seed=5)
    seen_premeld = seen_postmeld = 0
    for _ in range(200):
        mask = legal_actions(state)
        obs = encode(state)
        melded = has_melded(obs)
        for env in range(16):
            legal = agent.legal_macros(obs, env, mask)
            # extend_offset..end_macro spans EXTEND and STEAL, the two that touch it.
            offered = bool(legal[agent.extend_offset : agent.end_macro].any())
            if melded[env]:
                seen_postmeld += int(offered)
            else:
                assert not offered, "offered a table move before the opening meld"
                seen_premeld += 1
        step(state, act_on_state(agent, state, mask), mask)

    assert seen_premeld > 0, "never observed a pre-meld state, so this proved nothing"
    assert seen_postmeld > 0, "never offered a lay-off at all, so nor did this"


def test_laying_off_is_most_of_what_a_melding_agent_does():
    """84.2% of greedy's ASSIGNs land on an occupied slot, so a macro space that
    could only create new sets was missing the commonest move in the game -- worth
    76 points of mean score when it was added. This pins the capability by the same
    measure on both agents, which is what caught the last macro bug."""
    from rummi.agents import build
    from rummi.agents.base import act_on_state, table
    from rummi.agents.macro import MacroAgent, by_value
    from rummi.rules.actions import decode_batch

    cfg = STANDARD

    def extension_share(agent):
        agent.reset(16)
        state = reset(cfg, 16, seed=5)
        onto_occupied = onto_empty = 0
        for _ in range(200):
            mask = legal_actions(state)
            obs = encode(state)
            action = act_on_state(agent, state, mask)
            decoded = decode_batch(cfg, action)
            occupied = table(obs).max(-1) >= 0
            for env in np.flatnonzero(decoded.is_assign):
                if occupied[env, int(decoded.slot[env])]:
                    onto_occupied += 1
                else:
                    onto_empty += 1
            step(state, action, mask)
        assert onto_occupied + onto_empty > 0, "the state never advanced"
        return onto_occupied / (onto_occupied + onto_empty)

    greedy = extension_share(build("greedy", cfg))
    macro = extension_share(MacroAgent(cfg, choose=by_value(cfg)))
    assert greedy > 0.7, greedy
    # Within a third of the teacher's rate: the macro agent should be laying off at
    # a broadly similar rate, not merely be capable of it.
    assert macro > greedy / 1.5, (macro, greedy)


def test_a_macro_plays_a_set_without_ending_the_turn():
    """`to_actions.plan` commits the turn, because it exists for a solver deciding
    a whole turn at once. Left in, every turn is capped at one set -- and before the
    opening meld that means finding 30 points in a *single* set. It cost
    `first_legal` 130 points of mean score and left the END_TURN macro unreachable
    (chosen 0% of the time) until it was caught, so this pins all three halves of
    the property: no expansion ends the turn, ending is a choice the agent actually
    makes, and a turn can hold more than one set."""
    from rummi.agents.base import act_on_state
    from rummi.agents.macro import MacroAgent, by_value, set_templates

    cfg = STANDARD
    n_templates = len(set_templates(cfg))
    chose_end = 0
    sets_in_turn: list[int] = []
    running = np.zeros(16, dtype=int)

    def choose(obs, env, legal):
        nonlocal chose_end
        macro = by_value(cfg)(obs, env, legal)
        if macro < n_templates:
            running[env] += 1
        else:
            chose_end += int(macro == n_templates)
            sets_in_turn.append(int(running[env]))
            running[env] = 0
        return macro

    agent = MacroAgent(cfg, choose=choose)
    agent.reset(16)
    state = reset(cfg, 16, seed=5)
    for _ in range(300):
        mask = legal_actions(state)
        step(state, act_on_state(agent, state, mask), mask)

    # No expansion of a set may commit the turn.
    obs = encode(state)
    for env in range(16):
        legal = agent.legal_macros(obs, env, legal_actions(state))
        for macro in np.flatnonzero(legal[:n_templates])[:3]:
            assert cfg.end_turn_action not in agent.expand(obs, env, int(macro))

    assert chose_end > 0, "END_TURN was never chosen, so the macro is unreachable"
    assert max(sets_in_turn) > 1, f"no turn played more than one set: {set(sets_in_turn)}"


def test_delegating_with_always_play_is_exactly_its_inner_agent():
    """The sanity check that makes the comparison meaningful: the two-action agent
    at `always_play` must *be* `greedy`, action for action, or a difference in
    score could come from the wrapper rather than from holding."""
    from rummi.agents.base import act_on_state
    from rummi.agents.delegating import DelegatingAgent, always_play, tiles_at_least

    cfg = TINY_GROUPS
    # `k=1` must agree too: a valid turn always plays at least one tile, so the
    # threshold cannot bite at 1. If it ever does, `PlanSummary.tiles` is wrong.
    for decide in (always_play, tiles_at_least(1)):
        inner, agent = build("greedy", cfg), DelegatingAgent(cfg, inner="greedy", decide=decide)
        inner.reset(6)
        agent.reset(6)
        a_state, b_state = reset(cfg, 6, seed=3), reset(cfg, 6, seed=3)
        for _ in range(60):
            a_mask, b_mask = legal_actions(a_state), legal_actions(b_state)
            a_act = act_on_state(inner, a_state, a_mask)
            b_act = act_on_state(agent, b_state, b_mask)
            np.testing.assert_array_equal(a_act, b_act)
            step(a_state, a_act, a_mask)
            step(b_state, b_act, b_mask)
        assert agent.held == 0
        assert agent.played > 0, "the probe never reached a playable turn"


def test_a_holding_threshold_holds_turns_and_keeps_playing_legally():
    """Holding must cost tiles played, not legality: an empty plan is already
    `DRAW`, which is legal at every step."""
    from rummi.agents.delegating import DelegatingAgent, tiles_at_least

    cfg = TINY_GROUPS
    suite = Suite(TINY.name, cfg, opponent="greedy", games=8, seed_base=TINY.seed_base)
    held = {}
    for k in (1, 3):
        agent = DelegatingAgent(cfg, inner="greedy", decide=tiles_at_least(k))
        result = evaluate(f"delegate-{k}", suite, build_agent=lambda c, a=agent: a)
        assert result.illegal_attempts == 0
        assert not result.disqualified
        held[k] = agent.held
    assert held[1] == 0
    assert held[3] > 0, "a threshold of 3 tiles never held a turn, so it proved nothing"


def test_every_reference_agent_satisfies_the_protocol():
    for name in REGISTRY:
        agent = build(name, TINY_GROUPS)
        assert isinstance(agent, Agent), name
        assert agent.name


@pytest.mark.parametrize("name", ["greedy", "weighted-random"])
def test_an_agent_against_itself_scores_exactly_even(name: str):
    """The fairness guarantee. Seat rotation must cancel the turn-order advantage
    *and* the luck of the deal, so a self-match is 50% exactly -- not 50% within
    noise."""
    suite = Suite(TINY.name, TINY.cfg, opponent=name, games=40, seed_base=TINY.seed_base)
    result = evaluate(name, suite)
    assert result.win_rate == pytest.approx(0.5)
    assert result.mean_score == pytest.approx(0.0)
    assert result.wins + result.losses == result.games


@pytest.mark.parametrize("cfg,seats", [(STANDARD_3P, 3), (STANDARD_4P, 4)])
def test_a_self_match_is_exactly_one_over_the_seat_count(cfg, seats: int):
    """The same guarantee past two seats, which is the whole reason rotation
    replaced the swap. Exact, not approximate: with one policy in every seat the
    game is identical whichever seat is under test, so the agent wins exactly one
    of the `seats` rotations and the payouts cancel to zero."""
    suite = Suite(f"self-{seats}p", cfg, opponent="greedy", games=6, seed_base=1_000)
    result = evaluate("greedy", suite)
    assert result.win_rate == pytest.approx(1 / seats)
    assert result.mean_score == pytest.approx(0.0)
    assert result.games == suite.total_games


def test_baselines_rank_in_the_expected_order():
    """The ladder is the benchmark's whole value proposition: a submission needs
    rungs to place itself between."""
    scores = {
        name: evaluate(name, TINY, games=40).win_rate
        for name in ("random", "greedy", "rearrange", "optimal")
    }
    assert scores["random"] < scores["greedy"], scores
    assert scores["greedy"] <= scores["rearrange"] <= scores["optimal"], scores
    assert scores["greedy"] == pytest.approx(0.5), "greedy is the suite's own opponent"


def test_rearrange_never_plays_worse_than_greedy():
    """It falls back to greedy's plan whenever greedy has one, so it cannot be
    worse -- and when greedy is stuck it steals a tile instead of drawing."""
    from rummi.agents.greedy_agent import plan_turn
    from rummi.agents.rearrange_agent import RearrangeAgent

    cfg = TINY_GROUPS
    state = reset(cfg, 6, seed=29)
    agent = RearrangeAgent(cfg)
    agent.reset(6)
    stole = 0

    for _ in range(400):
        mask = legal_actions(state)
        obs = encode(state)
        actions = agent.act(obs, mask)
        for env in range(state.batch_size):
            if state.done[env] or state.micro_count[env] != 0:
                continue
            seat = int(state.current[env])
            greedy = plan_turn(
                cfg, state.racks[env, seat], state.table_sets[env], bool(state.melded[env, seat])
            )
            if greedy:
                assert int(actions[env]) == greedy[0], "should defer to greedy when it has a play"
            elif int(actions[env]) != cfg.draw_action:
                stole += 1
        step(state, actions, mask)
        if state.done.all():
            break
    assert stole > 0, "never found a steal, so the agent adds nothing over greedy"


def test_results_are_reproducible():
    a = evaluate("greedy", TINY, games=24)
    b = evaluate("greedy", TINY, games=24)
    assert (a.wins, a.losses, a.stalemates) == (b.wins, b.losses, b.stalemates)
    assert a.turns == b.turns
    assert a.scores == b.scores


def test_game_seeds_depend_only_on_position():
    """Batching must not change which deals are played, or a score would depend
    on batch_size."""
    first = protocol._game_seeds(TINY, 0, 8)
    later = protocol._game_seeds(TINY, 4, 4)
    assert [s.entropy for s in first[4:]] == [s.entropy for s in later]


def test_an_agent_proposing_illegal_actions_is_disqualified():
    class Cheater:
        name = "cheater"

        def __init__(self, cfg):
            self.cfg = cfg

        def reset(self, n_envs):
            pass

        def act(self, obs, mask, active=None):
            # END_TURN is illegal until a legal meld is on the table.
            return np.full(mask.shape[0], self.cfg.end_turn_action, dtype=np.int64)

    result = evaluate("cheater", TINY, build_agent=Cheater, games=8)
    assert result.disqualified
    assert result.illegal_attempts > 0
    assert result.games > 0, "the suite must still complete and report"


def test_a_passive_agent_is_legal_but_loses():
    """Always drawing is legal Rummikub, so it must not be disqualified -- it must
    simply lose."""

    class Passer:
        name = "passer"

        def __init__(self, cfg):
            self.cfg = cfg

        def reset(self, n_envs):
            pass

        def act(self, obs, mask, active=None):
            return np.full(mask.shape[0], self.cfg.draw_action, dtype=np.int64)

    result = evaluate("passer", TINY, build_agent=Passer, games=24)
    assert not result.disqualified
    assert result.win_rate < 0.5


def test_the_protocol_is_frozen():
    """A guard, not a formality: editing a suite silently invalidates every score
    published against this version."""
    assert PROTOCOL_VERSION == "2.0"
    fingerprint = [
        (s.name, s.cfg.n_players, s.opponent, s.games, s.seed_base) for s in SUITES
    ]
    assert fingerprint == [
        ("tiny", 2, "greedy", 100, 1_000),
        ("standard-greedy", 2, "greedy", 200, 2_000),
        ("standard-optimal", 2, "optimal", 100, 3_000),
        ("standard-3p", 3, "greedy", 70, 4_000),
        ("standard-4p", 4, "greedy", 55, 5_000),
    ]


def test_every_registered_env_has_a_suite():
    """The reason 2.0 exists: an id you can train on and cannot score against is
    a dead end for a submission."""
    from rummi.env import ENV_CONFIGS

    scored = {s.cfg.n_players for s in SUITES}
    for env_id, cfg in ENV_CONFIGS.items():
        assert cfg.n_players in scored, f"{env_id} has no suite at {cfg.n_players} seats"


def test_the_observation_is_sufficient_to_play():
    """The integrity property the benchmark rests on.

    Agents see only the observation, never the state. If an observation-driven
    agent makes exactly the same moves as the state-driven planner it wraps, then
    nothing a legal player needs is missing from the observation -- and no agent
    is handicapped by being denied the state.
    """
    from rummi.agents.greedy_agent import plan_turn

    cfg = TINY_GROUPS
    state = reset(cfg, 6, seed=17)
    agent = GreedyAgent(cfg)
    agent.reset(6)
    compared = 0

    for _ in range(300):
        mask = legal_actions(state)
        obs = encode(state)
        from_obs = agent.act(obs, mask)

        for env in range(state.batch_size):
            if state.done[env] or state.micro_count[env] != 0:
                continue
            seat = int(state.current[env])
            from_state = plan_turn(
                cfg, state.racks[env, seat], state.table_sets[env], bool(state.melded[env, seat])
            )
            expected = from_state[0] if from_state else cfg.draw_action
            assert int(from_obs[env]) == expected, f"env {env}: obs-driven plan differs"
            compared += 1

        step(state, from_obs, mask)
        if state.done.all():
            break
    assert compared > 20, f"only {compared} turns compared"


def test_planning_agents_honour_env_ownership():
    """A planning agent asked about envs it does not control must not consume
    those envs' plans -- this broke once already in the tournament harness."""
    cfg = TINY_GROUPS
    state = reset(cfg, 4, seed=5)
    agent = GreedyAgent(cfg)
    agent.reset(4)
    obs, mask = encode(state), legal_actions(state)

    only_first = np.array([True, False, False, False])
    agent.act(obs, mask, only_first)
    assert set(agent._plans) == {0}, "planned for envs it was not given"


def test_observation_helpers_read_the_documented_fields():
    cfg = TINY_GROUPS
    state = reset(cfg, 3, seed=1)
    obs = encode(state)
    assert turn_starting(obs).all(), "a freshly dealt game is at a turn boundary"
    assert not has_melded(obs).any()
    np.testing.assert_array_equal(table(obs), state.table_sets)
    np.testing.assert_array_equal(obs["rack"], state.racks[:, 0])
