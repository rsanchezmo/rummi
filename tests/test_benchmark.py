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
    from rummi.agents.macro import set_templates, template_points

    templates = set_templates(STANDARD)
    assert templates.shape == (329, STANDARD.n_kinds)
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


def test_choosing_which_set_to_play_is_what_the_macro_space_is_for():
    """`first_legal` takes the cheapest set, so before the opening meld it plays
    small and reverts; `by_value` takes the dearest. That one change is worth
    melding 33% of steps against 57% at this probe size (29.6% against 78.5% over
    32 envs and 400 steps) -- so a macro space whose policy could not tell the two
    apart would not be exposing the decision it exists for.

    The bound is deliberately loose: the gap grows with the sample, so pinning the
    measured ratio here would make the test a tripwire for probe size."""
    from rummi.agents.base import act_on_state
    from rummi.agents.macro import MacroAgent, by_value, first_legal
    from rummi.env.observation import encode

    melded = {}
    for label, choose in (("first_legal", first_legal), ("by_value", by_value(STANDARD))):
        agent = MacroAgent(STANDARD, choose=choose)
        agent.reset(16)
        state = reset(STANDARD, 16, seed=5)
        seen = 0.0
        for _ in range(200):
            mask = legal_actions(state)
            seen += float(encode(state)["melded"][:, 0].mean()) / 200
            step(state, act_on_state(agent, state, mask), mask)
        melded[label] = seen
    assert melded["by_value"] > 1.4 * melded["first_legal"], melded


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
