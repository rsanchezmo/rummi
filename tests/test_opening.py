"""The opening arms: what makes them one change rather than two.

`rummi/agents/opening.py` claims something narrow -- `frugal` with a different opening
turn -- and the experiment in `tools/opening_ab.py` is only about the opening if that
claim holds exactly. Four things carry it.

*Nothing after the meld moves.* Every arm delegates to `by_value` once the seat has
opened, so on post-meld states the two must choose the identical macro. That is the
whole attribution.

*The planner is `by_value` where it is meant to be.* The `full` arm rebuilds
`by_value`'s own opening through this file's own simulation of it, so it must play
identical games -- which is what makes the other arms' departures readable as their
own rule rather than as a reimplementation drifting.

*No arm gives up an opening.* An arm that could not build its opening its own way
plays `frugal`'s, because an opening delayed by a turn is a far larger effect than any
reshaping of one, and the two would be inseparable in a score.

*A joker is worth what the env says.* A template's own points are the wrong value for
any set the hand could only cover with a joker, and the meld threshold is decided on
the env's number.
"""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from denial_ab import play_deals

from rummi.agents.frugal_agent import FrugalAgent
from rummi.agents.macro import by_value, laid_tiles, n_macros, set_templates
from rummi.agents.opening import (
    ARMS,
    OpeningAgent,
    OpeningChoice,
    set_value,
    template_is_run,
)
from rummi.rules.config import STANDARD, TINY_GROUPS
from rummi.rules.encoding import EMPTY, kind_of
from rummi.env.numpy.deal import reset as deal_reset
from rummi.env.numpy.engine import step as engine_step
from rummi.env.numpy.masks import legal_actions
from rummi.env.numpy.sets import summarize
from rummi.env.observation import encode

C = STANDARD


def _counts(cfg, kinds) -> np.ndarray:
    out = np.zeros(cfg.n_kinds, dtype=np.int64)
    for k in kinds:
        out[k] += 1
    return out


def _template(cfg, kinds) -> int:
    rows = np.flatnonzero((set_templates(cfg) == _counts(cfg, kinds)).all(-1))
    assert rows.size == 1, kinds
    return int(rows[0])


def _obs(cfg, rack, placed=(), rows=(), melded=False):
    """The fields an opening arm reads, and nothing else."""
    table = np.full((cfg.max_sets, cfg.max_set_len), EMPTY, dtype=np.int16)
    for i, row in enumerate(rows):
        table[i, : len(row)] = sorted(row)
    return {
        "rack": _counts(cfg, rack)[None].astype(np.int16),
        "workbench": np.zeros((1, cfg.n_kinds), dtype=np.int16),
        "placed_this_turn": _counts(cfg, placed)[None].astype(np.int16),
        "table_sets": table[None],
        "melded": np.array([[int(melded), 0]], dtype=np.int8),
    }


def _legal(cfg, templates, end=False):
    out = np.zeros(n_macros(cfg, repartition=True), dtype=bool)
    out[list(templates)] = True
    out[-1] = True  # DRAW is never masked
    out[-2] = end
    return out


def _states(cfg, agent, games: int, steps: int, seed: int = 0):
    """Real states, reached by the agent that is being asked about them.

    Random play never reaches a meld on this config, so post-meld states have to be
    played into by something that plays.
    """
    state = deal_reset(cfg, games, seed=seed)
    agent.reset(games)
    for _ in range(steps):
        if state.done.all():
            break
        summary = summarize(cfg, state.table_sets)
        mask = legal_actions(state, summary)
        obs = encode(state, summary)
        yield state, obs, mask
        engine_step(state, agent.act(obs, mask), mask)


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_every_arm_plays_by_value_after_the_opening_meld(arm):
    """The attribution: a difference in score can only have come from the opening.

    Asked on the states the arm itself reaches, because those are the ones its score
    is made of -- and counted, since an assertion that never fires proves nothing.
    """
    pytest.importorskip("ortools")
    cfg = C
    agent = OpeningAgent(cfg, arm)
    base = by_value(cfg)
    decided = 0
    for state, obs, _mask in _states(cfg, agent, games=6, steps=200, seed=5):
        for env in range(state.batch_size):
            if state.done[env] or not bool(obs["melded"][env, 0]):
                continue
            legal = agent.legal_macros(obs, env)
            assert agent.choice(obs, env, legal) == base(obs, env, legal)
            decided += 1
    assert decided > 300, f"only {decided} post-meld decisions exercised"


def test_the_full_arm_plays_frugal_deal_for_deal():
    """`full` is the planner's own correctness check, so it has to be byte-identical.

    Played out rather than compared decision by decision: the arms replay a plan across
    the decisions of a turn, and only a whole game exercises the bookkeeping that
    carries it -- rebuilding at a turn boundary, and the fallback on the turns that
    cannot open at all.
    """
    cfg = TINY_GROUPS
    seeds = [np.random.SeedSequence([4242, i]) for i in range(6)]
    arm = [OpeningAgent(cfg, "full"), OpeningAgent(cfg, "base")]
    plain = [FrugalAgent(cfg), FrugalAgent(cfg)]
    assert all(np.array_equal(a, b) for a, b in zip(play_deals(cfg, arm, seeds),
                                                    play_deals(cfg, plain, seeds), strict=True))


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_no_arm_hands_back_an_opening_that_misses_the_threshold(arm):
    """An arm either opens, or plays `frugal`'s turn -- never a turn that opens later.

    `None` is the second of those, and it is what keeps the measurement about the shape
    of an opening instead of about its timing.
    """
    pytest.importorskip("ortools")
    cfg = C
    agent = OpeningAgent(cfg, arm)
    choice = agent.choice
    assert choice is not None
    built = 0
    for state, obs, _mask in _states(cfg, agent, games=8, steps=200, seed=9):
        for env in range(state.batch_size):
            if state.done[env] or bool(obs["melded"][env, 0]):
                continue
            if np.asarray(obs["placed_this_turn"][env]).any():
                continue
            hand = np.asarray(obs["rack"][env]).astype(np.int64)
            plan = choice._build(obs, env, hand)
            if plan is None:
                continue
            built += 1
            left = hand.copy()
            value = 0
            for t in plan:
                laid = laid_tiles(cfg, choice.templates[t], left)
                value += set_value(cfg, laid)
                left = left - laid
            assert (left >= 0).all(), "the plan laid a tile the rack does not hold"
            assert value >= cfg.initial_meld, f"{arm} planned a {value}-point opening"
    # One opening per seat per game, so this is a handful by construction.
    assert built > 5, f"only {built} plans built for {arm}"


def test_min_sets_stops_the_moment_the_threshold_is_met():
    """The first arm, and the one the oracle's cell points at directly.

    Two 30-point sets in one hand: `by_value` lays both, because pre-meld it ranks by
    points and keeps playing while anything is playable. The decision that separates
    them is the *second* one, which is why this drives two in sequence.
    """
    cfg = C
    run = [kind_of(cfg, 0, n) for n in (11, 12, 13)]
    group = [kind_of(cfg, c, 10) for c in (1, 2, 3)]
    dear, cheap = _template(cfg, run), _template(cfg, group)
    base = by_value(cfg)
    arm = OpeningChoice(cfg, "min_sets")

    first = _obs(cfg, rack=[*run, *group])
    legal = _legal(cfg, (dear, cheap))
    assert base(first, 0, legal) == dear
    assert arm(first, 0, legal) == dear, "the dearest set goes down either way"

    # The run is on the table and worth 36, so ending is legal and the group is not
    # worth playing -- which is the whole of what this arm claims.
    second = _obs(cfg, rack=group, placed=run, rows=[run])
    legal = _legal(cfg, (cheap,), end=True)
    assert base(second, 0, legal) == cheap
    assert arm(second, 0, legal) == len(legal) - 2


def test_minus_one_drops_the_last_set_only_when_the_rest_still_opens():
    """The smallest deviation available, and it has to refuse itself where it cannot pay.

    36 + 30 gives the last set up; 18 + 15 cannot, because 18 alone does not open.
    """
    cfg = C
    run = [kind_of(cfg, 0, n) for n in (11, 12, 13)]
    group = [kind_of(cfg, c, 10) for c in (1, 2, 3)]
    arm = OpeningChoice(cfg, "minus_one")
    obs = _obs(cfg, rack=[*run, *group])
    assert arm._build(obs, 0, np.asarray(obs["rack"][0]).astype(np.int64)) == [
        _template(cfg, run)
    ]

    sixes = [kind_of(cfg, c, 6) for c in (0, 1, 2)]
    fives = [kind_of(cfg, c, 5) for c in (0, 1, 2)]
    obs = _obs(cfg, rack=[*sixes, *fives])
    plan = arm._build(obs, 0, np.asarray(obs["rack"][0]).astype(np.int64))
    assert plan == [_template(cfg, sixes), _template(cfg, fives)]


def test_the_shape_arms_open_with_their_own_shape():
    """The shape test, and it has to overrule points or it is not one.

    A 36-point run against a 33-point group: `by_value` takes the run, so `groups_first`
    reversing that is the arm acting rather than agreeing.
    """
    cfg = C
    run = [kind_of(cfg, 0, n) for n in (9, 10, 11)]
    group = [kind_of(cfg, c, 12) for c in (1, 2, 3)]
    obs = _obs(cfg, rack=[*run, *group])
    hand = np.asarray(obs["rack"][0]).astype(np.int64)
    assert by_value(cfg)(obs, 0, _legal(cfg, (_template(cfg, run), _template(cfg, group)))) == (
        _template(cfg, group)
    ), "the group is the dearer set, so preferring the run overrules the ranking"
    assert OpeningChoice(cfg, "runs_first")._build(obs, 0, hand) == [_template(cfg, run)]
    assert OpeningChoice(cfg, "groups_first")._build(obs, 0, hand) == [_template(cfg, group)]


def test_cheap_opens_with_the_sets_by_value_ranks_last():
    """The inverse ordering, which separates value from tile count.

    15 + 18 opens on six cheap tiles where `by_value` opens on three dear ones, so the
    two differ in what is kept back rather than in how much.
    """
    cfg = C
    run = [kind_of(cfg, 0, n) for n in (11, 12, 13)]
    sixes = [kind_of(cfg, c, 6) for c in (0, 1, 2)]
    fives = [kind_of(cfg, c, 5) for c in (0, 1, 2)]
    obs = _obs(cfg, rack=[*run, *sixes, *fives])
    hand = np.asarray(obs["rack"][0]).astype(np.int64)
    assert OpeningChoice(cfg, "cheap")._build(obs, 0, hand) == [
        _template(cfg, fives),
        _template(cfg, sixes),
    ]


def test_no_joker_keeps_the_joker_where_the_rack_can_open_without_it():
    """The joker is the most flexible tile there is, so spending it opens rigidest.

    A joker covers the gap in red 10-11-_-13 and makes it the dearest set in the hand at
    46, which is what `by_value` opens on; yellow 11-12-13 opens on 36 and keeps it.
    """
    cfg = C
    gapped = [kind_of(cfg, 0, n) for n in (10, 11, 13)]
    whole = [kind_of(cfg, 3, n) for n in (11, 12, 13)]
    obs = _obs(cfg, rack=[*gapped, *whole, cfg.joker_kind])
    hand = np.asarray(obs["rack"][0]).astype(np.int64)

    def joker_tiles(arm: str) -> int:
        plan = OpeningChoice(cfg, arm)._build(obs, 0, hand)
        assert plan is not None
        left, spent = hand.copy(), 0
        for t in plan:
            laid = laid_tiles(cfg, set_templates(cfg)[t].astype(np.int64), left)
            spent += int(laid[cfg.joker_kind])
            left = left - laid
        return spent

    assert joker_tiles("full") == 1
    assert joker_tiles("no_joker") == 0


def test_the_solver_arms_bracket_the_tile_count():
    """`min_tiles` and `max_tiles` are the two ends of the axis under test.

    Both open, from the same hand, and the negative control sheds every tile it can --
    which is what makes it the opening `optimal` would have played.
    """
    pytest.importorskip("ortools")
    cfg = C
    group = [kind_of(cfg, c, 10) for c in (0, 1, 2)]
    run = [kind_of(cfg, 0, n) for n in (1, 2, 3, 4, 5)]
    obs = _obs(cfg, rack=[*group, *run])
    hand = np.asarray(obs["rack"][0]).astype(np.int64)

    def tiles(arm: str) -> int:
        plan = OpeningChoice(cfg, arm)._build(obs, 0, hand)
        assert plan is not None
        return int(sum(set_templates(cfg)[t].sum() for t in plan))

    assert tiles("min_tiles") == 3
    assert tiles("max_tiles") == 8


def test_set_value_reads_a_joker_at_the_value_the_env_credits():
    """Why the plan does not sum `template_points`.

    A joker in a run takes the value of the position it fills, and the env resolves
    that by the best window the slot admits -- so a hand of red 4-5 and a joker laid
    against the 3-4-5 template is credited 15, not the template's 12.
    """
    cfg = C
    template = _template(cfg, [kind_of(cfg, 0, n) for n in (3, 4, 5)])
    hand = _counts(cfg, [kind_of(cfg, 0, 4), kind_of(cfg, 0, 5), cfg.joker_kind])
    laid = laid_tiles(cfg, set_templates(cfg)[template].astype(np.int64), hand)
    assert laid[cfg.joker_kind] == 1
    assert set_value(cfg, laid) == 15


def test_template_is_run_splits_every_template():
    """Read off the kinds rather than off the order `set_templates` builds them in."""
    cfg = C
    is_run = template_is_run(cfg)
    runs = cfg.n_colors * sum(1 for length in range(cfg.min_set, cfg.max_set_len + 1)
                              for _ in range(1, cfg.n_numbers - length + 2))
    assert int(is_run.sum()) == runs
    assert int((~is_run).sum()) == len(set_templates(cfg)) - runs


def test_an_arm_abandons_a_plan_the_mask_refuses():
    """A stale plan is dropped for `by_value`'s pick, the same rule `MacroAgent` applies.

    Forced rather than waited for: the mask and the plan cannot disagree in a real game,
    and a fallback that only runs in an impossible state is a fallback that has never
    run.
    """
    cfg = C
    run = [kind_of(cfg, 0, n) for n in (11, 12, 13)]
    group = [kind_of(cfg, c, 10) for c in (1, 2, 3)]
    arm = OpeningChoice(cfg, "min_sets")
    obs = _obs(cfg, rack=[*run, *group])
    # Everything the plan wants is masked out, so only the group is left to play.
    legal = _legal(cfg, (_template(cfg, group),))
    assert arm(obs, 0, legal) == _template(cfg, group)
    assert arm.stats.stale == 1
