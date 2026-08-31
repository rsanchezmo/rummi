"""The fixed-opponent env: the learner only ever sees its own turn.

The base env is self-play, so these are the properties that are *new* here --
that opponents run to completion inside one step, that the learner keeps its
seat, and that reward covers the replies.
"""

import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")

from dataclasses import replace

from rummi.agents import build
from rummi.env.fixed_opponent import FixedOpponentEnv
from rummi.rules.config import STANDARD_4P, TINY_GROUPS, RummiConfig

C = TINY_GROUPS
# Four seats need a deck deep enough to deal them all: TINY_GROUPS holds 13 tiles
# and four racks of four would need 16.
FOUR = replace(TINY_GROUPS, n_players=4, n_numbers=6, n_copies=2, max_sets=6)


@pytest.fixture(autouse=True)
def _headless(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")


def sample_legal(mask, rng):
    return np.argmax(np.where(mask, rng.random(mask.shape), -1.0), axis=-1)


class Scripted:
    """An opponent the env cannot have built: it comes in already made.

    Draws every turn, and records which envs it was asked to play -- which is what
    a pool assignment is checked against.
    """

    def __init__(self, cfg: RummiConfig, name: str = "scripted") -> None:
        self.cfg = cfg
        self.name = name
        self.envs: set[int] = set()
        self.calls = 0

    def reset(self, n_envs: int) -> None:
        pass

    def act(self, obs, mask, active=None):
        played = np.ones(mask.shape[0], bool) if active is None else np.asarray(active)
        self.envs |= set(np.flatnonzero(played).tolist())
        self.calls += int(played.sum())
        return np.full(mask.shape[0], self.cfg.draw_action, dtype=np.int64)


class Watcher:
    """Wraps an opponent and checks the observation it is handed is this state's."""

    name = "watcher"

    def __init__(self, inner, state_of) -> None:
        self.inner = inner
        self.state_of = state_of
        self.calls = 0

    def reset(self, n_envs: int) -> None:
        self.inner.reset(n_envs)

    def act(self, obs, mask, active=None):
        from rummi.env.observation import encode

        fresh = encode(self.state_of())
        for key, value in fresh.items():
            np.testing.assert_array_equal(obs[key], value, err_msg=key)
        self.calls += 1
        return self.inner.act(obs, mask, active)


def test_the_opponents_see_the_state_they_are_acting_on():
    """The opponents read the observation the env encoded when the last advance
    left the state, instead of encoding one per seat. A stale one would be silent:
    the mask is built separately, so every action would still come out legal."""
    env = FixedOpponentEnv(num_envs=4, cfg=C, seed=3)
    watcher = Watcher(build("greedy", C), lambda: env.state)
    env._seats = [None if seat is None else watcher for seat in env._seats]
    rng = np.random.default_rng(0)
    try:
        obs, info = env.reset()
        for _ in range(120):
            obs, r, term, trunc, info = env.step(sample_legal(info["action_mask"], rng))
        assert watcher.calls, "the opponent never played, so nothing was checked"
    finally:
        env.close()


@pytest.mark.parametrize("cfg,seat", [(C, 0), (FOUR, 0), (FOUR, 2), (FOUR, 3)])
def test_the_learner_is_always_the_one_on_turn(cfg: RummiConfig, seat: int):
    """The whole point of the wrapper: every observation handed back is a
    position the learner can act in, whatever seat it was dealt."""
    env = FixedOpponentEnv(num_envs=6, cfg=cfg, seed=1, opponent="greedy", learner_seat=seat)
    try:
        obs, info = env.reset()
        assert (info["current_player"] == seat).all()
        rng = np.random.default_rng(0)
        for _ in range(200):
            obs, r, term, trunc, info = env.step(sample_legal(info["action_mask"], rng))
            # A finished env is exempt until the next step re-deals it.
            live = ~(term | trunc)
            assert (info["current_player"][live] == seat).all()
    finally:
        env.close()


def test_reward_is_the_learners_row_of_the_full_matrix():
    env = FixedOpponentEnv(num_envs=6, cfg=FOUR, seed=2, opponent="greedy", learner_seat=1)
    try:
        obs, info = env.reset()
        rng = np.random.default_rng(1)
        seen_nonzero = False
        for _ in range(400):
            obs, r, term, trunc, info = env.step(sample_legal(info["action_mask"], rng))
            np.testing.assert_allclose(r, info["rewards_all"][:, 1])
            seen_nonzero |= bool((r != 0).any())
        assert seen_nonzero, "no episode paid out, so the check proved nothing"
    finally:
        env.close()


def test_a_step_pays_out_the_opponents_replies_too():
    """WIN_LOSS is zero-sum, and a loss can only be credited on a step where an
    opponent moved. If the advance's reward were dropped, this would be all zeros."""
    env = FixedOpponentEnv(num_envs=8, cfg=C, seed=3, opponent="greedy")
    try:
        obs, info = env.reset()
        rng = np.random.default_rng(2)
        losses = 0
        for _ in range(600):
            obs, r, term, trunc, info = env.step(sample_legal(info["action_mask"], rng))
            np.testing.assert_allclose(info["rewards_all"].sum(-1), 0.0, atol=1e-6)
            losses += int((r < 0).sum())
        assert losses, "random play never lost, so opponent reward was never checked"
    finally:
        env.close()


def test_episodes_autoreset_and_the_learner_keeps_its_seat_across_them():
    env = FixedOpponentEnv(num_envs=8, cfg=C, seed=4, opponent="greedy", learner_seat=1)
    try:
        obs, info = env.reset()
        rng = np.random.default_rng(3)
        episodes = 0
        for _ in range(800):
            obs, r, term, trunc, info = env.step(sample_legal(info["action_mask"], rng))
            episodes += int((term | trunc).sum())
        assert episodes >= 4, f"only {episodes} episodes finished"
        assert env.observation_space.contains(obs)
    finally:
        env.close()


def test_a_bundled_opponent_never_proposes_an_illegal_action():
    env = FixedOpponentEnv(num_envs=6, cfg=C, seed=5, opponent="rearrange")
    try:
        obs, info = env.reset()
        rng = np.random.default_rng(4)
        for _ in range(300):
            obs, r, term, trunc, info = env.step(sample_legal(info["action_mask"], rng))
            assert info["opponent_illegal"] == 0
    finally:
        env.close()


def _learner_record(policy, seed: int = 6, steps: int = 600) -> tuple[int, int]:
    env = FixedOpponentEnv(num_envs=8, cfg=C, seed=seed, opponent="greedy")
    try:
        obs, info = env.reset()
        wins = losses = 0
        for _ in range(steps):
            obs, r, term, trunc, info = env.step(policy(env, obs, info))
            wins += int((r > 0).sum())
            losses += int((r < 0).sum())
        return wins, losses
    finally:
        env.close()


def test_a_learner_that_only_draws_does_far_worse_than_one_that_plays():
    """Sanity that the opponent is really playing and that reward can tell the
    difference.

    The playing side is ``greedy``, not random: random micro-actions never
    assemble a legal opening meld, so a random learner is not a stronger player
    than a passing one and would prove nothing here.

    Compared rather than thresholded, because passing does not lose *every* game
    even so: nearly every game on a small deck ends on an exhausted pool and is
    decided on lowest rack, where a player who never melds is behind but not
    eliminated.
    """
    passive_w, passive_l = _learner_record(
        lambda env, obs, info: np.full(env.num_envs, env.cfg.draw_action, dtype=np.int64)
    )
    learner = build("greedy", C)
    learner.reset(8)
    playing_w, playing_l = _learner_record(
        lambda env, obs, info: learner.act(obs, info["action_mask"])
    )
    assert passive_l > 5 * passive_w, f"passing won {passive_w}, lost {passive_l}"
    assert playing_w > passive_w, f"greedy won {playing_w}, passing won {passive_w}"


def _rollout(env, steps: int = 300, seed: int = 0) -> list:
    """Random legal learner actions, and everything the env said back."""
    obs, info = env.reset()
    rng = np.random.default_rng(seed)
    trace = []
    for _ in range(steps):
        obs, r, term, trunc, info = env.step(sample_legal(info["action_mask"], rng))
        env.state.check_invariants()
        trace.append((r.copy(), term.copy(), trunc.copy(), info["turn_count"].copy()))
    return trace


@pytest.mark.parametrize("cfg", [C, FOUR])
def test_an_already_built_agent_can_drive_the_opponent_seats(cfg: RummiConfig):
    """A learned opponent arrives as an instance -- there is no name to build. On
    four seats the same instance plays three of them, which is why it must key its
    memory by env."""
    scripted = Scripted(cfg)
    env = FixedOpponentEnv(num_envs=6, cfg=cfg, seed=11, opponent=scripted)
    try:
        trace = _rollout(env)
        assert scripted.calls, "the scripted opponent was never asked to act"
        assert max(int(t[3].max()) for t in trace) >= 2, "no opponent turn ever completed"
    finally:
        env.close()

    replay = Scripted(cfg)
    again = FixedOpponentEnv(num_envs=6, cfg=cfg, seed=11, opponent=replay)
    try:
        assert replay.calls == 0
        for before, after in zip(trace, _rollout(again), strict=True):
            for left, right in zip(before, after, strict=True):
                np.testing.assert_array_equal(left, right)
        assert replay.calls == scripted.calls
    finally:
        again.close()


def test_a_pool_splits_the_batch_by_env_index():
    """Round-robin, so both opponents are in every batch and which is where is fixed."""
    scripted = Scripted(C, name="odd")
    env = FixedOpponentEnv(num_envs=6, cfg=C, seed=12, opponent=["greedy", scripted])
    try:
        _rollout(env)
        assert scripted.envs == {1, 3, 5}
        assert env.opponent_pool == ("greedy", scripted)
    finally:
        env.close()


def test_a_pool_of_names_still_plays_legally():
    env = FixedOpponentEnv(num_envs=6, cfg=C, seed=13, opponent=["greedy", "rearrange"])
    try:
        obs, info = env.reset()
        rng = np.random.default_rng(0)
        for _ in range(300):
            obs, r, term, trunc, info = env.step(sample_legal(info["action_mask"], rng))
            assert info["opponent_illegal"] == 0
            env.state.check_invariants()
    finally:
        env.close()


def test_a_pool_of_one_is_the_bare_name():
    """The default path has to stay exactly what it was: a run given a one-member
    pool and a run given the name must be the same run."""
    named = FixedOpponentEnv(num_envs=6, cfg=C, seed=14, opponent="greedy")
    pooled = FixedOpponentEnv(num_envs=6, cfg=C, seed=14, opponent=["greedy"])
    try:
        for before, after in zip(_rollout(named), _rollout(pooled), strict=True):
            for left, right in zip(before, after, strict=True):
                np.testing.assert_array_equal(left, right)
    finally:
        named.close()
        pooled.close()


def test_an_empty_pool_is_refused():
    with pytest.raises(ValueError, match="pool is empty"):
        FixedOpponentEnv(num_envs=2, cfg=C, opponent=[])


def test_learner_seat_must_be_a_seat():
    with pytest.raises(ValueError, match="not a seat"):
        FixedOpponentEnv(num_envs=2, cfg=C, learner_seat=2)


def test_it_is_a_gymnasium_vector_env():
    env = FixedOpponentEnv(num_envs=2, cfg=C, seed=7)
    try:
        assert isinstance(env, gym.vector.VectorEnv)
        assert env.metadata["autoreset_mode"] is gym.vector.AutoresetMode.NEXT_STEP
    finally:
        env.close()


def test_the_standard_four_seat_config_runs():
    """STANDARD_4P is a registered id, so the opponent env has to survive it."""
    env = FixedOpponentEnv(num_envs=2, cfg=STANDARD_4P, seed=8, opponent="greedy")
    try:
        obs, info = env.reset()
        rng = np.random.default_rng(5)
        for _ in range(50):
            obs, r, term, trunc, info = env.step(sample_legal(info["action_mask"], rng))
        assert (info["current_player"] == 0).all()
    finally:
        env.close()
