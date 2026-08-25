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
