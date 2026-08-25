"""Gymnasium contract compliance and the self-play conventions layered on it."""

import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")

from rummi.rules.config import TINY_GROUPS
from rummi.rules.encoding import tables
from rummi.env.vector_env import RummiVectorEnv
from rummi.render.driver import RenderMode

C = TINY_GROUPS


@pytest.fixture(autouse=True)
def _headless(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")


def sample_legal(mask, rng):
    return np.argmax(np.where(mask, rng.random(mask.shape), -1.0), axis=-1)


@pytest.fixture
def env():
    e = RummiVectorEnv(num_envs=4, cfg=C, seed=1)
    yield e
    e.close()


def test_spaces_are_batched_consistently(env):
    assert env.num_envs == 4
    assert env.single_action_space.n == C.n_actions
    assert env.action_space.shape == (4,)
    obs, _ = env.reset()
    assert env.observation_space.contains(obs)
    for key, value in obs.items():
        assert value.shape[0] == 4
        assert env.single_observation_space[key].contains(value[0]), key


def test_step_returns_the_five_tuple_with_gymnasium_shapes(env):
    obs, info = env.reset()
    rng = np.random.default_rng(0)
    obs, rewards, terminated, truncated, info = env.step(sample_legal(info["action_mask"], rng))
    assert env.observation_space.contains(obs)
    assert rewards.shape == (4,) and rewards.dtype == np.float32
    assert terminated.shape == (4,) and terminated.dtype == bool
    assert truncated.shape == (4,) and truncated.dtype == bool
    assert info["rewards_all"].shape == (4, C.n_players)
    assert info["action_mask"].shape == (4, C.n_actions)
    assert info["current_player"].shape == (4,)


def test_reward_is_the_acting_seats_row_of_the_full_matrix(env):
    _, info = env.reset()
    rng = np.random.default_rng(3)
    for _ in range(60):
        before = info["current_player"].copy()
        obs, rewards, term, trunc, info = env.step(sample_legal(info["action_mask"], rng))
        # Only a committed turn advances the seat; on those steps the credited
        # reward must be the row of the seat that actually acted.
        moved = info["current_player"] != before
        if moved.any():
            idx = np.flatnonzero(moved)
            np.testing.assert_allclose(
                rewards[idx], info["rewards_all"][idx, before[idx]]
            )
            return
    pytest.fail("no turn was committed in 60 steps")


def test_observation_shows_the_acting_seats_rack(env):
    obs, info = env.reset()
    rng = np.random.default_rng(5)
    for _ in range(40):
        for e in range(env.num_envs):
            seat = int(info["current_player"][e])
            np.testing.assert_array_equal(obs["rack"][e], env.state.racks[e, seat])
            # Seat-relative rotation: index 0 is always the acting seat.
            assert obs["rack_sizes"][e, 0] == env.state.racks[e, seat].sum()
        obs, _, _, _, info = env.step(sample_legal(info["action_mask"], rng))


def test_unseen_accounts_for_every_tile_the_actor_cannot_locate(env):
    obs, info = env.reset()
    total = tables(C).copies
    for e in range(env.num_envs):
        located = obs["rack"][e] + env.state.table_counts()[e] + obs["workbench"][e]
        np.testing.assert_array_equal(obs["unseen"][e], total - located)


def test_next_step_autoreset_follows_gymnasium_1x(env):
    """The terminating step returns the final observation; the *next* step returns
    a fresh episode with zero reward and no flags, ignoring the action given."""
    _, info = env.reset()
    rng = np.random.default_rng(7)
    for _ in range(4000):
        obs, rewards, term, trunc, info = env.step(sample_legal(info["action_mask"], rng))
        done = term | trunc
        if done.any():
            which = np.flatnonzero(done)
            finished_turns = info["turn_count"][which].copy()
            obs2, rewards2, term2, trunc2, info2 = env.step(
                sample_legal(info["action_mask"], rng)
            )
            assert not term2[which].any() and not trunc2[which].any()
            np.testing.assert_allclose(rewards2[which], 0.0)
            assert (info2["turn_count"][which] == 0).all(), "episode should have restarted"
            assert (finished_turns > 0).all()
            return
    pytest.fail("no episode terminated")


def test_reset_is_reproducible_and_seedable():
    a = RummiVectorEnv(num_envs=3, cfg=C, seed=42)
    b = RummiVectorEnv(num_envs=3, cfg=C, seed=42)
    obs_a, _ = a.reset()
    obs_b, _ = b.reset()
    np.testing.assert_array_equal(obs_a["rack"], obs_b["rack"])

    obs_c, _ = a.reset(seed=43)
    assert not np.array_equal(obs_a["rack"], obs_c["rack"])
    a.close(), b.close()


def test_illegal_actions_are_rejected_when_validation_is_on(env):
    _, info = env.reset()
    illegal = np.full(env.num_envs, C.end_turn_action)
    assert not info["action_mask"][:, C.end_turn_action].any()
    with pytest.raises(ValueError, match="illegal action"):
        env.step(illegal)


def test_render_modes_produce_what_they_promise():
    """rgb_array returns a *tuple* of frames, per the Gymnasium vector
    convention -- a bare array breaks wrappers.vector.RecordVideo."""
    rgb = RummiVectorEnv(num_envs=2, cfg=C, seed=0, render_mode=RenderMode.RGB_ARRAY)
    rgb.reset()
    frames = rgb.render()
    assert isinstance(frames, tuple) and len(frames) == 1
    frame = frames[0]
    assert frame.ndim == 3 and frame.shape[2] == 3 and frame.dtype == np.uint8
    rgb.close()

    off = RummiVectorEnv(num_envs=2, cfg=C, seed=0, render_mode=RenderMode.NONE)
    off.reset()
    assert off.render() is None
    off.close()


def test_a_full_random_rollout_stays_inside_the_spaces():
    env = RummiVectorEnv(num_envs=6, cfg=C, seed=9)
    obs, info = env.reset()
    rng = np.random.default_rng(1)
    episodes = 0
    for _ in range(3000):
        actions = sample_legal(info["action_mask"], rng)
        assert env.action_space.contains(actions)
        obs, rewards, term, trunc, info = env.step(actions)
        assert np.isfinite(rewards).all()
        episodes += int((term | trunc).sum())
        if episodes >= 8:
            break
    assert episodes >= 8
    assert env.observation_space.contains(obs)
    env.close()


def test_gymnasium_record_video_can_drive_the_env():
    """Locks in the tuple contract: Gymnasium's vector recorder asserts
    ``len(frames) == num_envs``, so a bare array here would break it."""
    pytest.importorskip("moviepy")
    import tempfile

    from gymnasium.wrappers.vector import RecordVideo

    with tempfile.TemporaryDirectory() as parent:
        # A folder that does not exist yet: Gymnasium warns when asked to write
        # into an existing one, and warnings are errors here.
        folder = __import__("os").path.join(parent, "videos")
        env = RecordVideo(
            RummiVectorEnv(num_envs=1, cfg=C, seed=0, render_mode="rgb_array"),
            video_folder=folder,
            name_prefix="rummi",
            video_length=20,
        )
        obs, info = env.reset()
        rng = np.random.default_rng(0)
        for _ in range(20):
            obs, r, term, trunc, info = env.step(sample_legal(info["action_mask"], rng))
        env.close()
        written = [f for f in __import__("os").listdir(folder) if f.endswith(".mp4")]
        assert written, "no video was written"


@pytest.mark.parametrize("env_id,seats", [("Rummi-2p-v0", 2), ("Rummi-3p-v0", 3), ("Rummi-4p-v0", 4)])
def test_registered_ids_build_through_make_vec(env_id: str, seats: int):
    import rummi.env  # noqa: F401  -- importing the package is what registers

    env = gym.make_vec(env_id, num_envs=2)
    try:
        assert env.cfg.n_players == seats
        obs, info = env.reset(seed=0)
        assert info["rewards_all"].shape == (2, seats)
        rng = np.random.default_rng(0)
        obs, rewards, term, trunc, info = env.step(sample_legal(info["action_mask"], rng))
        assert rewards.shape == (2,)
        assert env.observation_space.contains(obs)
    finally:
        env.close()


def test_make_rejects_the_ids_because_there_is_no_single_env():
    """Registering only a vector entry point is deliberate: a batch of one is
    not a single-agent env, and silently wrapping one would hide that."""
    import rummi.env  # noqa: F401

    with pytest.raises(gym.error.Error):
        gym.make("Rummi-2p-v0")
