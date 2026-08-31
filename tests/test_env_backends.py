"""The Gymnasium env over each backend.

Conformance of the *simulator* lives in `test_backends.py`; what is new here is
that the env's own bookkeeping -- next-step autoreset, the acting-seat reward row,
telemetry -- comes out the same whichever backend is underneath, and that the
observation is not quietly copied to the host on the way.
"""

import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")

from rummi.env.api import available
from rummi.env.vector_env import RummiVectorEnv
from rummi.rules.config import TINY_GROUPS

C = TINY_GROUPS
BACKENDS = available()
OTHERS = [b for b in BACKENDS if b != "numpy"]


@pytest.fixture(autouse=True)
def _headless(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")


def drive(backend: str, steps: int = 240, seed: int = 3):
    """Record what an env produces, choosing actions from the mask on the host so
    every backend is handed the identical sequence."""
    env = RummiVectorEnv(num_envs=4, cfg=C, seed=seed, backend=backend)
    try:
        obs, info = env.reset()
        rewards, flags, seats = [], [], []
        for _ in range(steps):
            actions = env.backend.to_numpy(info["action_mask"]).argmax(-1)
            obs, r, term, trunc, info = env.step(actions)
            rewards.append(np.asarray(r).copy())
            flags.append((np.asarray(term).copy(), np.asarray(trunc).copy()))
            seats.append(info["current_player"].copy())
        final = {k: env.backend.to_numpy(v).copy() for k, v in obs.items()}
        return rewards, flags, seats, final
    finally:
        env.close()


def test_the_compile_suffix_names_a_backend():
    """`+compile` is part of the name, so a caller asks for it the way it asks for a
    device. It stays out of `available()`: compiling costs seconds a graph, and that
    list is what the sweeps here run by default."""
    from rummi.env.api import get_backend

    pytest.importorskip("torch")
    assert not get_backend("torch").compiled
    assert get_backend("torch+compile").compiled
    assert get_backend("torch-mps+compile").device == "mps"
    assert get_backend("torch-mps+compile").name == "torch-mps+compile"
    assert not any(name.endswith("+compile") for name in available())
    with pytest.raises(ValueError, match="suffix"):
        get_backend("torch+fused")


@pytest.mark.parametrize("backend", OTHERS)
def test_the_env_behaves_identically_on_every_backend(backend: str):
    want = drive("numpy")
    got = drive(backend)
    for i, (a, b) in enumerate(zip(want[0], got[0], strict=True)):
        np.testing.assert_allclose(b, a, atol=1e-6, err_msg=f"{backend}: reward at step {i}")
    for i, ((ta, ua), (tb, ub)) in enumerate(zip(want[1], got[1], strict=True)):
        np.testing.assert_array_equal(tb, ta, err_msg=f"{backend}: terminated at step {i}")
        np.testing.assert_array_equal(ub, ua, err_msg=f"{backend}: truncated at step {i}")
    for i, (a, b) in enumerate(zip(want[2], got[2], strict=True)):
        np.testing.assert_array_equal(b, a, err_msg=f"{backend}: current_player at step {i}")
    for name in want[3]:
        np.testing.assert_array_equal(
            got[3][name], want[3][name], err_msg=f"{backend}: final {name}"
        )


@pytest.mark.parametrize("backend", OTHERS)
def test_the_observation_stays_in_the_backends_own_array_type(backend: str):
    """The whole point of a device backend: copying the observation to the host
    every step would hand back the speedup it was chosen for."""
    env = RummiVectorEnv(num_envs=2, cfg=C, seed=0, backend=backend)
    try:
        obs, info = env.reset()
        assert not isinstance(obs["rack"], np.ndarray), f"{backend} returned NumPy"
        assert not isinstance(info["action_mask"], np.ndarray)
        # Telemetry is host-side on purpose, so a caller can read it directly.
        assert isinstance(info["current_player"], np.ndarray)
        assert isinstance(info["turn_count"], np.ndarray)
    finally:
        env.close()


@pytest.mark.parametrize("backend", OTHERS)
def test_rendering_a_non_numpy_backend_is_refused(backend: str):
    """The renderer reads a NumPy `BatchState`; failing at construction beats
    failing on the first frame."""
    with pytest.raises(ValueError, match="cannot draw"):
        RummiVectorEnv(num_envs=2, cfg=C, backend=backend, render_mode="rgb_array")


@pytest.mark.parametrize("backend", OTHERS)
def test_the_fixed_opponent_env_is_numpy_only(backend: str):
    from rummi.env.fixed_opponent import FixedOpponentEnv

    with pytest.raises(ValueError, match="cannot drive them"):
        FixedOpponentEnv(num_envs=2, cfg=C, backend=backend)


@pytest.mark.skipif("jax" not in available(), reason="needs the jax extra")
def test_the_jax_env_converts_through_gymnasiums_wrapper():
    """The sanctioned boundary: `JaxToTorch` goes through `from_dlpack`, so on one
    device this is a view rather than a copy."""
    pytest.importorskip("torch")
    pytest.importorskip("array_api_compat")
    import torch
    from gymnasium.wrappers.vector import JaxToTorch

    env = JaxToTorch(RummiVectorEnv(num_envs=2, cfg=C, seed=0, backend="jax"))
    try:
        obs, info = env.reset()
        assert isinstance(obs["rack"], torch.Tensor)
        actions = torch.as_tensor(np.asarray(info["action_mask"].cpu()).argmax(-1))
        obs, r, term, trunc, info = env.step(actions)
        assert isinstance(obs["rack"], torch.Tensor)
        assert isinstance(r, torch.Tensor)
    finally:
        env.close()
