"""`train_macro.py`'s warm start: what it restores, and what it refuses.

A resume that quietly reconfigured itself would load one run's weights into a net
of another shape and report what followed as a continuation of it. So the
architecture comes from the checkpoint and a flag that disagrees is an error, and a
head of the wrong width is refused outright -- the tensor names are the same across
action spaces, so `load_state_dict` accepts whatever happens to match in size.
"""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("gymnasium")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import train_macro as trainer

from rummi.agents.macro import n_macros
from rummi.rules.config import TINY_GROUPS

CFG = "tiny_groups"
MACROS = n_macros(TINY_GROUPS)
HIDDEN = 32


def build(head: str = "flat"):
    return trainer.MacroNet(TINY_GROUPS, MACROS, HIDDEN, head=head)


def save(path: Path, state, head: str = "flat", space: str = "macro"):
    torch.save(
        {"cfg": CFG, "space": space, "hidden": HIDDEN, "head": head, "state": state}, path
    )
    return path


@pytest.mark.parametrize("head", ["flat", "pointer"])
def test_a_saved_net_comes_back_with_the_same_weights(tmp_path: Path, head: str):
    saved = build(head)
    path = save(tmp_path / "ck.pt", saved.state_dict(), head=head)

    fresh = build(head)
    key = "trunk.0.weight"
    assert not torch.equal(saved.state_dict()[key], fresh.state_dict()[key]), (
        "two fresh nets started identical, so loading proves nothing"
    )

    checkpoint, hidden, restored = trainer.restore(path, CFG, "macro", MACROS, None, None)
    assert (hidden, restored) == (HIDDEN, head)
    fresh.load_state_dict(checkpoint["state"])
    for name, tensor in saved.state_dict().items():
        assert torch.equal(tensor, fresh.state_dict()[name]), name


@pytest.mark.parametrize(
    "override",
    [{"config": "standard"}, {"space": "hybrid"}, {"hidden": 64}, {"head": "pointer"}],
)
def test_a_flag_that_contradicts_the_checkpoint_is_refused(tmp_path: Path, override: dict):
    path = save(tmp_path / "ck.pt", build().state_dict())
    call = {"config": CFG, "space": "macro", "macros": MACROS, "hidden": None, "head": None}
    with pytest.raises(SystemExit, match="contradicts"):
        trainer.restore(path, **(call | override))


def test_a_head_of_the_wrong_width_is_refused(tmp_path: Path):
    """What a checkpoint from another action space looks like: same keys, fewer rows."""
    state = build().state_dict()
    state["pi.weight"] = state["pi.weight"][:-2]
    state["pi.bias"] = state["pi.bias"][:-2]
    path = save(tmp_path / "ck.pt", state)
    with pytest.raises(SystemExit, match="actions against"):
        trainer.restore(path, CFG, "macro", MACROS, None, None)
