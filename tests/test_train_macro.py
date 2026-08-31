"""`train_macro.py`'s warm starts: what they restore, and what they refuse.

A resume that quietly reconfigured itself would load one run's weights into a net
of another shape and report what followed as a continuation of it. So the
architecture comes from the checkpoint and a flag that disagrees is an error, and a
head of the wrong width is refused outright -- the tensor names are the same across
action spaces, so `load_state_dict` accepts whatever happens to match in size.

Crossing the two spaces is therefore a path of its own rather than a relaxation of
that check, and what it claims is exactness: the hybrid net it produces scores every
macro-space action exactly as the checkpoint did.
"""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("gymnasium")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import train_macro as trainer

from rummi.agents.hybrid import hybrid_action_features, macro_to_hybrid_actions
from rummi.agents.hybrid import n_actions as n_hybrid_actions
from rummi.agents.learned.features import feature_dim
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


@pytest.mark.parametrize("hybrid", [False, True])
def test_every_action_is_counted_in_exactly_one_block(hybrid: bool):
    """The telemetry's `end` and `draw` must be the ids the trainer acts on.

    In the hybrid space those are primitives at their own ids rather than the last
    two actions, so a block map restated from the macro layout would report
    ending the turn as a primitive and never move.
    """
    blocks = trainer.action_blocks(TINY_GROUPS, hybrid)
    width = n_hybrid_actions(TINY_GROUPS) if hybrid else n_macros(TINY_GROUPS)
    assert blocks.shape == (width,)
    assert blocks[TINY_GROUPS.end_turn_action if hybrid else width - 2] == trainer.BLOCKS.index("end")
    assert blocks[TINY_GROUPS.draw_action if hybrid else width - 1] == trainer.BLOCKS.index("draw")
    primitives = int((blocks == trainer.BLOCKS.index("prim")).sum())
    # Both committing primitives have their own block, and the macro space has none.
    assert primitives == (TINY_GROUPS.n_actions - 2 if hybrid else 0)


def test_a_head_of_the_wrong_width_is_refused(tmp_path: Path):
    """What a checkpoint from an older layout looks like: same keys, fewer rows."""
    state = build().state_dict()
    state["pi.weight"] = state["pi.weight"][:-2]
    state["pi.bias"] = state["pi.bias"][:-2]
    path = save(tmp_path / "ck.pt", state)
    with pytest.raises(SystemExit, match="actions against"):
        trainer.restore(path, CFG, "macro", MACROS, None, None)


@pytest.mark.parametrize("head", ["flat", "pointer"])
def test_a_macro_checkpoint_scores_its_macros_the_same_once_transferred(head: str):
    """The claim `--init-from-macro` rests on, for both heads.

    Exact, not approximate: the hybrid action table embeds the macro one column for
    column, so every shared action's logit is the same arithmetic on the same
    weights. What is *not* preserved is the softmax -- the primitives now share the
    denominator -- so the probabilities are asserted after renormalising over the
    shared block, which is what "the ranking survives" means here.
    """
    torch.manual_seed(0)
    source = build(head)
    with torch.no_grad():
        # A fresh net's action head is zeros (pointer) or gain 0.01 (flat), and
        # either would let a transfer that moved nothing pass.
        for parameter in source.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.1)

    target = trainer.MacroNet(
        TINY_GROUPS, n_hybrid_actions(TINY_GROUPS), HIDDEN, head=head,
        describe=hybrid_action_features(TINY_GROUPS),
    )
    trainer.transfer_from_macro(target, source.state_dict(), TINY_GROUPS)

    x = torch.randn(8, feature_dim(TINY_GROUPS))
    with torch.no_grad():
        theirs, their_value = source(x)
        mine, my_value = target(x)
    shared = mine[:, macro_to_hybrid_actions(TINY_GROUPS)]

    assert torch.allclose(shared, theirs, atol=1e-6), "a macro is scored differently"
    assert torch.allclose(my_value, their_value, atol=1e-6), "the critic moved"
    assert torch.allclose(
        torch.softmax(shared, -1), torch.softmax(theirs, -1), atol=1e-6
    ), "the ranking over the macros did not survive"
    assert float(torch.softmax(mine, -1)[:, macro_to_hybrid_actions(TINY_GROUPS)].sum(-1).max()) < 1.0, (
        "the primitives took no probability mass, so they are unreachable"
    )
