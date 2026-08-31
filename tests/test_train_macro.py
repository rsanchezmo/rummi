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

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("gymnasium")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import train_macro as trainer

from rummi.agents.hybrid import hybrid_action_features, macro_to_hybrid_actions
from rummi.agents.hybrid import n_actions as n_hybrid_actions
from rummi.agents.learned.features import feature_dim
from rummi.agents.learned.history import history_dim
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
    [
        {"config": "standard"},
        {"space": "hybrid"},
        {"hidden": 64},
        {"head": "pointer"},
        {"memory": "lstm"},
        {"oracle": True},
        {"repartition": True},
    ],
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


def test_the_repartition_macro_is_its_own_block():
    from rummi.agents.macro import repartition_offset

    blocks = trainer.action_blocks(TINY_GROUPS, hybrid=False, repartition=True)
    assert blocks.shape == (n_macros(TINY_GROUPS, True),)
    assert blocks[repartition_offset(TINY_GROUPS)] == trainer.BLOCKS.index("repart")
    assert blocks[-2] == trainer.BLOCKS.index("end")
    assert blocks[-1] == trainer.BLOCKS.index("draw")
    # Without the flag nothing lands in that block: the space ends before it.
    plain = trainer.action_blocks(TINY_GROUPS, hybrid=False)
    assert not (plain == trainer.BLOCKS.index("repart")).any()


def test_the_history_block_widens_the_input_and_nothing_else(tmp_path: Path):
    """At `extra=0` the net is the one it always was, which is what makes a
    features-off arm a control rather than a second treatment."""
    plain = build()
    off = trainer.MacroNet(TINY_GROUPS, MACROS, HIDDEN, extra=0)
    assert {k: v.shape for k, v in plain.state_dict().items()} == {
        k: v.shape for k, v in off.state_dict().items()
    }
    on = trainer.MacroNet(TINY_GROUPS, MACROS, HIDDEN, extra=history_dim(TINY_GROUPS))
    assert on.trunk[0].in_features - plain.trunk[0].in_features == history_dim(TINY_GROUPS)
    assert on.trunk[2].in_features == plain.trunk[2].in_features


def _act_with_memory(net, n_envs: int, decisions: int, terminal_rate: float):
    """Drive the cell exactly as the trainer's `choose` and done-handler do.

    One decision at a time on a randomly chosen env, storing each decision's
    pre-state, and zeroing the live state after a terminal -- the reference the
    replay has to reproduce.
    """
    rng = np.random.default_rng(3)
    dim = net.trunk[0].in_features
    mem = (torch.zeros(n_envs, net.memory_dim), torch.zeros(n_envs, net.memory_dim))
    rows = {"x": [], "pre_h": [], "pre_c": [], "terminal": [], "logits": []}
    sequences: dict[int, list[int]] = {}
    for i in range(decisions):
        env = int(rng.integers(n_envs))
        x = torch.as_tensor(rng.normal(size=dim).astype(np.float32))
        rows["pre_h"].append(mem[0][env].clone())
        rows["pre_c"].append(mem[1][env].clone())
        with torch.no_grad():
            logits, _, after = net(x[None], (mem[0][env : env + 1], mem[1][env : env + 1]))
        mem[0][env], mem[1][env] = after[0][0], after[1][0]
        terminal = float(rng.random() < terminal_rate)
        if terminal:
            mem[0][env] = 0.0
            mem[1][env] = 0.0
        rows["x"].append(x)
        rows["terminal"].append(terminal)
        rows["logits"].append(logits[0])
        sequences.setdefault(env, []).append(i)
    stacked = {k: torch.stack(v) if k != "terminal" else torch.as_tensor(v) for k, v in rows.items()}
    return stacked, list(sequences.values())


def test_the_replay_recomputes_exactly_what_acting_computed():
    """The update replays each env's decisions from their stored initial state, so
    a misalignment -- an off-by-one, a missed terminal reset, a crossed env -- would
    silently train on logits the policy never produced. Terminal resets included:
    a sequence spanning a re-deal must not carry one episode into the next."""
    torch.manual_seed(0)
    net = trainer.MacroNet(TINY_GROUPS, MACROS, HIDDEN, memory="lstm", memory_dim=8)
    rows, sequences = _act_with_memory(net, n_envs=3, decisions=40, terminal_rate=0.2)

    rep, _, _ = trainer.replay_features(
        net, rows["x"], rows["pre_h"], rows["pre_c"], rows["terminal"], sequences
    )
    logits, _ = net.heads(rep)
    assert torch.allclose(logits, rows["logits"], atol=1e-5), (
        "the replayed logits are not the ones the policy acted on"
    )


def test_the_gradient_reaches_a_write_through_the_decisions_that_read_it():
    """What full BPTT buys over replaying each step from its stored state: the loss
    at a late decision must move the *early* decision's input path, or the cell can
    never learn to store something that is useless now and useful later."""
    torch.manual_seed(0)
    net = trainer.MacroNet(TINY_GROUPS, MACROS, HIDDEN, memory="lstm", memory_dim=8)
    rows, sequences = _act_with_memory(net, n_envs=1, decisions=6, terminal_rate=0.0)

    x = rows["x"].clone().requires_grad_(True)
    rep, _, _ = trainer.replay_features(
        net, x, rows["pre_h"], rows["pre_c"], rows["terminal"], sequences
    )
    logits, _ = net.heads(rep)
    logits[-1].sum().backward()
    assert x.grad is not None
    assert float(x.grad[0].abs().sum()) > 0, (
        "the last decision's loss never reached the first decision's input"
    )


@pytest.mark.parametrize("head", ["flat", "pointer"])
def test_a_memoryless_checkpoint_seeds_a_memory_net_exactly(head: str):
    """The claim --init-memory-from rests on: with the heads' memory columns at
    zero, the warm-started net's forward equals the source's bitwise, whatever the
    fresh cell emits."""
    torch.manual_seed(0)
    source = build(head)
    with torch.no_grad():
        for parameter in source.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.1)

    target = trainer.MacroNet(
        TINY_GROUPS, MACROS, HIDDEN, head=head, memory="lstm", memory_dim=8
    )
    trainer.transfer_to_memory(target, source.state_dict())

    x = torch.randn(6, feature_dim(TINY_GROUPS))
    state = (torch.randn(6, 8), torch.randn(6, 8))
    with torch.no_grad():
        theirs, their_value = source(x)
        mine, my_value, _ = target(x, state)
    assert torch.allclose(mine, theirs, atol=1e-6), "a memory column moved the logits"
    assert torch.allclose(my_value, their_value, atol=1e-6), "the critic moved"


def test_memory_is_architecture_the_checkpoint_owns(tmp_path: Path):
    """The cell is tensors that exist or do not, so resuming with the flag flipped
    would load one run's weights into a net of another shape."""
    path = save(tmp_path / "ck.pt", build().state_dict())
    with pytest.raises(SystemExit, match="contradicts"):
        trainer.restore(path, CFG, "macro", MACROS, None, None, memory="lstm", memory_dim=8)


def test_history_is_architecture_the_checkpoint_owns(tmp_path: Path):
    """The block widens the trunk's input, so resuming with the flag flipped would
    load one run's weights into a net of another shape."""
    path = save(tmp_path / "ck.pt", build().state_dict())
    with pytest.raises(SystemExit, match="contradicts"):
        trainer.restore(path, CFG, "macro", MACROS, None, None, history=True)


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
