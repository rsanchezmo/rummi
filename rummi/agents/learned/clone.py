"""The ladder's learned rung: a network over the macro space, calling the solver.

The division of labour is what makes it worth a rung. `optimal` asks CP-SAT to
repartition the whole table every turn; this one carries `REPARTITION` as a single
macro among the templates, lay-offs and steals, so **the net decides *when* a turn
is worth a solve and CP-SAT decides *how***. That lands it at +47.32 and 99.6% on
`standard-greedy`, statistically even with `optimal` head-to-head (48.7%, n=600),
in the gap the ladder had between `rearrange` (85%) and the solver tier.

The weights are a DAgger clone of `by_value` driving this same macro space,
`REPARTITION` included -- an optimal-tier teacher, +48.01 -- and the clone
reproduces it to within noise from the observation alone, which is what keeps
"the observation is sufficient to play optimally" a tested claim.

One file of weights per bundled preset, named for it. Seat count widens the
observation (570, 572, 574 features), so a net trained on one preset cannot be
placed on another's suite, and a config with no weights is refused rather than
reshaped into something that would score meaninglessly.
"""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

import numpy as np

from rummi.agents.base import Observation
from rummi.agents.learned.features import FEATURE_FIELDS, feature_dim, feature_scale
from rummi.agents.macro import Choose, MacroAgent, action_features, n_macros
from rummi.rules.config import CONFIG_BY_NAME, RummiConfig

if TYPE_CHECKING:
    from rummi.agents.learned.macro_net import MacroNet

WEIGHTS = pathlib.Path(__file__).parent / "weights"
"""Bundled state dicts, one per preset in :data:`~rummi.rules.config.CONFIG_BY_NAME`
that has one, saved by ``tools/train_macro.py --out``."""


def preset_name(cfg: RummiConfig) -> str | None:
    """The `CONFIG_BY_NAME` key `cfg` equals, or None if it is not a bundled preset."""
    for name, preset in CONFIG_BY_NAME.items():
        if preset == cfg:
            return name
    return None


def weights_for(cfg: RummiConfig) -> pathlib.Path:
    """The bundled weights for `cfg`, or a ValueError saying what is shipped."""
    name = preset_name(cfg)
    path = WEIGHTS / f"{name}.pt" if name is not None else None
    if path is not None and path.exists():
        return path
    shipped = ", ".join(sorted(p.stem for p in WEIGHTS.glob("*.pt")))
    label = repr(name) if name is not None else f"an unregistered {feature_dim(cfg)}-feature config"
    raise ValueError(
        f"no bundled weights for {label}; the 'learned' rung ships weights for {shipped}. "
        "Train one with tools/train_macro.py --config <name> --repartition --clone by_value"
    )


def load_net(cfg: RummiConfig) -> MacroNet:
    """The bundled net for `cfg`, in eval mode."""
    import torch

    from rummi.agents.learned.macro_net import MacroNet

    path = weights_for(cfg)
    checkpoint = torch.load(path, weights_only=True)
    assert checkpoint["repartition"], f"{path} was trained without the REPARTITION macro"
    net = MacroNet(
        cfg,
        n_macros(cfg, True),
        checkpoint["hidden"],
        head=checkpoint["head"],
        describe=action_features(cfg, True),
    )
    net.load_state_dict(checkpoint["state"])
    net.eval()
    return net


def argmax_chooser(net: MacroNet, scale: np.ndarray) -> Choose:
    """Mask the logits, take the mode. Deterministic: a published score has to be
    reproducible, and the protocol pins every seed precisely so two people get the
    same number."""
    import torch

    from rummi.agents.learned.torch_net import MASKED

    def choose(obs: Observation, env: int, legal: np.ndarray) -> int:
        row = np.concatenate(
            [np.asarray(obs[f])[env].reshape(-1) for f in FEATURE_FIELDS]
        ).astype(np.float32) / scale
        with torch.no_grad():
            logits, _ = net(torch.as_tensor(row)[None])
        logits = torch.where(
            torch.as_tensor(legal)[None], logits, torch.full_like(logits, MASKED)
        )
        return int(logits[0].argmax())

    return choose


class ClonedMacroAgent(MacroAgent):
    """`MacroAgent` with the bundled network choosing, and `REPARTITION` enabled.

    Torch is imported when one is constructed, not when the package is: `import
    rummi.agents` has to work without it, exactly as it does without OR-Tools.
    """

    name = "learned"

    def __init__(self, cfg: RummiConfig) -> None:
        net = load_net(cfg)
        super().__init__(cfg, choose=argmax_chooser(net, feature_scale(cfg)), repartition=True)
        self.net = net
