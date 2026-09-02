"""Did these checkpoints ever become different policies?

    python tools/policy_divergence.py checkpoints/pool-anchor-s0.pt \
        checkpoints/pool-br-greedy-s0-u040.pt --opponent optimal --deals 30

A score that does not move is two different findings wearing the same face: the
mechanism is worthless, or the policy never left where it started. Every
information null in this repo is trusted because an ablation flipped 0.0% of
argmax decisions, and a *training* arm needs the same evidence in reverse -- that
it moved at all -- before its flat score means anything. The four best-response
arms of the mixed-pool matrix agreed with each other on 99.0% of decisions, which
is what turned that matrix from a null into a vacuum.

Decisions are compared **teacher-forced**: the first checkpoint drives a suite and
every decision's feature row and legal mask is recorded, then every checkpoint's
argmax is taken over those same rows. Letting each arm steer its own games would
measure drift -- two policies diverge after their first different pick and then see
different states -- rather than disagreement.

Off-protocol by construction, like `eval_macro --vs`: it substitutes an opponent
into a frozen suite's deals rather than editing a suite, so no published score is
touched. The number here is an agreement rate, not a score.
"""

from __future__ import annotations

import argparse
import dataclasses
import pathlib
import sys

import numpy as np
import torch

from rummi.agents import REGISTRY
from rummi.agents.learned.features import FEATURE_FIELDS, feature_scale
from rummi.agents.learned.macro_net import MacroNet
from rummi.agents.learned.torch_net import MASKED
from rummi.agents.macro import MacroAgent
from rummi.evaluate.protocol import SUITE_BY_NAME, Suite, evaluate
from rummi.rules.config import CONFIG_BY_NAME, RummiConfig

# The action table this tool must read a checkpoint through is decided in one
# place, and that place is the scorer: a second copy could disagree with it about
# which space a checkpoint names.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from eval_macro import layout


@dataclasses.dataclass(frozen=True, slots=True)
class Policy:
    """One checkpoint's net, with what it has to be driven through."""

    name: str
    net: MacroNet
    cfg: RummiConfig
    repartition: bool


def load(path: pathlib.Path) -> Policy:
    """Rebuild the net a checkpoint holds, refusing what cannot be compared."""
    ck = torch.load(path, weights_only=True)
    space = str(ck.get("space", "macro"))
    if space != "macro":
        raise SystemExit(f"{path}: {space} space; this compares macro-space argmaxes")
    if ck.get("memory", "none") != "none" or ck.get("history", False):
        raise SystemExit(
            f"{path}: carries a cell or a history block, so a decision depends on "
            "the turns before it -- there is no teacher-forced row to score"
        )
    cfg = CONFIG_BY_NAME[ck["cfg"]]
    repartition = bool(ck.get("repartition", False))
    table, action_count = layout(space, repartition)
    net = MacroNet(
        cfg, action_count(cfg), ck["hidden"], head=ck["head"], describe=table(cfg)
    )
    net.load_state_dict(ck["state"])
    net.eval()
    return Policy(path.stem, net, cfg, repartition)


def argmax_over(policy: Policy, x: torch.Tensor, legal: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        logits, _ = policy.net(x)
    return (
        torch.where(legal, logits, torch.full_like(logits, MASKED)).argmax(-1).numpy()
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoints", nargs="+", type=pathlib.Path)
    p.add_argument(
        "--opponent", default="greedy", choices=sorted(REGISTRY),
        help="who the recorded games are played against. It decides which states "
             "the comparison is made on, so a policy pair can agree everywhere one "
             "opponent leads and differ where another does",
    )
    p.add_argument("--deals", type=int, default=30)
    p.add_argument(
        "--suite", default="standard-greedy", choices=sorted(SUITE_BY_NAME),
        help="whose deals and seat rotation to reuse; its own opponent is replaced",
    )
    args = p.parse_args()

    if len(args.checkpoints) < 2:
        raise SystemExit("two or more checkpoints, or there is nothing to compare")
    policies = [load(path) for path in args.checkpoints]
    widths = {(q.cfg.n_players, q.repartition) for q in policies}
    if len(widths) > 1:
        raise SystemExit(f"these do not share an action space: {sorted(widths)}")

    driver = policies[0]
    scale = feature_scale(driver.cfg)
    rows: list[np.ndarray] = []
    masks: list[np.ndarray] = []

    def choose(o: dict, e: int, legal: np.ndarray) -> int:
        row = np.concatenate(
            [np.asarray(o[f])[e].reshape(-1) for f in FEATURE_FIELDS]
        ).astype(np.float32) / scale
        rows.append(row)
        masks.append(legal.copy())
        return int(argmax_over(driver, torch.as_tensor(row)[None],
                               torch.as_tensor(legal)[None])[0])

    base: Suite = SUITE_BY_NAME[args.suite]
    suite = dataclasses.replace(
        base, name=f"{args.suite}-vs-{args.opponent}", opponent=args.opponent
    )
    evaluate(
        driver.name, suite, games=args.deals,
        build_agent=lambda c: MacroAgent(
            c, choose=choose, repartition=driver.repartition
        ),
    )

    x = torch.as_tensor(np.stack(rows))
    legal = torch.as_tensor(np.stack(masks))
    print(
        f"{len(rows):,} decisions recorded against {args.opponent} over "
        f"{args.deals} deals of {args.suite}, driven by {driver.name}\n"
    )
    picks = {q.name: argmax_over(q, x, legal) for q in policies}

    names = list(picks)
    print(f"{'':>22}" + "".join(f"{n[-18:]:>20}" for n in names))
    for a in names:
        cells = "".join(f"{(picks[a] == picks[b]).mean():>19.1%} " for b in names)
        print(f"{a[-20:]:>22}{cells}")
    # The first checkpoint is the anchor by convention, and its column is the one
    # a training arm has to move away from before its score can mean anything.
    print(
        f"\nagainst {driver.name}: "
        + "  ".join(
            f"{n} moves {1 - (picks[n] == picks[driver.name]).mean():.1%}"
            for n in names[1:]
        )
    )


if __name__ == "__main__":
    main()
