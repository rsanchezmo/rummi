"""Score a saved macro- or hybrid-space checkpoint across the frozen protocol.

    python tools/eval_macro.py checkpoints/macro-pointer-s0.pt

`train_macro.py` saves weights (`--out`) and evaluates once, on one suite; this is
the loader side it lacks, for placing a finished run on every suite its config can
reach. The space comes from the checkpoint, never from a flag: it decides both the
head's width and which agent the weights drive. Suites are scored only when their
config's observation features, feature scale and action table all equal the
checkpoint's -- `tiny`'s differ, so a standard-trained net skips it with the reason
printed. The table must match *exactly*, not just in size: the action-space layout
has changed before (730 -> 713), and a checkpoint from an older layout indexes
different actions with the same ids, so refusing it is the point rather than a
limitation.

**Entropy is printed beside every score**, because the score is an argmax and the
two are not independent: above H ~1 the policy is near-uniform over ~20 legal
actions, its mode is close to arbitrary, and the number says how concentrated the
policy is rather than how well it plays.

Nothing here writes `docs/data/`: a training run is one seed of one recipe, and the
capture rule (`capture_experiments.py`) is that only reproducible agents are
captured.

Run as `python tools/eval_macro.py` from the repo root -- `MacroNet` is imported
from the sibling script, which relies on `tools/` being `sys.path[0]`.
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
import torch

from train_macro import MacroNet
from rummi.agents.hybrid import HybridAgent, hybrid_action_features
from rummi.agents.hybrid import n_actions as n_hybrid_actions
from rummi.agents.learned.features import FEATURE_FIELDS, feature_dim, feature_scale
from rummi.agents.learned.torch_net import MASKED
from rummi.agents.macro import MacroAgent, action_features, n_macros
from rummi.evaluate.protocol import SUITES, SUITE_BY_NAME, Suite, evaluate
from rummi.rules.config import CONFIG_BY_NAME, RummiConfig


def layout(space: str, repartition: bool = False):
    """The action table and its width, for whichever space the checkpoint names.

    One place decides both, so a hybrid checkpoint cannot be built with the macro
    table and pass the width check by coincidence. `repartition` widens the macro
    table by its one solver-backed action; the hybrid table has no such row.
    """
    if space == "hybrid":
        return hybrid_action_features, n_hybrid_actions
    return (
        lambda cfg: action_features(cfg, repartition),
        lambda cfg: n_macros(cfg, repartition),
    )


def incompatibility(
    space: str, ck_cfg: RummiConfig, suite_cfg: RummiConfig, repartition: bool = False
) -> str | None:
    """Why the checkpoint cannot be scored on this suite, or None if it can."""
    if feature_dim(suite_cfg) != feature_dim(ck_cfg):
        return f"feature dim {feature_dim(suite_cfg)} != checkpoint's {feature_dim(ck_cfg)}"
    if not np.array_equal(feature_scale(suite_cfg), feature_scale(ck_cfg)):
        return "feature scale differs from the checkpoint's"
    table, width = layout(space, repartition)
    if not np.array_equal(table(suite_cfg), table(ck_cfg)):
        return (
            f"{space} table differs from the checkpoint's "
            f"({width(suite_cfg)} vs {width(ck_cfg)} actions)"
        )
    return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=pathlib.Path)
    p.add_argument("--suites", nargs="+", default=[s.name for s in SUITES], choices=sorted(SUITE_BY_NAME))
    p.add_argument("--games", type=int, default=None, help="deals per suite; the suite's own count if omitted")
    p.add_argument(
        "--repartition", action="store_true",
        help="score against the macro space with the solver-backed REPARTITION macro. "
             "It is one more action, so only a checkpoint trained with it has the width",
    )
    args = p.parse_args()

    ck = torch.load(args.checkpoint, weights_only=True)
    cfg = CONFIG_BY_NAME[ck["cfg"]]
    space = str(ck.get("space", "macro"))
    if args.repartition and space == "hybrid":
        raise SystemExit("--repartition names a macro-space action; this checkpoint is hybrid")
    table, action_count = layout(space, args.repartition)

    # An older-layout checkpoint would load into the wrong rows silently
    # downstream, so the head width is checked against today's layout first.
    key = "action_bias" if ck["head"] == "pointer" else "pi.weight"
    width = int(ck["state"][key].shape[0])
    if width != action_count(cfg):
        raise SystemExit(
            f"{args.checkpoint}: {width} actions against {action_count(cfg)} in the current "
            f"{space} layout for '{ck['cfg']}' -- a checkpoint from a different action "
            "space, which this tool cannot place on the ladder"
        )

    net = MacroNet(
        cfg, action_count(cfg), ck["hidden"], head=ck["head"], describe=table(cfg)
    )
    net.load_state_dict(ck["state"])
    net.eval()
    scale = feature_scale(cfg)
    # Summed over decisions, so the score is read next to how concentrated the
    # policy taking it was. Reset per suite: they are different games.
    entropy = np.zeros(2)

    def choose(o: dict, e: int, legal: np.ndarray) -> int:
        row = np.concatenate(
            [np.asarray(o[f])[e].reshape(-1) for f in FEATURE_FIELDS]
        ).astype(np.float32) / scale
        with torch.no_grad():
            logits, _ = net(torch.as_tensor(row)[None])
        logits = torch.where(
            torch.as_tensor(legal)[None], logits, torch.full_like(logits, MASKED)
        )
        logp = torch.log_softmax(logits[0], -1)
        entropy[0] += float(-(logp.exp() * logp).sum())
        entropy[1] += 1.0
        return int(logits[0].argmax())

    def build(c: RummiConfig):
        if space == "hybrid":
            return HybridAgent(c, choose=choose)
        return MacroAgent(c, choose=choose, repartition=args.repartition)

    label = args.checkpoint.stem
    print(
        f"{label}: config {ck['cfg']}, {space} space, {ck['head']} head, "
        f"hidden {ck['hidden']}, {action_count(cfg)} actions"
    )
    for name in args.suites:
        suite: Suite = SUITE_BY_NAME[name]
        reason = incompatibility(space, cfg, suite.cfg, args.repartition)
        if reason is not None:
            print(f"{name:18s} skipped -- {reason}")
            continue
        entropy[:] = 0.0
        result = evaluate(label, suite, build_agent=build, games=args.games)
        assert not result.disqualified, f"{label} was disqualified on {name}"
        assert result.illegal_attempts == 0, f"{label} proposed a masked-out action on {name}"
        print(
            f"{name:18s} win {result.win_rate:>6.1%}  score {result.mean_score:>+8.2f}  "
            f"stalemate {result.stalemates / max(1, result.games):>6.1%}  "
            f"H {entropy[0] / max(entropy[1], 1):>5.3f}  n={result.games}",
            flush=True,
        )


if __name__ == "__main__":
    main()
