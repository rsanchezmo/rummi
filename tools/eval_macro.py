"""Score a saved macro-space checkpoint across the frozen protocol.

    python tools/eval_macro.py checkpoints/macro-pointer-s0.pt

`train_macro.py` saves weights (`--out`) and evaluates once, on one suite; this is
the loader side it lacks, for placing a finished run on every suite its config can
reach. Suites are scored only when their config's observation features, feature
scale and macro table all equal the checkpoint's -- `tiny`'s differ, so a
standard-trained net skips it with the reason printed. The macro table must match
*exactly*, not just in size: the action-space layout has changed before (730 -> 713),
and a checkpoint from an older layout indexes different actions with the same ids,
so refusing it is the point rather than a limitation.

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
from rummi.agents.learned.features import FEATURE_FIELDS, feature_dim, feature_scale
from rummi.agents.learned.torch_net import MASKED
from rummi.agents.macro import MacroAgent, action_features, n_macros
from rummi.evaluate.protocol import SUITES, SUITE_BY_NAME, Suite, evaluate
from rummi.rules.config import CONFIG_BY_NAME, RummiConfig


def incompatibility(ck_cfg: RummiConfig, suite_cfg: RummiConfig) -> str | None:
    """Why the checkpoint cannot be scored on this suite, or None if it can."""
    if feature_dim(suite_cfg) != feature_dim(ck_cfg):
        return f"feature dim {feature_dim(suite_cfg)} != checkpoint's {feature_dim(ck_cfg)}"
    if not np.array_equal(feature_scale(suite_cfg), feature_scale(ck_cfg)):
        return "feature scale differs from the checkpoint's"
    if not np.array_equal(action_features(suite_cfg), action_features(ck_cfg)):
        return f"macro table differs from the checkpoint's ({n_macros(suite_cfg)} vs {n_macros(ck_cfg)} macros)"
    return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=pathlib.Path)
    p.add_argument("--suites", nargs="+", default=[s.name for s in SUITES], choices=sorted(SUITE_BY_NAME))
    p.add_argument("--games", type=int, default=None, help="deals per suite; the suite's own count if omitted")
    args = p.parse_args()

    ck = torch.load(args.checkpoint, weights_only=True)
    cfg = CONFIG_BY_NAME[ck["cfg"]]

    # A hybrid-space or older-layout checkpoint would load into the wrong rows
    # silently downstream, so the head width is checked against today's layout first.
    key = "action_bias" if ck["head"] == "pointer" else "pi.weight"
    width = int(ck["state"][key].shape[0])
    if width != n_macros(cfg):
        raise SystemExit(
            f"{args.checkpoint}: {width} actions against {n_macros(cfg)} in the current "
            f"macro layout for '{ck['cfg']}' -- a hybrid-space or older-layout checkpoint, "
            "which this tool cannot place on the ladder"
        )

    net = MacroNet(cfg, n_macros(cfg), ck["hidden"], head=ck["head"])
    net.load_state_dict(ck["state"])
    net.eval()
    scale = feature_scale(cfg)

    def choose(o: dict, e: int, legal: np.ndarray) -> int:
        row = np.concatenate(
            [np.asarray(o[f])[e].reshape(-1) for f in FEATURE_FIELDS]
        ).astype(np.float32) / scale
        with torch.no_grad():
            logits, _ = net(torch.as_tensor(row)[None])
        logits = torch.where(
            torch.as_tensor(legal)[None], logits, torch.full_like(logits, MASKED)
        )
        return int(logits[0].argmax())

    label = args.checkpoint.stem
    print(f"{label}: config {ck['cfg']}, {ck['head']} head, hidden {ck['hidden']}")
    for name in args.suites:
        suite: Suite = SUITE_BY_NAME[name]
        reason = incompatibility(cfg, suite.cfg)
        if reason is not None:
            print(f"{name:18s} skipped -- {reason}")
            continue
        result = evaluate(
            label, suite, build_agent=lambda c: MacroAgent(c, choose=choose), games=args.games
        )
        assert not result.disqualified, f"{label} was disqualified on {name}"
        assert result.illegal_attempts == 0, f"{label} proposed a masked-out action on {name}"
        print(
            f"{name:18s} win {result.win_rate:>6.1%}  score {result.mean_score:>+8.2f}  "
            f"stalemate {result.stalemates / max(1, result.games):>6.1%}  n={result.games}",
            flush=True,
        )


if __name__ == "__main__":
    main()
