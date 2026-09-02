"""Does the cover head's attention actually route anything?

    python tools/probe_encoder.py --data checkpoints/repartitions-s*.npz \
        --checkpoint runs/encoder/enc-attn-s0.pt --against runs/encoder/enc-mlp-s0.pt

Every information null in this repo was confirmed by an ablation that flips 0.0% of
argmax decisions -- zeroing an LSTM's cell state, zeroing the oracle's rack block --
and that is what makes those nulls trustworthy rather than merely unresolved. The
same probe belongs beside a positive result: an encoder that wins has to be shown
*using* the pathway it added.

`uniform` replaces every attention weight with `1/N`, which keeps the token-wise
pathway, the learned kind identities and the permutation-invariant pooling and
removes only the content-based routing between kinds. `self` removes the mixing
altogether, so the summary token reads its own nine scalars and nothing else -- a
floor, not a control.

Two readings per ablation, because they answer different questions: what fraction
of the head's argmax cover decisions change (does the mechanism act at all), and
what the greedy decode's holdout playable rate becomes (is what it does worth
anything). `--against` scores a second checkpoint through the same two readings,
which is how "two arms decide differently" is separated from "two arms score
differently".

Decisions are compared **teacher-forced**, on the labelled phase-B rows: two
decodes diverge after their first different pick, so counting flips along two
trajectories would measure drift rather than disagreement.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import torch

from rummi.agents.learned.set_encoder import Attention, set_attention
from rummi.agents.learned.two_phase_net import (
    TwoPhaseNet,
    TwoPhaseScorer,
    two_phase_from_checkpoint,
)
from rummi.rules.config import CONFIG_BY_NAME, RummiConfig

# The trainer's own dataset and its own deliverable metric, read from where they
# are defined: a probe that restates either would be measuring a third thing.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from train_two_phase import TwoPhase, playable_rate


def cover_argmax(
    cfg: RummiConfig,
    data: TwoPhase,
    net: TwoPhaseNet,
    rows: np.ndarray,
    monotone: bool,
    chunk: int = 4096,
) -> np.ndarray:
    """The cover head's argmax on the given labelled phase-B rows."""
    out = np.empty(len(rows), dtype=np.int64)
    net.eval()
    with torch.no_grad():
        for begin in range(0, len(rows), chunk):
            piece = rows[begin : begin + chunk]
            state, dynamic, legal, _ = data.b_batch(piece, monotone)
            out[begin : begin + len(piece)] = net.cover(state, dynamic, legal).argmax(-1).numpy()
    return out


def report(
    label: str, flips: float | None, probe: dict[str, float], agree: float | None = None
) -> None:
    print(
        f"{label:22s} {'--' if flips is None else f'{flips:.1%}':>8s} "
        f"{'--' if agree is None else f'{agree:.1%}':>8s} "
        f"{probe['valid']:>7.1%} {probe['playable']:>9.1%} {probe['tiles']:>6.2f} "
        f"{1000 / probe['decodes_per_second']:>7.2f}",
        flush=True,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="standard", choices=sorted(CONFIG_BY_NAME))
    p.add_argument("--data", type=pathlib.Path, nargs="+", required=True)
    p.add_argument("--checkpoint", type=pathlib.Path, required=True)
    p.add_argument("--against", type=pathlib.Path, nargs="*", default=())
    p.add_argument("--states", type=int, default=2000, help="holdout states decoded per arm")
    p.add_argument("--rows", type=int, default=40_000, help="phase-B rows scored per arm")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    cfg = CONFIG_BY_NAME[args.config]
    rng = np.random.default_rng(args.seed)
    checkpoint = torch.load(args.checkpoint, weights_only=True)
    monotone = bool(checkpoint["monotone"])
    net = two_phase_from_checkpoint(cfg, checkpoint)

    data = TwoPhase(cfg, args.config, list(args.data), monotone)
    held = np.flatnonzero(data.holdout)
    decode_states = rng.permutation(held)[: args.states]
    rows = rng.permutation(np.flatnonzero(data.b_holdout))[: args.rows]
    target = data.b_target[rows]
    print(
        f"{args.checkpoint}  encoder={checkpoint.get('cover_encoder')}\n"
        f"{len(decode_states):,} holdout decodes, {len(rows):,} holdout decisions\n"
        f"\n{'arm':22s} {'flips':>8s} {'teacher':>8s} {'valid':>7s} {'playable':>9s} "
        f"{'tiles':>6s} {'ms':>7s}",
        flush=True,
    )

    base_decisions = cover_argmax(cfg, data, net, rows, monotone)
    report(
        "attend",
        None,
        playable_rate(cfg, data, TwoPhaseScorer(net), decode_states, 1, monotone),
        float((base_decisions == target).mean()),
    )

    ablations: tuple[Attention, ...] = ("uniform", "self")
    for mode in ablations:
        if not set_attention(net, mode):
            raise SystemExit(f"{args.checkpoint} holds no attention trunk to ablate")
        decisions = cover_argmax(cfg, data, net, rows, monotone)
        report(
            mode,
            float((decisions != base_decisions).mean()),
            playable_rate(cfg, data, TwoPhaseScorer(net), decode_states, 1, monotone),
            float((decisions == target).mean()),
        )
    set_attention(net, "attend")

    for path in args.against:
        other = torch.load(path, weights_only=True)
        if bool(other["monotone"]) != monotone:
            raise SystemExit(f"{path} was trained in the other order")
        rival = two_phase_from_checkpoint(cfg, other)
        decisions = cover_argmax(cfg, data, rival, rows, monotone)
        report(
            f"vs {pathlib.Path(path).stem}",
            float((decisions != base_decisions).mean()),
            playable_rate(cfg, data, TwoPhaseScorer(rival), decode_states, 1, monotone),
            float((decisions == target).mean()),
        )


if __name__ == "__main__":
    main()
