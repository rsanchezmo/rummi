"""A whole-table repartition, constructed one template at a time.

Deciding how the table plus the rack partitions into sets *is* the NP-hard content
of Rummikub, and it is the one thing `rummi/agents/macro.py` cannot say: where
`by_value` can only draw, CP-SAT still plays in 47.7% of the states, and 100% of
those plays dissolve at least one set. Enumerating whole repartitions to learn one
directly walks straight back into that wall. Picking **one set at a time** does
not -- the remaining multiset is a sufficient statistic for what may still be
built, so a repartition is a sequence of ~330 template choices under a mask that
costs three matmuls, and the solver's own answers are the labels.

The one hard constraint is that **every tile now on the table has to be
re-covered**; rack tiles are optional and playing more of them is the objective.
So the running state is two count vectors -- what is left to cover and what is
left to draw on -- and `STOP` is legal exactly where the first is empty.

Templates carry no joker, exactly as in `macro.set_templates`: a joker is a
*substitution*, and which tiles it stands in for is derived the way
`macro.laid_tiles` derives it, from what the remaining multiset cannot supply.
Every template count is 0 or 1 -- a run has distinct numbers, a group distinct
colours -- which is what turns "how many tiles does this template lack" into
`templates @ (remaining == 0)`: one matmul over the whole table where a search
would be the wall again.

A joker already *on the table* is the one place the derivation can dead-end. It
has to be re-covered like any other tile, and it can only be covered by standing
in for a gap, so a sequence whose templates all happen to be fully supplied
strands it and is rejected -- honestly, as an invalid decode, rather than by a
forcing rule that would have to choose which tile to hide behind it.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from itertools import combinations, permutations

import numpy as np
import torch
from torch import nn

from rummi.agents.macro import set_templates, template_points
from rummi.rules.config import RummiConfig
from rummi.rules.encoding import EMPTY, kinds_to_counts, tables

MASKED = -1e8
"""Not `-inf`, for the reason in `learned/torch_net.py`: the entropy term would
compute `0 * -inf` and end a run."""

CANDIDATE_DYNAMIC = 8
"""Per `(state, template)` features: what the template would take off the table,
what it would take out of the rack, what it would cost in jokers, whether it is a
set already sitting there, and how far above the previous pick it sits.

The fourth is not decoration. A solve at a stuck state plays a **mean of 1.6 tiles
across a mean of 12.6 sets**: almost the whole answer is the table as it already
stands, and the solver's own `keep_weight` tie-break says so. Without it the
network would have to re-derive the existing partition from counts alone, which is
the NP-hard problem in full rather than the local change the label is.

Nor is the fifth. The sequence is emitted in template order, so the answer is
overwhelmingly "the nearest feasible set above the last one" -- and a per-template
*bias* cannot say that, because "nearest" is relative to a state scalar. Without
the distance the head has no way to express the scan it is being taught."""

STATE_SCALARS = 9


@dataclass(frozen=True, slots=True)
class TemplateTable:
    """Everything about the ~330 sets that does not depend on the state.

    `by_content` maps a set's *real* tiles plus its length back to the templates
    that could have produced it, which is how a solver solution -- where jokers
    are already materialised and their role is gone -- is read back into this
    action space.
    """

    counts: np.ndarray
    """`(T, K)` int16, 0/1 per kind."""
    contents: tuple[tuple[int, ...], ...]
    length: np.ndarray
    points: np.ndarray
    is_run: np.ndarray
    static: np.ndarray
    """`(T + 1, K + 5)` float32 description of each action, `STOP` last."""
    by_content: dict[tuple[int, tuple[int, ...]], tuple[int, ...]]


@cache
def template_table(cfg: RummiConfig) -> TemplateTable:
    counts = set_templates(cfg).astype(np.int16)
    n_templates, n_kinds = counts.shape
    length = counts.sum(-1).astype(np.int16)
    points = template_points(cfg).astype(np.int32)
    colour = tables(cfg).color

    is_run = np.zeros(n_templates, dtype=bool)
    contents: list[tuple[int, ...]] = []
    for t in range(n_templates):
        kinds = tuple(int(k) for k in np.flatnonzero(counts[t]))
        contents.append(kinds)
        is_run[t] = bool((colour[list(kinds)] == colour[kinds[0]]).all())

    # A solver set holding `j` jokers is some template with `j` of its tiles
    # hidden, so every template registers itself under each of those readings.
    by_content: dict[tuple[int, tuple[int, ...]], list[int]] = {}
    for t, kinds in enumerate(contents):
        for hidden in range(min(cfg.n_jokers, len(kinds)) + 1):
            for gone in combinations(range(len(kinds)), hidden):
                left = tuple(k for i, k in enumerate(kinds) if i not in gone)
                by_content.setdefault((len(kinds), left), []).append(t)

    scale = float(cfg.n_numbers * cfg.max_set_len)
    static = np.zeros((n_templates + 1, n_kinds + 5), dtype=np.float32)
    static[:n_templates, :n_kinds] = counts
    static[:n_templates, n_kinds] = points / scale
    static[:n_templates, n_kinds + 1] = length / cfg.max_set_len
    static[:n_templates, n_kinds + 2] = is_run
    static[:n_templates, n_kinds + 3] = ~is_run
    static[n_templates, n_kinds + 4] = 1.0  # STOP describes itself and nothing else

    return TemplateTable(
        counts=counts,
        contents=tuple(contents),
        length=length,
        points=points,
        is_run=is_run,
        static=static,
        by_content={k: tuple(v) for k, v in by_content.items()},
    )


@cache
def colour_relabellings(cfg: RummiConfig) -> tuple[np.ndarray, np.ndarray]:
    """`(P, K)` and `(P, T)` forward maps, one per relabelling of the colours.

    A colour is a name, not a quantity: swap every red tile for a blue one
    throughout and a legal repartition maps onto another legal one, a state onto a
    state the solver would answer the same way, and the feasibility mask onto its
    own permutation. That is 24 readings of every labelled state on the standard
    config, and it is the only free data this problem has.
    """
    tt = template_table(cfg)
    index = {content: t for t, content in enumerate(tt.contents)}
    kinds, templates = [], []
    for order in permutations(range(cfg.n_colors)):
        kind_map = np.arange(cfg.n_kinds)
        for colour, renamed in enumerate(order):
            lo = colour * cfg.n_numbers
            kind_map[lo : lo + cfg.n_numbers] += (renamed - colour) * cfg.n_numbers
        kinds.append(kind_map)
        templates.append(
            np.array([index[tuple(sorted(kind_map[list(c)].tolist()))] for c in tt.contents])
        )
    return np.stack(kinds), np.stack(templates)


def n_actions(cfg: RummiConfig) -> int:
    """Templates, then `STOP`."""
    return len(set_templates(cfg)) + 1


def stop_action(cfg: RummiConfig) -> int:
    return len(set_templates(cfg))


def state_dim(cfg: RummiConfig) -> int:
    return 3 * cfg.n_kinds + STATE_SCALARS


def static_dim(cfg: RummiConfig) -> int:
    return cfg.n_kinds + 5


# --- the running multiset -------------------------------------------------


def initial_counts(cfg: RummiConfig, rack: np.ndarray, board: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`(need, avail)`: the table's tiles, and everything the turn may use.

    `board` is the `(S, L)` table as kind ids, which is what the observation
    carries and what `to_actions.plan` reads back.
    """
    board = np.asarray(board)
    need = np.zeros(cfg.n_kinds, dtype=np.int64)
    for kind in board[board >= 0].ravel():
        need[int(kind)] += 1
    return need, need + np.asarray(rack).astype(np.int64)


def candidate_stats(
    cfg: RummiConfig, need: np.ndarray, avail: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`(short, from_table, rack_value)`, each `(B, T)`.

    Template counts are 0/1, so "tiles this template lacks" is a dot product with
    the indicator of an exhausted kind rather than a per-kind maximum -- which is
    what keeps the whole ~330-way mask at three matmuls.
    """
    tt = template_table(cfg)
    counts = tt.counts.astype(np.float32)
    value = tables(cfg).value.astype(np.float32)

    absent = (np.asarray(avail) == 0).astype(np.float32)
    on_table = (np.asarray(need) > 0).astype(np.float32)
    spare = ((np.asarray(avail) > 0) & (np.asarray(need) == 0)).astype(np.float32) * value

    return absent @ counts.T, on_table @ counts.T, spare @ counts.T


def present_counts(cfg: RummiConfig, board: np.ndarray) -> np.ndarray:
    """`(T,)` how many of each template the table already holds.

    A slot carrying a joker matches every template it could be read as; the
    over-count is harmless in a feature and the alternative is deciding the
    joker's role before the construction that would determine it.
    """
    tt = template_table(cfg)
    out = np.zeros(len(tt.counts), dtype=np.float32)
    for row in np.asarray(board):
        kinds = sorted(int(k) for k in row if k >= 0)
        if not kinds:
            continue
        real = tuple(k for k in kinds if k != cfg.joker_kind)
        for template in tt.by_content.get((len(kinds), real), ()):
            out[template] += 1.0
    return out


def candidate_features(
    cfg: RummiConfig,
    need: np.ndarray,
    avail: np.ndarray,
    present: np.ndarray,
    last: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """`(dynamic, short)` -- `(B, T + 1, CANDIDATE_DYNAMIC)` and `(B, T)`."""
    tt = template_table(cfg)
    short, from_table, rack_value = candidate_stats(cfg, need, avail)
    n_templates = short.shape[1]
    length = tt.length.astype(np.float32)[None]
    scale = float(cfg.n_numbers * cfg.max_set_len)
    jokers = float(max(cfg.n_jokers, 1))
    ahead = np.arange(n_templates, dtype=np.float32)[None] - np.asarray(last, dtype=np.float32)[:, None]

    out = np.zeros((short.shape[0], n_templates + 1, CANDIDATE_DYNAMIC), dtype=np.float32)
    out[:, :-1, 0] = short / jokers
    out[:, :-1, 1] = from_table / cfg.max_set_len
    out[:, :-1, 2] = (length - short - from_table) / cfg.max_set_len
    out[:, :-1, 3] = from_table / length
    out[:, :-1, 4] = rack_value / scale
    out[:, :-1, 5] = np.minimum(np.asarray(present, dtype=np.float32), 1.0)
    out[:, :-1, 6] = ahead / n_templates
    out[:, :-1, 7] = np.where(ahead >= 0.0, 1.0 / (1.0 + np.maximum(ahead, 0.0)), 0.0)
    return out, short


def feasible(
    cfg: RummiConfig,
    need: np.ndarray,
    avail: np.ndarray,
    n_sets: np.ndarray,
    last: np.ndarray,
    short: np.ndarray,
    monotone: bool = True,
) -> np.ndarray:
    """`(B, T + 1)` -- which templates the remaining multiset can still lay.

    A gap is stood in for by a joker, so a template is feasible while it lacks no
    more tiles than there are jokers left. `STOP` is legal exactly where nothing
    on the table is still uncovered, which is the whole hard constraint.

    `monotone` holds the sequence in template order. A set of sets has no order of
    its own, so one is imposed to make the target well defined, and imposing it on
    the *mask* as well is what keeps decoding on the distribution the teacher
    forced -- at the cost that a template chosen too early closes everything below
    it, which is what makes the flag worth measuring rather than assuming.
    """
    need = np.asarray(need)
    avail = np.asarray(avail)
    room = (np.asarray(n_sets) < cfg.max_sets)[:, None]
    out = np.zeros((need.shape[0], short.shape[1] + 1), dtype=bool)
    out[:, :-1] = (short <= avail[:, cfg.joker_kind, None]) & room
    if monotone:
        order = np.arange(short.shape[1])[None]
        out[:, :-1] &= order >= np.asarray(last)[:, None]
    out[:, -1] = need.sum(-1) == 0
    return out


def state_features(
    cfg: RummiConfig,
    need: np.ndarray,
    avail: np.ndarray,
    n_sets: np.ndarray,
    last: np.ndarray,
    present: np.ndarray,
) -> np.ndarray:
    """`(B, state_dim)`. `last` is the template chosen previously, `-1` at the start.

    The teacher's sets come in template order, so what was chosen last is part of
    the state rather than trivia -- without it the same remaining multiset can sit
    at two different points of the same sequence.
    """
    need = np.asarray(need).astype(np.float32)
    avail = np.asarray(avail).astype(np.float32)
    tt = template_table(cfg)
    copies = float(cfg.n_copies)
    total = float(cfg.n_tiles)
    jokers = float(max(cfg.n_jokers, 1))
    sets = np.asarray(n_sets, dtype=np.float32)

    scalars = np.stack(
        [
            need.sum(-1) / total,
            avail.sum(-1) / total,
            (avail - need).sum(-1) / total,
            sets / cfg.max_sets,
            1.0 - sets / cfg.max_sets,
            (np.asarray(last, dtype=np.float32) + 1.0) / len(tt.counts),
            need[:, cfg.joker_kind] / jokers,
            avail[:, cfg.joker_kind] / jokers,
            np.asarray(present, dtype=np.float32).sum(-1) / cfg.max_sets,
        ],
        axis=-1,
    )
    return np.concatenate(
        [need / copies, avail / copies, (avail - need) / copies, scalars], axis=-1
    ).astype(np.float32)


def laid_by(cfg: RummiConfig, template: int, avail: np.ndarray) -> np.ndarray:
    """`(K,)` what one template actually puts on the table, `macro.laid_tiles`' rule.

    Whatever `avail` cannot supply is stood in for by a joker, so the set that
    lands is not the template and this -- not the template -- is what balances.
    """
    counts = template_table(cfg).counts[template].astype(np.int64)
    gap = np.maximum(counts - np.asarray(avail).astype(np.int64), 0)
    laid = counts - gap
    laid[cfg.joker_kind] += int(gap.sum())
    return laid


def apply_template(
    cfg: RummiConfig, need: np.ndarray, avail: np.ndarray, template: int
) -> tuple[np.ndarray, np.ndarray]:
    laid = laid_by(cfg, template, avail)
    return np.maximum(need - laid, 0), avail - laid


# --- reading a solve back into the action space ---------------------------


@dataclass(frozen=True, slots=True)
class Repartition:
    """A finished construction, in the form `to_actions.plan` takes."""

    templates: tuple[int, ...]
    sets: tuple[tuple[int, ...], ...]
    played: np.ndarray
    """`(K,)` tiles moved out of the rack."""

    @property
    def tiles_played(self) -> int:
        return int(self.played.sum())


def build(
    cfg: RummiConfig, need: np.ndarray, avail: np.ndarray, templates: list[int] | tuple[int, ...]
) -> Repartition | None:
    """Replay a template sequence, or `None` if it is not a legal repartition.

    Legality here is the construction's own: every step affordable, no more sets
    than slots, and -- the constraint the whole space exists to honour -- nothing
    left on the table uncovered when the sequence ends.
    """
    tt = template_table(cfg)
    table_counts = np.asarray(need).astype(np.int64)
    remaining_need = table_counts.copy()
    remaining = np.asarray(avail).astype(np.int64)
    sets: list[tuple[int, ...]] = []

    for template in templates:
        if len(sets) >= cfg.max_sets:
            return None
        counts = tt.counts[template].astype(np.int64)
        gap = np.maximum(counts - remaining, 0)
        if int(gap.sum()) > int(remaining[cfg.joker_kind]):
            return None
        laid = counts - gap
        laid[cfg.joker_kind] += int(gap.sum())
        sets.append(tuple(sorted(np.repeat(np.arange(cfg.n_kinds), laid).tolist())))
        remaining_need = np.maximum(remaining_need - laid, 0)
        remaining = remaining - laid

    if remaining_need.any():
        return None
    played = (np.asarray(avail).astype(np.int64) - remaining) - table_counts
    if (played < 0).any():
        return None
    return Repartition(
        templates=tuple(templates), sets=tuple(sets), played=played.astype(np.int16)
    )


def label_sequence(
    cfg: RummiConfig, need: np.ndarray, avail: np.ndarray, solution_sets
) -> tuple[tuple[int, ...], str]:
    """The solver's answer as a canonical template sequence.

    Sets are emitted in template order, which is what makes the sequence target
    well defined at all -- a set of sets has no order of its own. Reading a joker
    back is the only ambiguity: the solver materialises it, so which tile it stood
    in for is gone, and a template is accepted when the substitution rule
    reproduces the set the solver actually emitted.

    Returns the sequence and one of `exact` (every set reproduced tile for tile),
    `relaxed` (the same templates, a joker resting on a different tile -- same
    count played, so the same turn), or `none`.
    """
    tt = template_table(cfg)
    resolved: list[tuple[list[int], tuple[int, ...]]] = []
    for content in solution_sets:
        kinds = sorted(int(k) for k in content)
        real = tuple(k for k in kinds if k != cfg.joker_kind)
        options = tt.by_content.get((len(kinds), real))
        if not options:
            return (), "none"
        resolved.append((kinds, options))

    order = sorted(range(len(resolved)), key=lambda i: resolved[i][1][0])
    remaining = np.asarray(avail).astype(np.int64)
    sequence: list[int] = []
    for i in order:
        kinds, options = resolved[i]
        wanted = kinds_to_counts(cfg, kinds).astype(np.int64)
        chosen = next(
            (t for t in options if (laid_by(cfg, t, remaining) == wanted).all()), None
        )
        if chosen is None:
            # The same templates still cover the same tiles and play the same
            # count; only where the joker comes to rest differs.
            chosen = next(
                (
                    t
                    for t in options
                    if int(np.maximum(tt.counts[t] - remaining, 0).sum())
                    <= int(remaining[cfg.joker_kind])
                ),
                None,
            )
        if chosen is None:
            return (), "none"
        remaining = remaining - laid_by(cfg, chosen, remaining)
        sequence.append(chosen)

    sequence.sort()
    replayed = build(cfg, need, avail, sequence)
    if replayed is None:
        return (), "none"
    wanted_sets = sorted(tuple(sorted(int(k) for k in s)) for s in solution_sets)
    exact = sorted(replayed.sets) == wanted_sets
    return tuple(sequence), "exact" if exact else "relaxed"


# --- the network ----------------------------------------------------------


class RepartitionNet(nn.Module):
    """Logits over templates plus `STOP`, scored against what each one does.

    A flat head would have to learn template 147 from its index alone, and the
    macro-space experiments already measured what that costs: nothing learned
    about one set transfers to a similar one. So the score is bilinear -- a query
    from the state against a description of the candidate -- and the description
    is half static (the template's own tiles) and half dynamic (what it would take
    off *this* table, out of *this* rack, and in jokers).

    The dynamic half is folded through the query rather than into the candidate
    embedding: `q . (W d) == (q W) . d`, so the `(B, T, key)` tensor never exists
    and the whole head is two matmuls the size of the template table.
    """

    static: torch.Tensor

    def __init__(self, cfg: RummiConfig, hidden: int = 256, key: int = 64) -> None:
        super().__init__()
        tt = template_table(cfg)
        self.register_buffer("static", torch.as_tensor(tt.static))
        self.trunk = nn.Sequential(
            nn.Linear(state_dim(cfg), hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.query = nn.Linear(hidden, key)
        self.describe = nn.Sequential(
            nn.Linear(static_dim(cfg), key), nn.ReLU(), nn.Linear(key, key)
        )
        self.interact = nn.Linear(CANDIDATE_DYNAMIC, key, bias=False)
        self.prior = nn.Linear(CANDIDATE_DYNAMIC, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(n_actions(cfg)))

    def forward(
        self, state: torch.Tensor, dynamic: torch.Tensor, legal: torch.Tensor
    ) -> torch.Tensor:
        q = self.query(self.trunk(state))
        logits = q @ self.describe(self.static).T
        logits = logits + (dynamic * (q @ self.interact.weight)[:, None, :]).sum(-1)
        logits = logits + self.prior(dynamic).squeeze(-1) + self.bias
        return torch.where(legal, logits, torch.full_like(logits, MASKED))


class Scorer:
    """A trained net behind the one call `decode` makes, with no autograd tape."""

    def __init__(self, net: RepartitionNet) -> None:
        self.net = net.eval()

    def __call__(
        self, state: np.ndarray, dynamic: np.ndarray, legal: np.ndarray
    ) -> np.ndarray:
        with torch.no_grad():
            out = self.net(
                torch.from_numpy(state), torch.from_numpy(dynamic), torch.from_numpy(legal)
            )
        return out.numpy()


@dataclass(frozen=True, slots=True)
class _Partial:
    """One live construction: the sequence so far and the state it leaves."""

    logp: float
    sequence: tuple[int, ...]
    need: np.ndarray
    avail: np.ndarray
    present: np.ndarray
    last: int


def decode(
    cfg: RummiConfig,
    score,
    need: np.ndarray,
    avail: np.ndarray,
    present: np.ndarray,
    beam: int = 1,
    monotone: bool = True,
) -> Repartition | None:
    """Construct a repartition, best-first, `beam` partial sequences at a time.

    At `beam=1` this is plain greedy decoding and the loop runs one row wide; the
    two are one code path so the greedy arm cannot drift from the searched one.

    Every finished beam is a legal repartition -- `STOP` is masked until the table
    is covered -- so what a wider beam buys is a *choice* between them, and it is
    settled on tiles played rather than on likelihood. Imitation only ever aimed
    the model at the solver's answer; the turn is worth what it sheds.
    """
    stop = stop_action(cfg)
    finished: list[tuple[float, tuple[int, ...]]] = []
    live = [
        _Partial(
            0.0,
            (),
            np.asarray(need).astype(np.int64),
            np.asarray(avail).astype(np.int64),
            np.asarray(present, dtype=np.float32),
            0 if monotone else -1,
        )
    ]
    for _ in range(cfg.max_sets + 1):
        if not live:
            break
        needs = np.stack([row.need for row in live])
        avails = np.stack([row.avail for row in live])
        presents = np.stack([row.present for row in live])
        sizes = np.array([len(row.sequence) for row in live])
        lasts = np.array([row.last for row in live])

        dynamic, short = candidate_features(cfg, needs, avails, presents, lasts)
        legal = feasible(cfg, needs, avails, sizes, lasts, short, monotone)
        logits = score(
            state_features(cfg, needs, avails, sizes, lasts, presents), dynamic, legal
        )
        # Log-probabilities, so beams of different lengths compare on one scale.
        logits = logits - logits.max(-1, keepdims=True)
        logp = logits - np.log(np.exp(logits).sum(-1, keepdims=True))

        proposals: list[_Partial] = []
        for row, partial in enumerate(live):
            options = np.flatnonzero(legal[row])
            if options.size == 0:
                continue
            for action in options[np.argsort(-logp[row, options])][:beam].tolist():
                total = partial.logp + float(logp[row, action])
                if action == stop:
                    finished.append((total, partial.sequence))
                    continue
                after_need, after_avail = apply_template(
                    cfg, partial.need, partial.avail, action
                )
                after_present = partial.present.copy()
                after_present[action] = max(after_present[action] - 1.0, 0.0)
                proposals.append(
                    _Partial(
                        total,
                        (*partial.sequence, action),
                        after_need,
                        after_avail,
                        after_present,
                        action,
                    )
                )
        proposals.sort(key=lambda row: -row.logp)
        live = proposals[:beam]

    # Most likely first, so a tie on tiles played keeps the model's own preference.
    finished.sort(key=lambda row: -row[0])
    best: Repartition | None = None
    for _, sequence in finished[: 4 * beam]:
        found = build(cfg, need, avail, list(sequence))
        if found is not None and (best is None or found.tiles_played > best.tiles_played):
            best = found
    return best


def padded_sets(cfg: RummiConfig, sets) -> np.ndarray:
    """`(n, max_set_len)` kind ids, `EMPTY` padded -- what `evaluate_slots` reads."""
    out = np.full((len(sets), cfg.max_set_len), EMPTY, dtype=np.int16)
    for i, content in enumerate(sets):
        out[i, : len(content)] = content
    return out
