"""Does it pay to shape the table you leave behind?

    python tools/denial_ab.py --arena both --deals 600 --games 400

Every ranking in this repo -- `greedy`, `rearrange`, `by_value`, `frugal`, `optimal`,
the clone -- scores a candidate play by points before the opening meld and tiles shed
after, and none of them reads what the table it leaves gives the opponent. This
measures that axis directly, as a hand-written deterministic tie-break with no
learning and no reward term anywhere: among plays `by_value` ranks **equal**, prefer
the one leaving the least permeable table. Where the ranking has a single best play
the arm plays that, so the arm differs from `by_value` in exactly one component.

**Permeability** is the unseen-weighted count of lay-off doors: which kinds any slot
would accept (`greedy_agent.appendable`, the same matrix `macro.legal_macros` gates a
lay-off on), each weighted by the copies still unaccounted for in `unseen`. A run of
5-6-7 accepts a 4 and an 8, two copies each; a group of three 7s accepts one colour.
Nothing hidden is read -- denial asks whether the opponent can be *restricted*, which
needs only the table, and that is why it is live where the information arms are null.
The `deny+steal` arm adds `macro.removals`' side of it: how many tiles the table would
let an opponent take.

Four things make the answer readable, and none of them is optional. The **null** arm
breaks the same ties on a state-seeded coin flip down the same code path, so if it also
wins the harness is manufacturing the effect and any positive result is void. The
**open** arm prefers the *most* permeable table, so the axis has to show a sign: two
arms that read the same need no other refutation. The **value** arm is the positive
control -- a different tie-break, of the same shape and the same firing rate, that is
known to be worth something -- so a null on `deny` cannot be blamed on a measurement
that resolves nothing. And the head-to-head is **paired per deal**: every arm plays the
same deals from every seat, and the deal is most of the variance.

`--arena shape` is the mechanism behind whatever the score says: the permeability of
the table each seat is handed, split by which side left it. `left` against `met` is
the question a score cannot answer -- whether a door closed on the opponent is also
closed on the closer.

**What it measured.** Head-to-head against plain `by_value`, 1600 deals played from
both seats (3200 games per arm), and the same arms on `standard-greedy` at 400 deals:

| arm | head-to-head win | vs null, paired | `standard-greedy` |
|---|---|---|---|
| `deny` | 50.84% +-1.38% | +1.06pp +-1.57 | 83.5% +-2.6 / +29.73 |
| `deny+steal` | 50.59% +-1.35% | +0.81pp +-1.57 | 84.0% +-2.5 / +30.28 |
| `value` | 50.59% +-1.30% | +0.81pp +-1.56 | 84.0% +-2.5 / +29.73 |
| `open` | 49.97% +-1.09% | -- | 80.9% +-2.7 / +26.25 |
| `null` | 49.78% +-1.36% | -- | 83.5% +-2.6 / +29.99 |
| `base` | 50.00% +-0.00% | -- | 82.8% +-2.6 / +28.18 |

Every interval covers even, and **the null control is level with the arm on both
arenas** -- ahead of it on `standard-greedy`. `base` mirrored against itself reads
exactly 50.00% and exactly +0.00, so the rotation is exact and none of this is bias.
The positive control is null as well, so the bound is not denial-specific and is the
stronger statement for it: **no tie-break over `by_value`'s indifference set** -- the
axis's own two extremes and a face-value rule included -- **is worth more than ~1.6pp**
at this resolution.

Three numbers say why, and they are the transferable part:

- **The tie-break can barely act.** Ties are 12.4% of decisions (19.7% before the
  opening meld, 13.3% after) and 84% of them hold two members. In **58.9%** of ties
  the permeability spread is exactly zero -- the candidates leave equally permeable
  tables, so `argmin` returns `by_value`'s own pick. Net, the arm plays differently in
  **3.7%** of decisions.
- **Its effect on the table is real and small.** Mean permeability handed over falls
  13.17 -> 12.84 under `deny` and rises to 13.20 under `open`, so the axis does move
  in the intended direction. But the coin flip alone reaches 13.00: only 0.16 of the
  0.33, **1.2% of the level**, is denial rather than perturbation.
- **And it is not differential.** `left - met` is **0.00 +-0.00** for every arm. The
  table is common property and both players draw from the same pool, so a door closed
  on the opponent is closed on the closer by the same amount. That is why the axis has
  no sign to find, and it is not a fact about this tie-break: any table-shaping rule
  in this game pays its own cost, which no amount of search or learning removes.
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import time
from dataclasses import dataclass

import numpy as np

from rummi.agents.base import (
    Agent,
    Observation,
    act_by_seat,
    has_melded,
    table,
    turn_starting,
)
from rummi.agents.greedy_agent import appendable
from rummi.agents.macro import (
    MacroAgent,
    by_value,
    extend_offset,
    laid_tiles,
    removals,
    repartition_offset,
    set_templates,
    steal_offset,
    template_points,
)
from rummi.rules.config import CONFIG_BY_NAME, RummiConfig
from rummi.rules.encoding import EMPTY, tables
from rummi.env.numpy.deal import reset as deal_reset
from rummi.env.numpy.deal import reset_envs
from rummi.env.numpy.engine import step as engine_step
from rummi.env.numpy.masks import legal_actions
from rummi.env.numpy.sets import summarize
from rummi.env.numpy.state import counts_of
from rummi.env.observation import encode
from rummi.evaluate.protocol import evaluate, suite_for
from rummi.solver.to_actions import slot_contents


def shed_values(cfg: RummiConfig) -> np.ndarray:
    """`(K,)` rack penalty each kind sheds when played.

    The joker scores 0 face value because its value is positional; what it costs to
    keep is `joker_penalty`, which is what shedding it saves -- the same substitution
    `by_value` and `greedy` both make.
    """
    out = tables(cfg).value.astype(np.int64).copy()
    out[cfg.joker_kind] = cfg.joker_penalty
    return out


def rank_tables(cfg: RummiConfig) -> tuple[np.ndarray, np.ndarray]:
    """`by_value`'s two rankings over the template blocks: points, then tiles.

    Composed from the same public tables `by_value` reads, because a tie-break needs
    the rank *values* and the chooser only exposes its argmax. So this is the one
    thing here that has to agree with something it cannot call, and
    `test_the_recomputed_ranking_reproduces_by_value` pins the pair to its choices
    over random masks rather than trusting the composition.
    """
    n_extend = steal_offset(cfg) - extend_offset(cfg)
    set_points = template_points(cfg)
    set_tiles = set_templates(cfg).sum(-1).astype(np.int32)
    layoff = shed_values(cfg).astype(np.int32)[:n_extend]
    points = np.concatenate([set_points, layoff, set_points])
    tiles = np.concatenate(
        [set_tiles, np.ones(n_extend, np.int32), np.maximum(set_tiles - 1, 1)]
    )
    return points, tiles


def _row(cfg: RummiConfig, kinds) -> np.ndarray:
    """One slot row: kinds ascending, `EMPTY` trailing.

    The padding has to be contiguous at the end, because `appendable` grows a row by
    writing at its first empty position.
    """
    out = np.full(cfg.max_set_len, EMPTY, dtype=np.int16)
    ordered = sorted(int(k) for k in kinds)
    out[: len(ordered)] = ordered
    return out


def after_rows(
    cfg: RummiConfig, board: np.ndarray, hand: np.ndarray, macro: int
) -> np.ndarray:
    """The `(S, L)` table `macro` leaves behind, played from `hand`.

    The same construction `MacroAgent.expand` hands to `to_actions.plan` as its
    target, minus the micro-actions: which empty slot a new set lands in is left to
    the first free one, since permeability is a function of the rows and not of their
    order. `test_the_predicted_table_is_the_one_the_env_reaches` replays the real
    expansion against this.
    """
    templates = set_templates(cfg).astype(np.int64)
    rows = np.asarray(board, dtype=np.int16).copy()
    ext, steal = extend_offset(cfg), steal_offset(cfg)

    if macro < ext:
        laid = laid_tiles(cfg, templates[macro], hand)
        return _fill(cfg, rows, np.repeat(np.arange(cfg.n_kinds), laid))
    if macro < steal:
        kind = macro - ext
        # Indexed by tile, so the receiving slot is whichever takes it -- the same
        # first-true `expand` picks, off the same matrix.
        slot = int(np.flatnonzero(appendable(cfg, rows, hand)[:, kind])[0])
        rows[slot] = _row(cfg, [*(k for k in rows[slot] if k >= 0), kind])
        return rows

    template = templates[macro - steal]
    kind = int(np.maximum(template - hand, 0).argmax())
    contents = slot_contents(rows)
    donor = next(s for s, c in enumerate(contents) if c and kind in removals(cfg, c))
    left = list(contents[donor])
    left.remove(kind)
    rows[donor] = _row(cfg, left)
    return _fill(cfg, rows, np.repeat(np.arange(cfg.n_kinds), template))


def _fill(cfg: RummiConfig, rows: np.ndarray, kinds: np.ndarray) -> np.ndarray:
    """A new set into the lowest free slot, which `legal_macros` guarantees exists."""
    slot = int(np.flatnonzero((rows < 0).all(-1))[0])
    rows[slot] = _row(cfg, kinds)
    return rows


def permeability(
    cfg: RummiConfig,
    rows: np.ndarray,
    unseen: np.ndarray,
    steal_weight: float = 0.0,
) -> np.ndarray:
    """How much play a table offers an opponent. `rows` may carry a leading batch.

    Unseen-weighted lay-off doors: a kind counts once however many slots accept it --
    the opponent still sheds one tile per copy -- times the copies `unseen` says are
    still unaccounted for. Reading `appendable` against `unseen` as the hand does both
    at once, since it ands in `rack > 0`: a door for a kind wholly on the table is not
    a door.

    `unseen` broadcasts, so one call answers either shape a caller has: a tie set of
    candidate afterstates shares one `unseen`, while a batch of envs does not.
    """
    rows = np.asarray(rows, dtype=np.int16)
    batched = rows.reshape(-1, cfg.max_sets, cfg.max_set_len)
    hands = np.broadcast_to(np.asarray(unseen), (batched.shape[0], cfg.n_kinds))
    doors = appendable(cfg, batched, hands).any(1)
    out = (doors * hands).sum(-1).astype(np.float64)
    if steal_weight:
        out = out + steal_weight * np.array(
            [
                sum(len(removals(cfg, c)) for c in slot_contents(one))
                for one in batched
            ],
            dtype=np.float64,
        )
    return out.reshape(rows.shape[:-2])


@dataclass
class TieStats:
    """How often the tie-break gets to act, and how much room it has when it does."""

    decisions: int = 0
    ties: int = 0
    members: int = 0
    level: float = 0.0
    """Summed mean permeability of the tie's candidates, so `spread` can be read
    against the size of the thing it is a spread in."""
    spread: float = 0.0
    """Summed `max - min` permeability over the firing ties, so the mean says whether
    the candidates differ at all."""
    moved: int = 0
    """Ties where the preferred play is not the one `by_value` would have taken."""

    def report(self) -> str:
        ties = max(self.ties, 1)
        return (
            f"decisions {self.decisions:>7,}  ties {self.ties / max(self.decisions, 1):>6.1%}  "
            f"members {self.members / ties:>4.2f}  "
            f"perm {self.level / ties:>5.1f} +-{self.spread / ties:>4.2f}  "
            f"moved {self.moved / max(self.decisions, 1):>6.1%}"
        )


MODES = ("deny", "open", "null", "value")
"""The tie-break keys, all four minimised over the same tie set.

`deny` is the experiment: leave the least permeable table. The other three are what
make its answer readable.

`null` is a coin flip seeded from the state -- it says whether the harness manufactures
an effect, because a tie-break on nothing cannot win. `open` prefers the *most*
permeable table, so the axis has to show a sign: two arms that read the same need no
other refutation. `value` is the **positive** control, and it is the one that says the
measurement has resolution at all: post-meld `by_value` ranks a tie by tile count and
is blind to face value inside it, so shedding 11-12-13 over 1-2-3 is a real
improvement of the same shape as denial -- one deterministic tie-break, the same firing
rate. It came back null too, which does not weaken the verdict but widens it: the bound
is on the indifference set, not on denial alone.
"""


class TieBreak:
    """`by_value`, refined by one rule: among equal-ranked plays, minimise `mode`'s key.

    A strict refinement. Where the ranking has a single best play this returns it, and
    where several are tied it minimises `mode`'s key and falls back to `by_value`'s own
    index order beyond that -- `np.argmin` takes the first minimum. So an arm and the
    baseline differ in exactly the component under test.
    """

    def __init__(
        self,
        cfg: RummiConfig,
        mode: str = "deny",
        steal_weight: float = 0.0,
        stats: TieStats | None = None,
    ) -> None:
        assert mode in MODES, mode
        self.cfg = cfg
        self.mode = mode
        self.ranked = repartition_offset(cfg)
        self.points, self.tiles = rank_tables(cfg)
        self.steal_weight = steal_weight
        self.shed = shed_values(cfg)
        self.stats = stats if stats is not None else TieStats()

    def __call__(self, obs: Observation, env: int, legal: np.ndarray) -> int:
        options = np.flatnonzero(legal[: self.ranked])
        if not options.size:
            # Nothing a template describes is legal, so there is nothing to rank:
            # REPARTITION, END_TURN or DRAW, exactly as `by_value` leaves it.
            return int(np.flatnonzero(legal)[0])

        self.stats.decisions += 1
        rank = self.points if not bool(has_melded(obs)[env]) else self.tiles
        scores = rank[options]
        tied = options[scores == scores.max()]
        if tied.size == 1:
            return int(tied[0])

        cfg = self.cfg
        rack = np.asarray(obs["rack"][env]).astype(np.int64)
        hand = rack + np.asarray(obs["workbench"][env]).astype(np.int64)
        board = np.asarray(table(obs)[env])
        after = np.stack([after_rows(cfg, board, hand, int(m)) for m in tied])
        left = permeability(cfg, after, obs["unseen"][env], self.steal_weight)

        self.stats.ties += 1
        self.stats.members += int(tied.size)
        self.stats.level += float(left.mean())
        self.stats.spread += float(left.max() - left.min())
        key = self._key(after, board, rack, left)
        chosen = int(tied[int(np.argmin(key))])
        self.stats.moved += int(chosen != int(tied[0]))
        return chosen

    def _key(
        self, after: np.ndarray, board: np.ndarray, rack: np.ndarray, left: np.ndarray
    ) -> np.ndarray:
        if self.mode == "deny":
            return left
        if self.mode == "open":
            return -left
        if self.mode == "value":
            # What the table gained is what the hand played, in every block -- a steal
            # takes its missing tile off the table, so it cancels on both sides.
            played = counts_of(self.cfg, after).astype(np.int64) - counts_of(
                self.cfg, board[None]
            ).astype(np.int64)
            return -(played * self.shed).sum(-1).astype(np.float64)
        return self._coin(board, rack, len(left))

    def _coin(self, board: np.ndarray, rack: np.ndarray, n: int) -> np.ndarray:
        """A tie-break on nothing, fixed per state so the arm stays deterministic.

        `blake2b` rather than `hash`, whose seed varies per process: a control that
        played different games on every run could not be compared with anything.
        """
        digest = hashlib.blake2b(
            board.astype(np.int16).tobytes() + rack.astype(np.int64).tobytes(), digest_size=8
        )
        return np.random.default_rng(int.from_bytes(digest.digest(), "little")).random(n)


ARMS = ("deny", "deny+steal", "open", "null", "value")


def build_arm(cfg: RummiConfig, arm: str, repartition: bool, stats: TieStats) -> Agent:
    """One arm as an agent. `base` is `by_value` itself, the thing being refined."""
    if arm == "base":
        return MacroAgent(cfg, choose=by_value(cfg), repartition=repartition)
    choose = TieBreak(
        cfg,
        mode="deny" if arm == "deny+steal" else arm,
        steal_weight=1.0 if arm == "deny+steal" else 0.0,
        stats=stats,
    )
    return MacroAgent(cfg, choose=choose, repartition=repartition)


def play_deals(
    cfg: RummiConfig,
    seats: list[Agent],
    seeds: list[np.random.SeedSequence],
    max_steps: int = 20_000,
    watch=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Winner and final rack values of one game per seed, env slot *i* being seed *i*.

    `protocol._play_batch` drives the same pieces and folds the outcomes into a
    `Result`, which drops which env won which deal -- and the statistic here is per
    deal, because pairing on the deal is what removes the luck of it.

    `watch(obs, current, done)` sees every step, which is how the mechanism is read off
    the same games the score comes from rather than off a second set of them.
    """
    n = len(seeds)
    state = deal_reset(cfg, n, seed=0)
    reset_envs(state, np.arange(n), seeds)
    for agent in seats:
        agent.reset(n)

    for _ in range(max_steps):
        if state.done.all():
            break
        summary = summarize(cfg, state.table_sets)
        mask = legal_actions(state, summary)
        obs = encode(state, summary)
        if watch is not None:
            watch(obs, np.asarray(state.current), np.asarray(state.done))
        actions, illegal = act_by_seat(
            seats, cfg, state.current, state.done, mask, obs
        )
        assert illegal == 0, f"an arm proposed {illegal} masked-out actions"
        engine_step(state, actions, mask)

    assert state.done.all(), "a game did not finish inside max_steps"
    return np.asarray(state.winner).copy(), state.rack_values().copy()


def head_to_head(
    cfg: RummiConfig,
    make_a,
    make_b,
    deals: int,
    seed_base: int,
    batch: int = 32,
    offset: int = 0,
    watch=None,
) -> tuple[np.ndarray, np.ndarray]:
    """`(deals,)` win fraction and official score for A, seat-rotated over the deals.

    Every deal is played once per seat with A in it and B in the rest, which is what
    `protocol` does and the only thing that cancels turn order past two seats. Each
    deal collapses to one number, so the per-deal spread is the error bar and games
    sharing a deal are not counted as independent.

    `offset` shifts which deals these are without touching the seed base, so a caller
    may split one rotation across processes and concatenate the per-deal rows -- the
    deal a row belongs to is `offset + row`, and the seeds are the same either way.
    `watch(seat, first_deal)` returns an observer for :func:`play_deals`, which is how
    the mechanism is read off the same games the score comes from.
    """
    wins = np.zeros((deals, cfg.n_players))
    scores = np.zeros((deals, cfg.n_players))
    for start in range(0, deals, batch):
        count = min(batch, deals - start)
        seeds = [
            np.random.SeedSequence([seed_base, offset + start + i]) for i in range(count)
        ]
        for seat in range(cfg.n_players):
            seats: list[Agent] = [make_b(cfg) for _ in range(cfg.n_players)]
            seats[seat] = make_a(cfg)
            observer = None if watch is None else watch(seat, offset + start)
            winner, values = play_deals(cfg, seats, seeds, watch=observer)
            won = winner == seat
            wins[start : start + count, seat] = won
            others = [p for p in range(cfg.n_players) if p != seat]
            scores[start : start + count, seat] = np.where(
                won, values[:, others].sum(-1), -values[:, seat]
            )
    return wins.mean(-1), scores.mean(-1)


class FacedPermeability:
    """The permeability of the table each seat is handed at the top of its turn.

    The mechanism behind whatever the score says. A tie-break that moves a few percent
    of decisions can only pay if the table the opponent actually faces gets less
    permeable, and this is that quantity: read from the acting seat's own rotated
    observation, so `unseen` is what *that* seat cannot see.
    """

    def __init__(self, cfg: RummiConfig, steal_weight: float = 0.0) -> None:
        self.cfg = cfg
        self.steal_weight = steal_weight
        self.seat = 0
        self.deal = 0
        """Index of env slot 0, so a turn is attributed to the deal it belongs to."""
        self.left: dict[int, list[float]] = {}
        """Per deal, faced by an opponent -- so left behind by the arm."""
        self.met: dict[int, list[float]] = {}
        """Per deal, faced by the arm itself: the within-deal control, since a deal
        whose tiles simply build open runs raises both."""

    def __call__(self, obs: Observation, current: np.ndarray, done: np.ndarray) -> None:
        fresh = np.asarray(turn_starting(obs)) & ~done
        if not fresh.any():
            return
        envs = np.flatnonzero(fresh)
        rows = np.asarray(table(obs))[envs]
        # One `appendable` over every fresh table: `unseen` differs per env, so the
        # weighting cannot be shared, but the door matrix is the expensive half.
        for env, value in zip(
            envs.tolist(),
            permeability(self.cfg, rows, obs["unseen"][envs], self.steal_weight),
            strict=True,
        ):
            side = self.met if current[env] == self.seat else self.left
            side.setdefault(self.deal + env, []).append(float(value))

    def paired(self) -> tuple[float, float, np.ndarray]:
        """Mean left, mean met, and the per-deal difference between them."""
        deals = sorted(set(self.left) & set(self.met))
        left = np.array([np.mean(self.left[d]) for d in deals])
        met = np.array([np.mean(self.met[d]) for d in deals])
        return float(left.mean()), float(met.mean()), left - met


def shape_left(
    cfg: RummiConfig,
    make_a,
    make_b,
    deals: int,
    seed_base: int,
    steal_weight: float = 0.0,
    batch: int = 32,
) -> FacedPermeability:
    """Play the same rotation the score uses and watch the tables instead of the wins."""
    watcher = FacedPermeability(cfg, steal_weight)
    for start in range(0, deals, batch):
        count = min(batch, deals - start)
        seeds = [np.random.SeedSequence([seed_base, start + i]) for i in range(count)]
        for seat in range(cfg.n_players):
            seats: list[Agent] = [make_b(cfg) for _ in range(cfg.n_players)]
            seats[seat] = make_a(cfg)
            watcher.seat, watcher.deal = seat, start
            play_deals(cfg, seats, seeds, watch=watcher)
    return watcher


def interval(x: np.ndarray) -> tuple[float, float]:
    """Mean and the 95% half-width of it, over whatever the rows are paired on."""
    x = np.asarray(x, dtype=np.float64)
    if x.size < 2:
        return float(x.mean()), float("nan")
    return float(x.mean()), float(1.96 * x.std(ddof=1) / np.sqrt(x.size))


def _deals_for(half_width: float, sd: float) -> int:
    """Deals a paired mean needs to resolve `half_width` at 95%."""
    return int(np.ceil((1.96 * sd / half_width) ** 2))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="standard", choices=sorted(CONFIG_BY_NAME))
    p.add_argument(
        "--arena", default="both", choices=("head2head", "suite", "shape", "both"),
        help="'shape' skips scoring and measures only the table each arm hands over",
    )
    p.add_argument("--deals", type=int, default=600, help="distinct deals per head-to-head arm")
    p.add_argument("--games", type=int, default=200, help="deals for the standard-greedy arm")
    p.add_argument("--arms", nargs="+", default=list(ARMS), choices=sorted({*ARMS, "base"}))
    p.add_argument(
        "--opponent", default="base", choices=("base", *ARMS, "optimal", "frugal", "greedy"),
        help="who the head-to-head is against; 'base' is plain by_value",
    )
    p.add_argument(
        "--repartition", action="store_true",
        help="run every arm and the baseline at frugal tier, with the stuck-state solve on",
    )
    p.add_argument("--seed-base", type=int, default=91_000, help="ad-hoc suite, outside the frozen ones")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument(
        "--out", type=pathlib.Path, default=None,
        help="npz of the per-deal head-to-head outcomes, for pairing arms across runs",
    )
    args = p.parse_args()

    cfg = CONFIG_BY_NAME[args.config]
    stats: dict[str, TieStats] = {}

    def make(arm: str):
        stats.setdefault(arm, TieStats())
        return lambda c: build_arm(c, arm, args.repartition, stats[arm])

    if args.opponent in ("base", *ARMS):
        make_b = make(args.opponent)
        opponent = f"{args.opponent}{'+repartition' if args.repartition else ''}"
    else:
        from rummi.agents import build as build_registry

        make_b = lambda c: build_registry(args.opponent, c)  # noqa: E731
        opponent = args.opponent

    print(
        f"config {args.config}  repartition {args.repartition}  "
        f"baseline {'by_value' if args.opponent == 'base' else opponent}"
    )

    if args.arena in ("head2head", "both"):
        print(f"\nhead-to-head against {opponent}, {args.deals} deals x {cfg.n_players} seats")
        played: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for arm in args.arms:
            stats[arm] = TieStats()  # one count per measurement, not per process
            began = time.perf_counter()
            wins, scores = head_to_head(
                cfg, make(arm), make_b, args.deals, args.seed_base, args.batch
            )
            played[arm] = (wins, scores)
            win, win_ci = interval(wins)
            score, score_ci = interval(scores)
            even = 1.0 / cfg.n_players
            print(
                f"  {arm:<12} win {win:>6.2%} +-{win_ci:>5.2%} (even {even:.1%})  "
                f"score {score:>+7.2f} +-{score_ci:>5.2f}  "
                f"{time.perf_counter() - began:.0f}s",
                flush=True,
            )
            print(f"  {'':<12} {stats[arm].report()}")
            # What the per-deal spread implies for anyone reading the interval.
            print(
                f"  {'':<12} 2pp needs {_deals_for(0.02, float(np.std(wins, ddof=1))):,} deals",
                flush=True,
            )

        if args.out:
            # Per deal, so any pair of arms measured against the same baseline at the
            # same `--seed-base` can be paired afterwards -- including across runs, which
            # is what a control added after the fact needs.
            args.out.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                args.out,
                deals=args.deals,
                seed_base=args.seed_base,
                **{f"{arm}_{field}": played[arm][i] for arm in played
                   for i, field in enumerate(("wins", "scores"))},
            )
            print(f"\n  wrote {args.out}")

        # Every arm met the same baseline on the same deals, so the difference between
        # two of them pairs on the deal as well -- which is the tighter comparison, and
        # the one that says whether an arm beats its own control rather than the mean.
        if "null" in played:
            print(f"\n  paired against the null control, {args.deals} deals")
            base_wins, base_scores = played["null"]
            for arm in args.arms:
                if arm == "null":
                    continue
                win, win_ci = interval(played[arm][0] - base_wins)
                score, score_ci = interval(played[arm][1] - base_scores)
                print(
                    f"  {arm + ' - null':<14} win {win:>+6.2%} +-{win_ci:>5.2%}  "
                    f"score {score:>+7.2f} +-{score_ci:>5.2f}"
                )

    if args.arena == "shape":
        print(f"\ntable handed over, {args.deals} deals x {cfg.n_players} seats")
        for arm in dict.fromkeys(("base", *args.arms)):
            stats[arm] = TieStats()
            began = time.perf_counter()
            watcher = shape_left(
                cfg, make(arm), make_b, args.deals, args.seed_base, batch=args.batch
            )
            left, met, difference = watcher.paired()
            gap, gap_ci = interval(difference)
            print(
                f"  {arm:<12} left {left:>6.2f}  met {met:>6.2f}  "
                f"left-met {gap:>+5.2f} +-{gap_ci:>4.2f}  "
                f"{time.perf_counter() - began:.0f}s",
                flush=True,
            )
            if stats[arm].decisions:
                print(f"  {'':<12} {stats[arm].report()}")

    if args.arena in ("suite", "both"):
        suite = suite_for(args.config)
        print(f"\n{suite.name}, {args.games} deals x {cfg.n_players} seats")
        for arm in dict.fromkeys(("base", *args.arms)):
            stats[arm] = TieStats()
            began = time.perf_counter()
            result = evaluate(
                arm, suite, build_agent=make(arm), games=args.games
            )
            assert not result.disqualified, f"{arm} was disqualified"
            # Error bars over games rather than deals, because `evaluate` reports the
            # suite in aggregate: the two seats of a deal are correlated, so these are
            # the *narrowest* the intervals could honestly be, not the true widths.
            games = max(result.games, 1)
            win_ci = 1.96 * np.sqrt(result.win_rate * (1 - result.win_rate) / games)
            score_ci = 1.96 * float(np.std(result.scores, ddof=1)) / np.sqrt(games)
            print(
                f"  {arm:<12} win {result.win_rate:>6.1%} +-{win_ci:>4.1%}  "
                f"score {result.mean_score:>+8.2f} +-{score_ci:>5.2f}  "
                f"stale {result.stalemates / games:>5.1%}  n={games}  "
                f"{time.perf_counter() - began:.0f}s",
                flush=True,
            )
            if stats[arm].decisions:
                print(f"  {'':<12} {stats[arm].report()}")


if __name__ == "__main__":
    main()
