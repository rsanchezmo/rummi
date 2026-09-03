"""Is the opening the one turn worth shedding fewer tiles on?

    python tools/opening_ab.py --arena head2head --deals 1600 --seed-base 93000
    python tools/opening_ab.py --arena head2head --opponent optimal --deals 600 \
        --arms base full min_sets --seed-base 93000
    python tools/opening_ab.py --config standard_3p --deals 600 --arms base min_sets
    python tools/opening_ab.py --arena suite --games 400

`tools/oracle_regret.py` rolled out 39,903 alternative turns against the real deck and
found no single-turn deviation from per-turn maximisation worth anything -- except one
cell. Pre-meld, an alternative shedding **fewer** tiles than the maximum was worth
**+9.3% +-5.7%** (428 alternatives, 242 deals), against -1.2% midgame and -8.1% in the
endgame; split against what `frugal` actually opened with, shedding fewer was +6.2
(n=384) and CP-SAT's max-tiles opening -1.1 (n=90). One cell of fourteen at that width
is not a finding, so this is the pre-sized, paired test of it.

`rummi/agents/opening.py` holds the arms: each is `frugal` with the opening turn
rebuilt and every later decision delegated to `by_value` untouched. `full` is the
correctness check -- `by_value`'s own opening, rebuilt through the same planner, which
must therefore score *exactly* even -- and `base` is plain `frugal` mirrored against
itself, which pins the rotation.

The telemetry is read off the same games as the score, per arm: what the opening
consisted of (tiles, meld value, sets, which of the seat's turns it was) and **what the
opponent did with it** -- the tiles it shed on the turn immediately after. That last is
the mechanism: an opening is handed over as rigid sets, so a larger one should be worth
more to the opponent.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, fields
from multiprocessing import get_context

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from rummi.agents.base import Agent, Observation, turn_starting
from rummi.agents.opening import ARMS, OpeningAgent, OpeningStats
from rummi.evaluate.protocol import SUITE_BY_NAME, evaluate
from rummi.rules.config import CONFIG_BY_NAME, RummiConfig
from rummi.rules.encoding import EMPTY
from rummi.rules.observation import F_IS_NEW, MELD_PROGRESS

# The two preceding experiments' runner, imported rather than rewritten: it already
# pins the self-mirror to exactly 50.00% and the seat rotation that cancels turn order.
from denial_ab import head_to_head, interval, permeability

ARM_NAMES = ("base", *ARMS)


@dataclass
class Telemetry:
    """What the opening looked like, and what it was worth to everyone afterwards.

    Summed rather than averaged, so chunks running in different processes merge by
    addition and the means are taken once at the end.
    """

    openings: int = 0
    tiles: int = 0
    value: int = 0
    sets: int = 0
    turn: int = 0
    """Which of the seat's own turns the opening was, counting the drawn ones."""
    jokers: int = 0
    """Openings that spent the rack's joker."""
    doors_new: float = 0.0
    """`denial_ab.permeability` of the sets the opening created: the unseen-weighted
    lay-off doors it hands over. This is the mechanism the hypothesis names -- a set laid
    pre-meld is handed to the opponent rigid -- and it is measured rather than assumed,
    because a *smaller* opening is not automatically a tighter one."""
    doors_all: float = 0.0
    """The same for the whole table the reply then faces, which is the level `doors_new`
    has to be read against: the opponent's own sets are most of it."""
    replies: int = 0
    reply_shed: int = 0
    """Tiles the next seat sheds on the turn immediately after the opening.

    Tiles shed, not tiles laid off: every macro in this space rebuilds the slot it
    touches rather than appending to it, so `slot_new` marks a lay-off's target as new
    and "growth of a set that was already there" is structurally zero -- measured, not
    assumed."""
    opp_turns: int = 0
    opp_shed: int = 0
    """Every opponent turn from the opening to the end of the deal. The reply is one
    turn and usually a pre-meld one, so it cannot carry the mechanism alone: this is the
    window in which a set handed over can be used at all."""
    own_turns: int = 0
    own_shed: int = 0
    """The same for the arm after its own opening -- the within-deal control, since an
    opening that keeps tiles back has to shed them later or it kept nothing."""
    game_turns: int = 0
    """Turns in the deal, all seats: an opening that stalls would lengthen it."""

    def merge(self, other: Telemetry) -> None:
        for f in fields(self):
            setattr(self, f.name, getattr(self, f.name) + getattr(other, f.name))

    def report(self) -> tuple[str, str]:
        openings = max(self.openings, 1)
        replies = max(self.replies, 1)
        opp, own = max(self.opp_turns, 1), max(self.own_turns, 1)
        return (
            f"opened {self.openings:>5,}  tiles {self.tiles / openings:>4.2f}  "
            f"value {self.value / openings:>5.2f}  sets {self.sets / openings:>4.2f}  "
            f"turn {self.turn / openings:>5.2f}  joker {self.jokers / openings:>5.1%}  "
            f"doors {self.doors_new / openings:>5.2f} of {self.doors_all / openings:>6.2f}",
            f"reply sheds {self.reply_shed / replies:>4.2f}  "
            f"per opponent turn after it {self.opp_shed / opp:>4.2f}  "
            f"own {self.own_shed / own:>4.2f}  deal {self.game_turns / openings:>5.1f} turns",
        )


@dataclass
class _Turn:
    """One turn in progress, overwritten on every step of it."""

    seat: int
    index: int
    """Turn number in the deal, all seats -- so "the turn after" is `index + 1`."""
    own_index: int
    """Turn number for this seat alone."""
    shed: int = 0
    value: int = 0
    sets: int = 0
    jokers: int = 0
    rows: np.ndarray | None = None
    """The table as it stands, kept only for the arm's own pre-meld turns: the opening's
    permeability is read once per deal and cannot be read after the turn has closed."""
    new: np.ndarray | None = None
    unseen: np.ndarray | None = None


class OpeningWatch:
    """Per env, the arm's opening turn and what every turn after it shed.

    Everything is read from the acting seat's rotated observation, so the arm's meld
    flag comes from `(seat - current) % n_players`. A turn's record is overwritten on
    every step and committed when the next turn starts, which lands it on the last step
    before the `END_TURN` or `DRAW` that closed it -- where `placed_this_turn` holds the
    whole turn and `MELD_PROGRESS` what it declared.
    """

    def __init__(self, cfg: RummiConfig, seat: int) -> None:
        self.cfg = cfg
        self.seat = seat
        self.live: dict[int, _Turn] = {}
        self.seat_turns: dict[int, int] = defaultdict(int)
        self.deal_turns: dict[int, int] = defaultdict(int)
        self.opened_at: dict[int, int] = {}
        self.out = Telemetry()

    def __call__(self, obs: Observation, current: np.ndarray, done: np.ndarray) -> None:
        cfg = self.cfg
        fresh = np.asarray(turn_starting(obs))
        melded = np.asarray(obs["melded"])
        placed = np.asarray(obs["placed_this_turn"])
        progress = np.asarray(obs["scalars"])[:, MELD_PROGRESS]
        slots = np.asarray(obs["slot_features"])
        is_new = slots[..., F_IS_NEW] > 0

        for env in range(len(current)):
            if done[env]:
                continue
            seat = int(current[env])
            mine = bool(melded[env, (self.seat - seat) % cfg.n_players])
            if fresh[env]:
                self._commit(env, mine)
                self.deal_turns[env] += 1
                if seat == self.seat:
                    self.seat_turns[env] += 1
                self.live[env] = _Turn(seat, self.deal_turns[env], self.seat_turns[env])
            turn = self.live.get(env)
            if turn is None:  # an env first seen mid-turn has no turn to attribute to
                continue
            turn.shed = int(placed[env].sum())
            turn.value = int(progress[env])
            turn.sets = int(is_new[env].sum())
            turn.jokers = int(placed[env, cfg.joker_kind])
            if seat == self.seat and not mine:
                turn.rows = np.asarray(obs["table_sets"][env]).copy()
                turn.new = is_new[env].copy()
                turn.unseen = np.asarray(obs["unseen"][env]).copy()

    def _commit(self, env: int, arm_melded: bool) -> None:
        turn = self.live.pop(env, None)
        if turn is None:
            return
        out = self.out
        if turn.seat == self.seat:
            if env in self.opened_at:
                out.own_turns += 1
                out.own_shed += turn.shed
                return
            # The flag flips at the `END_TURN` that opened, so the turn that just closed
            # is the opening exactly when the flag is set now and was not before.
            if not arm_melded or turn.rows is None or turn.new is None:
                return
            self.opened_at[env] = turn.index
            out.openings += 1
            out.tiles += turn.shed
            out.value += turn.value
            out.sets += turn.sets
            out.turn += turn.own_index
            out.jokers += int(turn.jokers > 0)
            opened = np.where(turn.new[:, None], turn.rows, np.int16(EMPTY))
            out.doors_new += float(permeability(self.cfg, opened, turn.unseen))
            out.doors_all += float(permeability(self.cfg, turn.rows, turn.unseen))
            return
        if self.opened_at.get(env) is None:
            return
        if turn.index == self.opened_at[env] + 1:
            out.replies += 1
            out.reply_shed += turn.shed
        out.opp_turns += 1
        out.opp_shed += turn.shed

    def totals(self) -> Telemetry:
        for env in list(self.live):
            self._commit(env, False)
        for env in self.opened_at:
            self.out.game_turns += self.deal_turns[env]
        return self.out


def build(cfg: RummiConfig, name: str, stats: OpeningStats | None = None) -> Agent:
    """One seat. An arm name builds an `OpeningAgent`; anything else is a registry rung."""
    if name in ARM_NAMES:
        return OpeningAgent(cfg, name, stats)
    from rummi.agents import build as build_registry

    return build_registry(name, cfg)


def _chunk(job: tuple) -> dict:
    """One slice of deals for one arm, in its own process.

    The deal index is `offset + row`, which is what makes the slices reassemble into the
    same rotation a single process would have played -- the seeds do not depend on how
    the work was split.
    """
    config, arm, opponent, deals, offset, seed_base, batch = job
    cfg = CONFIG_BY_NAME[config]
    stats = OpeningStats()
    watchers: list[OpeningWatch] = []

    def watch(seat: int, _first_deal: int) -> OpeningWatch:
        watchers.append(OpeningWatch(cfg, seat))
        return watchers[-1]

    began = time.perf_counter()
    wins, scores = head_to_head(
        cfg,
        lambda c: build(c, arm, stats),
        lambda c: build(c, opponent),
        deals,
        seed_base,
        batch,
        offset=offset,
        watch=watch,
    )
    telemetry = Telemetry()
    for watcher in watchers:
        telemetry.merge(watcher.totals())
    return {
        "arm": arm,
        "offset": offset,
        "wins": wins.tolist(),
        "scores": scores.tolist(),
        "stats": asdict(stats),
        "telemetry": asdict(telemetry),
        "seconds": time.perf_counter() - began,
    }


def _suite_job(job: tuple) -> dict:
    config, arm, suite_name, games = job
    cfg = CONFIG_BY_NAME[config]
    suite = SUITE_BY_NAME[suite_name]
    stats = OpeningStats()
    began = time.perf_counter()
    result = evaluate(arm, suite, build_agent=lambda c: build(c, arm, stats), games=games)
    assert not result.disqualified, f"{arm} was disqualified"
    played = max(result.games, 1)
    return {
        "arm": arm,
        "suite": suite_name,
        "games": played,
        "win_rate": result.win_rate,
        "win_ci": float(1.96 * np.sqrt(result.win_rate * (1 - result.win_rate) / played)),
        "mean_score": result.mean_score,
        "score_ci": float(1.96 * np.std(result.scores, ddof=1) / np.sqrt(played)),
        "stalemates": result.stalemates / played,
        "stats": asdict(stats),
        "seconds": time.perf_counter() - began,
        "n_players": cfg.n_players,
    }


def _run(
    jobs: list[tuple], worker, workers: int, label: str, cache: pathlib.Path | None = None
) -> list[dict]:
    """Every job, resuming whatever `cache` already holds and appending as they land.

    A full arena is an hour of wall clock and the chunks only exist in this process's
    memory until it finishes, so one interruption used to cost the whole run. Each
    chunk is a line of JSON the moment it returns instead, keyed by what produced it,
    and a re-run skips what is already there.
    """
    done: list[dict] = []
    if cache is not None and cache.exists():
        done = [json.loads(line) for line in cache.read_text().splitlines() if line.strip()]
        print(f"resuming {len(done)} chunks from {cache}", flush=True)
    seen = {(out["arm"], out.get("offset", -1)) for out in done}
    todo = [job for job in jobs if (job[1], job[4] if len(job) > 4 else -1) not in seen]

    started = time.perf_counter()
    handle = None if cache is None else cache.open("a")
    try:
        results = (
            get_context("spawn").Pool(workers).imap_unordered(worker, todo)
            if workers > 1
            else map(worker, todo)
        )
        for out in results:
            done.append(out)
            if handle is not None:
                handle.write(json.dumps(out) + "\n")
                handle.flush()
            print(
                f"[{len(done):>4}/{len(jobs)}] {label} {out['arm']:<12} "
                f"{out['seconds']:.0f}s  ({time.perf_counter() - started:.0f}s elapsed)",
                flush=True,
            )
    finally:
        if handle is not None:
            handle.close()
    return done


def _assemble(chunks: list[dict], arms: list[str], deals: int) -> dict[str, dict]:
    """Per arm, the per-deal rows in deal order plus the summed counters."""
    out: dict[str, dict] = {}
    for arm in arms:
        wins = np.full(deals, np.nan)
        scores = np.full(deals, np.nan)
        stats, telemetry = OpeningStats(), Telemetry()
        for chunk in (c for c in chunks if c["arm"] == arm):
            rows = slice(chunk["offset"], chunk["offset"] + len(chunk["wins"]))
            wins[rows] = chunk["wins"]
            scores[rows] = chunk["scores"]
            stats.merge(OpeningStats(**chunk["stats"]))
            telemetry.merge(Telemetry(**chunk["telemetry"]))
        assert not np.isnan(wins).any(), f"{arm} is missing deals"
        out[arm] = {"wins": wins, "scores": scores, "stats": stats, "telemetry": telemetry}
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="standard", choices=sorted(CONFIG_BY_NAME))
    p.add_argument("--arena", default="head2head", choices=("head2head", "suite"))
    p.add_argument("--arms", nargs="+", default=list(ARM_NAMES))
    p.add_argument("--opponent", default="frugal", help="who the head-to-head is against")
    p.add_argument("--deals", type=int, default=1600)
    p.add_argument("--games", type=int, default=400, help="deals for the suite arena")
    p.add_argument("--suite", default=None, help="defaults to the config's own suite")
    p.add_argument("--seed-base", type=int, default=93_000)
    p.add_argument("--batch", type=int, default=32, help="envs per process, and the chunk size")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--out", type=pathlib.Path, default=pathlib.Path("runs/opening-ab"))
    p.add_argument("--tag", default="", help="suffix for the results file")
    args = p.parse_args()

    cfg = CONFIG_BY_NAME[args.config]
    args.out.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"{args.arena}-{args.opponent}-{args.config}-{args.deals}"
    lines: list[str] = []

    def say(line: str = "") -> None:
        lines.append(line)
        print(line, flush=True)

    started = time.perf_counter()
    if args.arena == "suite":
        # The suites are named with hyphens and the configs with underscores, and
        # `standard`'s own suite is named after its opponent.
        default = "standard-greedy" if args.config == "standard" else args.config.replace("_", "-")
        suite = args.suite or default
        jobs = [(args.config, arm, suite, args.games) for arm in args.arms]
        results = _run(
            jobs, _suite_job, min(args.workers, len(jobs)), suite, cache=args.out / f"{tag}.jsonl"
        )
        results.sort(key=lambda r: args.arms.index(r["arm"]))
        say(f"\n{suite}, {args.games} deals x {cfg.n_players} seats")
        for r in results:
            say(
                f"  {r['arm']:<12} win {r['win_rate']:>6.1%} +-{r['win_ci']:>4.1%}  "
                f"score {r['mean_score']:>+8.2f} +-{r['score_ci']:>5.2f}  "
                f"stale {r['stalemates']:>5.1%}  n={r['games']}"
            )
            say(f"  {'':<12} {OpeningStats(**r['stats']).report()}")
        payload: dict = {"arena": "suite", "suite": suite, "results": results}
    else:
        jobs = [
            (args.config, arm, args.opponent, min(args.batch, args.deals - start), start,
             args.seed_base, args.batch)
            for arm in args.arms
            for start in range(0, args.deals, args.batch)
        ]
        chunks = _run(jobs, _chunk, args.workers, args.opponent, cache=args.out / f"{tag}.jsonl")
        played = _assemble(chunks, args.arms, args.deals)

        even = 1.0 / cfg.n_players
        reference = played["base"]["wins"] if "base" in played else np.full(args.deals, even)
        say(
            f"\nhead-to-head against {args.opponent} on {args.config}, "
            f"{args.deals} deals x {cfg.n_players} seats, seed base {args.seed_base}"
        )
        say(f"  even is {even:.2%}; the paired delta is against `base`, deal by deal")
        for arm in args.arms:
            wins, scores = played[arm]["wins"], played[arm]["scores"]
            win, win_ci = interval(wins)
            score, score_ci = interval(scores)
            # Paired on the deal against the mirror, which is where the variance is:
            # every arm met the same opponent on the same deals from every seat.
            delta, delta_ci = interval(wins - reference)
            say(
                f"  {arm:<12} win {win:>6.2%} +-{win_ci:>5.2%}  "
                f"delta {delta:>+6.2%} +-{delta_ci:>5.2%}  "
                f"score {score:>+7.2f} +-{score_ci:>5.2f}"
            )
            say(f"  {'':<12} {played[arm]['stats'].report()}")
            for line in played[arm]["telemetry"].report():
                say(f"  {'':<12} {line}")
        sd = float(np.std(played[args.arms[-1]]["wins"], ddof=1))
        say(f"\n  per-deal sd {sd:.3f}; 2pp at 95% needs {int(np.ceil((1.96 * sd / 0.02) ** 2)):,} deals")
        payload = {
            "arena": "head2head",
            "opponent": args.opponent,
            "arms": {
                arm: {
                    "wins": played[arm]["wins"].tolist(),
                    "scores": played[arm]["scores"].tolist(),
                    "stats": asdict(played[arm]["stats"]),
                    "telemetry": asdict(played[arm]["telemetry"]),
                }
                for arm in args.arms
            },
        }

    wall = time.perf_counter() - started
    say(f"\nwall clock {wall:.0f}s over {args.workers} workers")

    path = args.out / f"{tag}.json"
    path.write_text(
        json.dumps(
            {
                "config": args.config,
                "deals": args.deals,
                "games": args.games,
                "seed_base": args.seed_base,
                "wall_seconds": wall,
                "summary": lines,
                **payload,
            },
            indent=1,
        )
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
