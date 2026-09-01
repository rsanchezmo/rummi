"""Optimal-tier play that pays for a solve only when it is stuck.

The ladder's one idea is how much of the table an agent will take apart.
`rearrange` stops at stealing one tile; `optimal` repartitions the whole table
every turn, which is what CP-SAT is for. This rung sits between them in
mechanism and level with `optimal` in strength: template sets dearest-first
before the opening meld and most-tiles-first after, lay-offs, single-tile
steals -- and the solver invoked only where nothing else plays. Measured even
with `optimal` head-to-head (48.7% at n=600 two-seat; 33.9% at three seats,
where even is 33.3%) at a tenth of the compute, which is the measured claim
that per-turn play above this buys nothing the stuck-state solve does not
already deliver.
"""

from rummi.agents.macro import MacroAgent, by_value
from rummi.rules.config import RummiConfig


class FrugalAgent(MacroAgent):
    name = "frugal"

    def __init__(self, cfg: RummiConfig) -> None:
        super().__init__(cfg, choose=by_value(cfg), repartition=True)
