"""A reference learned agent: the feature layout, two networks, and the adapter.

Under `rummi/agents/` rather than in a package of its own, because a parallel
"policies" concept existed here once and was deliberately removed -- an agent is
an agent, and this one is not privileged either.

One thing built here *is* privileged, and had to earn it: `clone.py` is the
ladder's `learned` rung, registered in :data:`rummi.agents.REGISTRY` because it
landed between `rearrange` and the solver tier -- the bar a rung carrying weights
has to clear. Everything else here is scaffolding for the next attempt.

Score one the same way you would score your own::

    from rummi.evaluate.protocol import SUITE_BY_NAME, evaluate
    evaluate("mine", SUITE_BY_NAME["tiny"], build_agent=lambda cfg: my_agent)
"""

from rummi.agents.learned.features import (
    FEATURE_FIELDS,
    feature_dim,
    feature_scale,
)

__all__ = ["FEATURE_FIELDS", "feature_dim", "feature_scale"]
