"""A reference learned agent: the feature layout, two networks, and the adapter.

Under `rummi/agents/` rather than in a package of its own, because a parallel
"policies" concept existed here once and was deliberately removed -- an agent is
an agent, and this one is not privileged either. It is **not** in
:data:`rummi.agents.REGISTRY`: the bundled ladder is `random` through `optimal`,
and a rung with weights in it has to earn its place by landing between
`rearrange` and `optimal` first.

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
