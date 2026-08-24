"""Independent JAX implementation of the simulator, written against SPEC.md."""

from rummi.backends.jax_backend.sim import (
    JaxState,
    check_actions,
    counts_of,
    deck_orders,
    derived_deck_orders,
    digest,
    legal_actions,
    reset,
    reset_envs,
    step,
)

__all__ = [
    "JaxState",
    "check_actions",
    "counts_of",
    "deck_orders",
    "derived_deck_orders",
    "digest",
    "legal_actions",
    "reset",
    "reset_envs",
    "step",
]
