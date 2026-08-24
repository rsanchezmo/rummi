"""Independent torch implementation of the simulator, written against SPEC.md."""

from rummi.backends.torch_backend.sim import (
    TorchState,
    allocate,
    counts_of,
    derived_deck_orders,
    legal_actions,
    reset,
    reset_envs,
    step,
)

__all__ = [
    "TorchState",
    "allocate",
    "counts_of",
    "derived_deck_orders",
    "legal_actions",
    "reset",
    "reset_envs",
    "step",
]
