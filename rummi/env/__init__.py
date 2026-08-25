"""The environment: one subpackage per implementation, plus the Gymnasium wrapper.

:mod:`rummi.env.numpy` is the reference; :mod:`rummi.env.torch` and
:mod:`rummi.env.jax` are written independently against ``SPEC.md`` rather than
against a shared abstraction, so comparing them measures implementations and
not the cost of a common layer. :mod:`rummi.env.api` reconciles the three at
the boundary, which is what makes them swappable by name.
"""

from rummi.rules.config import STANDARD, STANDARD_3P, STANDARD_4P, RummiConfig

ENV_CONFIGS: dict[str, RummiConfig] = {
    "Rummi-2p-v0": STANDARD,
    "Rummi-3p-v0": STANDARD_3P,
    "Rummi-4p-v0": STANDARD_4P,
}
"""Registry ids, one per seat count of the standard deck."""


def register_envs() -> None:
    """Register :data:`ENV_CONFIGS` with Gymnasium, if Gymnasium is installed.

    Only a ``vector_entry_point`` is registered: there is no single-env
    implementation to fall back on, so ``gymnasium.make("Rummi-2p-v0")`` should
    fail loudly rather than silently wrapping a batch of one. The entry point
    stays a string so importing :mod:`rummi.env` does not pull in pygame.
    """
    try:
        import gymnasium
    except ModuleNotFoundError:  # the Gymnasium extra is optional
        return

    for env_id, cfg in ENV_CONFIGS.items():
        if env_id in gymnasium.registry:
            continue
        gymnasium.register(
            env_id,
            vector_entry_point="rummi.env.vector_env:RummiVectorEnv",
            kwargs={"cfg": cfg},
        )


register_envs()
