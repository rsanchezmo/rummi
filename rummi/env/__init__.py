"""The environment: one subpackage per implementation, plus the Gymnasium wrapper.

:mod:`rummi.env.numpy` is the reference; :mod:`rummi.env.torch` and
:mod:`rummi.env.jax` are written independently against ``SPEC.md`` rather than
against a shared abstraction, so comparing them measures implementations and
not the cost of a common layer. :mod:`rummi.env.api` reconciles the three at
the boundary, which is what makes them swappable by name.
"""
