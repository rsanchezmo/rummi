"""Every module imports.

A trivial-looking test that pays for itself: `rummi/bench/bench_render.py` once
imported a name another module had stopped exporting, and nothing in the suite
noticed -- the CLI entry points are exercised by hand, not by tests.

Missing *optional extras* are skipped rather than failed, so this stays honest on
a checkout installed without torch, jax or pygame. Anything else that fails to
import is a broken module.
"""

import importlib
import pkgutil

import pytest

import rummi

OPTIONAL = {"torch", "jax", "jaxlib", "pygame", "ortools", "gymnasium", "moviepy", "PIL"}

MODULES = sorted(
    m.name for m in pkgutil.walk_packages(rummi.__path__, prefix="rummi.")
    if not m.ispkg
)


def test_the_walk_found_the_modules():
    """Guards the guard: a bad prefix would make this file vacuously pass."""
    assert len(MODULES) > 20, MODULES
    assert "rummi.bench.bench_render" in MODULES


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name: str):
    try:
        importlib.import_module(name)
    except ModuleNotFoundError as exc:
        root = (exc.name or "").split(".")[0]
        if root in OPTIONAL:
            pytest.skip(f"needs the optional {root!r}")
        raise
