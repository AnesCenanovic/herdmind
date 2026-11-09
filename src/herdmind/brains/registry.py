"""Build brains from their config names, so nothing outside ``brains/`` imports one."""

from __future__ import annotations

from collections.abc import Mapping

from herdmind.brains.base import Brain
from herdmind.brains.rules import RulesBrain

# jev and decider register here once they exist.
_BRAINS = {"rules": RulesBrain}


def make_brain(name: str) -> Brain:
    """Build the brain registered as ``name``."""
    if name not in _BRAINS:
        raise ValueError(f"unknown brain {name!r}; known: {sorted(_BRAINS)}")
    return _BRAINS[name]()


def make_brains(by_species: Mapping[str, str]) -> dict[str, Brain]:
    """Map species to brains; species naming the same brain share one instance."""
    built = {name: make_brain(name) for name in set(by_species.values())}
    return {species: built[name] for species, name in by_species.items()}
