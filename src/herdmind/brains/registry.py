"""Build a brain from its config name, so nothing outside ``brains/`` imports one."""

from __future__ import annotations

from herdmind.brains.base import Brain
from herdmind.brains.rules import RulesBrain

# jev and decider register here once they exist.
_BRAINS = {"rules": RulesBrain}


def make_brain(name: str) -> Brain:
    if name not in _BRAINS:
        raise ValueError(f"unknown brain {name!r}; known: {sorted(_BRAINS)}")
    return _BRAINS[name]()
