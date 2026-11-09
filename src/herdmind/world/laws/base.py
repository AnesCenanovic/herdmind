"""The shape every law shares: what it does, and how to say it in English."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Law:
    """One named rule.

    ``apply`` takes the world first; its other arguments depend on the family.
    ``describe`` turns the world spec into one plain sentence, stated as a fact.
    """

    apply: Callable[..., Any]
    describe: Callable[[Any], str]


def pct(p: float) -> str:
    return f"{p:.0%}"
