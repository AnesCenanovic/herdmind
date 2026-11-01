"""The Brain protocol and its Decision result.

Shared contract. A brain returns probabilities, never an action; the world
samples with the run's RNG. All brains (rules, jev, decider) return the same
type so they can be swapped by config alone.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from herdmind.encoding.question import Question

PROB_SUM_TOL = 1e-6


@dataclass(frozen=True)
class Decision:
    """A distribution over ``Question.options`` (same order) plus provenance."""

    probs: tuple[float, ...]
    latency_s: float
    brain: str
    model_version: str
    cached: bool = False

    def __post_init__(self) -> None:
        # A bad distribution must fail here, not silently skew sampling later.
        if not all(math.isfinite(p) and p >= 0 for p in self.probs):
            raise ValueError(f"probs must be finite and non-negative: {self.probs}")
        if abs(sum(self.probs) - 1.0) > PROB_SUM_TOL:
            raise ValueError(f"probs must sum to 1 (+-{PROB_SUM_TOL}): sum={sum(self.probs)}")


class Brain(Protocol):
    """Decision model. ``decide_batch`` is the only entry point (one world step)."""

    name: str

    def decide_batch(self, questions: Sequence[Question]) -> list[Decision]: ...
