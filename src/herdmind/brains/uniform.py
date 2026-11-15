"""Null baseline brain: every offered option is equally likely.

It shows what a world produces with no intelligence at all, so a broken world
config can be told apart from a weak model, and emergent behaviour only counts
when a brain clearly beats it. The world still does the sampling.
"""

from __future__ import annotations

from collections.abc import Sequence

from herdmind.brains.base import Decision
from herdmind.encoding.question import Question


class RandomBrain:
    """Uniform over ``Question.options``; reads nothing else."""

    name: str = "random"
    model_version: str = "random-1.0.0"

    def decide_batch(self, questions: Sequence[Question]) -> list[Decision]:
        return [
            Decision((1.0 / len(q.options),) * len(q.options), 0.0, self.name, self.model_version)
            for q in questions
        ]
