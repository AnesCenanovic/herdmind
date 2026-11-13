"""The question engine: the one place that turns what an animal senses into a Question.

Every brain sees Questions built here and only here, so a change in how
questions are formed is a change for all brains at once. Given the same
perception, actions, briefing and RNG state it always builds the same Question.
Option order is re-drawn for every query because decision models have
first-option bias; mapping back needs no permutation since labels are unique.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from herdmind.encoding.actions import Action
from herdmind.encoding.question import Question
from herdmind.encoding.state import Perception, render


def encode(
    agent_id: int,
    p: Perception,
    actions: Sequence[Action],
    briefing: str,
    rng: np.random.Generator,
) -> Question:
    """Build one animal's Question with its options in a fresh random order."""
    labels = [a.label for a in actions]
    order = rng.permutation(len(labels))
    return Question(
        agent_id=agent_id,
        species=p.species,  # type: ignore[arg-type]
        state_text=render(p),
        options=tuple(labels[i] for i in order),
        briefing=briefing,
    )
