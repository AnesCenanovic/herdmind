"""Perception -> Question, with option order shuffled by the run's RNG.

Decision models have first-option bias, so order is re-drawn for every query.
Mapping back needs no permutation: options are unique labels (see ``actions``).
"""

from __future__ import annotations

import numpy as np

from herdmind.encoding.actions import ACTIONS_BY_SPECIES
from herdmind.encoding.question import Question
from herdmind.encoding.state import Perception, render


def encode(agent_id: int, p: Perception, rng: np.random.Generator) -> Question:
    labels = [a.value for a in ACTIONS_BY_SPECIES[p.species]]
    order = rng.permutation(len(labels))
    return Question(
        agent_id=agent_id,
        species=p.species,  # type: ignore[arg-type]
        state_text=render(p),
        options=tuple(labels[i] for i in order),
    )
