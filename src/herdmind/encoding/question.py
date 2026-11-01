"""The Question a brain is asked: perceptions plus runtime options.

Shared contract. Brains only ever see this type, so the world and the encoder
can change without touching them. Shuffling and un-shuffling happen in
``encoding/``; by the time a Question exists, ``options`` is already shuffled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Question:
    """One animal's decision problem for one world step.

    ``state_text`` holds perceptions only ("wolf 2 cells north"), never advice,
    otherwise we measure our prompt instead of the model.
    """

    agent_id: int
    species: Literal["wolf", "sheep"]
    state_text: str
    options: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.species not in ("wolf", "sheep"):
            raise ValueError(f"unknown species: {self.species!r}")
        # Duplicates would make mapping probabilities back to actions ambiguous.
        if len(self.options) < 2:
            raise ValueError(f"need at least 2 options, got {len(self.options)}")
        if len(set(self.options)) != len(self.options):
            raise ValueError(f"options must be unique: {self.options}")
