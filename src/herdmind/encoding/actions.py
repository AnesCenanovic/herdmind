"""The action vocabulary and the neutral labels shown to brains as options.

Labels describe what happens, never what is wise. The world maps a chosen
option back to an ``Action`` by label, so shuffled order cannot corrupt it.
Speed shows up as distance: a speed-2 wolf gets "move north 1 cell" and
"move north 2 cells".
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

# y grows southwards, as on a screen.
DIRECTIONS: dict[str, tuple[int, int]] = {
    "north": (0, -1),
    "south": (0, 1),
    "east": (1, 0),
    "west": (-1, 0),
}
STAY_LABEL = "stay in place"
EAT_LABEL = "eat grass here"
GRAZERS = ("sheep",)  # a wolf eats by moving onto a sheep


@dataclass(frozen=True)
class Action:
    kind: Literal["move", "stay", "eat"]
    direction: str = ""
    distance: int = 0

    @property
    def label(self) -> str:
        if self.kind == "move":
            unit = "cell" if self.distance == 1 else "cells"
            return f"move {self.direction} {self.distance} {unit}"
        return STAY_LABEL if self.kind == "stay" else EAT_LABEL

    @property
    def delta(self) -> tuple[int, int]:
        if self.kind != "move":
            return (0, 0)
        dx, dy = DIRECTIONS[self.direction]
        return (dx * self.distance, dy * self.distance)


def actions_for(species: str, speed: int) -> tuple[Action, ...]:
    """Every action an animal of ``species`` with ``speed`` may take, in a fixed order."""
    moves = [Action("move", d, n) for d in DIRECTIONS for n in range(1, speed + 1)]
    extra = [Action("eat")] if species in GRAZERS else []
    return (*moves, Action("stay"), *extra)


_MOVE = re.compile(r"^move (north|south|east|west) (\d+) cells?$")


def from_label(label: str) -> Action:
    """Map an option label back to its ``Action``; unknown labels are an error."""
    if label == STAY_LABEL:
        return Action("stay")
    if label == EAT_LABEL:
        return Action("eat")
    if m := _MOVE.match(label):
        return Action("move", m[1], int(m[2]))
    raise ValueError(f"unknown action label: {label!r}")
