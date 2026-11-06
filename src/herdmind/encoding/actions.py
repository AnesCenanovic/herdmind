"""The action vocabulary and the neutral labels shown to brains as options.

Labels describe what happens, never what is wise. The world maps a chosen
option back to an ``Action`` by label, so shuffled order cannot corrupt it.
"""

from __future__ import annotations

from enum import Enum


class Action(Enum):
    NORTH = "move north"
    SOUTH = "move south"
    EAST = "move east"
    WEST = "move west"
    STAY = "stay in place"
    EAT = "eat grass here"


# y grows southwards, as on a screen.
DELTAS: dict[Action, tuple[int, int]] = {
    Action.NORTH: (0, -1),
    Action.SOUTH: (0, 1),
    Action.EAST: (1, 0),
    Action.WEST: (-1, 0),
    Action.STAY: (0, 0),
    Action.EAT: (0, 0),
}

_MOVES = (Action.NORTH, Action.SOUTH, Action.EAST, Action.WEST, Action.STAY)
ACTIONS_BY_SPECIES: dict[str, tuple[Action, ...]] = {
    "sheep": (*_MOVES, Action.EAT),
    "wolf": _MOVES,  # a wolf eats by moving onto a sheep
}


def from_label(label: str) -> Action:
    """Map an option label back to its ``Action``."""
    return Action(label)
