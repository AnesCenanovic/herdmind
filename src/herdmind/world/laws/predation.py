"""Predation: what happens when a wolf ends its move on a sheep's cell.

``apply(world, wolf, prey) -> bool``: True if the sheep dies.
"""

from __future__ import annotations

from herdmind.world.laws.base import Law, pct
from herdmind.world.perception import count_near


def _always_kill(world, wolf, prey) -> bool:
    return True


def _chance(world, wolf, prey) -> bool:
    return world.rng.random() < world.traits("wolf").predation_success


def _pack(world, wolf, prey) -> bool:
    wolves = count_near(world, "wolf", prey.x, prey.y, radius=1)
    return wolves >= world.spec.laws.pack_size and _chance(world, wolf, prey)


KINDS = {
    "always_kill": Law(
        _always_kill, lambda s: "A wolf that moves onto a sheep's cell always kills and eats it."
    ),
    "chance": Law(
        _chance,
        lambda s: (
            "A wolf that moves onto a sheep's cell kills and eats it "
            f"{pct(s.species.wolf.predation_success)} of the time."
        ),
    ),
    "pack": Law(
        _pack,
        lambda s: (
            "A wolf that moves onto a sheep's cell can only kill it when at least "
            f"{s.laws.pack_size} wolves are within 1 cell of the sheep, and then succeeds "
            f"{pct(s.species.wolf.predation_success)} of the time."
        ),
    ),
}
