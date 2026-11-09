"""Breeding: whether an animal has one offspring this step.

``apply(world, animal) -> bool``. The world then gives the child half the parent's energy.
"""

from __future__ import annotations

from herdmind.world.laws.base import Law
from herdmind.world.perception import count_near


def _split(world, a) -> bool:
    t = world.traits(a.species)
    return a.energy >= t.breed_threshold and world.rng.random() < t.breed_chance


def _needs_mate(world, a) -> bool:
    has_mate = count_near(world, a.species, a.x, a.y, radius=1, exclude=a) > 0
    return has_mate and _split(world, a)


_CHILD = "The offspring takes half of the parent's energy."
KINDS = {
    "split": Law(
        _split,
        lambda s: "An animal with enough energy may have one offspring on its own. " + _CHILD,
    ),
    "needs_mate": Law(
        _needs_mate,
        lambda s: (
            "An animal with enough energy may have one offspring, but only while "
            "another animal of its kind is within 1 cell. " + _CHILD
        ),
    ),
}
