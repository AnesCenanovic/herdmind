"""Metabolism: how much energy an animal burns this step.

``apply(world, animal) -> int``. Uses the animal's ``moved`` (cells crossed this step).
"""

from __future__ import annotations

from herdmind.world.laws.base import Law


def _flat(world, a) -> int:
    return world.traits(a.species).metabolism


def _movement(world, a) -> int:
    return world.traits(a.species).metabolism + a.moved


KINDS = {
    "flat": Law(_flat, lambda s: "Animals burn the same energy every step, whatever they do."),
    "movement": Law(
        _movement,
        lambda s: (
            "Animals burn their usual energy every step, plus 1 energy for every cell they move."
        ),
    ),
}
