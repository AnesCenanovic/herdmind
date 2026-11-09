"""Disease: whether an animal dies of illness this step.

``apply(world, animal) -> bool``: True if it dies.
"""

from __future__ import annotations

from herdmind.world.laws.base import Law, pct
from herdmind.world.perception import count_near


def _none(world, a) -> bool:
    return False


def _fatal(world, a) -> bool:
    return world.rng.random() < world.traits(a.species).disease_chance


def _contagious(world, a) -> bool:
    neighbours = count_near(world, a.species, a.x, a.y, radius=1, exclude=a)
    chance = min(1.0, world.traits(a.species).disease_chance * (1 + neighbours))
    return world.rng.random() < chance


def _chances(s) -> str:
    return (
        f"{pct(s.species.sheep.disease_chance)} for a sheep and "
        f"{pct(s.species.wolf.disease_chance)} for a wolf"
    )


KINDS = {
    "none": Law(_none, lambda s: "There is no disease."),
    "fatal": Law(
        _fatal, lambda s: f"Each step, an animal dies of disease with a chance of {_chances(s)}."
    ),
    "contagious": Law(
        _contagious,
        lambda s: (
            f"Each step, an animal dies of disease with a base chance of {_chances(s)}; "
            "the chance grows with every animal of its kind within 1 cell."
        ),
    ),
}
