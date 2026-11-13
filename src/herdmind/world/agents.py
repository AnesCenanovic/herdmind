"""The beings in the world."""

from __future__ import annotations

from typing import TYPE_CHECKING

import mesa

if TYPE_CHECKING:
    from herdmind.world.model import WorldModel


class Animal(mesa.Agent):
    """One wolf or sheep.

    ``moved`` is the cells crossed this step, which a metabolism law may charge for.
    """

    def __init__(self, model: WorldModel, species: str, x: int, y: int, energy: int) -> None:
        super().__init__(model)
        self.species, self.x, self.y, self.energy = species, x, y, energy
        self.moved = 0
