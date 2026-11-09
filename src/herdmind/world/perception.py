"""What an animal senses, and the torus geometry behind it.

Perception is the world's half of the question engine: it decides *what* an
animal knows; ``encoding/`` decides how that is written down.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from herdmind.encoding.state import Perception, Sighting

if TYPE_CHECKING:
    from herdmind.world.agents import Animal
    from herdmind.world.model import WorldModel

_KIND_ORDER = {"wolf": 0, "sheep": 1, "grass": 2}


def wrap_delta(a: int, b: int, size: int) -> int:
    """Shortest signed step from a to b on a ring of ``size``."""
    d = (b - a) % size
    return d - size if d > size // 2 else d


def direction(dx: int, dy: int) -> str:
    # Lossy on purpose: diagonals collapse onto the longer axis; ties go vertical.
    if abs(dy) >= abs(dx):
        return "south" if dy > 0 else "north"
    return "east" if dx > 0 else "west"


def count_near(
    world: WorldModel,
    species: str,
    x: int,
    y: int,
    radius: int = 1,
    exclude: Animal | None = None,
) -> int:
    """Animals of ``species`` within ``radius`` cells (Chebyshev, so diagonals count)."""
    g = world.spec.grid
    return sum(
        1
        for o in world.animals(species)
        if o is not exclude
        and max(abs(wrap_delta(x, o.x, g.width)), abs(wrap_delta(y, o.y, g.height))) <= radius
    )


def perceive(world: WorldModel, a: Animal) -> Perception:
    """Nearest thing of each kind within the animal's vision (Manhattan distance)."""
    g, r = world.spec.grid, world.traits(a.species).vision
    best: dict[str, Sighting] = {}

    def consider(kind: str, x: int, y: int) -> None:
        dx, dy = wrap_delta(a.x, x, g.width), wrap_delta(a.y, y, g.height)
        dist = abs(dx) + abs(dy)
        if 0 < dist <= r and (kind not in best or dist < best[kind].distance):
            best[kind] = Sighting(kind, dist, direction(dx, dy))

    for o in world.animals():
        if o is not a:
            consider(o.species, o.x, o.y)
    if a.species == "sheep":
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                x, y = (a.x + dx) % g.width, (a.y + dy) % g.height
                if world.grass[y, x]:
                    consider("grass", x, y)
    sightings = tuple(sorted(best.values(), key=lambda s: _KIND_ORDER[s.kind]))
    grass_here = bool(world.grass[a.y, a.x]) if a.species == "sheep" else None
    return Perception(a.species, a.energy, grass_here, sightings)
