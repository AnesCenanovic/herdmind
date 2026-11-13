"""Perceptions as text: what an animal senses, rendered without advice.

``render`` and ``parse`` are inverses so the rules brain can read the very text
other brains read; it never gets a side channel into the world.

TODO(verify): the wording is a first draft. Changing it changes every brain's
input, so treat it as a shared contract once Jev logs exist.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Sighting:
    """Nearest thing of one kind within vision, e.g. ``wolf 2 cells north``."""

    kind: str
    distance: int
    direction: str


@dataclass(frozen=True)
class Perception:
    species: str
    energy: int
    grass_here: bool | None  # None for species that do not graze
    sightings: tuple[Sighting, ...] = ()

    def nearest(self, kind: str) -> Sighting | None:
        return next((s for s in self.sightings if s.kind == kind), None)


def render(p: Perception) -> str:
    lines = [f"species: {p.species}", f"energy: {p.energy}"]
    if p.grass_here is not None:
        lines.append(f"grass here: {'yes' if p.grass_here else 'no'}")
    for s in p.sightings:
        unit = "cell" if s.distance == 1 else "cells"
        lines.append(f"{s.kind} {s.distance} {unit} {s.direction}")
    return "\n".join(lines)


_SIGHTING = re.compile(r"^(\w+) (\d+) cells? (north|south|east|west)$")


def parse(text: str) -> Perception:
    species, energy, grass = "", 0, None
    sightings = []
    for line in text.splitlines():
        if line.startswith("species: "):
            species = line.removeprefix("species: ")
        elif line.startswith("energy: "):
            energy = int(line.removeprefix("energy: "))
        elif line.startswith("grass here: "):
            grass = line.removeprefix("grass here: ") == "yes"
        elif m := _SIGHTING.match(line):
            sightings.append(Sighting(m[1], int(m[2]), m[3]))
        else:
            raise ValueError(f"unparseable perception line: {line!r}")
    return Perception(species, energy, grass, tuple(sightings))
