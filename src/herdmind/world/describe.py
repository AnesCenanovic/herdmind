"""The world in plain English, for people (UI, run folders) and for brains.

Facts only, never strategy: "wolves always kill a sheep they reach" is a law;
"you should flee" is advice and would mean we measure our prompt, not the model.
"""

from __future__ import annotations

from herdmind.world.laws import describe_laws
from herdmind.world.laws.base import pct
from herdmind.world.spec import SPECIES, SpeciesTraits, WorldSpec

PLURAL = {"sheep": "sheep", "wolf": "wolves"}
_FOOD = {"sheep": "grass", "wolf": "sheep"}


def _cells(n: int) -> str:
    return f"{n} cell" if n == 1 else f"{n} cells"


def describe_species(species: str, t: SpeciesTraits) -> list[str]:
    name = PLURAL[species].capitalize()
    return [
        f"{name} start with {t.start_energy} energy and burn {t.metabolism} energy per step.",
        f"{name} move up to {_cells(t.speed)} per step in a straight line and see up to "
        f"{_cells(t.vision)} away, counted along rows and columns.",
        f"{name} gain {t.energy_gain} energy for each {_FOOD[species]} they eat.",
        f"{name} with at least {t.breed_threshold} energy breed with a "
        f"{pct(t.breed_chance)} chance per step.",
    ]


def _world_facts(spec: WorldSpec) -> list[str]:
    g = spec.grid
    return [
        f"The world is a {g.width} by {g.height} grid whose edges wrap around.",
        f"Sheep eat grass on their own cell; eaten grass grows back after "
        f"{g.grass_regrow_steps} steps.",
        "An animal whose energy reaches 0 dies.",
        "Each step, all animals move at once; then sheep eat, then wolves hunt.",
        '"Within 1 cell" includes the diagonal cells.',
        *describe_laws(spec),
        *(line for s in SPECIES for line in describe_species(s, spec.species.of(s))),
    ]


def briefing(spec: WorldSpec, species: str) -> str:
    """What every animal of ``species`` is told about its world and its goal."""
    goal = spec.species.of(species).goal
    return "\n".join([*_world_facts(spec), f"You are a {species}. Your goal: {goal}."])


def describe_world(spec: WorldSpec) -> str:
    """The full declaration in English, including what the animals are not told."""
    counts = ", ".join(f"{spec.species.of(s).count} {PLURAL[s]}" for s in SPECIES)
    header = [f"World: {spec.name}"]
    if spec.description:
        header.append(spec.description)
    return "\n".join(
        [
            *header,
            f"It starts with {counts} and runs for {spec.steps} steps.",
            f"Layout seed {spec.world_seed}, run seed {spec.seed}.",
            f"Brains: sheep use {spec.brains.sheep}, wolves use {spec.brains.wolf}.",
            *_world_facts(spec),
            *(f"Goal of {PLURAL[s]}: {spec.species.of(s).goal}." for s in SPECIES),
        ]
    )
