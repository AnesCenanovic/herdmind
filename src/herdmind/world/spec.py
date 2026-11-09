"""The world declaration: one YAML file describes a whole world.

Kept in separate layers on purpose, so each can change without touching the others:

- ``grid``: the environment (size, grass).
- ``species``: traits, numbers that say *how much* (energy, speed, chances).
- ``laws``: named rules that say *what happens* (does a wolf always kill?).
- ``brains``: which decision model answers for each species.

Laws read traits, so numbers stay tweakable and logic stays swappable. Every
part renders to English (``world/describe.py``) for people and for brains.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields

from herdmind.world.laws import FAMILIES

SPECIES = ("sheep", "wolf")
# Options grow as 4 * speed + 2; speed 3 gives a sheep 14 options.
MAX_SPEED = 3


@dataclass(frozen=True)
class GridParams:
    width: int = 20
    height: int = 20
    grass_fraction: float = 0.5  # share of cells with grass at the start
    grass_regrow_steps: int = 10

    def __post_init__(self) -> None:
        if self.width < 1 or self.height < 1 or self.grass_regrow_steps < 1:
            raise ValueError("width, height and grass_regrow_steps must be at least 1")
        if not 0.0 <= self.grass_fraction <= 1.0:
            raise ValueError(f"grass_fraction must be in [0, 1], got {self.grass_fraction}")


@dataclass(frozen=True)
class SpeciesTraits:
    """Per-species numbers. Laws decide which of them matter in a given world."""

    count: int
    start_energy: int
    metabolism: int = 1  # energy burned per step
    speed: int = 1  # most cells crossed in one step
    vision: int = 4  # how far it sees (cells, Manhattan)
    energy_gain: int = 6  # per grass eaten (sheep) or sheep eaten (wolf)
    breed_threshold: int = 20  # "eagerness": energy needed before breeding
    breed_chance: float = 0.1  # per step, once over the threshold
    disease_chance: float = 0.0  # per step; used only by a disease law
    predation_success: float = 1.0  # per kill attempt; used only by predators
    # Shown verbatim to brains: keep it an end ("stay alive"), never a strategy
    # ("avoid wolves"), or we measure our prompt instead of the model.
    goal: str = "stay alive and have offspring"

    def __post_init__(self) -> None:
        if not 1 <= self.speed <= MAX_SPEED:
            raise ValueError(f"speed must be 1..{MAX_SPEED}, got {self.speed}")
        for name in ("breed_chance", "disease_chance", "predation_success"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {getattr(self, name)}")
        if self.count < 0 or self.start_energy <= 0 or self.vision < 0:
            raise ValueError("count >= 0, start_energy > 0 and vision >= 0 are required")
        if self.breed_threshold < 2:  # below 2 a child would be born with 0 energy
            raise ValueError(f"breed_threshold must be at least 2, got {self.breed_threshold}")


def _sheep() -> SpeciesTraits:
    return SpeciesTraits(count=40, start_energy=10, energy_gain=6, breed_threshold=20)


def _wolf() -> SpeciesTraits:
    return SpeciesTraits(
        count=8, start_energy=20, energy_gain=15, breed_threshold=40, breed_chance=0.05
    )


@dataclass(frozen=True)
class Species:
    sheep: SpeciesTraits = field(default_factory=_sheep)
    wolf: SpeciesTraits = field(default_factory=_wolf)

    def of(self, species: str) -> SpeciesTraits:
        return getattr(self, species)


@dataclass(frozen=True)
class LawsConfig:
    """Which variant of each law family holds. Kinds live in ``world/laws/``."""

    predation: str = "always_kill"
    pack_size: int = 2  # used by predation: pack
    breeding: str = "split"
    disease: str = "none"
    metabolism: str = "flat"

    def __post_init__(self) -> None:
        if self.pack_size < 1:
            raise ValueError(f"pack_size must be at least 1, got {self.pack_size}")
        for family, kinds in FAMILIES.items():
            kind = getattr(self, family)
            if kind not in kinds:
                raise ValueError(f"unknown {family} law {kind!r}; known: {sorted(kinds)}")


@dataclass(frozen=True)
class BrainsConfig:
    sheep: str = "rules"
    wolf: str = "rules"


@dataclass(frozen=True)
class WorldSpec:
    """A whole world. ``world_seed`` fixes the starting map, ``seed`` everything after."""

    name: str = "micro"
    description: str = ""
    steps: int = 100
    seed: int = 0
    world_seed: int = 0
    grid: GridParams = field(default_factory=GridParams)
    species: Species = field(default_factory=Species)
    laws: LawsConfig = field(default_factory=LawsConfig)
    brains: BrainsConfig = field(default_factory=BrainsConfig)

    def brain_names(self) -> dict[str, str]:
        return {f.name: getattr(self.brains, f.name) for f in fields(self.brains)}
