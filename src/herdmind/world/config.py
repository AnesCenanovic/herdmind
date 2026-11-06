"""Experiment config: frozen dataclasses, loaded from one YAML per experiment."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SpeciesConfig:
    initial: int
    start_energy: int
    energy_gain: int  # per grass eaten (sheep) or sheep eaten (wolf)
    breed_threshold: int
    breed_prob: float


@dataclass(frozen=True)
class WorldConfig:
    width: int = 20
    height: int = 20
    grass_fraction: float = 0.5
    grass_regrow_steps: int = 10
    vision_radius: int = 4
    sheep: SpeciesConfig = field(default_factory=lambda: SpeciesConfig(40, 10, 6, 20, 0.1))
    wolf: SpeciesConfig = field(default_factory=lambda: SpeciesConfig(8, 20, 15, 40, 0.05))


@dataclass(frozen=True)
class ExperimentConfig:
    name: str = "micro"
    seed: int = 0
    steps: int = 100
    brain: str = "rules"
    world: WorldConfig = field(default_factory=WorldConfig)
