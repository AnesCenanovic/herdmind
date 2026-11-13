from pathlib import Path

import pytest

from herdmind.common.config import load_config
from herdmind.world.describe import briefing, describe_world
from herdmind.world.spec import LawsConfig, SpeciesTraits, WorldSpec

CONFIGS = Path(__file__).parents[2] / "configs"


@pytest.mark.parametrize("path", sorted(CONFIGS.glob("*.yaml")), ids=lambda p: p.name)
def test_every_shipped_config_loads(path):
    assert isinstance(load_config(path, WorldSpec), WorldSpec)


def test_partial_config_keeps_defaults():
    spec = load_config(CONFIGS / "pack-hunters.yaml", WorldSpec)
    assert spec.laws.predation == "pack" and spec.species.wolf.speed == 2
    assert spec.grid.width == 20 and spec.species.sheep.vision == 4  # defaults


def test_bad_values_are_rejected():
    with pytest.raises(ValueError, match="speed"):
        SpeciesTraits(count=1, start_energy=5, speed=0)
    with pytest.raises(ValueError, match="breed_chance"):
        SpeciesTraits(count=1, start_energy=5, breed_chance=1.5)
    with pytest.raises(ValueError, match="predation law"):
        LawsConfig(predation="merciful")


def test_briefing_states_laws_and_goal_as_facts():
    text = briefing(load_config(CONFIGS / "pack-hunters.yaml", WorldSpec), "sheep")
    assert "at least 2 wolves are within 1 cell" in text
    assert "70% of the time" in text
    assert "Wolves move up to 2 cells per step" in text
    assert text.endswith("You are a sheep. Your goal: stay alive and have offspring.")
    assert "should" not in text.lower()  # facts, never advice


def test_description_includes_what_animals_are_not_told():
    text = describe_world(WorldSpec())
    assert "40 sheep, 8 wolves" in text and "Layout seed 0, run seed 0" in text
    assert "always kills" in text
