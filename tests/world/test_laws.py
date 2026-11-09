"""Each law variant on a tiny hand-placed world."""

import dataclasses

from herdmind.encoding.actions import Action
from herdmind.world.agents import Animal
from herdmind.world.model import WorldModel
from herdmind.world.spec import LawsConfig, Species, SpeciesTraits, WorldSpec
from tests.world.test_model import ScriptedBrain, UniformBrain


def _world(laws=None, sheep=None, wolf=None):
    laws = laws or LawsConfig()
    sp = Species(
        sheep=sheep or SpeciesTraits(count=0, start_energy=10, breed_chance=0.0),
        wolf=wolf or SpeciesTraits(count=0, start_energy=20, energy_gain=15, breed_chance=0.0),
    )
    b = UniformBrain()
    return WorldModel(WorldSpec(species=sp, laws=laws), {"sheep": b, "wolf": b})


def _place(w, species, x, y, energy=10):
    return Animal(w, species, x, y, energy)


def test_always_kill():
    w = _world()
    sheep, wolf = _place(w, "sheep", 5, 5), _place(w, "wolf", 5, 5, 20)
    w._hunt(wolf)
    assert not sheep.alive and wolf.energy == 35


def test_chance_with_zero_success_never_kills():
    wolf_t = SpeciesTraits(count=0, start_energy=20, predation_success=0.0, breed_chance=0.0)
    w = _world(LawsConfig(predation="chance"), wolf=wolf_t)
    sheep, wolf = _place(w, "sheep", 5, 5), _place(w, "wolf", 5, 5, 20)
    w._hunt(wolf)
    assert sheep.alive


def test_pack_needs_enough_wolves():
    w = _world(LawsConfig(predation="pack", pack_size=2))
    sheep, wolf = _place(w, "sheep", 5, 5), _place(w, "wolf", 5, 5, 20)
    w._hunt(wolf)
    assert sheep.alive  # alone
    _place(w, "wolf", 6, 6, 20)  # diagonal neighbour counts
    w._hunt(wolf)
    assert not sheep.alive


def test_needs_mate_breeding():
    eager = SpeciesTraits(count=0, start_energy=10, breed_threshold=5, breed_chance=1.0)
    w = _world(LawsConfig(breeding="needs_mate"), sheep=eager)
    w.grass[:] = False
    lone = _place(w, "sheep", 0, 0)
    assert not w.laws.breeding.apply(w, lone)
    _place(w, "sheep", 1, 0)
    assert w.laws.breeding.apply(w, lone)


def test_fatal_disease_and_movement_metabolism():
    sick = SpeciesTraits(count=0, start_energy=10, disease_chance=1.0, breed_chance=0.0)
    w = _world(LawsConfig(disease="fatal", metabolism="movement"), sheep=sick)
    a = _place(w, "sheep", 0, 0)
    assert w.laws.disease.apply(w, a)
    a.moved = 2
    assert w.laws.metabolism.apply(w, a) == 3  # 1 per step + 1 per cell


def test_speed_moves_several_cells():
    fast = dataclasses.replace(SpeciesTraits(count=0, start_energy=20), speed=3)
    w = _world(wolf=fast)
    wolf = _place(w, "wolf", 0, 0, 20)
    w._move(wolf, Action("move", "west", 3))
    assert (wolf.x, wolf.y, wolf.moved) == (17, 0, 3)


def test_pack_counts_wolves_where_they_end_up():
    """A wolf next to the sheep that walks away is no pack member, whatever the turn order."""
    for b_choice, survives in (("move east 1 cell", True), ("stay in place", False)):
        w = _world(LawsConfig(predation="pack", pack_size=2))
        w.grass[:] = False
        sheep = _place(w, "sheep", 5, 5)
        a, b = _place(w, "wolf", 5, 5, 20), _place(w, "wolf", 6, 6, 20)
        w.brains = dict.fromkeys(("sheep", "wolf"), ScriptedBrain({b.unique_id: b_choice}))
        w.step()
        assert sheep.alive is survives and a.alive
