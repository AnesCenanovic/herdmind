import dataclasses

import pytest

from herdmind.brains.base import Decision
from herdmind.brains.registry import make_brain, make_brains
from herdmind.world.model import WorldModel
from herdmind.world.spec import Species, SpeciesTraits, WorldSpec

RULES = make_brains({"sheep": "rules", "wolf": "rules"})


class UniformBrain:
    """Fake brain: every option equally likely, so argmax would always pick option 0."""

    name = "uniform"

    def __init__(self) -> None:
        self.calls: list[int] = []

    def decide_batch(self, questions):
        self.calls.append(len(questions))
        out = []
        for q in questions:
            n = len(q.options)
            out.append(Decision((1 / n,) * n, 0.0, self.name, "v0"))
        return out


class ScriptedBrain:
    """Fake brain: probability 1 on a scripted label per agent, else on staying put."""

    name = "scripted"

    def __init__(self, script: dict[int, str]) -> None:
        self.script = script

    def decide_batch(self, questions):
        out = []
        for q in questions:
            want = self.script.get(q.agent_id, "stay in place")
            out.append(Decision(tuple(float(o == want) for o in q.options), 0.0, self.name, "v0"))
        return out


def _run(spec, steps=30, brains=RULES):
    w = WorldModel(spec, brains)
    for _ in range(steps):
        w.step()
    return w


def _positions(w):
    return sorted((a.species, a.x, a.y) for a in w.animals())


def test_same_seeds_same_history():
    assert _run(WorldSpec(seed=3)).history == _run(WorldSpec(seed=3)).history


def test_world_seed_fixes_the_map_and_seed_the_history():
    a, b = WorldSpec(world_seed=1, seed=1), WorldSpec(world_seed=1, seed=2)
    assert _positions(WorldModel(a, RULES)) == _positions(WorldModel(b, RULES))
    assert _run(a).history != _run(b).history
    other_map = WorldSpec(world_seed=2, seed=1)
    assert _positions(WorldModel(a, RULES)) != _positions(WorldModel(other_map, RULES))


def test_populations_are_sane():
    h = _run(WorldSpec(seed=1)).history
    g = WorldSpec().grid
    assert len(h) == 31
    assert all(
        r["sheep"] >= 0 and r["wolf"] >= 0 and 0 <= r["grass"] <= g.width * g.height for r in h
    )
    assert h[0] != h[-1]


def _solo_sheep(**traits):
    t = SpeciesTraits(count=1, start_energy=1000, breed_chance=0.0)
    return WorldSpec(
        species=Species(
            sheep=dataclasses.replace(t, **traits), wolf=dataclasses.replace(t, count=0)
        )
    )


def test_actions_are_sampled_not_argmax():
    b = UniformBrain()
    w = WorldModel(_solo_sheep(), {"sheep": b, "wolf": b})
    seen = set()
    for _ in range(40):
        a = w.animals("sheep")[0]
        before = (a.x, a.y)
        w.step()
        seen.add((a.x - before[0], a.y - before[1]))
    assert len(seen) > 2  # several different moves, not option 0 every time


def test_each_brain_gets_one_batch_per_step():
    sheep_b, wolf_b = UniformBrain(), UniformBrain()
    w = WorldModel(WorldSpec(), {"sheep": sheep_b, "wolf": wolf_b})
    w.step()
    assert sheep_b.calls == [40] and wolf_b.calls == [8]
    shared = UniformBrain()
    WorldModel(WorldSpec(), {"sheep": shared, "wolf": shared}).step()
    assert shared.calls == [48]


def test_sheep_eats_grass_and_it_regrows():
    b = UniformBrain()
    w = WorldModel(_solo_sheep(), {"sheep": b, "wolf": b})
    sheep = w.animals("sheep")[0]
    w.grass[sheep.y, sheep.x] = True
    w._graze(sheep)
    assert sheep.energy == 1006 and not w.grass[sheep.y, sheep.x]
    assert w.regrow[sheep.y, sheep.x] == w.spec.grid.grass_regrow_steps


def test_breeding_splits_energy_and_starvation_kills():
    b = UniformBrain()
    w = WorldModel(
        _solo_sheep(count=2, start_energy=10, breed_threshold=5, breed_chance=1.0),
        {"sheep": b, "wolf": b},
    )
    w.step()  # both sheep are above the threshold and breed with probability 1
    assert len(w.animals("sheep")) == 4
    starving = WorldModel(_solo_sheep(count=3, start_energy=1), {"sheep": b, "wolf": b})
    starving.grass[:] = False  # nothing to eat, so energy 1 -> 0 and they starve
    starving.step()
    assert starving.animals("sheep") == []


def test_bare_cells_regrow():
    w = WorldModel(WorldSpec(world_seed=2), RULES)
    bare = ~w.grass
    assert bare.any() and (w.regrow[bare] > 0).all()


def test_make_brain_rejects_unknown_name():
    with pytest.raises(ValueError):
        make_brain("nope")
