import pytest

from herdmind.brains.base import Decision
from herdmind.brains.registry import make_brain
from herdmind.common.seeding import make_rng
from herdmind.encoding.actions import Action
from herdmind.world.config import ExperimentConfig, SpeciesConfig, WorldConfig
from herdmind.world.model import WorldModel


def _run(seed, steps=30):
    w = WorldModel(ExperimentConfig(seed=seed), make_brain("rules"), make_rng(seed))
    for _ in range(steps):
        w.step()
    return w.history


def test_same_seed_same_history():
    assert _run(3) == _run(3)


def test_populations_are_sane():
    h = _run(1)
    assert len(h) == 31
    assert all(r["sheep"] >= 0 and r["wolf"] >= 0 and 0 <= r["grass"] <= 20 * 20 for r in h)
    assert h[0] != h[-1]


def test_bare_cells_regrow():
    w = WorldModel(ExperimentConfig(seed=2), make_brain("rules"), make_rng(2))
    bare = ~w.grass
    assert bare.any() and (w.regrow[bare] > 0).all()


class _FixedBrain:
    """Fake brain: every option equally likely, so argmax would always pick option 0."""

    name = "fixed"

    def decide_batch(self, questions):
        return [
            Decision((1 / len(q.options),) * len(q.options), 0.0, "fixed", "v0") for q in questions
        ]


def _world(brain, **world):
    cfg = ExperimentConfig(world=WorldConfig(**world))
    return WorldModel(cfg, brain, make_rng(0))


def test_actions_are_sampled_not_argmax():
    w = _world(
        _FixedBrain(),
        sheep=SpeciesConfig(1, 1000, 6, 9999, 0.0),
        wolf=SpeciesConfig(0, 10, 15, 9999, 0.0),
    )
    seen = set()
    for _ in range(40):
        a = w.animals("sheep")[0]
        before = (a.x, a.y)
        w.step()
        seen.add((a.x - before[0]) % 20 + 100 * ((a.y - before[1]) % 20))
    assert len(seen) > 2  # several different moves, not option 0 every time


def test_wolf_eats_sheep_on_its_cell():
    w = _world(
        _FixedBrain(),
        sheep=SpeciesConfig(1, 10, 6, 9999, 0.0),
        wolf=SpeciesConfig(1, 20, 15, 9999, 0.0),
    )
    sheep, wolf = w.animals("sheep")[0], w.animals("wolf")[0]
    sheep.x, sheep.y, wolf.x, wolf.y = 5, 5, 5, 5
    w._apply(wolf, Action.STAY)
    assert not sheep.alive and w.animals("sheep") == [] and wolf.energy == 20 + 15


def test_sheep_eats_grass_and_it_regrows():
    w = _world(
        _FixedBrain(),
        sheep=SpeciesConfig(1, 10, 6, 9999, 0.0),
        wolf=SpeciesConfig(0, 20, 15, 9999, 0.0),
    )
    sheep = w.animals("sheep")[0]
    w.grass[sheep.y, sheep.x] = True
    w._apply(sheep, Action.EAT)
    assert sheep.energy == 16 and not w.grass[sheep.y, sheep.x]
    assert w.regrow[sheep.y, sheep.x] == w.cfg.grass_regrow_steps


def test_breeding_splits_energy_and_starvation_kills():
    w = _world(
        _FixedBrain(),
        sheep=SpeciesConfig(2, 10, 6, 5, 1.0),
        wolf=SpeciesConfig(0, 20, 15, 9999, 0.0),
    )
    w.step()  # both sheep are above the threshold and breed with probability 1
    assert len(w.animals("sheep")) == 4
    starving = _world(
        _FixedBrain(),
        sheep=SpeciesConfig(3, 1, 6, 9999, 0.0),
        wolf=SpeciesConfig(0, 20, 15, 9999, 0.0),
    )
    starving.grass[:] = False  # nothing to eat, so energy 1 -> 0 and they starve
    starving.step()
    assert starving.animals("sheep") == []


def test_make_brain_rejects_unknown_name():
    with pytest.raises(ValueError):
        make_brain("nope")
