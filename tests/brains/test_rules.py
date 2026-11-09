import numpy as np

from herdmind.brains.registry import make_brains
from herdmind.brains.rules import RulesBrain
from herdmind.brains.testing import check_brain_contract
from herdmind.encoding.actions import actions_for
from herdmind.encoding.engine import encode
from herdmind.encoding.state import Perception, Sighting


def _qs():
    rng = np.random.default_rng(0)
    sheep = Perception("sheep", 5, False, (Sighting("wolf", 2, "north"),))
    wolf = Perception("wolf", 20, None, (Sighting("sheep", 3, "east"),))
    return [
        encode(1, sheep, actions_for("sheep", 1), "", rng),
        encode(2, wolf, actions_for("wolf", 2), "", rng),
    ]


def test_contract():
    check_brain_contract(RulesBrain(), _qs())


def test_sheep_flees_and_wolf_chases():
    qs = _qs()
    sheep, wolf = RulesBrain().decide_batch(qs)
    assert qs[0].options[int(np.argmax(sheep.probs))] == "move south 1 cell"
    top = max(wolf.probs)
    best = {o for o, p in zip(qs[1].options, wolf.probs, strict=True) if p == top}
    assert best == {"move east 1 cell", "move east 2 cells"}


def test_species_naming_one_brain_share_it():
    brains = make_brains({"sheep": "rules", "wolf": "rules"})
    assert brains["sheep"] is brains["wolf"]
