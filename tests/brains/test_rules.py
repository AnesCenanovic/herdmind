import numpy as np

from herdmind.brains.rules import RulesBrain
from herdmind.brains.testing import check_brain_contract
from herdmind.encoding.encode import encode
from herdmind.encoding.state import Perception, Sighting


def _qs():
    rng = np.random.default_rng(0)
    return [
        encode(1, Perception("sheep", 5, False, (Sighting("wolf", 2, "north"),)), rng),
        encode(2, Perception("wolf", 20, None, (Sighting("sheep", 3, "east"),)), rng),
    ]


def test_contract():
    check_brain_contract(RulesBrain(), _qs())


def test_sheep_flees_and_wolf_chases():
    sheep, wolf = RulesBrain().decide_batch(_qs())
    qs = _qs()
    assert qs[0].options[int(np.argmax(sheep.probs))] == "move south"
    assert qs[1].options[int(np.argmax(wolf.probs))] == "move east"
