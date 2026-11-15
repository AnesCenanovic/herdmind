import numpy as np
import pytest

from herdmind.brains.registry import make_brain
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
    check_brain_contract(make_brain("random"), _qs())


def test_every_option_gets_one_over_k():
    qs = _qs()
    for q, d in zip(qs, make_brain("random").decide_batch(qs), strict=True):
        assert d.probs == pytest.approx([1 / len(q.options)] * len(q.options))
