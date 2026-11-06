import numpy as np

from herdmind.encoding.actions import ACTIONS_BY_SPECIES
from herdmind.encoding.encode import encode
from herdmind.encoding.state import Perception, Sighting, parse, render

P = Perception("sheep", 12, True, (Sighting("wolf", 2, "north"), Sighting("grass", 1, "east")))


def test_render_is_perceptions_only():
    assert render(P) == (
        "species: sheep\nenergy: 12\ngrass here: yes\nwolf 2 cells north\ngrass 1 cell east"
    )


def test_parse_inverts_render():
    assert parse(render(P)) == P


def test_encode_shuffles_but_keeps_option_set():
    labels = {a.value for a in ACTIONS_BY_SPECIES["sheep"]}
    orders = {encode(1, P, np.random.default_rng(s)).options for s in range(20)}
    assert len(orders) > 1
    assert all(set(o) == labels for o in orders)
