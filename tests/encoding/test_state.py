import numpy as np

from herdmind.encoding.actions import actions_for
from herdmind.encoding.engine import encode
from herdmind.encoding.state import Perception, Sighting, parse, render

P = Perception("sheep", 12, True, (Sighting("wolf", 2, "north"), Sighting("grass", 1, "east")))


def test_render_is_perceptions_only():
    assert render(P) == (
        "species: sheep\nenergy: 12\ngrass here: yes\nwolf 2 cells north\ngrass 1 cell east"
    )


def test_parse_inverts_render():
    assert parse(render(P)) == P


def test_engine_shuffles_but_keeps_option_set_and_briefing():
    actions = actions_for("sheep", 1)
    labels = {a.label for a in actions}
    qs = [encode(1, P, actions, "facts", np.random.default_rng(s)) for s in range(20)]
    assert len({q.options for q in qs}) > 1
    assert all(set(q.options) == labels and q.briefing == "facts" for q in qs)


def test_engine_is_deterministic_for_a_seed():
    actions = actions_for("wolf", 2)
    a = encode(1, P, actions, "", np.random.default_rng(5))
    b = encode(1, P, actions, "", np.random.default_rng(5))
    assert a == b
