import pytest

from herdmind.encoding.actions import Action, actions_for, from_label


def test_option_counts_grow_with_speed():
    assert len(actions_for("sheep", 1)) == 6  # 4 moves + stay + eat
    assert len(actions_for("wolf", 2)) == 9  # 8 moves + stay, wolves do not graze


def test_labels_round_trip():
    for a in actions_for("sheep", 3):
        assert from_label(a.label) == a
    assert Action("move", "north", 2).label == "move north 2 cells"
    assert Action("move", "west", 2).delta == (-2, 0)


def test_unknown_label_is_an_error():
    with pytest.raises(ValueError):
        from_label("run away")
