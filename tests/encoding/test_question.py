from __future__ import annotations

import pytest

from herdmind.encoding.question import Question


def make(options: tuple[str, ...] = ("north", "south", "stay")) -> Question:
    return Question(agent_id=1, species="wolf", state_text="sheep 2 cells north", options=options)


def test_valid_question() -> None:
    assert make().options == ("north", "south", "stay")


def test_too_few_options() -> None:
    with pytest.raises(ValueError):
        make(("stay",))


def test_duplicate_options() -> None:
    with pytest.raises(ValueError):
        make(("north", "north", "stay"))


def test_unknown_species() -> None:
    with pytest.raises(ValueError, match="species"):
        Question(1, "cow", "grass", ("north", "stay"))  # type: ignore[arg-type]
