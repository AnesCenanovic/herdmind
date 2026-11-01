from __future__ import annotations

from collections.abc import Sequence

import pytest

from herdmind.brains.base import Decision
from herdmind.brains.testing import check_brain_contract
from herdmind.encoding.question import Question


def dec(probs: tuple[float, ...]) -> Decision:
    return Decision(probs=probs, latency_s=0.0, brain="fake", model_version="fake-1")


def test_valid_decision() -> None:
    assert dec((0.5, 0.25, 0.25)).cached is False


@pytest.mark.parametrize(
    "probs",
    [(0.5, 0.4), (0.7, 0.7), (-0.1, 1.1), (float("nan"), 1.0), (float("inf"), 0.0)],
)
def test_invalid_probs(probs: tuple[float, ...]) -> None:
    with pytest.raises(ValueError):
        dec(probs)


class FakeBrain:
    name = "fake"

    def decide_batch(self, questions: Sequence[Question]) -> list[Decision]:
        return [dec(tuple(1 / len(q.options) for _ in q.options)) for q in questions]


class WrongLengthBrain(FakeBrain):
    def decide_batch(self, questions: Sequence[Question]) -> list[Decision]:
        return [dec((0.5, 0.5)) for _ in questions]


QUESTIONS = [
    Question(1, "wolf", "sheep 2 cells north", ("north", "south", "stay")),
    Question(2, "sheep", "wolf 3 cells east", ("east", "west", "stay", "call")),
]


def test_contract_passes_for_fake_brain() -> None:
    assert len(check_brain_contract(FakeBrain(), QUESTIONS)) == 2


def test_contract_catches_misaligned_probs() -> None:
    with pytest.raises(AssertionError):
        check_brain_contract(WrongLengthBrain(), QUESTIONS)


class RenamedBrain(FakeBrain):
    name = "other"


class FlakyBrain(FakeBrain):
    calls = 0

    def decide_batch(self, questions: Sequence[Question]) -> list[Decision]:
        self.calls += 1
        p = 0.5 if self.calls % 2 else 0.25
        return [dec((p, 1 - p)) for _ in questions]


def test_contract_catches_name_mismatch() -> None:
    with pytest.raises(AssertionError, match="Brain.name"):
        check_brain_contract(RenamedBrain(), QUESTIONS)


def test_contract_catches_nondeterminism() -> None:
    two_options = [Question(1, "wolf", "sheep 2 cells north", ("north", "stay"))]
    with pytest.raises(AssertionError, match="deterministic"):
        check_brain_contract(FlakyBrain(), two_options)
