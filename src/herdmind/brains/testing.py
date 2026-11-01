"""Reusable contract check every brain's test runs.

Lives in ``src`` so rules, jev and decider tests share one definition of
"a valid brain" instead of drifting apart.
"""

from __future__ import annotations

from collections.abc import Sequence

from herdmind.brains.base import Brain, Decision
from herdmind.encoding.question import Question


def check_brain_contract(brain: Brain, questions: Sequence[Question]) -> list[Decision]:
    """Assert ``brain`` honours the Brain contract on ``questions``; return its decisions.

    Brains return probabilities and never sample, so identical input must give
    identical probabilities (use a fake backend / fixed seed for stochastic ones).
    The brain is called twice, so never pass one with a live backend: it would hit
    the network twice, and the second call may come back ``cached=True``.
    """
    decisions = brain.decide_batch(questions)
    assert len(decisions) == len(questions), "one Decision per Question, in order"
    for q, d in zip(questions, decisions, strict=True):
        assert len(d.probs) == len(q.options), "probs must align with the offered options"
        assert d.brain == brain.name, "Decision.brain must equal Brain.name"
        assert d.model_version, "model_version must be pinned"
    again = brain.decide_batch(questions)
    assert [d.probs for d in again] == [d.probs for d in decisions], "must be deterministic"
    return decisions
