"""Hand-written baseline brain: fixed heuristics turned into probabilities.

Weights are soft (never zero) so the world still samples and runs stay
stochastic like the model brains. It reads only the Question text.
"""

from __future__ import annotations

from collections.abc import Sequence

from herdmind.brains.base import Decision
from herdmind.encoding.actions import Action, from_label
from herdmind.encoding.question import Question
from herdmind.encoding.state import Perception, parse

# Weight for actions the heuristic favours / allows / discourages.
HIGH, MID, LOW = 8.0, 2.0, 0.5
FLEE_DISTANCE = 3
HUNGRY_ENERGY = 10
_AWAY = {"north": Action.SOUTH, "south": Action.NORTH, "east": Action.WEST, "west": Action.EAST}
_TOWARD = {"north": Action.NORTH, "south": Action.SOUTH, "east": Action.EAST, "west": Action.WEST}


def _weights(p: Perception) -> dict[Action, float]:
    w = {a: MID for a in Action}
    w[Action.EAT] = LOW
    if p.species == "sheep":
        wolf = p.nearest("wolf")
        grass = p.nearest("grass")
        if wolf and wolf.distance <= FLEE_DISTANCE:
            w[_AWAY[wolf.direction]] = HIGH
        elif p.grass_here and p.energy < 2 * HUNGRY_ENERGY:
            w[Action.EAT] = HIGH
        elif grass:
            w[_TOWARD[grass.direction]] = HIGH
    else:
        prey = p.nearest("sheep")
        if prey:
            w[_TOWARD[prey.direction]] = HIGH
    return w


class RulesBrain:
    name = "rules"
    model_version = "rules-0.1.0"

    def decide_batch(self, questions: Sequence[Question]) -> list[Decision]:
        out = []
        for q in questions:
            w = _weights(parse(q.state_text))
            raw = [w[from_label(o)] for o in q.options]
            total = sum(raw)
            out.append(Decision(tuple(r / total for r in raw), 0.0, self.name, self.model_version))
        return out
