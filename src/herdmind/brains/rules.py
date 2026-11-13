"""Hand-written baseline brain: fixed heuristics turned into probabilities.

Weights are soft (never zero) so the world still samples and runs stay
stochastic like the model brains. It reads only the Question's perception
text and option labels; it ignores the briefing, because its policy is fixed
by hand and does not adapt to a world's laws.
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
HUNGRY_ENERGY = 20  # sheep eat when grass is here and energy is below this
_AWAY = {"north": "south", "south": "north", "east": "west", "west": "east"}


def _weights(p: Perception, actions: Sequence[Action]) -> list[float]:
    target, reach, eat = None, None, LOW  # reach None: any distance toward target
    if p.species == "sheep":
        wolf, grass = p.nearest("wolf"), p.nearest("grass")
        if wolf and wolf.distance <= FLEE_DISTANCE:
            target = _AWAY[wolf.direction]
        elif p.grass_here and p.energy < HUNGRY_ENERGY:
            eat = HIGH
        elif grass:
            target, reach = grass.direction, grass.distance
    elif prey := p.nearest("sheep"):
        target, reach = prey.direction, prey.distance
    out = []
    for a in actions:
        if a.kind == "eat":
            out.append(eat)
        elif a.direction == target and (reach is None or a.distance <= reach):
            out.append(HIGH)
        else:
            out.append(MID)
    return out


class RulesBrain:
    """Sheep flee close wolves, eat when hungry, else head for grass; wolves chase."""

    name: str = "rules"
    model_version: str = "rules-0.2.0"

    def decide_batch(self, questions: Sequence[Question]) -> list[Decision]:
        out = []
        for q in questions:
            raw = _weights(parse(q.state_text), [from_label(o) for o in q.options])
            total = sum(raw)
            out.append(Decision(tuple(r / total for r in raw), 0.0, self.name, self.model_version))
        return out
