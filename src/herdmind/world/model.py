"""Mesa wolf-sheep world on a torus.

One step = one batch call to the brain; the world samples each action from the
returned probabilities with the run's RNG. The world never knows which brain it has.
"""

from __future__ import annotations

import mesa
import numpy as np

from herdmind.brains.base import Brain
from herdmind.encoding.actions import DELTAS, Action, from_label
from herdmind.encoding.encode import encode
from herdmind.encoding.state import Perception, Sighting
from herdmind.world.config import ExperimentConfig, SpeciesConfig

_KIND_ORDER = {"wolf": 0, "sheep": 1, "grass": 2}


class Animal(mesa.Agent):
    """One wolf or sheep; ``alive`` guards against acting after being eaten."""

    def __init__(self, model: WorldModel, species: str, x: int, y: int, energy: int) -> None:
        super().__init__(model)
        self.species, self.x, self.y, self.energy = species, x, y, energy
        self.alive = True


def _delta(a: int, b: int, size: int) -> int:
    """Shortest signed step from a to b on a ring of ``size``."""
    d = (b - a) % size
    return d - size if d > size // 2 else d


def _direction(dx: int, dy: int) -> str:
    # Ties go to the vertical axis; y grows southwards.
    if abs(dy) >= abs(dx):
        return "south" if dy > 0 else "north"
    return "east" if dx > 0 else "west"


class WorldModel(mesa.Model):
    """Torus grid world; ``history`` holds per-step sheep/wolf/grass counts."""

    def __init__(self, cfg: ExperimentConfig, brain: Brain, rng: np.random.Generator) -> None:
        super().__init__(rng=rng)
        self.cfg, self.brain, self.rng = cfg.world, brain, rng
        w = self.cfg
        self.grass = rng.random((w.height, w.width)) < w.grass_fraction
        # Bare cells start mid-regrowth, otherwise they would stay barren forever.
        self.regrow = np.where(
            self.grass, 0, rng.integers(1, w.grass_regrow_steps + 1, self.grass.shape)
        )
        for species, sc in (("sheep", w.sheep), ("wolf", w.wolf)):
            for _ in range(sc.initial):
                x, y = int(rng.integers(w.width)), int(rng.integers(w.height))
                Animal(self, species, x, y, sc.start_energy)
        self.history: list[dict[str, int]] = []
        self._record()

    def _species_cfg(self, species: str) -> SpeciesConfig:
        return self.cfg.sheep if species == "sheep" else self.cfg.wolf

    def animals(self, species: str | None = None) -> list[Animal]:
        return [a for a in self.agents if species in (None, a.species)]

    def perceive(self, a: Animal) -> Perception:
        """What ``a`` senses: nearest thing of each kind within vision (Manhattan)."""
        w, r = self.cfg, self.cfg.vision_radius
        best: dict[str, Sighting] = {}

        def consider(kind: str, x: int, y: int) -> None:
            dx, dy = _delta(a.x, x, w.width), _delta(a.y, y, w.height)
            dist = abs(dx) + abs(dy)
            if 0 < dist <= r and (kind not in best or dist < best[kind].distance):
                best[kind] = Sighting(kind, dist, _direction(dx, dy))

        for o in self.animals():
            if o is not a:
                consider(o.species, o.x, o.y)
        if a.species == "sheep":
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    x, y = (a.x + dx) % w.width, (a.y + dy) % w.height
                    if self.grass[y, x]:
                        consider("grass", x, y)
        sightings = tuple(sorted(best.values(), key=lambda s: _KIND_ORDER[s.kind]))
        grass_here = bool(self.grass[a.y, a.x]) if a.species == "sheep" else None
        return Perception(a.species, a.energy, grass_here, sightings)

    def step(self) -> None:
        acting = self.animals()
        questions = [encode(a.unique_id, self.perceive(a), self.rng) for a in acting]
        decisions = self.brain.decide_batch(questions) if questions else []
        for i in self.rng.permutation(len(acting)):
            a = acting[i]
            if not a.alive:  # eaten earlier this step
                continue
            p = np.asarray(decisions[i].probs)
            choice = self.rng.choice(len(p), p=p / p.sum())  # sample, never argmax
            self._apply(a, from_label(questions[i].options[choice]))
        for a in self.animals():
            a.energy -= 1
            if a.energy <= 0:
                self._kill(a)
                continue
            sc = self._species_cfg(a.species)
            if a.energy >= sc.breed_threshold and self.rng.random() < sc.breed_prob:
                child = a.energy // 2
                a.energy -= child
                Animal(self, a.species, a.x, a.y, child)
        growing = self.regrow > 0
        self.regrow[growing] -= 1
        self.grass |= growing & (self.regrow == 0)
        self._record()

    def _apply(self, a: Animal, action: Action) -> None:
        w = self.cfg
        dx, dy = DELTAS[action]
        a.x, a.y = (a.x + dx) % w.width, (a.y + dy) % w.height
        if a.species == "sheep" and action is Action.EAT and self.grass[a.y, a.x]:
            self.grass[a.y, a.x] = False
            self.regrow[a.y, a.x] = w.grass_regrow_steps
            a.energy += w.sheep.energy_gain
        elif a.species == "wolf":
            prey = [s for s in self.animals("sheep") if (s.x, s.y) == (a.x, a.y)]
            if prey:
                self._kill(prey[0])
                a.energy += w.wolf.energy_gain

    def _kill(self, a: Animal) -> None:
        a.alive = False
        a.remove()

    def _record(self) -> None:
        self.history.append(
            {
                "sheep": len(self.animals("sheep")),
                "wolf": len(self.animals("wolf")),
                "grass": int(self.grass.sum()),
            }
        )
