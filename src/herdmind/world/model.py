"""Mesa wolf-sheep world on a torus: the step loop.

Each step, every animal gets a Question from the question engine, each brain
answers its species' Questions in one batch call, and the world samples every
action from the returned probabilities. What happens next (kills, breeding,
disease, energy burn) is decided by the world's laws, not hard-coded here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import mesa
import numpy as np

from herdmind.brains.base import Brain, Decision
from herdmind.common.seeding import make_rng
from herdmind.encoding.actions import Action, actions_for, from_label
from herdmind.encoding.engine import encode
from herdmind.encoding.question import Question
from herdmind.world.agents import Animal
from herdmind.world.describe import briefing
from herdmind.world.laws import make_laws
from herdmind.world.log import DecisionRecord, EventRecord, RunLog
from herdmind.world.perception import perceive
from herdmind.world.spec import SPECIES, SpeciesTraits, WorldSpec


class WorldModel(mesa.Model):
    """One world. ``history`` holds per-step sheep/wolf/grass counts.

    Two RNGs from two seeds: ``world_seed`` lays out the start (animals, grass),
    ``seed`` drives everything after (shuffles, sampling, law rolls). Same map,
    different history, or the other way round.
    """

    def __init__(
        self, spec: WorldSpec, brains: Mapping[str, Brain], log: RunLog | None = None
    ) -> None:
        self.rng = make_rng(spec.seed)
        super().__init__(rng=self.rng)
        self.spec, self.brains, self.log = spec, brains, log
        self.laws = make_laws(spec.laws)
        self.actions = {s: actions_for(s, spec.species.of(s).speed) for s in SPECIES}
        self.briefings = {s: briefing(spec, s) for s in SPECIES}
        self.step_no = 0
        self._lay_out(make_rng(spec.world_seed))
        self.history: list[dict[str, int]] = []
        self._record()

    def _lay_out(self, rng: np.random.Generator) -> None:
        g = self.spec.grid
        self.grass = rng.random((g.height, g.width)) < g.grass_fraction
        # Bare cells start mid-regrowth, otherwise they would stay barren forever.
        self.regrow = np.where(
            self.grass, 0, rng.integers(1, g.grass_regrow_steps + 1, self.grass.shape)
        )
        for s in SPECIES:
            t = self.traits(s)
            for _ in range(t.count):
                x, y = int(rng.integers(g.width)), int(rng.integers(g.height))
                self._event("birth", Animal(self, s, x, y, t.start_energy))  # step 0

    def traits(self, species: str) -> SpeciesTraits:
        return self.spec.species.of(species)

    def animals(self, species: str | None = None) -> list[Animal]:
        return [a for a in self.agents if species in (None, a.species)]

    def step(self) -> None:
        self.step_no += 1
        acting = self.animals()
        questions = [
            encode(
                a.unique_id,
                perceive(self, a),
                self.actions[a.species],
                self.briefings[a.species],
                self.rng,
            )
            for a in acting
        ]
        decisions = self._decide(acting, questions)
        # Phases, so no outcome depends on who happened to act first:
        # every animal moves, then sheep graze, then wolves hunt on the final positions.
        chosen: list[Action] = []
        for a, q, d in zip(acting, questions, decisions, strict=True):
            p = np.asarray(d.probs)
            label = q.options[self.rng.choice(len(p), p=p / p.sum())]  # sample, never argmax
            self._log_decision(a, q, d, label)  # every decision, including soon-eaten sheep
            chosen.append(from_label(label))
        for a, action in zip(acting, chosen, strict=True):
            self._move(a, action)
        for a, action in zip(acting, chosen, strict=True):
            if a.species == "sheep" and action.kind == "eat":
                self._graze(a)
        wolves = [a for a in acting if a.species == "wolf"]
        for i in self.rng.permutation(len(wolves)):
            self._hunt(wolves[i])
        self._upkeep()
        if self.log:
            self.log.flush()
        growing = self.regrow > 0
        self.regrow[growing] -= 1
        self.grass |= growing & (self.regrow == 0)
        self._record()

    def _decide(self, acting: Sequence[Animal], questions: Sequence[Question]) -> list[Decision]:
        """One batch call per distinct brain; a brain shared by both species is called once."""
        out: list[Decision | None] = [None] * len(questions)
        for brain in {id(b): b for b in self.brains.values()}.values():
            idx = [i for i, a in enumerate(acting) if self.brains[a.species] is brain]
            if idx:
                for i, d in zip(idx, brain.decide_batch([questions[i] for i in idx]), strict=True):
                    out[i] = d
        return out  # type: ignore[return-value]

    def _move(self, a: Animal, action: Action) -> None:
        g = self.spec.grid
        dx, dy = action.delta
        a.x, a.y = (a.x + dx) % g.width, (a.y + dy) % g.height
        a.moved = action.distance

    def _graze(self, a: Animal) -> None:
        if self.grass[a.y, a.x]:
            self.grass[a.y, a.x] = False
            self.regrow[a.y, a.x] = self.spec.grid.grass_regrow_steps
            a.energy += self.traits("sheep").energy_gain

    def _hunt(self, wolf: Animal) -> None:
        """A wolf sharing a cell with sheep tries one of them; the predation law decides."""
        prey = [s for s in self.animals("sheep") if (s.x, s.y) == (wolf.x, wolf.y)]
        if not prey:
            return
        target = prey[int(self.rng.integers(len(prey)))]
        if self.laws.predation.apply(self, wolf, target):
            self._kill(target, "eaten", other_id=wolf.unique_id)
            wolf.energy += self.traits("wolf").energy_gain

    def _upkeep(self) -> None:
        """Energy burn, then disease and breeding, judged on one snapshot of the survivors.

        Deciding first and applying after means a death or birth earlier in the
        list cannot change a later animal's mates or contagion neighbours.
        """
        for a in self.animals():
            a.energy -= self.laws.metabolism.apply(self, a)
            a.moved = 0
            if a.energy <= 0:
                self._kill(a, "starved")
        survivors = self.animals()
        sick = [a for a in survivors if self.laws.disease.apply(self, a)]
        sick_ids = {a.unique_id for a in sick}
        breeders = [
            a
            for a in survivors
            if a.unique_id not in sick_ids and self.laws.breeding.apply(self, a)
        ]
        for a in sick:
            self._kill(a, "disease")
        for a in breeders:
            child_energy = a.energy // 2
            a.energy -= child_energy
            child = Animal(self, a.species, a.x, a.y, child_energy)
            self._event("birth", child, other_id=a.unique_id)

    def _kill(self, a: Animal, cause: str, other_id: int = -1) -> None:
        a.alive = False
        a.remove()
        self._event("death", a, cause=cause, other_id=other_id)

    def _event(self, kind: str, a: Animal, cause: str = "", other_id: int = -1) -> None:
        if self.log:
            self.log.event(EventRecord(self.step_no, kind, a.unique_id, a.species, cause, other_id))

    def _log_decision(self, a: Animal, q: Question, d: Decision, chosen: str) -> None:
        if self.log:
            self.log.decision(
                DecisionRecord(
                    self.step_no,
                    a.unique_id,
                    a.species,
                    d.brain,
                    d.model_version,
                    q.state_text,
                    q.options,
                    d.probs,
                    chosen,
                    d.latency_s,
                    d.cached,
                )
            )

    def _record(self) -> None:
        self.history.append(
            {
                "sheep": len(self.animals("sheep")),
                "wolf": len(self.animals("wolf")),
                "grass": int(self.grass.sum()),
            }
        )
