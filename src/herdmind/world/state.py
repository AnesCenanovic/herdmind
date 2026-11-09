"""The ledger: the exact board at the end of every step of a run.

One ``BoardState`` per step (step 0 is the starting layout), written in order
to ``ledger.jsonl`` in the run folder. It is the single record of how the world
looked; replays, a UI, and world metrics read it instead of re-running the world.
It holds the board only, not RNG state, so it shows a run but cannot resume one.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from herdmind.world.model import WorldModel

GRASS, BARE = "#", "."
LEDGER_FILE = "ledger.jsonl"


@dataclass(frozen=True)
class AnimalState:
    agent_id: int
    species: str
    x: int
    y: int
    energy: int


@dataclass(frozen=True)
class BoardState:
    """The whole board after ``step``. Grid rows are indexed by y (north at the top)."""

    step: int
    grass: tuple[str, ...]  # one string per row: "#" grass, "." bare
    regrow: tuple[tuple[int, ...], ...]  # steps until a bare cell grows back; 0 = has grass
    animals: tuple[AnimalState, ...]  # living animals, sorted by id

    def count(self, species: str) -> int:
        return sum(a.species == species for a in self.animals)

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, line: str) -> BoardState:
        d = json.loads(line)
        return cls(
            step=d["step"],
            grass=tuple(d["grass"]),
            regrow=tuple(tuple(r) for r in d["regrow"]),
            animals=tuple(AnimalState(**a) for a in d["animals"]),
        )


def snapshot(world: WorldModel) -> BoardState:
    """Freeze the world as it is right now."""
    return BoardState(
        step=world.step_no,
        grass=tuple("".join(GRASS if g else BARE for g in row) for row in world.grass),
        regrow=tuple(tuple(int(v) for v in row) for row in world.regrow),
        animals=tuple(
            AnimalState(a.unique_id, a.species, a.x, a.y, a.energy)
            for a in sorted(world.animals(), key=lambda a: a.unique_id)
        ),
    )


def read_ledger(run_dir: str | Path) -> list[BoardState]:
    """Every step's board from a run folder, in step order."""
    lines = (Path(run_dir) / LEDGER_FILE).read_text(encoding="utf-8").splitlines()
    return [BoardState.from_json(line) for line in lines]
