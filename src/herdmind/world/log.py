"""Run logs, one JSON object per line: decisions (the distillation dataset),
events (births and deaths, for emergence metrics) and the board ledger.

shortcut: JSONL, not parquet; convert when the dataset gets big (item 23).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from types import TracebackType

from herdmind.world.ledger import LEDGER_FILE, BoardState


@dataclass(frozen=True)
class DecisionRecord:
    step: int
    agent_id: int
    species: str
    brain: str
    model_version: str
    state_text: str
    options: tuple[str, ...]
    probs: tuple[float, ...]
    chosen: str  # the sampled option, not the argmax
    latency_s: float
    cached: bool


@dataclass(frozen=True)
class EventRecord:
    step: int
    kind: str  # "birth" | "death"
    agent_id: int
    species: str
    cause: str = ""  # death: "starved" | "eaten" | "disease"
    other_id: int = -1  # birth: parent; eaten: the wolf


class RunLog:
    """Writes ``decisions.jsonl``, ``events.jsonl`` and the ledger in a run folder."""

    def __init__(self, run_dir: str | Path) -> None:
        run_dir = Path(run_dir)
        # "x": refuse to append into an earlier run's log and silently mix two runs.
        self._decisions = (run_dir / "decisions.jsonl").open("x", encoding="utf-8")
        self._events = (run_dir / "events.jsonl").open("x", encoding="utf-8")
        self._ledger = (run_dir / LEDGER_FILE).open("x", encoding="utf-8")
        self._files = (self._decisions, self._events, self._ledger)

    def flush(self) -> None:
        """Called once per step, so a killed run keeps every finished step."""
        for f in self._files:
            f.flush()

    def decision(self, rec: DecisionRecord) -> None:
        self._decisions.write(json.dumps(asdict(rec)) + "\n")

    def event(self, rec: EventRecord) -> None:
        self._events.write(json.dumps(asdict(rec)) + "\n")

    def board(self, board: BoardState) -> None:
        self._ledger.write(board.to_json() + "\n")

    def close(self) -> None:
        for f in self._files:
            f.close()

    def __enter__(self) -> RunLog:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
