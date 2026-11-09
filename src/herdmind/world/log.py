"""Run logs: every decision and every birth or death, one JSON object per line.

Decisions are the distillation dataset (state, options, full distribution,
chosen option); events feed the emergence metrics. JSONL keeps it dependency
free and append-only; logs are flushed every step, so a crashed run keeps
every finished step.
shortcut: JSONL, not parquet; convert when the dataset gets big (item 23).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from types import TracebackType


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
    """Appends records to ``decisions.jsonl`` and ``events.jsonl`` in a run folder."""

    def __init__(self, run_dir: str | Path) -> None:
        run_dir = Path(run_dir)
        # "x": refuse to append into an earlier run's log and silently mix two runs.
        self._decisions = (run_dir / "decisions.jsonl").open("x", encoding="utf-8")
        self._events = (run_dir / "events.jsonl").open("x", encoding="utf-8")

    def flush(self) -> None:
        """Called once per step, so a killed run keeps every finished step."""
        self._decisions.flush()
        self._events.flush()

    def decision(self, rec: DecisionRecord) -> None:
        self._decisions.write(json.dumps(asdict(rec)) + "\n")

    def event(self, rec: EventRecord) -> None:
        self._events.write(json.dumps(asdict(rec)) + "\n")

    def close(self) -> None:
        self._decisions.close()
        self._events.close()

    def __enter__(self) -> RunLog:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
