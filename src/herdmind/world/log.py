"""Run logs, one JSON object per line: decisions (the distillation dataset),
events (births and deaths, for emergence metrics) and the board ledger.

JSONL is the write path because it survives a killed run (flushed every step).
``export_decisions`` turns the decision log into ``decisions.parquet`` with the
world and seed ids on every row, so runs can be concatenated into one dataset.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from types import TracebackType

import pyarrow as pa
import pyarrow.parquet as pq

from herdmind.world.ledger import LEDGER_FILE, BoardState

logger = logging.getLogger(__name__)

DECISIONS_FILE = "decisions.jsonl"
EVENTS_FILE = "events.jsonl"
DECISIONS_PARQUET = "decisions.parquet"
# Rows per parquet row group; bounds memory while converting a long run.
PARQUET_BATCH_ROWS = 10_000

# The distillation dataset's columns. The first four identify the run, so rows
# from many runs stay traceable (and splittable by world) after concatenation.
DECISION_SCHEMA = pa.schema(
    [
        ("world", pa.string()),
        ("world_seed", pa.int64()),
        ("seed", pa.int64()),
        ("run", pa.string()),
        ("step", pa.int32()),
        ("agent_id", pa.int64()),
        ("species", pa.string()),
        ("brain", pa.string()),
        ("model_version", pa.string()),
        ("state_text", pa.string()),
        ("briefing", pa.string()),
        ("options", pa.list_(pa.string())),
        ("probs", pa.list_(pa.float64())),  # aligned with options
        ("chosen", pa.string()),  # the sampled option, not the argmax
        ("latency_s", pa.float64()),
        ("cached", pa.bool_()),
    ]
)


@dataclass(frozen=True)
class DecisionRecord:
    step: int
    agent_id: int
    species: str
    brain: str
    model_version: str
    state_text: str
    briefing: str  # as the brain saw it; differs per world and species
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
        self._decisions = (run_dir / DECISIONS_FILE).open("x", encoding="utf-8")
        self._events = (run_dir / EVENTS_FILE).open("x", encoding="utf-8")
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


def _read_decisions(path: Path) -> Iterator[dict]:
    """Yield the decision records; a cut-off last line (killed run) is dropped."""
    with path.open(encoding="utf-8") as f:
        line = f.readline()
        while line:
            following = f.readline()
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                if following:  # damage in the middle is not a killed run
                    raise
                logger.warning("%s: dropped a cut-off last line", path)
            line = following


def export_decisions(run_dir: str | Path, world: str, world_seed: int, seed: int) -> Path:
    """Convert a run's ``decisions.jsonl`` to ``decisions.parquet`` (``DECISION_SCHEMA``).

    Args:
        run_dir: Run folder holding ``decisions.jsonl``; its name becomes the ``run`` column.
        world: World name from the spec.
        world_seed: Seed of the starting layout.
        seed: Seed of everything after the layout.

    Returns:
        Path of the written parquet file.

    Raises:
        FileNotFoundError: If the run folder has no decision log.
        json.JSONDecodeError: If a line other than the last is not valid JSON.
        KeyError: If a record lacks a schema column.
    """
    run_dir = Path(run_dir)
    ids = {"world": world, "world_seed": world_seed, "seed": seed, "run": run_dir.name}
    out = run_dir / DECISIONS_PARQUET
    # Written under a temp name: a failed export must not leave a file that
    # reads like a complete dataset.
    tmp = out.with_name(out.name + ".tmp")
    batch: list[dict] = []
    try:
        with pq.ParquetWriter(tmp, DECISION_SCHEMA) as writer:

            def write_batch() -> None:
                # Columns by name: a record missing one fails here instead of becoming nulls.
                columns = [[row[name] for row in batch] for name in DECISION_SCHEMA.names]
                writer.write_table(pa.Table.from_arrays(columns, schema=DECISION_SCHEMA))
                batch.clear()

            for rec in _read_decisions(run_dir / DECISIONS_FILE):
                batch.append(ids | rec)
                if len(batch) >= PARQUET_BATCH_ROWS:
                    write_batch()
            if batch:
                write_batch()
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    tmp.replace(out)
    return out
