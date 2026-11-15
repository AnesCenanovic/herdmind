"""Training rows for distillation, read back from a run folder.

A run's ``decisions.jsonl`` already is the dataset: every row holds what a brain
was shown and the full distribution it answered with. This module only reads
it, so training never re-derives a question outside the question engine. No
torch here; the rows are plain records that any trainer can consume.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from herdmind.brains.base import PROB_SUM_TOL
from herdmind.world.log import DECISIONS_FILE

BRIEFINGS_FILE = "briefings.json"


@dataclass(frozen=True)
class DistillRow:
    """One logged decision as a training example.

    ``probs`` is the teacher's distribution, aligned with ``options`` in the
    order the teacher saw them (already shuffled by ``encoding/``).
    """

    context: str
    options: tuple[str, ...]
    probs: tuple[float, ...]

    def __post_init__(self) -> None:
        # A missing probability is an error, not a silent zero (Brain contract).
        if len(self.probs) != len(self.options):
            raise ValueError(f"{len(self.probs)} probs for {len(self.options)} options")
        if abs(sum(self.probs) - 1.0) > PROB_SUM_TOL:
            raise ValueError(f"probs must sum to 1, got {sum(self.probs)}")

    @property
    def teacher_argmax(self) -> int:
        """Index of the teacher's top option: the hard label."""
        return int(np.argmax(self.probs))


def make_context(briefing: str, state_text: str) -> str:
    """Join briefing and perceptions into the student's context text.

    TODO(verify): provisional layout for the pilot. How the briefing reaches
    Decider is settled with the Decider brain (item 22); training and inference
    must then use the same layout.
    """
    return f"{briefing}\n\n{state_text}" if briefing else state_text


def load_rows(run_dir: str | Path, limit: int, rng: np.random.Generator) -> list[DistillRow]:
    """Read up to ``limit`` decisions from a run folder, sampled without replacement.

    Args:
        run_dir: Run folder holding ``decisions.jsonl`` (and ``briefings.json``
            for logs written before the briefing was stored per decision).
        limit: Largest number of rows to return; fewer if the run is smaller.
        rng: Picks which rows are kept, so a subset is reproducible.

    Returns:
        The kept rows in a random order.

    Raises:
        ValueError: If the run has no decisions, or a row is malformed.
    """
    run_dir = Path(run_dir)
    briefings_path = run_dir / BRIEFINGS_FILE
    briefings: dict[str, str] = (
        json.loads(briefings_path.read_text(encoding="utf-8")) if briefings_path.exists() else {}
    )
    with (run_dir / DECISIONS_FILE).open(encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    if not records:
        raise ValueError(f"no decisions in {run_dir}")
    keep = rng.permutation(len(records))[:limit]
    return [_to_row(records[i], briefings) for i in keep]


def _to_row(rec: dict, briefings: dict[str, str]) -> DistillRow:
    briefing = rec.get("briefing") or briefings.get(rec["species"], "")
    return DistillRow(
        context=make_context(briefing, rec["state_text"]),
        options=tuple(rec["options"]),
        probs=tuple(rec["probs"]),
    )
