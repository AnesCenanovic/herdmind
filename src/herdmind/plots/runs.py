"""Read finished run folders for plotting.

Figures read runs only through this module, never a live model, so a figure can
always be rebuilt from ``runs/`` alone.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from omegaconf import OmegaConf

from herdmind.world.ledger import LEDGER_FILE, BoardState, read_ledger

logger = logging.getLogger(__name__)

META_FILE = "meta.json"
CONFIG_FILE = "config.yaml"


@dataclass(frozen=True)
class RunData:
    """What a figure may know about one run."""

    path: Path
    name: str  # world name from config.yaml; runs with the same name are seeds of one world
    seed: int
    commit: str
    dirty: bool
    boards: tuple[BoardState, ...]


def is_run_dir(path: Path) -> bool:
    """A run folder has a meta file and a ledger; anything else (e.g. figures/) is skipped."""
    return (path / META_FILE).is_file() and (path / LEDGER_FILE).is_file()


def load_run(run_dir: str | Path) -> RunData:
    """Load one run folder written by ``world/experiment.py``.

    Raises:
        FileNotFoundError: If the folder lacks ``meta.json``, ``config.yaml`` or the ledger.
    """
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / META_FILE).read_text(encoding="utf-8"))
    cfg = OmegaConf.load(run_dir / CONFIG_FILE)
    return RunData(
        path=run_dir,
        name=str(cfg.name),
        seed=int(meta["seed"]),
        commit=str(meta["commit"]),
        dirty=bool(meta["dirty"]),
        boards=tuple(read_ledger(run_dir)),
    )


def find_runs(runs_dir: str | Path, names: Sequence[str] = ()) -> list[RunData]:
    """Load every run folder directly under ``runs_dir``, sorted by folder name.

    Args:
        runs_dir: Parent folder of the run folders.
        names: Keep only runs of these worlds; empty keeps all.

    Returns:
        The loaded runs; empty if none match.
    """
    runs = []
    for path in sorted(Path(runs_dir).iterdir()):
        if not path.is_dir() or not is_run_dir(path):
            continue
        run = load_run(path)
        if names and run.name not in names:
            continue
        if run.dirty:
            logger.warning("%s ran from uncommitted code; its commit hash is not the code", path)
        runs.append(run)
    return runs


def group_by_name(runs: Sequence[RunData]) -> dict[str, list[RunData]]:
    """Group runs into worlds (same name, different seeds), keeping input order."""
    groups: dict[str, list[RunData]] = {}
    for run in runs:
        groups.setdefault(run.name, []).append(run)
    return groups
