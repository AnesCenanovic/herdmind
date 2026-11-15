"""Fake run folders with hand-picked head counts, written the way real runs are."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from herdmind.common import run as run_module
from herdmind.common.run import start_run
from herdmind.world.ledger import LEDGER_FILE, AnimalState, BoardState
from herdmind.world.spec import WorldSpec


def write_run(runs_dir: Path, name: str, seed: int, counts: Sequence[tuple[int, int]]) -> Path:
    """A run folder whose ledger has ``counts[step] = (sheep, wolves)``."""
    spec = WorldSpec(name=name, seed=seed)
    brains = {"sheep": {"brain": "rules", "model_version": "rules-test"}}
    run_dir = start_run(spec, name, seed, brains, runs_dir)
    lines = []
    for step, (n_sheep, n_wolf) in enumerate(counts):
        animals = [AnimalState(i, "sheep", 0, 0, 5) for i in range(n_sheep)]
        animals += [AnimalState(100 + i, "wolf", 0, 0, 5) for i in range(n_wolf)]
        board = BoardState(step=step, grass=("#",), regrow=((0,),), animals=tuple(animals))
        lines.append(board.to_json())
    (run_dir / LEDGER_FILE).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return run_dir


@pytest.fixture(autouse=True)
def _no_git(monkeypatch: pytest.MonkeyPatch) -> None:
    """start_run shells out to git per run, which is slow and not what these tests check."""
    monkeypatch.setattr(run_module, "_git_commit", lambda: "test-commit")
    monkeypatch.setattr(run_module, "_git_dirty", lambda: False)


@pytest.fixture
def runs_dir(tmp_path: Path) -> Path:
    """Two seeds of world "a" and one of world "b"."""
    d = tmp_path / "runs"
    write_run(d, "a", 0, [(4, 1), (6, 2), (2, 3)])
    write_run(d, "a", 1, [(8, 1), (2, 2), (4, 1)])
    write_run(d, "b", 0, [(1, 1), (1, 1)])
    return d
