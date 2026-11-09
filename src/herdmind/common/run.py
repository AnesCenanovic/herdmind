"""Run folders: every run records what is needed to reproduce it."""

from __future__ import annotations

import json
import platform
import subprocess
from collections.abc import Mapping
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

from omegaconf import OmegaConf

PACKAGES = ("herdmind", "numpy", "mesa", "omegaconf")


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _git_dirty() -> bool:
    """True if the tree has uncommitted changes, so the commit hash alone is not the code."""
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        )
        return bool(out.stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        return True


def start_run(
    cfg: object,
    name: str,
    seed: int,
    brains: Mapping[str, Mapping[str, str]],
    runs_dir: str | Path = "runs",
) -> Path:
    """Create ``runs_dir/<name>-seed<seed>-<utc time>/`` with config and meta files.

    ``brains`` maps each species to its brain's name and pinned ``model_version``.
    """
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = Path(runs_dir) / f"{name}-seed{seed}-{stamp}"
    run_dir.mkdir(parents=True)
    OmegaConf.save(OmegaConf.structured(cfg), run_dir / "config.yaml")
    meta = {
        "seed": seed,
        "brains": {s: dict(b) for s, b in brains.items()},
        "commit": _git_commit(),
        "dirty": _git_dirty(),
        "python": platform.python_version(),
        "versions": {p: metadata.version(p) for p in PACKAGES},
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return run_dir
