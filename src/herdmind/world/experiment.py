"""Run one experiment end to end, so scripts stay thin and the runner can reuse it."""

from __future__ import annotations

import json
from pathlib import Path

from herdmind.brains.registry import make_brain
from herdmind.common.run import start_run
from herdmind.common.seeding import make_rng
from herdmind.world.config import ExperimentConfig
from herdmind.world.model import WorldModel


def run_experiment(cfg: ExperimentConfig, runs_dir: str | Path = "runs") -> Path:
    """Step the world ``cfg.steps`` times; write config, meta and populations; return the dir."""
    brain = make_brain(cfg.brain)
    # TODO(verify): model_version is not on the Brain protocol; jev/decider must expose it.
    version = getattr(brain, "model_version", "unknown")
    run_dir = start_run(cfg, cfg.name, cfg.seed, brain.name, version, runs_dir)
    world = WorldModel(cfg, brain, make_rng(cfg.seed))
    for _ in range(cfg.steps):
        world.step()
    (run_dir / "populations.json").write_text(json.dumps(world.history))
    return run_dir
