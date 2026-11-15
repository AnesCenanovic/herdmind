"""Run one world end to end, so scripts stay thin and the multi-seed runner can reuse it."""

from __future__ import annotations

import json
from pathlib import Path

from herdmind.brains.registry import make_brains
from herdmind.common.run import start_run
from herdmind.world.describe import describe_world
from herdmind.world.log import RunLog, export_decisions
from herdmind.world.model import WorldModel
from herdmind.world.spec import WorldSpec


def run_experiment(spec: WorldSpec, runs_dir: str | Path = "runs") -> Path:
    """Run ``spec`` and return its run folder."""
    brains = make_brains(spec.brain_names())
    # TODO(verify): model_version is not on the Brain protocol; jev/decider must expose it.
    versions = {
        s: {"brain": b.name, "model_version": getattr(b, "model_version", "unknown")}
        for s, b in brains.items()
    }
    run_dir = start_run(spec, spec.name, spec.seed, versions, runs_dir)
    (run_dir / "world.txt").write_text(describe_world(spec) + "\n", encoding="utf-8")
    try:
        with RunLog(run_dir) as log:
            world = WorldModel(spec, brains, log)
            (run_dir / "briefings.json").write_text(json.dumps(world.briefings, indent=2))
            for _ in range(spec.steps):
                world.step()
    finally:
        # Also after a crash or Ctrl-C: the finished steps are still training data.
        export_decisions(run_dir, spec.name, spec.world_seed, spec.seed)
    return run_dir
