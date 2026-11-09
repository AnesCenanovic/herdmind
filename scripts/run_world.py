"""Run one world declaration: ``uv run python scripts/run_world.py configs/micro.yaml``."""

from __future__ import annotations

import logging
import sys

from herdmind.common.config import load_config
from herdmind.world.experiment import run_experiment
from herdmind.world.spec import WorldSpec


def main(path: str) -> None:
    logging.basicConfig(level=logging.INFO)
    run_dir = run_experiment(load_config(path, WorldSpec))
    logging.getLogger(__name__).info("wrote %s", run_dir)


if __name__ == "__main__":
    main(sys.argv[1])
