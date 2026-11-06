"""Run one experiment config: ``uv run python scripts/run_world.py configs/micro.yaml``."""

from __future__ import annotations

import logging
import sys

from herdmind.common.config import load_config
from herdmind.world.config import ExperimentConfig
from herdmind.world.experiment import run_experiment


def main(path: str) -> None:
    logging.basicConfig(level=logging.INFO)
    run_dir = run_experiment(load_config(path, ExperimentConfig))
    logging.getLogger(__name__).info("wrote %s", run_dir)


if __name__ == "__main__":
    main(sys.argv[1])
