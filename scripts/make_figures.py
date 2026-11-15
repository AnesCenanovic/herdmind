"""Build figures from run folders: ``uv run python scripts/make_figures.py configs/figures/default.yaml``."""

from __future__ import annotations

import argparse
import logging

from herdmind.common.config import load_config
from herdmind.plots.figures import FiguresConfig, make_figures


def main() -> None:
    """Parse the config path and build the figures it lists."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="figures YAML, e.g. configs/figures/default.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    make_figures(load_config(args.config, FiguresConfig))


if __name__ == "__main__":
    main()
