"""Run the LoRA pilot: ``python scripts/pilot_lora.py configs/train/pilot-lora.yaml``.

Needs the ``train`` extra (``pip install -e ".[train]"``) and, realistically, a GPU.
"""

from __future__ import annotations

import logging
import sys

from herdmind.common.config import load_config
from herdmind.distill.pilot import PilotConfig, run_pilot


def main(path: str) -> None:
    logging.basicConfig(level=logging.INFO)
    run_dir = run_pilot(load_config(path, PilotConfig))
    logging.getLogger(__name__).info("wrote %s", run_dir)


if __name__ == "__main__":
    main(sys.argv[1])
