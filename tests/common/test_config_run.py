import json
from pathlib import Path

import pytest

from herdmind.common.config import load_config
from herdmind.common.run import start_run
from herdmind.world.config import ExperimentConfig

ROOT = Path(__file__).parents[2]


def test_load_micro_config():
    cfg = load_config(ROOT / "configs" / "micro.yaml", ExperimentConfig)
    assert cfg.brain == "rules" and cfg.world.width == 20 and cfg.world.sheep.initial == 40


def test_unknown_key_fails(tmp_path):
    f = tmp_path / "bad.yaml"
    f.write_text("sead: 1\n")
    with pytest.raises(Exception):  # noqa: B017 - omegaconf's error type is internal
        load_config(f, ExperimentConfig)


def test_start_run_writes_snapshot(tmp_path):
    cfg = ExperimentConfig(seed=7)
    run = start_run(cfg, cfg.name, cfg.seed, "rules", "rules-0.1.0", tmp_path)
    meta = json.loads((run / "meta.json").read_text())
    assert meta["brain"] == "rules" and meta["model_version"] == "rules-0.1.0"
    assert isinstance(meta["dirty"], bool)
    assert meta["seed"] == 7 and "mesa" in meta["versions"] and meta["commit"]
    assert (run / "config.yaml").exists()
