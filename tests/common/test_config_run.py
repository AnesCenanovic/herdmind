import json

import pytest

from herdmind.common.config import load_config
from herdmind.common.run import start_run
from herdmind.world.spec import WorldSpec


def test_unknown_key_fails(tmp_path):
    f = tmp_path / "bad.yaml"
    f.write_text("sead: 1\n")
    with pytest.raises(Exception):  # noqa: B017 - omegaconf's error type is internal
        load_config(f, WorldSpec)


def test_start_run_writes_snapshot(tmp_path):
    spec = WorldSpec(seed=7)
    brains = {"sheep": {"brain": "rules", "model_version": "rules-0.2.0"}}
    run = start_run(spec, spec.name, spec.seed, brains, tmp_path)
    meta = json.loads((run / "meta.json").read_text())
    assert meta["brains"] == brains
    assert isinstance(meta["dirty"], bool)
    assert meta["seed"] == 7 and "mesa" in meta["versions"] and meta["commit"]
    assert (run / "config.yaml").exists()
