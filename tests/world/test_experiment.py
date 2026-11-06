import json

from herdmind.world.config import ExperimentConfig
from herdmind.world.experiment import run_experiment


def test_run_experiment_writes_run_folder(tmp_path):
    run = run_experiment(ExperimentConfig(steps=5), tmp_path)
    assert len(json.loads((run / "populations.json").read_text())) == 6
    assert json.loads((run / "meta.json").read_text())["model_version"] == "rules-0.1.0"
