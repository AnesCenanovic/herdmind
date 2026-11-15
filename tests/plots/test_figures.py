import json
import logging
from pathlib import Path

import numpy as np
import pytest

from herdmind.common.config import load_config
from herdmind.plots.figures import (
    MANIFEST_FILE,
    FiguresConfig,
    make_figures,
    population_band,
)
from herdmind.plots.runs import find_runs
from herdmind.plots.style import PlotStyle
from tests.plots.conftest import write_run

CONFIGS = Path(__file__).resolve().parents[2] / "configs"
FAST = PlotStyle(dpi=50, formats=("png",))


def test_population_band_hand_computed(runs_dir):
    a = [r for r in find_runs(runs_dir) if r.name == "a"]
    steps, mean, std = population_band(a, "sheep")
    # sheep per step: seed0 = 4, 6, 2; seed1 = 8, 2, 4
    np.testing.assert_array_equal(steps, [0, 1, 2])
    np.testing.assert_allclose(mean, [6.0, 4.0, 3.0])
    # sample std of two values x, y is |x - y| / sqrt(2)
    np.testing.assert_allclose(std, np.array([4.0, 4.0, 2.0]) / np.sqrt(2))


def test_population_band_single_run_has_zero_std(runs_dir):
    b = [r for r in find_runs(runs_dir) if r.name == "b"]
    _, mean, std = population_band(b, "wolf")
    np.testing.assert_allclose(mean, [1.0, 1.0])
    np.testing.assert_array_equal(std, [0.0, 0.0])


def test_population_band_cuts_to_shortest_run(tmp_path):
    d = tmp_path / "runs"
    write_run(d, "a", 0, [(2, 0), (4, 0), (6, 0)])
    write_run(d, "a", 1, [(4, 0), (8, 0)])
    steps, mean, _ = population_band(find_runs(d), "sheep")
    np.testing.assert_array_equal(steps, [0, 1])
    np.testing.assert_allclose(mean, [3.0, 6.0])


def test_population_band_rejects_empty():
    with pytest.raises(ValueError):
        population_band([], "sheep")


def test_make_figures_writes_files_and_manifest(runs_dir, tmp_path, caplog):
    out = tmp_path / "figs"
    cfg = FiguresConfig(runs_dir=str(runs_dir), out_dir=str(out), style=FAST)
    with caplog.at_level(logging.WARNING):
        written = make_figures(cfg)
    assert [p.name for p in written] == ["population.png", MANIFEST_FILE]
    manifest = json.loads((out / MANIFEST_FILE).read_text())
    assert manifest["figures"]["population"]["files"] == ["population.png"]
    assert len(manifest["figures"]["population"]["runs"]) == 3
    assert sorted((r["name"], r["seed"]) for r in manifest["runs"]) == [
        ("a", 0),
        ("a", 1),
        ("b", 0),
    ]
    assert "has 2 seeds" in caplog.text  # below the default min_seeds=5


def test_make_figures_unknown_figure(runs_dir, tmp_path):
    cfg = FiguresConfig(runs_dir=str(runs_dir), out_dir=str(tmp_path), figures=("nope",))
    with pytest.raises(KeyError):
        make_figures(cfg)


def test_make_figures_no_matching_runs(runs_dir, tmp_path):
    cfg = FiguresConfig(runs_dir=str(runs_dir), out_dir=str(tmp_path), run_names=("zzz",))
    with pytest.raises(FileNotFoundError):
        make_figures(cfg)


def test_shipped_config_loads():
    cfg = load_config(CONFIGS / "figures" / "default.yaml", FiguresConfig)
    assert cfg.figures == ("population",) and cfg.style.formats == ("png", "pdf")
