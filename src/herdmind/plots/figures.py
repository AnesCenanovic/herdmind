"""Figure registry and the ``make_figures`` pipeline.

Each figure is a function ``(runs, cfg) -> Figure`` registered in ``FIGURES``.
It gets loaded run folders only, so a figure cannot depend on state that a run
did not write down. ``make_figures`` builds the figures a config asks for and
writes ``manifest.json`` naming the runs (and commits) behind each file.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from herdmind.plots.runs import RunData, find_runs, group_by_name
from herdmind.plots.style import SPECIES_COLORS, PlotStyle, plot_style, save_figure

logger = logging.getLogger(__name__)

MANIFEST_FILE = "manifest.json"
SPECIES = ("sheep", "wolf")


@dataclass(frozen=True)
class FiguresConfig:
    """Which runs to read, which figures to build and where to write them."""

    runs_dir: str = "runs"
    # Inside runs_dir is safe: find_runs skips folders without meta.json + ledger.
    out_dir: str = "runs/figures"
    run_names: tuple[str, ...] = ()  # world names to include; empty = every run found
    figures: tuple[str, ...] = ("population",)
    min_seeds: int = 5  # fewer seeds per world than this is logged as a warning
    style: PlotStyle = field(default_factory=PlotStyle)


FigureFn = Callable[[Sequence[RunData], FiguresConfig], Figure]


def population_band(
    runs: Sequence[RunData], species: str
) -> tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.float64]]:
    """Mean and sample std (ddof=1) of a species' head count per step, across seeds.

    Runs of different length are cut to the shortest, so every point averages
    the same seeds. With one run the std is zero.

    Args:
        runs: Seeds of one world.
        species: ``"sheep"`` or ``"wolf"``.

    Returns:
        ``(steps, mean, std)``, all of the same length.

    Raises:
        ValueError: If ``runs`` is empty or a run has no boards.
    """
    if not runs or any(not r.boards for r in runs):
        raise ValueError("population_band needs at least one run with a non-empty ledger")
    n_steps = min(len(r.boards) for r in runs)
    if any(len(r.boards) != n_steps for r in runs):
        logger.warning("runs differ in length; cutting all to %d boards", n_steps)
    counts = np.array(
        [[b.count(species) for b in r.boards[:n_steps]] for r in runs], dtype=np.float64
    )
    steps = np.array([b.step for b in runs[0].boards[:n_steps]], dtype=np.int64)
    std = counts.std(axis=0, ddof=1) if len(runs) > 1 else np.zeros(n_steps)
    return steps, counts.mean(axis=0), std


def population_figure(runs: Sequence[RunData], cfg: FiguresConfig) -> Figure:
    """Sheep and wolf counts over time, mean ± std across seeds, one panel per world.

    Caption: line = mean head count over seeds, band = ± one sample std; n = seeds.
    """
    groups = group_by_name(runs)
    fig = Figure(figsize=(cfg.style.width_in, cfg.style.height_in * len(groups)))
    axes = fig.subplots(len(groups), 1, squeeze=False, sharex=True)[:, 0]
    for ax, (name, group) in zip(axes, groups.items(), strict=True):
        for species in SPECIES:
            steps, mean, std = population_band(group, species)
            color = SPECIES_COLORS[species]
            ax.plot(steps, mean, color=color, label=species)
            # Head counts cannot be negative, so the band stops at zero.
            low = np.maximum(mean - std, 0.0)
            ax.fill_between(steps, low, mean + std, color=color, alpha=cfg.style.band_alpha, lw=0)
        ax.set_title(f"{name} (n={len(group)} seeds)")
        ax.set_ylabel("animals")
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("step")
    return fig


FIGURES: Mapping[str, FigureFn] = {"population": population_figure}


def _check_seeds(runs: Sequence[RunData], min_seeds: int) -> None:
    """Results must aggregate several seeds; say loudly when a world has too few."""
    for name, group in group_by_name(runs).items():
        if len(group) < min_seeds:
            logger.warning("world %r has %d seeds (< %d)", name, len(group), min_seeds)


def make_figures(cfg: FiguresConfig) -> list[Path]:
    """Build every figure in ``cfg.figures`` from the runs under ``cfg.runs_dir``.

    Args:
        cfg: The figures config.

    Returns:
        All written figure files, plus the manifest last.

    Raises:
        KeyError: If a figure name is not in ``FIGURES``.
        FileNotFoundError: If no run folders match.
    """
    unknown = [f for f in cfg.figures if f not in FIGURES]
    if unknown:
        raise KeyError(f"unknown figures {unknown}; known: {sorted(FIGURES)}")
    runs = find_runs(cfg.runs_dir, cfg.run_names)
    if not runs:
        raise FileNotFoundError(f"no run folders in {cfg.runs_dir} matching {cfg.run_names}")
    _check_seeds(runs, cfg.min_seeds)

    out_dir = Path(cfg.out_dir)
    written: list[Path] = []
    figures: dict[str, dict[str, list[str]]] = {}
    with plot_style(cfg.style):
        for name in cfg.figures:
            fig = FIGURES[name](runs, cfg)
            files = save_figure(fig, out_dir, name, cfg.style.formats)
            written += files
            # Per figure, so a later figure that reads only some runs stays traceable.
            figures[name] = {
                "files": [p.name for p in files],
                "runs": [r.path.name for r in runs],
            }
            logger.info("wrote %s", ", ".join(str(p) for p in files))

    manifest = {
        "config": asdict(cfg),
        "figures": figures,
        "runs": [
            {
                "path": r.path.as_posix(),
                "name": r.name,
                "seed": r.seed,
                "commit": r.commit,
                "dirty": r.dirty,
            }
            for r in runs
        ],
    }
    manifest_path = out_dir / MANIFEST_FILE
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return [*written, manifest_path]
