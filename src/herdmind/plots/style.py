"""One plot style for every figure, so plots from different scripts and people match.

The style is applied with a context manager instead of changing matplotlib's
global ``rcParams``, so importing this module never changes anyone else's plots.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
from matplotlib.figure import Figure

# Colour-blind-safe pair (Okabe–Ito blue and vermillion); one colour per species everywhere.
SPECIES_COLORS: Mapping[str, str] = {"sheep": "#0072B2", "wolf": "#D55E00"}
# Line colour for anything that is not a species (e.g. a reference line).
NEUTRAL_COLOR = "#555555"

# Strip timestamps so the same runs give byte-identical figure files.
_SAVE_METADATA: Mapping[str, Mapping[str, None]] = {
    "pdf": {"CreationDate": None, "ModDate": None},
    "svg": {"Date": None},
}


@dataclass(frozen=True)
class PlotStyle:
    """Sizes and output settings shared by all figures.

    The default width fits one column of a two-column paper.
    """

    width_in: float = 3.5
    height_in: float = 2.4
    font_size: float = 8.0
    line_width: float = 1.2
    band_alpha: float = 0.25  # opacity of mean ± std bands
    dpi: int = 300
    formats: tuple[str, ...] = ("png", "pdf")


def rc_params(style: PlotStyle) -> dict[str, object]:
    """Matplotlib settings for ``style``; a pure function so it can be tested and reused."""
    return {
        "figure.figsize": (style.width_in, style.height_in),
        "figure.dpi": style.dpi,
        "savefig.dpi": style.dpi,
        "savefig.bbox": "tight",
        "font.size": style.font_size,
        "axes.titlesize": style.font_size,
        "axes.labelsize": style.font_size,
        "legend.fontsize": style.font_size - 1,
        "xtick.labelsize": style.font_size - 1,
        "ytick.labelsize": style.font_size - 1,
        "lines.linewidth": style.line_width,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "legend.frameon": False,
        # Embed TrueType fonts so PDFs pass journal checks.
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        # Same SVG ids on every run, so regenerated files diff cleanly.
        "svg.hashsalt": "herdmind",
    }


@contextmanager
def plot_style(style: PlotStyle) -> Iterator[None]:
    """Apply ``style`` while building and saving figures, then restore the old settings."""
    with mpl.rc_context(rc_params(style)):
        yield


def save_figure(fig: Figure, out_dir: Path, name: str, formats: Sequence[str]) -> list[Path]:
    """Write ``fig`` as ``out_dir/name.<fmt>`` for every format.

    Args:
        fig: The figure; call inside ``plot_style`` so the style's save settings apply.
        out_dir: Folder to write into; created if missing.
        name: File name without extension.
        formats: File extensions matplotlib can write, e.g. ``("png", "pdf")``.

    Returns:
        The written paths, in the order of ``formats``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in formats:
        path = out_dir / f"{name}.{fmt}"
        fig.savefig(path, format=fmt, metadata=_SAVE_METADATA.get(fmt))
        paths.append(path)
    return paths
