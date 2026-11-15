import matplotlib as mpl
from matplotlib.figure import Figure

from herdmind.plots.style import PlotStyle, plot_style, rc_params, save_figure


def test_rc_params_follow_style():
    rc = rc_params(PlotStyle(width_in=4.0, height_in=2.0, font_size=10.0, dpi=150))
    assert rc["figure.figsize"] == (4.0, 2.0)
    assert rc["font.size"] == 10.0 and rc["savefig.dpi"] == 150
    assert rc["pdf.fonttype"] == 42


def test_plot_style_does_not_leak():
    before = mpl.rcParams["font.size"]
    with plot_style(PlotStyle(font_size=before + 3)):
        assert mpl.rcParams["font.size"] == before + 3
    assert mpl.rcParams["font.size"] == before


def test_save_figure_writes_every_format(tmp_path):
    fig = Figure()
    fig.subplots().plot([0, 1], [0, 1])
    with plot_style(PlotStyle(dpi=50)):
        paths = save_figure(fig, tmp_path / "out", "line", ("png", "pdf", "svg"))
    assert [p.name for p in paths] == ["line.png", "line.pdf", "line.svg"]
    assert all(p.stat().st_size > 0 for p in paths)


def test_saved_pdf_is_reproducible(tmp_path):
    """No timestamps in the file, so rebuilding a figure gives the same bytes."""

    def build(out):
        fig = Figure()
        fig.subplots().plot([0, 1], [1, 0])
        with plot_style(PlotStyle()):
            return save_figure(fig, out, "f", ("pdf", "svg"))

    first, second = build(tmp_path / "1"), build(tmp_path / "2")
    for a, b in zip(first, second, strict=True):
        assert a.read_bytes() == b.read_bytes()
