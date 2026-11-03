# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""Behaviour of the NEB profile and energy landscape figures.

Every figure is drawn from the Au/Al(100) EMT runs in data/au_al100.json,
written by data/au_al100.py.
"""

import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_hex

pytest.importorskip("scipy")
pytest.importorskip("ase")

from chemparseplot.plot import journal as jr
from chemparseplot.plot.chemgp import plot_surface_contour
from chemparseplot.plot.neb import _tick_decimals, plot_neb_profile
from chemparseplot.plot.palette import RUHI_CYCLE

DATA = Path(__file__).parent / "data"
FACE = "DejaVu Sans"


@pytest.fixture(scope="module")
def run():
    return json.loads((DATA / "au_al100.json").read_text())


@pytest.fixture(autouse=True)
def dejavu(monkeypatch):
    monkeypatch.setattr(jr, "resolve_font", lambda candidates: FACE)
    with mpl.rc_context():
        yield
    plt.close("all")


def _band(row, e0):
    return np.array(row["s"]), np.array(row["energies"]) - e0, np.array(row["f_para"])


@pytest.fixture
def final(run):
    """The converged BFGS band: path length, relative energy, force."""
    return _band(run["neb"]["BFGS"]["history"][-1], run["e_initial"])


def _band_line(ax):
    return max(ax.get_lines(), key=lambda line: len(line.get_xdata()))


class TestNebProfile:
    def test_column_width_and_sampling(self, final):
        fig = plot_neb_profile(*final)
        assert fig.get_figwidth() == jr.ACS.width("single")
        band = _band_line(fig.axes[0])
        pixels = math.ceil(fig.get_figwidth() * jr.ACS.raster_dpi)
        assert len(band.get_xdata()) == pixels + 1
        style = jr.figure_style()
        assert band.get_linewidth() == style.wide
        assert band.get_color() == RUHI_CYCLE[0]

    def test_barrier_has_one_more_decimal_than_the_ticks(self, final):
        energy = final[1]
        fig = plot_neb_profile(*final)
        ax = fig.axes[0]
        fig.canvas.draw()
        (label,) = [t for t in ax.texts if "ddagger" in t.get_text()]
        value = re.search(r"= ([\d.]+)\$", label.get_text()).group(1)
        decimals = len(value.split(".")[1])
        assert decimals == _tick_decimals(ax.yaxis) + 1
        barrier = energy.max() - energy[0]
        assert float(value) == pytest.approx(barrier, abs=0.5 * 10.0**-decimals)
        renderer = fig.canvas.get_renderer()
        assert label.get_window_extent(renderer).y1 <= ax.get_window_extent(renderer).y1

    def test_no_barrier_when_the_peak_is_an_end(self, final):
        # From the hollow to the bridge saddle the band rises throughout.
        half = int(np.argmax(final[1])) + 1
        fig = plot_neb_profile(*(column[:half] for column in final))
        assert not [t for t in fig.axes[0].texts if "ddagger" in t.get_text()]

    def test_history_reference_and_label(self, run, final):
        e0 = run["e_initial"]
        rows = run["neb"]["BFGS"]["history"]
        earlier = [rows[0], rows[-2]]
        steps = [row["iteration"] for row in earlier]
        reference = run["saddle"]["energy"] - e0
        # The first earlier band goes in without forces, the second with them.
        history = [_band(earlier[0], e0)[:2], _band(earlier[1], e0)]
        fig = plot_neb_profile(
            *final,
            history=history,
            iterations=steps,
            reference=reference,
            reference_label="Bridge saddle",
            label="BFGS",
        )
        ax, cbar_ax = fig.axes
        assert cbar_ax.get_ylabel() == "Iteration"
        assert cbar_ax.get_ylim() == (float(steps[0]), float(steps[1]))
        style = jr.figure_style()
        context = [ln for ln in ax.get_lines() if ln.get_linewidth() == style.narrow]
        (ref,) = [ln for ln in context if ln.get_linestyle() == "--"]
        assert ref.get_ydata()[0] == reference
        texts = [t.get_text() for t in ax.texts]
        legend = ax.get_legend()
        if legend is not None:
            texts += [t.get_text() for t in legend.get_texts()]
        assert "Bridge saddle" in texts
        assert "BFGS" in texts

    def test_given_axes_keep_the_callers_rcparams(self, final):
        fig, ax = plt.subplots()
        size = mpl.rcParams["font.size"]
        out = plot_neb_profile(*final, ax=ax)
        assert out is fig
        assert mpl.rcParams["font.size"] == size

    def test_curve_fallbacks(self, final):
        rc, energy, _ = final
        fig = plot_neb_profile(rc, energy)
        assert len(_band_line(fig.axes[0]).get_xdata()) > len(rc)
        fig = plot_neb_profile(rc[:3], energy[:3], method="spline")
        assert len(_band_line(fig.axes[0]).get_xdata()) == len(rc[:3])


class TestSurfaceContour:
    @pytest.fixture
    def surface(self, run):
        """The converged adatom map, relative to its lowest node."""
        grid = run["map"]
        x, y = np.meshgrid(grid["x"], grid["y"])
        z = np.array(grid["energy"])
        return x, y, z - z.min()

    def test_shared_levels_and_line_colours(self, surface):
        fig = plot_surface_contour(*surface)
        ax = fig.axes[0]
        filled, lines = ax.collections[:2]
        style = jr.figure_style()
        assert set(lines.levels) <= set(filled.levels)
        colours = {to_hex(c) for c in lines.get_edgecolor()}
        assert colours <= {to_hex(style.textcolor), to_hex(style.facecolor)}

    def test_colour_bar_matches_the_map(self, surface):
        fig = plot_surface_contour(*surface)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        ax = fig.axes[0]
        (cax,) = ax.child_axes
        box, cbox = ax.get_window_extent(renderer), cax.get_window_extent(renderer)
        assert cbox.y0 == pytest.approx(box.y0)
        assert cbox.y1 == pytest.approx(box.y1)

    def test_square_map_is_cropped_to_its_content(self, surface):
        fig = plot_surface_contour(*surface)
        assert fig.get_figwidth() == jr.ACS.width("single")
        assert fig.get_figheight() < fig.get_figwidth()
