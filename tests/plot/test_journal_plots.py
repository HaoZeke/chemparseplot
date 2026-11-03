# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""Behaviour of the energy landscape figure.

The map is the converged adatom map of the Au/Al(100) EMT runs in
data/au_al100.json, written by data/au_al100.py.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_hex

from chemparseplot.plot import journal as jr
from chemparseplot.plot.chemgp import plot_surface_contour

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
