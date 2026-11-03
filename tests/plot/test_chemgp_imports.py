# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""The chemgp surface plots import and draw without plotnine and pandas."""

import importlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest

import chemparseplot.plot as plot_pkg

MODULE = "chemparseplot.plot.chemgp"
DATA = Path(__file__).parent / "data"


@pytest.fixture
def chemgp_alone(monkeypatch):
    """chemgp imported afresh while plotnine and pandas cannot be imported."""
    for name in ("plotnine", "pandas"):
        monkeypatch.setitem(sys.modules, name, None)
    monkeypatch.delitem(sys.modules, MODULE, raising=False)
    monkeypatch.delattr(plot_pkg, "chemgp", raising=False)
    return importlib.import_module(MODULE)


def test_surface_contour_without_plotnine(chemgp_alone):
    # The converged Au/Al(100) adatom map written by data/au_al100.py.
    grid = json.loads((DATA / "au_al100.json").read_text())["map"]
    gx, gy = np.meshgrid(grid["x"], grid["y"])
    z = np.array(grid["energy"])
    with mpl.rc_context():
        fig = chemgp_alone.plot_surface_contour(gx, gy, z - z.min())
    assert fig.axes
    plt.close(fig)


def test_line_charts_ask_for_plotnine(chemgp_alone):
    with pytest.raises(ImportError):
        chemgp_alone._gg()
