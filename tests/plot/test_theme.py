# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""Regression checks for the Ruhi theme colour maps."""

import inspect

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_hex

from chemparseplot.plot.chemgp import (
    plot_gp_progression,
    plot_surface_contour,
    plot_variance_overlay,
)
from chemparseplot.plot.journal import figure_style
from chemparseplot.plot.neb import (
    plot_landscape_path_overlay,
    plot_landscape_surface,
    plot_mmf_peaks_overlay,
)
from chemparseplot.plot.optimization import (
    plot_optimization_landscape,
    render_single_ended_landscape,
)
from chemparseplot.plot.palette import RUHI_SEQUENTIAL
from chemparseplot.plot.theme import RUHI_THEME

# The linear blend registered as ruhi_diverging, teal through sunshine.
_BLEND = "ruhi_diverging"
_DIVERGING_START = "#004D40"
_DIVERGING_END = "#F1DB4B"


def test_ruhi_theme_uses_the_diverging_blend():
    assert RUHI_THEME.cmap_profile == RUHI_THEME.cmap_landscape == _BLEND
    cmap = mpl.colormaps[RUHI_THEME.cmap_profile]
    assert to_hex(cmap(0.0)).upper() == _DIVERGING_START
    assert to_hex(cmap(1.0)).upper() == _DIVERGING_END
    assert RUHI_SEQUENTIAL in mpl.colormaps


def test_surface_defaults_use_ruhi_diverging():
    for fn in (
        plot_landscape_surface,
        plot_mmf_peaks_overlay,
        plot_optimization_landscape,
        render_single_ended_landscape,
    ):
        assert inspect.signature(fn).parameters["cmap"].default == _BLEND
    assert figure_style("acs").rc()["image.cmap"] == _BLEND


def test_unknown_landscape_cmap_falls_back_to_ruhi_diverging():
    fig, ax = plt.subplots()
    plot_landscape_path_overlay(
        ax,
        np.array([0.0, 1.0, 2.0]),
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.2]),
        "not-a-cmap",
        "E",
        project_path=False,
    )
    drawn = [coll.get_cmap().name for coll in ax.collections if hasattr(coll, "get_cmap")]
    assert _BLEND in drawn
    plt.close(fig)


def _filled_cmap_names(fig):
    names = []
    for ax in fig.axes:
        for coll in getattr(ax, "collections", ()):
            getter = getattr(coll, "get_cmap", None)
            if getter is None:
                continue
            cmap = getter()
            if cmap is not None:
                names.append(cmap.name)
    return names


def test_chemgp_energy_surfaces_use_ruhi_diverging():
    x = np.linspace(-1.0, 1.0, 12)
    grid_x, grid_y = np.meshgrid(x, x)
    energy = grid_x**2 + grid_y**2
    mean = {"gp_mean": energy, "train_x": [0.0], "train_y": [0.0]}

    surface = plot_surface_contour(grid_x, grid_y, energy)
    progression = plot_gp_progression({1: mean}, energy, x, x, clamp_lo=0.0, clamp_hi=2.0)
    variance = plot_variance_overlay(
        grid_x, grid_y, energy, energy, clamp_lo=0.0, clamp_hi=2.0
    )
    try:
        assert _BLEND in _filled_cmap_names(surface)
        assert _BLEND in _filled_cmap_names(progression)
        assert _BLEND in _filled_cmap_names(variance)
    finally:
        plt.close(surface)
        plt.close(progression)
        plt.close(variance)
