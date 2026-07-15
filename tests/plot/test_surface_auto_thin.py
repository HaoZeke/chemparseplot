# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""Tests for surface fit thinning (auto_thin)."""

from __future__ import annotations

import sys
import types

import matplotlib.pyplot as plt
import numpy as np
import pytest


class TestSurfaceFitIndices:
    def test_short_cloud_unchanged(self):
        from chemparseplot.plot.neb import surface_fit_indices

        idx = surface_fit_indices(10, 64)
        assert idx.tolist() == list(range(10))

    def test_dense_cloud_preserves_endpoints_and_cap(self):
        from chemparseplot.plot.neb import surface_fit_indices

        n, max_pts = 194, 64
        idx = surface_fit_indices(n, max_pts)
        assert idx[0] == 0
        assert idx[-1] == n - 1
        assert len(idx) <= max_pts
        assert len(idx) == len(np.unique(idx))
        assert np.all(np.diff(idx) > 0)

    def test_max_points_too_small(self):
        from chemparseplot.plot.neb import surface_fit_indices

        with pytest.raises(ValueError, match="max_points"):
            surface_fit_indices(10, 1)


def _install_dummy_surfaces(seen: dict):
    class DummyModel:
        def __init__(self, **kwargs):
            x = kwargs.get("x", kwargs.get("x_obs"))
            seen["n"] = len(x)
            self.ls = 0.5
            self.noise = 1e-2

        def __call__(self, pts):
            return np.zeros(len(pts))

        def predict_var(self, pts):
            return np.ones(len(pts)) * 0.01

    fake = types.ModuleType("rgpycrumbs.surfaces")
    fake.NYSTROM_N_INDUCING_DEFAULT = 300
    fake.NYSTROM_THRESHOLD = 1000
    fake.get_surface_model = lambda method: DummyModel
    fake.nystrom_paths_needed = lambda *a, **k: 1
    sys.modules["rgpycrumbs.surfaces"] = fake
    if "rgpycrumbs" not in sys.modules:
        sys.modules["rgpycrumbs"] = types.ModuleType("rgpycrumbs")
    return DummyModel


class TestPlotLandscapeSurfaceAutoThin:
    def test_auto_thin_false_by_default_uses_all_points(self):
        from chemparseplot.plot import neb as neb_mod

        seen: dict = {}
        _install_dummy_surfaces(seen)

        n = 100
        r = np.linspace(0.0, 1.0, n)
        p = np.linspace(1.0, 0.0, n)
        z = np.linspace(0.0, 0.2, n)
        gr = np.zeros(n)
        gp = np.zeros(n)
        fig, ax = plt.subplots()
        neb_mod.plot_landscape_surface(
            ax, r, p, gr, gp, z, method="rbf", project_path=False, show_pts=False
        )
        assert seen["n"] == n
        plt.close(fig)

    def test_auto_thin_reduces_fit_points(self):
        from chemparseplot.plot import neb as neb_mod

        seen: dict = {}
        _install_dummy_surfaces(seen)

        n = 194
        r = np.linspace(0.0, 1.0, n)
        p = np.linspace(1.0, 0.0, n)
        z = np.linspace(0.0, 0.2, n)
        gr = np.zeros(n)
        gp = np.zeros(n)
        fig, ax = plt.subplots()
        neb_mod.plot_landscape_surface(
            ax,
            r,
            p,
            gr,
            gp,
            z,
            method="rbf",
            project_path=False,
            show_pts=False,
            auto_thin=True,
            max_surface_points=64,
        )
        assert seen["n"] <= 64
        assert seen["n"] < n
        plt.close(fig)
