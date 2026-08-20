# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""landfold FES plotter consumes landfold.fes.v1 and returns a figure."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pytest

pytest.importorskip("matplotlib")

from chemparseplot.parse.types import LANDFOLD_FES_SCHEMA, LandfoldFesResult
from chemparseplot.plot.landfold import fes_observations, plot_fes

pytestmark = pytest.mark.pure


def _bowl() -> LandfoldFesResult:
    x = np.linspace(-1.0, 1.0, 12)
    y = np.linspace(-1.0, 1.0, 10)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    fes = xx**2 + yy**2
    return LandfoldFesResult.from_mapping(
        {
            "schema": LANDFOLD_FES_SCHEMA,
            "x": x,
            "y": y,
            "free_energy": fes,
            "kt": 1.0,
        }
    )


def test_fes_observations_match_grid_slopes() -> None:
    result = _bowl()
    s1, s2, g1, g2, z = fes_observations(result)
    assert s1.shape == s2.shape == g1.shape == g2.shape == z.shape
    assert s1.size == result.x.size * result.y.size
    # Bowl: dF/ds1 ~ 2 s1 at interior points.
    mid = np.argmin(np.abs(s1) + np.abs(s2))
    assert z[mid] == pytest.approx(0.0, abs=0.05)
    interior = (np.abs(s1) > 0.3) & (np.abs(s2) < 0.3)
    assert np.corrcoef(g1[interior], 2.0 * s1[interior])[0, 1] > 0.9


def test_plot_fes_uses_landscape_surface(monkeypatch) -> None:
    called = {}

    def fake_surface(ax, r, p, gr, gp, z, **kwargs):
        called["project_path"] = kwargs.get("project_path")
        called["method"] = kwargs.get("method")
        called["n"] = len(r)
        called["has_grad"] = gr is not None and gp is not None
        ax.contourf([[0.0, 1.0], [0.0, 1.0]], [[0.0, 0.0], [1.0, 1.0]], [[0.0, 1.0], [1.0, 2.0]])

    monkeypatch.setattr(
        "chemparseplot.plot.landfold.plot_landscape_surface", fake_surface
    )
    fig = plot_fes(_bowl(), fmax=2.0, clabel=r"$F/kT$", method="grad_imq")
    assert called["project_path"] is False
    assert called["method"] == "grad_imq"
    assert called["has_grad"] is True
    assert called["n"] > 0
    assert fig.axes[0].get_xlabel() == r"$s_1$"
    assert fig.axes[0].get_ylabel() == r"$s_2$"


def test_plot_fes_rejects_bad_fmax() -> None:
    with pytest.raises(ValueError, match="fmax"):
        plot_fes(_bowl(), fmax=0.0)
