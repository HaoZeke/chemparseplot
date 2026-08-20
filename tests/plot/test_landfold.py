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
    rho = np.exp(-fes)
    return LandfoldFesResult.from_mapping(
        {
            "schema": LANDFOLD_FES_SCHEMA,
            "x": x,
            "y": y,
            "free_energy": fes,
            "density": rho,
            "kt": 1.0,
        }
    )


def test_fes_observations_fit_log_density_not_clipped_f() -> None:
    result = _bowl()
    s1, s2, g1, g2, z = fes_observations(result, on="density")
    assert s1.shape == s2.shape == g1.shape == g2.shape == z.shape
    assert s1.size == result.x.size * result.y.size
    mid = np.argmin(np.abs(s1) + np.abs(s2))
    assert z[mid] == pytest.approx(0.0, abs=0.05)
    interior = (np.abs(s1) > 0.3) & (np.abs(s2) < 0.3)
    assert np.corrcoef(g1[interior], 2.0 * s1[interior])[0, 1] > 0.9
    # fmax drops the ceiling from the fit set; it does not clip z.
    s1b, _, _, _, zb = fes_observations(result, on="density", fmax=1.0)
    assert s1b.size < s1.size
    assert zb.max() < 1.0


def test_cloud_observations_keep_finite_z() -> None:
    from chemparseplot.plot.landfold import cloud_observations

    s1, s2, g1, g2, z = cloud_observations(
        [0.0, 1.0, np.nan],
        [0.0, 1.0, 2.0],
        [0.5, 1.5, 9.0],
    )
    assert g1 is None and g2 is None
    np.testing.assert_allclose(z, [0.5, 1.5])


def test_plot_fes_uses_landscape_surface(monkeypatch) -> None:
    called = {}

    def fake_surface(ax, r, p, gr, gp, z, **kwargs):
        called["project_path"] = kwargs.get("project_path")
        called["method"] = kwargs.get("method")
        called["rbf_smooth"] = kwargs.get("rbf_smooth")
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
    assert called["rbf_smooth"] > 0.0
    assert called["n"] > 0
    assert fig.axes[0].get_xlabel() == r"$s_1$"
    assert fig.axes[0].get_ylabel() == r"$s_2$"


def test_plot_fes_rejects_bad_fmax() -> None:
    with pytest.raises(ValueError, match="fmax"):
        plot_fes(_bowl(), fmax=0.0)
