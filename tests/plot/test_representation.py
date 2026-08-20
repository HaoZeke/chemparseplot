# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""Energy representation plot: E on the metric plane, not occupancy F."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pytest

from chemparseplot.parse.types import (
    LANDFOLD_FES_SCHEMA,
    EnergyRepresentation,
    LandfoldFesResult,
)
from chemparseplot.plot.representation import plot_energy

pytestmark = pytest.mark.pure


def _path() -> EnergyRepresentation:
    r = np.array([0.0, 1.0, 2.0, 3.0])
    p = np.array([3.0, 2.0, 1.0, 0.0])
    e = np.array([0.0, 0.5, 0.4, -0.2])
    f = np.array([0.2, 0.1, 0.0, -0.1])
    return EnergyRepresentation(
        x=r,
        y=p,
        energy=e,
        f_para=f,
        frame="rmsd",
        xlabel=r"RMSD-R",
        ylabel=r"RMSD-P",
    )


def test_plot_energy_rejects_occupancy_fes() -> None:
    x = np.linspace(-1.0, 1.0, 4)
    y = np.linspace(-1.0, 1.0, 3)
    fes = np.ones((3, 4))
    result = LandfoldFesResult.from_mapping(
        {
            "schema": LANDFOLD_FES_SCHEMA,
            "x": x,
            "y": y,
            "free_energy": fes,
            "density": np.exp(-fes),
            "kt": 1.0,
        }
    )
    with pytest.raises(TypeError, match="EnergyRepresentation"):
        plot_energy(result)


def test_plot_energy_uses_landscape_of_e(monkeypatch) -> None:
    called = {}

    def fake_surface(ax, r, p, gr, gp, z, **kwargs):
        called["project_path"] = kwargs.get("project_path")
        called["method"] = kwargs.get("method")
        called["z"] = np.asarray(z)
        called["has_grad"] = gr is not None and gp is not None
        ax.contourf(
            [[0.0, 1.0], [0.0, 1.0]],
            [[0.0, 0.0], [1.0, 1.0]],
            [[0.0, 1.0], [1.0, 2.0]],
        )

    monkeypatch.setattr(
        "chemparseplot.plot.representation.plot_landscape_surface", fake_surface
    )
    fig = plot_energy(_path(), method="grad_imq")
    assert called["project_path"] is True
    assert called["method"] == "grad_imq"
    assert called["has_grad"] is True
    np.testing.assert_allclose(called["z"], [0.0, 0.5, 0.4, -0.2])
    assert fig.axes[0].get_xlabel() == r"$s$"
    assert fig.axes[0].get_ylabel() == r"$d$"
    cbar = fig.axes[-1]
    assert cbar.get_ylabel() == r"$E$"


def test_plot_energy_landfold_plane_stays_unrotated(monkeypatch) -> None:
    called = {}

    def fake_surface(ax, r, p, gr, gp, z, **kwargs):
        called["project_path"] = kwargs.get("project_path")
        z = [[0.0, 0.0], [1.0, 1.0]]
        ax.contourf([[0.0, 1.0], [0.0, 1.0]], [[0.0, 0.0], [1.0, 1.0]], z)

    monkeypatch.setattr(
        "chemparseplot.plot.representation.plot_landscape_surface", fake_surface
    )
    rep = EnergyRepresentation(
        x=np.array([0.0, 1.0, 2.0]),
        y=np.array([0.5, 0.4, 0.1]),
        energy=np.array([-1.0, -0.5, -1.2]),
        frame="landfold",
        xlabel=r"$s_1$",
        ylabel=r"$s_2$",
    )
    fig = plot_energy(rep)
    assert called["project_path"] is False
    assert fig.axes[0].get_xlabel() == r"$s_1$"
    assert fig.axes[0].get_ylabel() == r"$s_2$"


def test_plot_energy_basin_coordinate_labels_xi(monkeypatch) -> None:
    called = {}

    def fake_surface(ax, r, p, gr, gp, z, **kwargs):
        called["z"] = np.asarray(z)
        ax.contourf(
            [[0.0, 1.0], [0.0, 1.0]],
            [[0.0, 0.0], [1.0, 1.0]],
            [[0.0, 0.5], [0.5, 1.0]],
        )

    monkeypatch.setattr(
        "chemparseplot.plot.representation.plot_landscape_surface", fake_surface
    )
    from chemparseplot.parse.representation import from_descriptor_cloud

    fig = plot_energy(
        from_descriptor_cloud(
            [0.0, 1.0, 2.0],
            [0.0, 0.5, 1.0],
            [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]],
            ref_a=[1.0, 0.0],
            ref_b=[0.0, 1.0],
        )
    )
    np.testing.assert_allclose(called["z"][0], 0.0, atol=1e-12)
    np.testing.assert_allclose(called["z"][-1], 1.0, atol=1e-12)
    assert fig.axes[-1].get_ylabel() == r"$\xi$"
