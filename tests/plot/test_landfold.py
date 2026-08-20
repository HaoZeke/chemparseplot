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

from chemparseplot.parse.types import LANDFOLD_FES_SCHEMA
from chemparseplot.plot.landfold import plot_fes

pytestmark = pytest.mark.pure


def test_plot_fes_from_v1_mapping() -> None:
    x = np.linspace(-1.0, 1.0, 12)
    y = np.linspace(-1.0, 1.0, 10)
    xx, yy = np.meshgrid(x, y)
    fes = xx**2 + yy**2
    fig = plot_fes(
        {
            "schema": LANDFOLD_FES_SCHEMA,
            "x": x,
            "y": y,
            "free_energy": fes,
            "kt": 1.0,
        },
        fmax=2.0,
        clabel=r"$F/kT$",
    )
    ax = fig.axes[0]
    assert len(ax.collections) >= 1
    assert ax.get_xlabel() == r"$s_1$"
    assert ax.get_ylabel() == r"$s_2$"


def test_plot_fes_rejects_bad_fmax() -> None:
    with pytest.raises(ValueError, match="fmax"):
        plot_fes(
            {
                "schema": LANDFOLD_FES_SCHEMA,
                "x": [0.0, 1.0],
                "y": [0.0, 1.0],
                "free_energy": [[0.0, 0.5], [0.5, 1.0]],
            },
            fmax=0.0,
        )
