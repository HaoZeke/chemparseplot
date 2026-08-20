# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""Plot an energy representation: E on a metric 2D plane.

This is the MethodsX landscape: energies and synthetic path-tangent
gradients feed a gradient-enhanced IMQ-GP. Occupancy invert is not a
valid field here.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from chemparseplot.parse.neb_utils import compute_synthetic_gradients
from chemparseplot.parse.types import EnergyRepresentation, LandfoldFesResult
from chemparseplot.plot.neb import SurfaceFitConfig, plot_landscape_surface
from chemparseplot.plot.theme import RUHI_THEME, get_theme, setup_publication_theme

__all__ = ["plot_energy"]


def plot_energy(
    rep: EnergyRepresentation | Mapping[str, Any],
    *,
    project_path: bool | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    clabel: str | None = None,
    method: str = "grad_imq",
    surface_fit: SurfaceFitConfig | Mapping[str, Any] | None = None,
    rbf_smooth: float | None = None,
    n_inducing: int | None = None,
    show_pts: bool = True,
    extra_points: dict[str, tuple] | None = None,
    figsize: tuple[float, float] = (5.6, 4.8),
    dpi: int = 170,
):
    """GP landscape of potential energy on the representation plane.

    Parameters
    ----------
    rep
        :class:`~chemparseplot.parse.types.EnergyRepresentation` or a
        ``chemparseplot.energy.v1`` mapping. Occupancy FES results are
        rejected.
    project_path
        Rotate RMSD ``(r, p)`` to ``(s, d)``. Default true for
        ``frame="rmsd"``, false for an already-rotated or landfold plane.
    """
    import matplotlib.pyplot as plt

    if isinstance(rep, LandfoldFesResult):
        msg = "plot_energy needs EnergyRepresentation, not a landfold occupancy FES"
        raise TypeError(msg)
    if not isinstance(rep, EnergyRepresentation):
        try:
            rep = EnergyRepresentation.from_mapping(rep)
        except (TypeError, ValueError, KeyError) as exc:
            msg = "plot_energy needs EnergyRepresentation"
            raise TypeError(msg) from exc

    setup_publication_theme(get_theme("ruhi"))
    rotate = project_path if project_path is not None else rep.frame == "rmsd"
    x, y = rep.x, rep.y
    gx, gy = rep.grad_x, rep.grad_y
    if (gx is None or gy is None) and rep.f_para is not None:
        gx, gy = compute_synthetic_gradients(x, y, rep.f_para)
    if gx is None and method.startswith("grad_"):
        method = "rbf"
    if rbf_smooth is None:
        span = float(max(np.ptp(x), np.ptp(y)))
        rbf_smooth = max(0.1 * span, 1e-3)
    fit = (
        surface_fit
        if isinstance(surface_fit, SurfaceFitConfig)
        else SurfaceFitConfig.from_mapping(surface_fit)
        if surface_fit is not None
        else SurfaceFitConfig(auto_thin=False, max_surface_points=300)
    )
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, facecolor="white")
    extra = None
    scatter_pts = extra_points
    if extra_points:
        extra = np.column_stack(
            [
                np.concatenate([np.atleast_1d(pt[0]) for pt in extra_points.values()]),
                np.concatenate([np.atleast_1d(pt[1]) for pt in extra_points.values()]),
            ]
        )
        if rotate:
            from chemparseplot.parse.projection import (
                compute_projection_basis,
                project_to_sd,
            )

            basis = compute_projection_basis(x, y)
            scatter_pts = {
                label: project_to_sd(np.atleast_1d(pt[0]), np.atleast_1d(pt[1]), basis)
                for label, pt in extra_points.items()
            }
    plot_landscape_surface(
        ax,
        x,
        y,
        gx,
        gy,
        rep.energy,
        step_data=rep.step,
        method=method,
        rbf_smooth=rbf_smooth,
        project_path=rotate,
        cmap=RUHI_THEME.cmap_landscape,
        show_pts=show_pts,
        surface_fit=fit,
        n_inducing=n_inducing,
        extra_points=extra,
        variance_threshold=0.5,
    )
    if scatter_pts:
        for label, (xs, ys) in scatter_pts.items():
            ax.scatter(xs, ys, s=36, marker="*", label=label, zorder=50)
        ax.legend(fontsize=8, loc="best", frameon=True)
    if rotate:
        ax.set_xlabel(xlabel or r"$s$")
        ax.set_ylabel(ylabel or r"$d$")
    else:
        ax.set_xlabel(xlabel or rep.xlabel)
        ax.set_ylabel(ylabel or rep.ylabel)
    filled = next(
        (c for c in ax.collections if getattr(c, "filled", False)),
        ax.collections[0] if ax.collections else None,
    )
    if filled is not None:
        fig.colorbar(filled, ax=ax, fraction=0.046, pad=0.04).set_label(
            clabel if clabel is not None else r"$E$"
        )
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    return fig
