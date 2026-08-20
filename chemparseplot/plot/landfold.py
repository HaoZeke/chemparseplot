# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""Landfold FES figures via the same GP landscape as NEB.

Landfold owns ``F = -kT ln(rho/rhomax)``. This module turns the regular
``landfold.fes.v1`` grid into ``(s1, s2, F)`` observations with
finite-difference slopes and hands them to
:func:`chemparseplot.plot.neb.plot_landscape_surface`
(``project_path=False``: the plane is already the sketch-map). Surface
models come from ``rgpycrumbs.surfaces`` (default ``grad_imq``, same
as the eOn NEB plot TOML).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from chemparseplot.parse.landfold import load_fes_result
from chemparseplot.parse.types import LandfoldFesResult
from chemparseplot.plot.neb import SurfaceFitConfig, plot_landscape_surface
from chemparseplot.plot.theme import RUHI_THEME, get_theme, setup_publication_theme

__all__ = ["fes_observations", "plot_fes"]


def fes_observations(
    fes_result: LandfoldFesResult,
    *,
    fmax: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Finite FES cells as GP observations plus grid slopes.

    Returns ``(s1, s2, grad_s1, grad_s2, F)``. Slopes are ``np.gradient``
    on the regular landfold grid (the NEB path uses
    :func:`~chemparseplot.parse.neb_utils.compute_synthetic_gradients`
    because those points lie on a string, not a lattice).
    """
    z = np.array(fes_result.free_energy, dtype=float, copy=True)
    if fmax is not None:
        if not np.isfinite(fmax) or fmax <= 0.0:
            msg = "fmax must be finite and > 0"
            raise ValueError(msg)
        finite = np.isfinite(z)
        z[finite] = np.clip(z[finite], 0.0, fmax)
    if fes_result.x.size < 2 or fes_result.y.size < 2:
        msg = "landfold FES grid must be at least 2x2 for slopes"
        raise ValueError(msg)
    d_dy, d_dx = np.gradient(z, fes_result.y, fes_result.x)
    yy, xx = np.meshgrid(fes_result.y, fes_result.x, indexing="ij")
    keep = np.isfinite(z) & np.isfinite(d_dx) & np.isfinite(d_dy)
    if not np.any(keep):
        msg = "landfold FES has no finite cells to fit"
        raise ValueError(msg)
    return xx[keep], yy[keep], d_dx[keep], d_dy[keep], z[keep]


def plot_fes(
    fes_result: LandfoldFesResult | Mapping[str, Any],
    *,
    xlabel: str = r"$s_1$",
    ylabel: str = r"$s_2$",
    clabel: str | None = None,
    fmax: float | None = None,
    points: dict[str, tuple] | None = None,
    method: str = "grad_imq",
    surface_fit: SurfaceFitConfig | Mapping[str, Any] | None = None,
    rbf_smooth: float | None = None,
    n_inducing: int | None = None,
    show_pts: bool = False,
    figsize: tuple[float, float] = (5.6, 4.8),
    dpi: int = 170,
):
    """GP landscape of a landfold FES (same stack as NEB visuals).

    Parameters
    ----------
    fes_result
        :class:`~chemparseplot.parse.types.LandfoldFesResult` or a
        ``landfold.fes.v1`` mapping.
    method
        ``rgpycrumbs.surfaces`` name. Default ``grad_imq`` matches the
        eOn NEB plot TOML.
    surface_fit
        :class:`~chemparseplot.plot.neb.SurfaceFitConfig`. Default thins
        the dense histogram cloud to 300 fit points (Nyström inducing
        default).
    n_inducing
        Forwarded when the fit switches to ``grad_imq_ny``.
    show_pts
        Scatter the histogram cells (off: the FES is the figure).
    """
    import matplotlib.pyplot as plt

    setup_publication_theme(get_theme("ruhi"))
    result = (
        fes_result
        if isinstance(fes_result, LandfoldFesResult)
        else load_fes_result(fes_result)
    )
    s1, s2, g1, g2, z = fes_observations(result, fmax=fmax)
    fit = surface_fit or SurfaceFitConfig(auto_thin=True, max_surface_points=300)
    # NEB RMSD planes are a few angstrom; sketch-map s spans tens of units.
    # Default IMQ length 0.5 then predicts NaNs off the thinned cloud.
    if rbf_smooth is None:
        span = float(max(np.ptp(s1), np.ptp(s2)))
        rbf_smooth = max(0.1 * span, 1e-3)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, facecolor="white")
    plot_landscape_surface(
        ax,
        s1,
        s2,
        g1,
        g2,
        z,
        method=method,
        rbf_smooth=rbf_smooth,
        project_path=False,
        cmap=RUHI_THEME.cmap_landscape,
        show_pts=show_pts,
        surface_fit=fit,
        n_inducing=n_inducing,
        variance_threshold=0.5,
    )
    if points:
        for label, (xs, ys) in points.items():
            ax.scatter(xs, ys, s=12, label=label, zorder=50, linewidths=0)
        ax.legend(fontsize=8, loc="lower left", frameon=True)
    if clabel is None:
        clabel = r"$F/kT$" if abs(result.kt - 1.0) < 1e-12 else r"$F$"
    filled = next(
        (c for c in ax.collections if getattr(c, "filled", False)),
        ax.collections[0] if ax.collections else None,
    )
    if filled is not None:
        fig.colorbar(filled, ax=ax, fraction=0.046, pad=0.04).set_label(clabel)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    return fig
