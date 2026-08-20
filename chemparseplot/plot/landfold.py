# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""Landfold FES figures via the same GP landscape as NEB.

The fit target is the observation cloud, not a clipped histogram of
``F``. Default: occupied-bin log-density
``z = -kT ln(rho/rhomax)`` (invert after the counts, never clip to
``fmax`` before the GP). ``on="free-energy"`` uses landfold's ``F``
grid instead. A raw ``(s1, s2, z)`` cloud (frames, landmarks, or
minima energies) is ``cloud_observations``.

Those points go to :func:`chemparseplot.plot.neb.plot_landscape_surface`
(``project_path=False``). Default kernel ``grad_imq``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from chemparseplot.parse.landfold import load_fes_result
from chemparseplot.parse.types import LandfoldFesResult
from chemparseplot.plot.neb import SurfaceFitConfig, plot_landscape_surface
from chemparseplot.plot.representer import (
    coalesce_sites,
    farthest_indices,
    training_residual,
)
from chemparseplot.plot.theme import RUHI_THEME, get_theme, setup_publication_theme

__all__ = ["cloud_observations", "fes_observations", "plot_fes"]

_ON = ("density", "free-energy")


def _validate_fmax(fmax: float | None) -> None:
    if fmax is None:
        return
    if not np.isfinite(fmax) or fmax <= 0.0:
        msg = "fmax must be finite and > 0"
        raise ValueError(msg)


def cloud_observations(
    s1,
    s2,
    z,
    grad_s1=None,
    grad_s2=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray]:
    """Raw map points as GP observations (frames, landmarks, or energies)."""
    s1 = np.asarray(s1, dtype=float).reshape(-1)
    s2 = np.asarray(s2, dtype=float).reshape(-1)
    z = np.asarray(z, dtype=float).reshape(-1)
    if s1.shape != s2.shape or s1.shape != z.shape:
        msg = "cloud s1, s2, and z must have the same length"
        raise ValueError(msg)
    keep = np.isfinite(s1) & np.isfinite(s2) & np.isfinite(z)
    g1 = g2 = None
    if grad_s1 is not None and grad_s2 is not None:
        g1 = np.asarray(grad_s1, dtype=float).reshape(-1)
        g2 = np.asarray(grad_s2, dtype=float).reshape(-1)
        if g1.shape != s1.shape or g2.shape != s1.shape:
            msg = "cloud gradients must match s1/s2/z"
            raise ValueError(msg)
        keep = keep & np.isfinite(g1) & np.isfinite(g2)
        g1, g2 = g1[keep], g2[keep]
    if not np.any(keep):
        msg = "cloud has no finite observations to fit"
        raise ValueError(msg)
    return s1[keep], s2[keep], g1, g2, z[keep]


def fes_observations(
    fes_result: LandfoldFesResult,
    *,
    on: str = "density",
    floor: float = 0.006,
    fmax: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Grid cells as GP observations plus lattice slopes.

    ``on="density"`` (default) fits unclipped
    ``z = -kT ln(rho/rhomax)`` on occupied bins. ``fmax`` drops the
    ceiling (``z >= fmax``) from the fit set; it does not clip ``z``.
    ``on="free-energy"`` uses landfold's ``F`` grid the same way.

    Slopes are ``np.gradient`` on the lattice. NEB strings use
    :func:`~chemparseplot.parse.neb_utils.compute_synthetic_gradients`.
    """
    if on not in _ON:
        msg = f"on must be one of {_ON}, got {on!r}"
        raise ValueError(msg)
    _validate_fmax(fmax)
    if not np.isfinite(floor) or floor < 0.0:
        msg = "floor must be finite and >= 0"
        raise ValueError(msg)
    if fes_result.x.size < 2 or fes_result.y.size < 2:
        msg = "landfold FES grid must be at least 2x2 for slopes"
        raise ValueError(msg)
    rho = np.asarray(fes_result.density, dtype=float)
    rmax = float(np.nanmax(rho)) if rho.size else 0.0
    if on == "density" and rmax > 0.0:
        occupied = np.isfinite(rho) & (rho > floor * rmax)
        z = np.full(rho.shape, np.nan, dtype=float)
        z[occupied] = -fes_result.kt * np.log(np.clip(rho[occupied] / rmax, 1e-12, 1.0))
    else:
        z = np.array(fes_result.free_energy, dtype=float, copy=True)
        occupied = np.isfinite(z)
        if floor > 0.0 and rmax > 0.0:
            occupied = occupied & (rho > floor * rmax)
    if fmax is not None:
        occupied = occupied & (z < fmax)
    d_dy, d_dx = np.gradient(z, fes_result.y, fes_result.x)
    yy, xx = np.meshgrid(fes_result.y, fes_result.x, indexing="ij")
    keep = occupied & np.isfinite(z) & np.isfinite(d_dx) & np.isfinite(d_dy)
    if not np.any(keep):
        msg = "landfold FES has no finite cells to fit"
        raise ValueError(msg)
    return xx[keep], yy[keep], d_dx[keep], d_dy[keep], z[keep]


def plot_fes(
    fes_result: LandfoldFesResult | Mapping[str, Any] | None = None,
    *,
    on: str = "density",
    floor: float = 0.006,
    cloud: tuple | None = None,
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
        ``landfold.fes.v1`` mapping or typed result. Ignored when
        *cloud* is set.
    on
        ``density`` (default): fit unclipped ``-kT ln(rho/rhomax)``.
        ``free-energy``: fit landfold's ``F`` grid.
    cloud
        ``(s1, s2, z)`` or ``(s1, s2, z, grad_s1, grad_s2)`` for frames,
        landmarks, or minima energies.
    method
        ``rgpycrumbs.surfaces`` name. Default ``grad_imq``. Value-only
        clouds without slopes fall back to ``rbf``.
    surface_fit
        Default thins the cloud to 300 fit points.
    """
    import matplotlib.pyplot as plt

    setup_publication_theme(get_theme("ruhi"))
    _validate_fmax(fmax)
    kt = 1.0
    if cloud is not None:
        if len(cloud) == 3:
            s1, s2, g1, g2, z = cloud_observations(cloud[0], cloud[1], cloud[2])
        elif len(cloud) == 5:
            s1, s2, g1, g2, z = cloud_observations(*cloud)
        else:
            msg = "cloud must be (s1, s2, z) or (s1, s2, z, grad_s1, grad_s2)"
            raise ValueError(msg)
        if g1 is None and method.startswith("grad_"):
            method = "rbf"
    elif fes_result is not None:
        result = (
            fes_result
            if isinstance(fes_result, LandfoldFesResult)
            else load_fes_result(fes_result)
        )
        kt = result.kt
        s1, s2, g1, g2, z = fes_observations(
            result, on=on, floor=floor, fmax=fmax
        )
    else:
        msg = "plot_fes needs fes_result or cloud"
        raise ValueError(msg)
    s1, s2, g1, g2, z, _ = coalesce_sites(s1, s2, z, g1, g2)
    if s1.size < 2:
        msg = "representer needs at least two distinct sites"
        raise ValueError(msg)
    fit = (
        surface_fit
        if isinstance(surface_fit, SurfaceFitConfig)
        else SurfaceFitConfig.from_mapping(surface_fit)
        if surface_fit is not None
        else SurfaceFitConfig(auto_thin=False, max_surface_points=300)
    )
    max_pts = fit.max_surface_points
    if s1.size > max_pts:
        idx = farthest_indices(np.column_stack([s1, s2]), max_pts)
        s1, s2, z = s1[idx], s2[idx], z[idx]
        if g1 is not None and g2 is not None:
            g1, g2 = g1[idx], g2[idx]
    if rbf_smooth is None:
        span = float(max(np.ptp(s1), np.ptp(s2)))
        rbf_smooth = max(0.1 * span, 1e-3)
    xy = np.column_stack([s1, s2])
    resid = training_residual(xy, z, rbf_smooth)
    scale = max(float(np.ptp(z)), 1.0)
    if float(np.max(np.abs(resid))) > 1e-4 * scale:
        msg = (
            "observation table is not a section of the IMQ Gram "
            f"(max |Kα − z| = {float(np.max(np.abs(resid)))})"
        )
        raise ValueError(msg)
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
        surface_fit=SurfaceFitConfig(auto_thin=False, max_surface_points=max_pts),
        n_inducing=n_inducing,
        variance_threshold=0.5,
    )
    if points:
        for label, (xs, ys) in points.items():
            ax.scatter(xs, ys, s=12, label=label, zorder=50, linewidths=0)
        ax.legend(fontsize=8, loc="lower left", frameon=True)
    if clabel is None:
        clabel = r"$F/kT$" if abs(kt - 1.0) < 1e-12 else r"$F$"
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
