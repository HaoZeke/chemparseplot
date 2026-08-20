# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""Publication FES figures for landfold embeddings.

Landfold computes the surface (``landfold.fes.v1`` / ``landfold fes --csv``).
This module only draws it, using the Ruhi theme and the same filled-contour
plus thin isoline style as :func:`chemparseplot.plot.chemgp.plot_surface_contour`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from chemparseplot.parse.landfold import load_fes_result
from chemparseplot.parse.types import LandfoldFesResult
from chemparseplot.plot.theme import get_theme, setup_publication_theme

__all__ = ["plot_fes"]


def plot_fes(
    fes_result: LandfoldFesResult | Mapping[str, Any],
    *,
    xlabel: str = r"$s_1$",
    ylabel: str = r"$s_2$",
    clabel: str | None = None,
    fmax: float | None = None,
    points: dict[str, tuple] | None = None,
    figsize: tuple[float, float] = (5.6, 4.8),
    dpi: int = 170,
):
    """Filled-contour landfold FES with thin isolines.

    Parameters
    ----------
    fes_result
        A :class:`~chemparseplot.parse.types.LandfoldFesResult` or a
        ``landfold.fes.v1`` mapping from ``fes_xy_result``.
    xlabel, ylabel
        Axis labels (sketch-map coordinates by default).
    clabel
        Colorbar label. Default is ``F/kT`` when ``kt == 1``, else ``F``.
    fmax
        Clip finite ``F`` to ``[0, fmax]`` (JCTC 2013 panel uses 2).
    points
        Optional ``{label: (xs, ys)}`` overlays.
    figsize, dpi
        Figure size in inches and raster resolution.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    setup_publication_theme(get_theme("ruhi"))
    result = (
        fes_result
        if isinstance(fes_result, LandfoldFesResult)
        else load_fes_result(fes_result)
    )
    z = np.array(result.free_energy, dtype=float, copy=True)
    if fmax is not None:
        if not np.isfinite(fmax) or fmax <= 0.0:
            msg = "fmax must be finite and > 0"
            raise ValueError(msg)
        finite = np.isfinite(z)
        z[finite] = np.clip(z[finite], 0.0, fmax)
    masked = np.ma.masked_invalid(z)
    if masked.mask is False or not np.all(masked.mask):
        lo = 0.0
        hi = float(fmax) if fmax is not None else float(np.nanmax(z))
        if not np.isfinite(hi) or hi <= lo:
            hi = lo + 1.0
    else:
        lo, hi = 0.0, 1.0
    levels = np.linspace(lo, hi, 21)
    isolines = np.linspace(lo + 0.08 * (hi - lo), hi - 0.08 * (hi - lo), 12)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, facecolor="white")
    mesh = ax.contourf(
        result.x, result.y, masked, levels=levels, cmap="ruhi_diverging", extend="max"
    )
    ax.contour(
        result.x,
        result.y,
        masked,
        levels=isolines,
        colors="black",
        linewidths=0.35,
    )
    if points:
        for label, (xs, ys) in points.items():
            ax.scatter(xs, ys, s=12, label=label, zorder=3, linewidths=0)
        ax.legend(fontsize=8, loc="lower left", frameon=True)
    if clabel is None:
        clabel = r"$F/kT$" if abs(result.kt - 1.0) < 1e-12 else r"$F$"
    fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04).set_label(clabel)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    return fig
