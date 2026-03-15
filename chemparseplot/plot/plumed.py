# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT

"""PLUMED FES plotting functions.

Provides publication-quality 1D and 2D Free Energy Surface plots
using the Ruhi theme and cmcrameri colormaps.
"""

import cmcrameri.cm as cmc
import matplotlib.pyplot as plt

from chemparseplot.plot.theme import get_theme, setup_publication_theme


def plot_fes_2d(fes_result, minima_result=None, cmap=None, figsize=(8, 6), dpi=300):
    """Plot 2D Free Energy Surface with optional minima markers.

    Parameters
    ----------
    fes_result : dict
        Output from ``calculate_fes_from_hills``. Must contain keys
        ``fes``, ``x``, ``y``, and ``dimension == 2``.
    minima_result : dict or None
        Output from ``find_fes_minima``. If provided, minima are
        marked with red X markers and labeled.
    cmap : matplotlib colormap or None
        Colormap for the contour plot. Defaults to ``cmcrameri.cm.batlow``.
    figsize : tuple
        Figure size in inches.
    dpi : int
        Figure resolution.

    Returns
    -------
    matplotlib.figure.Figure
        The 2D FES contour figure.
    """
    setup_publication_theme(get_theme("ruhi"))

    if cmap is None:
        cmap = cmc.batlow

    fes_data = fes_result["fes"]
    x = fes_result["x"]
    y = fes_result["y"]

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    contour = ax.contourf(x, y, fes_data.T, levels=25, cmap=cmap)
    ax.contour(x, y, fes_data.T, levels=contour.levels, colors="black", linewidths=0.5)
    fig.colorbar(contour, ax=ax, label="Free Energy (kJ/mol)")
    ax.set_xlabel("Collective Variable 1")
    ax.set_ylabel("Collective Variable 2")

    # Add markers for minima if found
    if minima_result is not None:
        minima_df = minima_result["minima"]
        ax.scatter(
            minima_df["CV1"],
            minima_df["CV2"],
            s=100,
            c="red",
            marker="x",
            label="Minima",
        )
        for _, row in minima_df.iterrows():
            ax.text(
                row["CV1"],
                row["CV2"],
                f"  {row['letter']}",
                color="white",
                fontsize=12,
                fontweight="bold",
            )
        ax.legend()

    fig.tight_layout()
    return fig


def plot_fes_1d(fes_result, minima_result=None, figsize=(8, 5), dpi=300):
    """Plot 1D Free Energy Surface with optional minima markers.

    Parameters
    ----------
    fes_result : dict
        Output from ``calculate_fes_from_hills``. Must contain keys
        ``fes``, ``x``, and ``dimension == 1``.
    minima_result : dict or None
        Output from ``find_fes_minima``. If provided, minima are
        marked with red X markers and labeled.
    figsize : tuple
        Figure size in inches.
    dpi : int
        Figure resolution.

    Returns
    -------
    matplotlib.figure.Figure
        The 1D FES line plot figure.
    """
    setup_publication_theme(get_theme("ruhi"))

    fes_data = fes_result["fes"]
    x = fes_result["x"]

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.plot(x, fes_data)
    ax.set_xlabel("Collective Variable 1")
    ax.set_ylabel("Free Energy (kJ/mol)")
    ax.grid(True)

    # Add markers for minima if found
    if minima_result is not None:
        minima_df = minima_result["minima"]
        ax.scatter(
            minima_df["CV1"],
            minima_df["free_energy"],
            s=100,
            c="red",
            marker="x",
            zorder=5,
            label="Minima",
        )
        for _, row in minima_df.iterrows():
            ax.text(
                row["CV1"],
                row["free_energy"],
                f"  {row['letter']}",
                color="black",
                fontsize=12,
            )
        ax.legend()

    fig.tight_layout()
    return fig
