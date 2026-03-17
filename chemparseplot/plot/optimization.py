"""Visualization functions for optimization trajectories.

Provides 2D landscape and 1D profile plots for single-ended methods
(dimer/saddle search, minimization) using the generalized (s, d)
reaction valley projection.

The key semantic difference from NEB plots:
- s = optimization progress (toward saddle or minimum)
- d = lateral deviation (wasted sideways motion)

.. versionadded:: 1.3.0
"""

from __future__ import annotations

import logging
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from chemparseplot.parse.projection import (
    compute_projection_basis,
    project_to_sd,
)
from chemparseplot.plot.neb import (
    plot_landscape_path_overlay,
    plot_landscape_surface,
)

log = logging.getLogger(__name__)

# --- Label modes ---
_LABELS = {
    "reaction": {
        "x": r"Reaction progress $s$ ($\AA$)",
        "y": r"Orthogonal deviation $d$ ($\AA$)",
    },
    "optimization": {
        "x": r"Optimization progress $s$ ($\AA$)",
        "y": r"Lateral deviation $d$ ($\AA$)",
    },
}


def plot_optimization_landscape(
    ax,
    rmsd_a: np.ndarray,
    rmsd_b: np.ndarray,
    grad_a: np.ndarray,
    grad_b: np.ndarray,
    z_data: np.ndarray,
    *,
    label_mode: str = "optimization",
    project_path: bool = True,
    method: str = "grad_matern",
    cmap: str = "viridis",
    z_label: str = "Energy (eV)",
    **surface_kwargs,
) -> Any:
    """Plot 2D landscape for an optimization trajectory.

    Wraps :func:`plot_landscape_surface` and :func:`plot_landscape_path_overlay`
    with semantically correct axis labels for single-ended methods.

    Parameters
    ----------
    ax
        Matplotlib axes.
    rmsd_a, rmsd_b
        RMSD distances to reference A and B.
    grad_a, grad_b
        Synthetic gradients in A and B directions.
    z_data
        Energy values for surface fitting.
    label_mode
        ``"optimization"`` (default) or ``"reaction"`` for NEB-style labels.
    project_path
        Whether to project into (s, d) coordinates.
    method
        Surface fitting method (passed to plot_landscape_surface).
    cmap
        Colormap name.
    z_label
        Label for the colorbar.
    **surface_kwargs
        Extra keyword arguments passed to :func:`plot_landscape_surface`.

    Returns
    -------
    colorbar or None
    """
    plot_landscape_surface(
        ax,
        rmsd_a,
        rmsd_b,
        grad_a,
        grad_b,
        z_data,
        method=method,
        cmap=cmap,
        project_path=project_path,
        **surface_kwargs,
    )

    cb = plot_landscape_path_overlay(
        ax,
        rmsd_a,
        rmsd_b,
        z_data,
        cmap=cmap,
        z_label=z_label,
        project_path=project_path,
    )

    labels = _LABELS.get(label_mode, _LABELS["optimization"])
    if project_path:
        ax.set_xlabel(labels["x"])
        ax.set_ylabel(labels["y"])
    else:
        ax.set_xlabel(r"RMSD from ref A ($\AA$)")
        ax.set_ylabel(r"RMSD from ref B ($\AA$)")

    return cb


def plot_optimization_profile(
    ax,
    iterations: np.ndarray,
    energies: np.ndarray,
    *,
    eigenvalues: np.ndarray | None = None,
    ax_eigen: Any = None,
    color: str = "#004D40",
    eigen_color: str = "#FF655D",
) -> None:
    """Plot energy (and optionally eigenvalue) vs iteration.

    Parameters
    ----------
    ax
        Matplotlib axes for the energy profile.
    iterations
        Iteration numbers.
    energies
        Energy values per iteration.
    eigenvalues
        Eigenvalue per iteration (dimer only). Plotted on ``ax_eigen``.
    ax_eigen
        Secondary axes for eigenvalue subplot. Required if eigenvalues given.
    color
        Energy line color.
    eigen_color
        Eigenvalue line color.
    """
    ax.plot(iterations, energies, "o-", color=color, markersize=4, linewidth=1.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Energy (eV)")

    if eigenvalues is not None and ax_eigen is not None:
        ax_eigen.plot(
            iterations,
            eigenvalues,
            "s-",
            color=eigen_color,
            markersize=3,
            linewidth=1.2,
        )
        ax_eigen.axhline(0, color="gray", linestyle=":", linewidth=1, alpha=0.6)
        ax_eigen.set_xlabel("Iteration")
        ax_eigen.set_ylabel("Eigenvalue (eV/$\\AA^2$)")


def plot_convergence_panel(
    ax_force,
    ax_step,
    dat_df: pl.DataFrame,
    *,
    force_col: str = "convergence",
    step_col: str = "step_size",
    iter_col: str = "iteration",
    color: str = "#004D40",
) -> None:
    """Plot convergence metrics from a trajectory DataFrame.

    Parameters
    ----------
    ax_force
        Axes for force/convergence metric.
    ax_step
        Axes for step size.
    dat_df
        Polars DataFrame with iteration data (from climb.dat or min.dat).
    force_col, step_col, iter_col
        Column names.
    color
        Line color.
    """
    iters = dat_df[iter_col].to_numpy()
    forces = dat_df[force_col].to_numpy()
    steps = dat_df[step_col].to_numpy()

    ax_force.semilogy(iters, np.abs(forces), "o-", color=color, markersize=3)
    ax_force.set_xlabel("Iteration")
    ax_force.set_ylabel("Convergence")

    ax_step.plot(iters, steps, "o-", color=color, markersize=3)
    ax_step.set_xlabel("Iteration")
    ax_step.set_ylabel("Step size")


def plot_dimer_mode_evolution(
    ax,
    mode_vectors: list[np.ndarray],
    *,
    color: str = "#004D40",
) -> None:
    """Plot how the dimer mode aligns with the final mode over iterations.

    Shows the dot product of each iteration's mode vector with the
    final converged mode, indicating rotation convergence.

    Parameters
    ----------
    ax
        Matplotlib axes.
    mode_vectors
        List of mode vectors per iteration. The last is the reference.
    color
        Line color.
    """
    if len(mode_vectors) < 2:
        return

    ref = mode_vectors[-1].ravel()
    ref_norm = np.linalg.norm(ref)
    if ref_norm < 1e-12:
        return

    ref = ref / ref_norm
    dots = []
    for mv in mode_vectors:
        v = mv.ravel()
        vnorm = np.linalg.norm(v)
        if vnorm > 1e-12:
            dots.append(abs(np.dot(v / vnorm, ref)))
        else:
            dots.append(0.0)

    iters = np.arange(len(dots))
    ax.plot(iters, dots, "o-", color=color, markersize=3, linewidth=1.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("|cos(mode, final mode)|")
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1, alpha=0.6)
