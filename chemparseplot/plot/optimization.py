"""Visualization functions for optimization trajectories.

Provides 2D landscape and 1D profile plots for single-ended methods
(dimer/saddle search, minimization) using the generalized (s, d)
reaction valley projection.

The key semantic difference from NEB plots:
- s = optimization progress (toward saddle or minimum)
- d = lateral deviation (wasted sideways motion)

.. versionadded:: 1.5.0
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.gridspec import GridSpec

from chemparseplot.parse.projection import compute_projection_basis, project_to_sd
from chemparseplot.plot.neb import (
    plot_landscape_path_overlay,
    plot_landscape_surface,
    plot_structure_strip,
)
from chemparseplot.plot.structs import (
    convert_energy,
    convert_energy_curvature,
    eigenvalue_axis_label,
    energy_axis_label,
)
from chemparseplot.plot.theme import apply_axis_theme

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

OVERLAY_COLORS = ["#004D40", "#FF655D", "#3F51B5", "#FF9800", "#9C27B0", "#009688"]


def create_landscape_axes(*, dpi: int, has_strip: bool, theme, base_size: float = 5.37):
    """Create a landscape figure with an optional structure strip axis."""

    fig = plt.figure(figsize=(base_size, base_size + (0.95 if has_strip else 0)), dpi=dpi)
    if has_strip:
        gs = GridSpec(2, 1, height_ratios=[1, 0.16], hspace=0.22, figure=fig)
        ax = fig.add_subplot(gs[0])
        ax_strip = fig.add_subplot(gs[1])
        if theme:
            apply_axis_theme(ax_strip, theme)
    else:
        ax = fig.add_subplot(111)
        ax_strip = None

    if theme:
        apply_axis_theme(ax, theme)
    return fig, ax, ax_strip


def project_landscape_path(rmsd_a, rmsd_b, *, project_path: bool, basis=None):
    """Project an optimization path into display coordinates."""

    if not project_path:
        return rmsd_a, rmsd_b, None
    if basis is None:
        basis = compute_projection_basis(rmsd_a, rmsd_b)
    plot_x, plot_y = project_to_sd(rmsd_a, rmsd_b, basis)
    return plot_x, plot_y, basis


def annotate_endpoint(ax, x: float, y: float, label: str, *, boxed: bool):
    """Annotate an optimization endpoint consistently."""

    kwargs = {
        "fontsize": 10,
        "fontweight": "bold",
        "ha": "center",
        "va": "bottom",
        "zorder": 60,
    }
    if boxed:
        kwargs.update(
            {
                "xytext": (0, 6),
                "textcoords": "offset points",
                "bbox": {
                    "boxstyle": "round,pad=0.2",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.85,
                },
            }
        )
    ax.annotate(label, (x, y), **kwargs)


def default_strip_zoom(structs) -> float:
    """Scale strip zoom gently with atom count."""

    max_atoms = max(len(s) for s in structs) if structs else 10
    return max(0.14, 0.38 * (20 / max(max_atoms, 20)) ** 0.25)


def render_endpoint_strip(
    ax_strip,
    structs,
    labels,
    *,
    strip_zoom,
    rotation,
    theme,
    strip_renderer,
    strip_spacing,
    strip_dividers,
    perspective_tilt,
    xyzrender_config,
):
    """Render the standard endpoint strip for single-ended plots."""

    zoom = strip_zoom if strip_zoom is not None else default_strip_zoom(structs)
    plot_structure_strip(
        ax_strip,
        structs,
        labels,
        zoom=zoom,
        rotation=rotation,
        theme_color=theme.textcolor if theme else "black",
        renderer=strip_renderer,
        col_spacing=strip_spacing,
        show_dividers=strip_dividers,
        perspective_tilt=perspective_tilt,
        xyzrender_config=xyzrender_config,
        max_display_height_px=44.0,
        width_fill_fraction=0.78,
    )


def save_landscape_figure(fig, output: Path, *, dpi: int, has_strip: bool) -> None:
    """Save optimization landscapes without tight-layout strip warnings."""

    if not has_strip:
        fig.tight_layout()
        fig.savefig(str(output), dpi=dpi, bbox_inches="tight")
    else:
        fig.savefig(str(output), dpi=dpi)
    plt.close(fig)


def save_standard_figure(fig, output: Path, *, dpi: int) -> None:
    """Save a standard figure with tight layout."""

    fig.tight_layout()
    fig.savefig(str(output), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_single_ended_profile(
    trajs,
    labels,
    output: Path,
    dpi: int,
    *,
    energy_unit: str,
    energy_column: str,
    title: str,
    eigen_column: str | None = None,
) -> None:
    """Plot shared single-ended optimization profiles."""

    has_eigen = bool(eigen_column) and any(
        eigen_column in trajectory.dat_df.columns for trajectory in trajs
    )
    fig, axes = plt.subplots(
        1, 2 if has_eigen else 1, figsize=(10 if has_eigen else 5.37, 4), dpi=dpi
    )
    if not has_eigen:
        axes = [axes]

    for idx, (traj, lbl) in enumerate(zip(trajs, labels, strict=False)):
        dat = traj.dat_df
        color = OVERLAY_COLORS[idx % len(OVERLAY_COLORS)]
        iters = dat["iteration"].to_numpy()
        energies = convert_energy(dat[energy_column].to_numpy(), energy_unit)
        axes[0].plot(
            iters, energies, "o-", color=color, markersize=4, linewidth=1.5, label=lbl
        )

        if has_eigen and eigen_column and eigen_column in dat.columns:
            eigenvalues = convert_energy_curvature(
                dat[eigen_column].to_numpy(), energy_unit
            )
            axes[1].plot(
                iters,
                eigenvalues,
                "s-",
                color=color,
                markersize=3,
                linewidth=1.2,
                label=lbl,
            )

    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel(energy_axis_label(energy_unit))
    axes[0].set_title(title)
    if len(trajs) > 1:
        axes[0].legend(frameon=False)

    if has_eigen:
        axes[1].axhline(0, color="gray", linestyle=":", linewidth=1, alpha=0.6)
        axes[1].set_xlabel("Iteration")
        axes[1].set_ylabel(eigenvalue_axis_label(energy_unit))
        axes[1].set_title("Eigenvalue vs Iteration")
        if len(trajs) > 1:
            axes[1].legend(frameon=False)

    save_standard_figure(fig, output, dpi=dpi)


def plot_single_ended_convergence(trajs, labels, output: Path, dpi: int) -> None:
    """Plot shared convergence panels for single-ended optimizers."""

    fig, (ax_force, ax_step) = plt.subplots(1, 2, figsize=(10, 4), dpi=dpi)
    for idx, (traj, lbl) in enumerate(zip(trajs, labels, strict=False)):
        color = OVERLAY_COLORS[idx % len(OVERLAY_COLORS)]
        plot_convergence_panel(ax_force, ax_step, traj.dat_df, color=color)
        ax_force.plot([], [], color=color, label=lbl)
    if len(trajs) > 1:
        ax_force.legend(frameon=False)
    save_standard_figure(fig, output, dpi=dpi)


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
    energy_unit: str = "eV",
    z_label: str | None = None,
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
    z_values = convert_energy(z_data, energy_unit)
    grad_a_values = convert_energy(grad_a, energy_unit)
    grad_b_values = convert_energy(grad_b, energy_unit)

    plot_landscape_surface(
        ax,
        rmsd_a,
        rmsd_b,
        grad_a_values,
        grad_b_values,
        z_values,
        method=method,
        cmap=cmap,
        project_path=project_path,
        **surface_kwargs,
    )

    cb = plot_landscape_path_overlay(
        ax,
        rmsd_a,
        rmsd_b,
        z_values,
        cmap=cmap,
        z_label=z_label or energy_axis_label(energy_unit),
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
    energy_unit: str = "eV",
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
    converted_energies = convert_energy(energies, energy_unit)
    ax.plot(
        iterations,
        converted_energies,
        "o-",
        color=color,
        markersize=4,
        linewidth=1.5,
    )
    ax.set_xlabel("Iteration")
    ax.set_ylabel(energy_axis_label(energy_unit))

    if eigenvalues is not None and ax_eigen is not None:
        converted_eigenvalues = convert_energy_curvature(eigenvalues, energy_unit)
        ax_eigen.plot(
            iterations,
            converted_eigenvalues,
            "s-",
            color=eigen_color,
            markersize=3,
            linewidth=1.2,
        )
        ax_eigen.axhline(0, color="gray", linestyle=":", linewidth=1, alpha=0.6)
        ax_eigen.set_xlabel("Iteration")
        ax_eigen.set_ylabel(eigenvalue_axis_label(energy_unit))


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
        Polars DataFrame with iteration data, usually reconstructed from
        embedded CON metadata and only falling back to sidecar TSV data when
        needed for compatibility.
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
    if len(mode_vectors) < 2:  # noqa: PLR2004
        return

    ref = mode_vectors[-1].ravel()
    ref_norm = np.linalg.norm(ref)
    if ref_norm < 1e-12:  # noqa: PLR2004
        return

    ref = ref / ref_norm
    dots = []
    for mv in mode_vectors:
        v = mv.ravel()
        vnorm = np.linalg.norm(v)
        if vnorm > 1e-12:  # noqa: PLR2004
            dots.append(abs(np.dot(v / vnorm, ref)))
        else:
            dots.append(0.0)

    iters = np.arange(len(dots))
    ax.plot(iters, dots, "o-", color=color, markersize=3, linewidth=1.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("|cos(mode, final mode)|")
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1, alpha=0.6)
