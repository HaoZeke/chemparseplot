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
from collections.abc import Sequence
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


def create_landscape_axes(
    *,
    dpi: int,
    has_strip: bool,
    theme,
    base_size: float = 6.0,
    strip_height_ratio: float = 0.38,
    strip_hspace: float = 0.22,
):
    """Create a landscape figure with an optional structure strip axis."""

    # Extra bottom room so x-label / ticks are not crushed by the strip.
    fig_h = base_size + (2.15 if has_strip else 0.35)
    fig = plt.figure(figsize=(base_size + 1.1, fig_h), dpi=dpi)
    if has_strip:
        gs = GridSpec(
            2,
            1,
            height_ratios=[1, strip_height_ratio],
            hspace=strip_hspace,
            figure=fig,
            left=0.14,
            right=0.86,
            top=0.90,
            bottom=0.08,
        )
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
        "fontsize": 12,
        "fontweight": "bold",
        "color": "black",
        "ha": "center",
        "va": "bottom",
        "zorder": 120,
    }
    if boxed:
        kwargs.update(
            {
                "xytext": (0, 8),
                "textcoords": "offset points",
                "bbox": {
                    "boxstyle": "round,pad=0.25",
                    "facecolor": "white",
                    "edgecolor": "black",
                    "linewidth": 0.8,
                    "alpha": 0.95,
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
        width_fill_fraction=0.72,
        prefer_single_row=True,
    )


def enforce_strip_clearance(
    fig,
    ax,
    ax_strip,
    *,
    min_clearance_px: float = 32.0,
) -> None:
    """Ensure a minimum pixel gap between x-axis text and the strip axis."""

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    x_text_bottoms = [ax.xaxis.label.get_window_extent(renderer).y0]
    x_text_bottoms.extend(
        tick.get_window_extent(renderer).y0
        for tick in ax.get_xticklabels()
        if tick.get_text()
    )
    target_bottom = min(x_text_bottoms)
    strip_bbox = ax_strip.get_window_extent(renderer)
    current_gap = target_bottom - strip_bbox.y1
    if current_gap >= min_clearance_px:
        return

    delta_px = min_clearance_px - current_gap
    fig_height_px = fig.get_size_inches()[1] * fig.dpi
    delta_fig = delta_px / fig_height_px
    pos = ax_strip.get_position()
    ax_strip.set_position([pos.x0, pos.y0 - delta_fig, pos.width, pos.height])


def save_landscape_figure(
    fig,
    output: Path,
    *,
    dpi: int,
    has_strip: bool,
    ax=None,
    ax_strip=None,
) -> None:
    """Save optimization landscapes; always crop with a pad so labels stay visible."""

    if has_strip and ax is not None and ax_strip is not None:
        enforce_strip_clearance(fig, ax, ax_strip, min_clearance_px=40.0)
    elif not has_strip:
        fig.tight_layout()
    # Always tight-crop: without it strip layouts left large white margins and
    # could hide axis labels depending on the viewer/frame.
    fig.savefig(
        str(output),
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.25,
        facecolor=fig.get_facecolor(),
    )
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


def render_single_ended_landscape(
    *,
    atoms_list: Sequence[Any],
    energies_eV: np.ndarray,
    ref_a: Any,
    ref_b: Any,
    overlay_atom_lists: Sequence[Sequence[Any]] | None = None,
    overlay_labels: Sequence[str] | None = None,
    ira_instance: Any | None = None,
    ira_kmax: float = 14.0,
    project_path: bool = True,
    surface_type: str = "grad_matern",
    energy_unit: str = "eV",
    energy_cap: float | None = None,
    energy_cap_window: float | None = None,
    relative_energy: bool = True,
    title: str | None = None,
    cmap: str = "viridis",
    output: Path,
    dpi: int = 150,
    theme: Any | None = None,
    plot_structures: str = "none",
    strip_structs: Sequence[Any] | None = None,
    strip_labels: Sequence[str] | None = None,
    endpoint_start_label: str = "initial",
    endpoint_end_label: str = "minimized",
    endpoint_boxed: bool = True,
    annotate_overlay_starts: bool = False,
    overlay_start_label: str = "R",
    strip_renderer: str = "xyzrender",
    xyzrender_config: str = "paton",
    strip_spacing: float = 1.5,
    strip_zoom: float | None = None,
    strip_dividers: bool = False,
    rotation: str = "auto",
    perspective_tilt: float = 0.0,
) -> None:
    """Full single-ended landscape pipeline shared by min/saddle CLIs.

    Coordinates use :func:`~chemparseplot.parse.neb_utils.calculate_landscape_coords`
    and synthetic gradients; the surface/path overlay uses
    :func:`plot_optimization_landscape`. Energies are passed in **eV** and
    converted once for display (avoids double conversion when ``energy_unit``
    is not eV).

    ```{versionadded} 1.8.1
    ```
    """
    from matplotlib import ticker

    from chemparseplot.parse.neb_utils import (
        calculate_landscape_coords,
        compute_synthetic_gradients,
    )

    has_strip = plot_structures == "endpoints" and strip_structs is not None
    fig, ax, ax_strip = create_landscape_axes(dpi=dpi, has_strip=has_strip, theme=theme)

    rmsd_a, rmsd_b = calculate_landscape_coords(
        atoms_list,
        ira_instance,
        ira_kmax,
        ref_a=ref_a,
        ref_b=ref_b,
    )
    energies = np.asarray(energies_eV, dtype=float)
    n = min(len(rmsd_a), len(energies))
    rmsd_a, rmsd_b, energies = rmsd_a[:n], rmsd_b[:n], energies[:n]
    # Relative energy keeps colorbars readable (absolute eV often span <0.2 eV
    # and all ticks collapse to the same printed value).
    e_ref = float(np.min(energies)) if relative_energy else 0.0
    energies_plot = energies - e_ref
    display_e = convert_energy(energies_plot, energy_unit)
    cap = energy_cap
    if cap is None and energy_cap_window is not None:
        cap = float(np.min(display_e)) + energy_cap_window
    if cap is not None:
        factor = float(convert_energy(np.array([1.0]), energy_unit)[0])
        energies_plot = np.minimum(
            energies_plot, cap / factor if factor else cap
        )
        display_e = convert_energy(energies_plot, energy_unit)

    f_para = -np.gradient(energies_plot)
    grad_a, grad_b = compute_synthetic_gradients(rmsd_a, rmsd_b, f_para)

    z_label = (
        f"Relative energy ({energy_unit})"
        if relative_energy
        else energy_axis_label(energy_unit)
    )
    cb = plot_optimization_landscape(
        ax,
        rmsd_a,
        rmsd_b,
        grad_a,
        grad_b,
        energies_plot,
        project_path=project_path,
        method=surface_type,
        cmap=cmap,
        label_mode="optimization",
        energy_unit=energy_unit,
        z_label=z_label,
    )
    if cb is not None:
        cb.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
        cb.set_label(z_label, rotation=270, labelpad=18)

    basis = None
    if project_path:
        _, _, basis = project_landscape_path(rmsd_a, rmsd_b, project_path=True)

    overlays = overlay_atom_lists or ()
    labels = list(overlay_labels or ())
    for idx, atoms_ov in enumerate(overlays):
        lbl = labels[idx] if idx < len(labels) else f"path{idx}"
        ra, rb = calculate_landscape_coords(
            atoms_ov,
            ira_instance,
            ira_kmax,
            ref_a=ref_a,
            ref_b=ref_b,
        )
        m = min(len(ra), len(atoms_ov))
        ra, rb = ra[:m], rb[:m]
        px, py, _ = project_landscape_path(
            ra, rb, project_path=project_path, basis=basis
        )
        color = OVERLAY_COLORS[idx % len(OVERLAY_COLORS)]
        if len(overlays) > 1:
            ax.plot(
                px,
                py,
                "o-",
                color=color,
                markersize=3,
                linewidth=1.5,
                alpha=0.8,
                zorder=55,
                label=lbl,
            )
        if annotate_overlay_starts and len(px):
            annotate_endpoint(
                ax, float(px[0]), float(py[0]), overlay_start_label, boxed=False
            )

    plot_x, plot_y, _ = project_landscape_path(
        rmsd_a, rmsd_b, project_path=project_path, basis=basis
    )
    annotate_endpoint(
        ax,
        float(plot_x[0]),
        float(plot_y[0]),
        endpoint_start_label,
        boxed=endpoint_boxed,
    )
    annotate_endpoint(
        ax,
        float(plot_x[-1]),
        float(plot_y[-1]),
        endpoint_end_label,
        boxed=endpoint_boxed,
    )

    # True 1:1 Å panel: Δd window matches Δs (same RMSD metric).
    if project_path and len(plot_x) > 1:
        x0, x1 = float(np.min(plot_x)), float(np.max(plot_x))
        y0, y1 = float(np.min(plot_y)), float(np.max(plot_y))
        s_pad = max((x1 - x0) * 0.06, 0.01)
        x0, x1 = x0 - s_pad, x1 + s_pad
        half = max(0.5 * (x1 - x0), abs(y0), abs(y1), 0.02)
        x_mid = 0.5 * (x0 + x1)
        ax.set_xlim(x_mid - half, x_mid + half)
        ax.set_ylim(-half, half)
        ax.set_aspect("equal", adjustable="box")

    # Re-assert axis labels after surface/path (bold, large enough to read).
    if project_path:
        ax.set_xlabel(_LABELS["optimization"]["x"], fontweight="bold", fontsize=12)
        ax.set_ylabel(_LABELS["optimization"]["y"], fontweight="bold", fontsize=12)
    else:
        ax.set_xlabel(r"RMSD from ref A ($\AA$)", fontweight="bold", fontsize=12)
        ax.set_ylabel(r"RMSD from ref B ($\AA$)", fontweight="bold", fontsize=12)
    ax.tick_params(axis="both", labelsize=10)
    if title:
        ax.set_title(title, fontweight="bold", fontsize=13, pad=10)

    if overlays and len(overlays) > 1:
        ax.legend(frameon=True, framealpha=0.95, loc="best")

    if has_strip and ax_strip is not None and strip_structs is not None:
        zoom = (
            strip_zoom
            if strip_zoom is not None
            else max(0.45, default_strip_zoom(list(strip_structs)) * 1.6)
        )
        render_endpoint_strip(
            ax_strip,
            list(strip_structs),
            list(strip_labels or []),
            strip_zoom=zoom,
            rotation=rotation,
            theme=theme,
            strip_renderer=strip_renderer,
            strip_spacing=max(strip_spacing, 2.2),
            strip_dividers=strip_dividers,
            perspective_tilt=perspective_tilt,
            xyzrender_config=xyzrender_config,
        )

    save_landscape_figure(
        fig, output, dpi=dpi, has_strip=has_strip, ax=ax, ax_strip=ax_strip
    )


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
        ax.set_xlabel(labels["x"], fontweight="bold", fontsize=12)
        ax.set_ylabel(labels["y"], fontweight="bold", fontsize=12)
    else:
        ax.set_xlabel(r"RMSD from ref A ($\AA$)", fontweight="bold", fontsize=12)
        ax.set_ylabel(r"RMSD from ref B ($\AA$)", fontweight="bold", fontsize=12)

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
