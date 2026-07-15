# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""Plot helpers for general readcon CON trajectories.

Complements eOn-specific landscape/NEB plots: these functions take a plain
:class:`~chemparseplot.parse.con.trajectory.ConTrajectory` (any CON movie)
and draw energy / force profiles with optional structure strips.

.. versionadded:: 1.9.12
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from chemparseplot.plot.structs import convert_energy
from chemparseplot.units import normalize_energy_unit

__all__ = [
    "plot_con_energy_profile",
    "plot_con_force_profile",
    "plot_con_overview",
]


def _x_axis(traj) -> np.ndarray:
    table = traj.table
    if table is not None and not table.is_empty() and "frame_index" in table.columns:
        x = np.asarray(table["frame_index"].to_numpy(), dtype=float)
        if np.all(np.isfinite(x)):
            return x
    return np.arange(traj.n_frames, dtype=float)


def plot_con_energy_profile(
    traj,
    ax=None,
    *,
    energy_unit: str = "eV",
    relative: bool = True,
    color: str | None = None,
    marker: str = "o",
    label: str | None = None,
    title: str | None = None,
) -> Any:
    """Plot energy vs frame index for a CON trajectory.

    Parameters
    ----------
    traj:
        :class:`~chemparseplot.parse.con.trajectory.ConTrajectory`.
    relative:
        If True, subtract the first finite energy (ΔE).
    energy_unit:
        Presentation unit (``eV``, ``kcal/mol``, ``kJ/mol``).
    """
    import matplotlib.pyplot as plt

    energy_unit = normalize_energy_unit(energy_unit)
    if ax is None:
        _, ax = plt.subplots(figsize=(5.37, 3.5))

    x = _x_axis(traj)
    e = np.asarray(traj.energies, dtype=float)
    e = convert_energy(e, energy_unit, source_unit="eV")
    if relative:
        finite = e[np.isfinite(e)]
        if finite.size:
            e = e - finite[0]

    ax.plot(x, e, marker=marker, color=color, label=label)
    ax.set_xlabel("frame")
    ylab = r"$\Delta E$" if relative else r"$E$"
    ax.set_ylabel(f"{ylab} ({energy_unit})")
    if title:
        ax.set_title(title)
    elif traj.path is not None:
        ax.set_title(Path(traj.path).name)
    if label:
        ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    return ax


def plot_con_force_profile(
    traj,
    ax=None,
    *,
    which: str = "fmax",
    color: str | None = None,
    marker: str = "s",
    label: str | None = None,
    title: str | None = None,
) -> Any:
    """Plot per-frame force norm from CON metadata/forces.

    Parameters
    ----------
    which:
        ``fmax`` (max atomic force) or ``frms`` (RMS force).
    """
    import matplotlib.pyplot as plt

    if which not in {"fmax", "frms"}:
        msg = f"which must be 'fmax' or 'frms', got {which!r}"
        raise ValueError(msg)

    table = traj.table
    if (
        table is None
        or table.is_empty()
        or which not in table.columns
        or table[which].null_count() == table.height
    ):
        msg = f"trajectory has no {which} data (frames lack forces)"
        raise ValueError(msg)

    if ax is None:
        _, ax = plt.subplots(figsize=(5.37, 3.5))

    x = _x_axis(traj)
    y = np.asarray(table[which].to_numpy(), dtype=float)
    ax.plot(x, y, marker=marker, color=color, label=label or which)
    ax.set_xlabel("frame")
    ax.set_ylabel(f"{which} (eV/Å)")
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if label:
        ax.legend(frameon=False)
    return ax


def plot_con_overview(
    traj,
    *,
    energy_unit: str = "eV",
    relative: bool = True,
    show_forces: bool = True,
    title: str | None = None,
    figsize: tuple[float, float] = (5.37, 5.0),
) -> Any:
    """Two-panel overview: energy profile and optional force norms.

    Returns the matplotlib ``Figure``.
    """
    import matplotlib.pyplot as plt

    has_f = (
        traj.table is not None
        and not traj.table.is_empty()
        and "fmax" in traj.table.columns
        and traj.table["fmax"].null_count() < traj.table.height
    )
    n_panels = 2 if (show_forces and has_f) else 1
    fig, axes = plt.subplots(
        n_panels, 1, figsize=figsize, sharex=True, constrained_layout=True
    )
    if n_panels == 1:
        axes = [axes]

    plot_con_energy_profile(
        traj,
        ax=axes[0],
        energy_unit=energy_unit,
        relative=relative,
        title=title,
    )
    if n_panels == 2:
        plot_con_force_profile(traj, ax=axes[1], which="fmax")
    return fig
