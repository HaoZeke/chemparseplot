# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""Draw CON trajectory profiles. These helpers do not parse files."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from chemparseplot.plot.structs import convert_energy, energy_axis_label
from chemparseplot.plot.theme import RUHI_COLORS

_COLORS = (
    RUHI_COLORS["teal"],
    RUHI_COLORS["sky"],
    RUHI_COLORS["coral"],
    RUHI_COLORS["magenta"],
    RUHI_COLORS["sunshine"],
)

_STRUCTURE_MODES = ("none", "endpoints", "linspace", "all")


def _display_energies(
    energies: Sequence[float], unit: str, *, relative: bool
) -> np.ndarray:
    values = np.asarray(list(energies), dtype=float)
    if relative:
        finite = values[np.isfinite(values)]
        if finite.size:
            values = values - finite[0]
    return convert_energy(values, unit)


def _finite_fmax(traj: Any) -> np.ndarray | None:
    table = getattr(traj, "table", None)
    if table is None or getattr(table, "is_empty", lambda: True)():
        return None
    if "fmax" not in table.columns:
        return None
    values = np.asarray(table["fmax"].to_numpy(), dtype=float)
    if not np.isfinite(values).any():
        return None
    return values


def _structure_indices(n_frames: int, mode: str, max_structs: int) -> list[int]:
    if mode not in _STRUCTURE_MODES:
        msg = f"structures must be one of {_STRUCTURE_MODES}, got {mode!r}"
        raise ValueError(msg)
    if n_frames <= 0 or mode == "none":
        return []
    if mode == "endpoints":
        if n_frames == 1:
            return [0]
        return [0, n_frames - 1]
    cap = n_frames if mode == "all" and n_frames <= max_structs else max_structs
    cap = max(1, min(cap, n_frames))
    if cap == 1:
        return [0]
    if cap >= n_frames:
        return list(range(n_frames))
    return [round(i * (n_frames - 1) / (cap - 1)) for i in range(cap)]


def _atoms_for_strip(traj: Any) -> list[Any]:
    frames = list(traj.frames)
    if getattr(traj, "source", None) == "con" or (
        frames and callable(getattr(frames[0], "to_ase", None))
    ):
        from chemparseplot.parse.eon.frame_series import atoms_list_from_frames

        return atoms_list_from_frames(frames)
    return frames


def _energy_ylabel(unit: str, *, relative: bool) -> str:
    label = "Relative energy" if relative else "Energy"
    return energy_axis_label(unit, label=label)


def plot_con_overview(
    traj: Any,
    *,
    energy_unit: str = "eV",
    relative: bool = True,
    show_forces: bool = True,
    structures: str = "none",
    max_structs: int = 8,
    strip_renderer: str = "ase",
    title: str | None = None,
) -> Figure:
    """Draw one trajectory: energy, optional fmax, optional structure strip.

    Positions for the strip come from
    :func:`chemparseplot.parse.eon.frame_series.atoms_list_from_frames`.
    This function does not read a file.
    """
    energies = _display_energies(traj.energies, energy_unit, relative=relative)
    forces = _finite_fmax(traj) if show_forces else None
    if forces is not None:
        forces = convert_energy(forces, energy_unit)
    show_strip = structures != "none"
    n_panels = 1 + int(forces is not None) + int(show_strip)
    fig, axes = plt.subplots(
        n_panels,
        1,
        figsize=(5.37, 3.1 * n_panels),
        constrained_layout=True,
        squeeze=False,
    )
    panels = list(axes[:, 0])
    ax_energy = panels.pop(0)
    x = np.arange(int(traj.n_frames))
    ax_energy.plot(x, energies, marker="o", color=_COLORS[0], linewidth=1.5)
    ax_energy.set_ylabel(_energy_ylabel(energy_unit, relative=relative))
    if title:
        ax_energy.set_title(title)
    bottom = ax_energy
    if forces is not None:
        ax_force = panels.pop(0)
        ax_force.plot(x, forces, marker="o", color=_COLORS[1], linewidth=1.5)
        ax_force.set_ylabel(energy_axis_label(energy_unit, label="fmax"))
        bottom = ax_force
    if show_strip:
        ax_strip = panels.pop(0)
        atoms = _atoms_for_strip(traj)
        indices = _structure_indices(len(atoms), structures, max_structs)
        from chemparseplot.plot.neb import plot_structure_strip

        plot_structure_strip(
            ax_strip,
            [atoms[index] for index in indices],
            labels=[str(index) for index in indices],
            renderer=strip_renderer,
        )
    else:
        bottom.set_xlabel("Frame")
    return fig


def plot_con_overlay(
    trajs: Sequence[Any],
    *,
    labels: Sequence[str] | None = None,
    ax: Axes | None = None,
    energy_unit: str = "eV",
    relative: bool = True,
    title: str | None = None,
) -> Figure:
    """Draw energy curves for several trajectories on one axes.

    Each curve is shifted by its own first finite energy when *relative* is
    true. This function does not read a file.
    """
    if labels is not None and len(labels) != len(trajs):
        msg = f"labels ({len(labels)}) must match trajectories ({len(trajs)})"
        raise ValueError(msg)
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.37, 3.8), constrained_layout=True)
    else:
        fig = ax.figure
    for index, traj in enumerate(trajs):
        energies = _display_energies(traj.energies, energy_unit, relative=relative)
        label = None if labels is None else labels[index]
        ax.plot(
            np.arange(int(traj.n_frames)),
            energies,
            marker="o",
            color=_COLORS[index % len(_COLORS)],
            linewidth=1.5,
            label=label,
        )
    ax.set_xlabel("Frame")
    ax.set_ylabel(_energy_ylabel(energy_unit, relative=relative))
    if title:
        ax.set_title(title)
    if labels:
        ax.legend(frameon=False)
    return fig
