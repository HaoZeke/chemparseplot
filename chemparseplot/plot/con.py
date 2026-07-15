# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""Plot helpers for general readcon CON trajectories.

Complements eOn-specific landscape/NEB plots: these functions take a plain
:class:`~chemparseplot.parse.con.trajectory.ConTrajectory` (any CON movie)
and draw energy / force profiles, multi-trajectory overlays, and optional
structure strips (reusing :func:`~chemparseplot.plot.neb.plot_structure_strip`).

.. versionadded:: 1.9.12
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np

from chemparseplot.plot.structs import convert_energy
from chemparseplot.units import normalize_energy_unit

__all__ = [
    "plot_con_energy_profile",
    "plot_con_force_profile",
    "plot_con_overlay",
    "plot_con_overview",
    "plot_con_structure_strip",
    "select_structure_indices",
]

StripMode = Literal["none", "endpoints", "all", "linspace"]


def _x_axis(traj) -> np.ndarray:
    table = traj.table
    if table is not None and not table.is_empty() and "frame_index" in table.columns:
        x = np.asarray(table["frame_index"].to_numpy(), dtype=float)
        if np.all(np.isfinite(x)):
            return x
    return np.arange(traj.n_frames, dtype=float)


def _energy_series(traj, *, energy_unit: str, relative: bool) -> np.ndarray:
    energy_unit = normalize_energy_unit(energy_unit)
    e = np.asarray(traj.energies, dtype=float)
    e = convert_energy(e, energy_unit, source_unit="eV")
    if relative:
        finite = e[np.isfinite(e)]
        if finite.size:
            e = e - finite[0]
    return e


def _has_force_column(traj, which: str = "fmax") -> bool:
    table = traj.table
    return (
        table is not None
        and not table.is_empty()
        and which in table.columns
        and table[which].null_count() < table.height
    )


def select_structure_indices(
    n_frames: int,
    mode: StripMode = "endpoints",
    *,
    max_structs: int = 8,
    energies: np.ndarray | None = None,
) -> list[int]:
    """Pick frame indices for a structure strip.

    Parameters
    ----------
    mode
        ``endpoints`` — first/last (and max-energy as SP when useful);
        ``all`` — every frame (capped by *max_structs* via linspace);
        ``linspace`` — evenly spaced including ends;
        ``none`` — empty.
    """
    if n_frames <= 0 or mode == "none":
        return []
    if mode == "all" and n_frames <= max_structs:
        return list(range(n_frames))
    if mode in {"linspace", "all"}:
        k = min(max_structs, n_frames)
        idx = np.unique(np.linspace(0, n_frames - 1, num=k, dtype=int))
        return idx.tolist()
    # endpoints (+ saddle if energy available)
    if n_frames == 1:
        return [0]
    chosen = {0, n_frames - 1}
    if energies is not None and n_frames > 2:
        e = np.asarray(energies, dtype=float)
        mid = e[1:-1]
        if mid.size and np.any(np.isfinite(mid)):
            # argmax on mid; map back
            j = int(np.nanargmax(mid)) + 1
            chosen.add(j)
    return sorted(chosen)


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
    """Plot energy vs frame index for a CON trajectory."""
    import matplotlib.pyplot as plt

    energy_unit = normalize_energy_unit(energy_unit)
    if ax is None:
        _, ax = plt.subplots(figsize=(5.37, 3.5))

    x = _x_axis(traj)
    e = _energy_series(traj, energy_unit=energy_unit, relative=relative)

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
    """Plot per-frame force norm from CON metadata/forces."""
    import matplotlib.pyplot as plt

    if which not in {"fmax", "frms"}:
        msg = f"which must be 'fmax' or 'frms', got {which!r}"
        raise ValueError(msg)

    if not _has_force_column(traj, which):
        msg = f"trajectory has no {which} data (frames lack forces)"
        raise ValueError(msg)

    if ax is None:
        _, ax = plt.subplots(figsize=(5.37, 3.5))

    x = _x_axis(traj)
    y = np.asarray(traj.table[which].to_numpy(), dtype=float)
    ax.plot(x, y, marker=marker, color=color, label=label or which)
    ax.set_xlabel("frame")
    ax.set_ylabel(f"{which} (eV/Å)")
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if label:
        ax.legend(frameon=False)
    return ax


def plot_con_overlay(
    trajs: Sequence[Any],
    labels: Sequence[str] | None = None,
    ax=None,
    *,
    energy_unit: str = "eV",
    relative: bool = True,
    title: str | None = None,
    markers: Sequence[str] | None = None,
) -> Any:
    """Overlay energy profiles from multiple CON trajectories.

    Each trajectory uses its own frame axis (not time-aligned). Relative
    energies are per-trajectory (each starts at 0 when *relative* is True).
    """
    import matplotlib.pyplot as plt

    if not trajs:
        msg = "trajs must be non-empty"
        raise ValueError(msg)
    energy_unit = normalize_energy_unit(energy_unit)
    if ax is None:
        _, ax = plt.subplots(figsize=(5.37, 3.5))

    n = len(trajs)
    if labels is None:
        labels = []
        for i, t in enumerate(trajs):
            if t.path is not None:
                labels.append(Path(t.path).stem)
            else:
                labels.append(f"traj{i}")
    if len(labels) != n:
        msg = f"labels length {len(labels)} != n_trajs {n}"
        raise ValueError(msg)

    default_markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    if markers is None:
        markers = [default_markers[i % len(default_markers)] for i in range(n)]

    for traj, lab, mk in zip(trajs, labels, markers):
        x = _x_axis(traj)
        e = _energy_series(traj, energy_unit=energy_unit, relative=relative)
        ax.plot(x, e, marker=mk, label=lab)

    ax.set_xlabel("frame")
    ylab = r"$\Delta E$" if relative else r"$E$"
    ax.set_ylabel(f"{ylab} ({energy_unit})")
    if title:
        ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    return ax


def plot_con_structure_strip(
    traj,
    ax=None,
    *,
    mode: StripMode = "endpoints",
    max_structs: int = 8,
    renderer: str = "ase",
    rotation: str = "0x,90y,0z",
    xyzrender_config: str = "paton",
    title: str | None = None,
) -> Any:
    """Render selected frames from *traj* as a structure strip.

    Uses :func:`~chemparseplot.plot.neb.plot_structure_strip`. Default renderer
    is ``ase`` so bare installs work without ``xyzrender`` on PATH.
    """
    import matplotlib.pyplot as plt

    from chemparseplot.plot.neb import plot_structure_strip

    if ax is None:
        _, ax = plt.subplots(figsize=(5.37, 1.8))

    e = np.asarray(traj.energies, dtype=float)
    indices = select_structure_indices(
        traj.n_frames, mode, max_structs=max_structs, energies=e
    )
    if not indices:
        ax.axis("off")
        ax.text(0.5, 0.5, "no structures", ha="center", va="center", transform=ax.transAxes)
        return ax

    atoms_list = []
    labels = []
    for i in indices:
        atoms = traj.atoms_list[i]
        if atoms is None:
            continue
        atoms_list.append(atoms)
        if i == 0:
            labels.append("start")
        elif i == traj.n_frames - 1:
            labels.append("end")
        else:
            labels.append(str(i))

    if not atoms_list:
        ax.axis("off")
        ax.text(0.5, 0.5, "no ASE atoms", ha="center", va="center", transform=ax.transAxes)
        return ax

    plot_structure_strip(
        ax,
        atoms_list,
        labels=labels,
        renderer=renderer,
        rotation=rotation,
        xyzrender_config=xyzrender_config,
        prefer_single_row=True,
    )
    if title:
        ax.set_title(title, fontsize=10)
    return ax


def plot_con_overview(
    traj,
    *,
    energy_unit: str = "eV",
    relative: bool = True,
    show_forces: bool = True,
    structures: StripMode = "none",
    max_structs: int = 8,
    strip_renderer: str = "ase",
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
) -> Any:
    """Energy profile (+ optional force panel + optional structure strip).

    Parameters
    ----------
    structures
        Structure strip mode: ``none``, ``endpoints``, ``linspace``, ``all``.
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    has_f = show_forces and _has_force_column(traj, "fmax")
    has_strip = structures != "none" and traj.n_frames > 0

    n_main = 1 + int(has_f)
    if figsize is None:
        h = 3.2 * n_main + (1.6 if has_strip else 0.0)
        figsize = (5.37, max(3.2, h))

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    if has_strip:
        height_ratios = [1.0] * n_main + [0.55]
        gs = GridSpec(n_main + 1, 1, figure=fig, height_ratios=height_ratios)
    else:
        gs = GridSpec(n_main, 1, figure=fig)

    ax0 = fig.add_subplot(gs[0, 0])
    plot_con_energy_profile(
        traj,
        ax=ax0,
        energy_unit=energy_unit,
        relative=relative,
        title=title,
    )
    row = 1
    if has_f:
        ax1 = fig.add_subplot(gs[row, 0], sharex=ax0)
        plot_con_force_profile(traj, ax=ax1, which="fmax")
        row += 1
    if has_strip:
        ax_s = fig.add_subplot(gs[row, 0])
        plot_con_structure_strip(
            traj,
            ax=ax_s,
            mode=structures,
            max_structs=max_structs,
            renderer=strip_renderer,
        )
    return fig
