# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
# SPDX-License-Identifier: MIT
"""Plot helpers for general CON trajectories."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("matplotlib")

from chemparseplot.parse.con.trajectory import ConTrajectory, frames_to_table
from chemparseplot.plot.con import (
    plot_con_energy_profile,
    plot_con_force_profile,
    plot_con_overview,
)


def _traj_with_energy(energies, forces_list=None):
    frames = []
    for i, e in enumerate(energies):
        forces = None
        if forces_list is not None:
            forces = forces_list[i]
        fr = SimpleNamespace(
            energy=e,
            frame_index=i,
            neb_bead=None,
            neb_band=None,
            time=None,
            timestep=None,
            potential_type=None,
            spec_version=2,
            atoms=[0, 1],
            has_forces=forces is not None,
            has_velocities=False,
            forces_array=forces,
            metadata={},
        )
        frames.append(fr)
    table = frames_to_table(frames)
    return ConTrajectory(path=None, frames=frames, atoms_list=[None] * len(frames), table=table)


def test_plot_con_energy_profile_relative():
    traj = _traj_with_energy([-2.0, -2.1, -1.9])
    ax = plot_con_energy_profile(traj, relative=True, label="run")
    assert ax is not None
    # relative: first point ~0
    line = ax.lines[0]
    y = line.get_ydata()
    assert y[0] == pytest.approx(0.0)


def test_plot_con_force_profile():
    f0 = np.array([[0, 0, 0], [0.5, 0, 0]])
    f1 = np.array([[0, 0, 0], [1.0, 0, 0]])
    traj = _traj_with_energy([-1.0, -1.0], forces_list=[f0, f1])
    ax = plot_con_force_profile(traj, which="fmax")
    assert ax is not None


def test_plot_con_force_missing_raises():
    traj = _traj_with_energy([-1.0])
    with pytest.raises(ValueError, match="fmax"):
        plot_con_force_profile(traj)


def test_plot_con_overview_energy_only():
    traj = _traj_with_energy([-1.0, -1.1])
    fig = plot_con_overview(traj, show_forces=True, title="demo")
    assert fig is not None


def test_plot_con_overview_with_forces():
    f0 = np.zeros((2, 3))
    f1 = np.array([[0, 0, 0], [0, 2, 0]])
    traj = _traj_with_energy([-1.0, -1.2], forces_list=[f0, f1])
    fig = plot_con_overview(traj, show_forces=True)
    assert len(fig.axes) == 2
