# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
# SPDX-License-Identifier: MIT
"""Plot helpers for general CON trajectories."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("ase")

from ase.build import molecule

from chemparseplot.parse.con.trajectory import ConTrajectory, frames_to_table
from chemparseplot.plot.con import (
    plot_con_energy_profile,
    plot_con_force_profile,
    plot_con_overlay,
    plot_con_overview,
    plot_con_structure_strip,
    select_structure_indices,
)


def _fake_frame(energy=None, forces=None, frame_index=None, atoms=None):
    if atoms is None:
        atoms_list = [0, 1]
    else:
        atoms_list = list(range(len(atoms)))
    return SimpleNamespace(
        energy=energy,
        frame_index=frame_index,
        neb_bead=None,
        neb_band=None,
        time=None,
        timestep=None,
        potential_type=None,
        spec_version=2,
        atoms=atoms_list,
        has_forces=forces is not None,
        has_velocities=False,
        forces_array=forces,
        metadata={},
    )


def _traj_with_energy(energies, forces_list=None, real_atoms=False):
    frames = []
    ase_list = []
    base = molecule("H2O") if real_atoms else None
    for i, e in enumerate(energies):
        forces = None if forces_list is None else forces_list[i]
        if real_atoms:
            a = base.copy()
            a.positions = a.positions * (1.0 + 0.01 * i)
            ase_list.append(a)
            fr = _fake_frame(energy=e, forces=forces, frame_index=i, atoms=a)
        else:
            ase_list.append(None)
            fr = _fake_frame(energy=e, forces=forces, frame_index=i)
        frames.append(fr)
    table = frames_to_table(frames)
    return ConTrajectory(
        path=None, frames=frames, atoms_list=ase_list, table=table
    )


def test_select_structure_indices():
    assert select_structure_indices(0, "endpoints") == []
    assert select_structure_indices(5, "none") == []
    assert select_structure_indices(5, "endpoints") == [0, 4]
    e = np.array([0.0, 0.1, 1.0, 0.2, 0.0])
    idx = select_structure_indices(5, "endpoints", energies=e)
    assert 0 in idx and 4 in idx and 2 in idx
    assert select_structure_indices(10, "linspace", max_structs=4)[0] == 0
    assert select_structure_indices(10, "linspace", max_structs=4)[-1] == 9
    assert len(select_structure_indices(3, "all")) == 3


def test_plot_con_energy_profile_relative():
    traj = _traj_with_energy([-2.0, -2.1, -1.9])
    ax = plot_con_energy_profile(traj, relative=True, label="run")
    assert ax.lines[0].get_ydata()[0] == pytest.approx(0.0)


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


def test_plot_con_overlay():
    t1 = _traj_with_energy([-1.0, -1.2, -1.1])
    t2 = _traj_with_energy([-1.0, -0.9, -0.8, -0.85])
    ax = plot_con_overlay([t1, t2], labels=["a", "b"], relative=True)
    assert len(ax.lines) == 2


def test_plot_con_overlay_label_mismatch():
    t1 = _traj_with_energy([-1.0, -1.1])
    with pytest.raises(ValueError, match="labels"):
        plot_con_overlay([t1, t1], labels=["only_one"])


def test_plot_con_structure_strip():
    traj = _traj_with_energy([-1.0, -0.5, -1.1], real_atoms=True)
    ax = plot_con_structure_strip(traj, mode="endpoints", renderer="ase")
    assert ax is not None


def test_plot_con_overview_with_strip():
    traj = _traj_with_energy([-1.0, -0.8, -1.05], real_atoms=True)
    fig = plot_con_overview(traj, structures="endpoints", strip_renderer="ase")
    # energy + strip
    assert len(fig.axes) >= 2


def test_plot_con_overview_energy_only():
    traj = _traj_with_energy([-1.0, -1.1])
    fig = plot_con_overview(traj, show_forces=True, structures="none", title="demo")
    assert fig is not None


def test_plot_con_overview_with_forces():
    f0 = np.zeros((2, 3))
    f1 = np.array([[0, 0, 0], [0, 2, 0]])
    traj = _traj_with_energy([-1.0, -1.2], forces_list=[f0, f1])
    fig = plot_con_overview(traj, show_forces=True, structures="none")
    assert len(fig.axes) == 2
