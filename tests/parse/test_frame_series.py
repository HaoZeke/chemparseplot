# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""ConFrame sequence → series used by eOn plot paths."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("readcon")
pytest.importorskip("ase")
pytest.importorskip("polars")

from ase import Atoms

from chemparseplot.parse.eon.frame_series import (
    atoms_list_from_frames,
    dimer_trajectory_from_frames,
    energies_from_frames,
    min_trajectory_from_frames,
    neb_path_arrays,
)


def _h_frame(energy: float, z: float, *, frame_index: int, **scalars):
    import readcon

    atoms = Atoms("H", positions=[[0.0, 0.0, z]], cell=[10, 10, 10], pbc=False)
    frame = readcon.ConFrame.from_ase(atoms)
    frame.set_energy(float(energy))
    frame.set_frame_index(int(frame_index))
    for key, value in scalars.items():
        frame.set_scalar_metadata(
            key, float(value) if isinstance(value, float) else value
        )
    return frame


def test_energies_from_frames_match_stamps():
    frames = [_h_frame(1.0 + 0.1 * i, float(i), frame_index=i) for i in range(4)]
    e = energies_from_frames(frames)
    assert e.shape == (4,)
    assert np.allclose(e, [1.0, 1.1, 1.2, 1.3])


def test_energies_from_frames_missing_raises():
    import readcon
    from ase import Atoms

    bare = readcon.ConFrame.from_ase(Atoms("H", positions=[[0, 0, 0]]))
    with pytest.raises(ValueError, match="no energy"):
        energies_from_frames([bare])


def test_neb_path_arrays_length_and_stamps():
    frames = []
    for i in range(5):
        fr = _h_frame(
            -10.0 + i,
            0.1 * i,
            frame_index=i,
            reaction_coordinate=0.2 * i,
            relative_energy=float(i),
            parallel_force=-0.01 * i,
        )
        fr.set_neb_bead(i)
        frames.append(fr)
    path = neb_path_arrays(frames)
    assert path["n_frames"] == 5
    assert len(path["atoms_list"]) == 5
    assert np.allclose(path["energies"], [-10, -9, -8, -7, -6])
    assert np.allclose(path["reaction_coordinate"], [0.0, 0.2, 0.4, 0.6, 0.8])
    assert np.allclose(path["relative_energy"], [0, 1, 2, 3, 4])
    assert path["atoms_list"][0].info["energy"] == pytest.approx(-10.0)


def test_min_trajectory_from_frames_metrics():
    frames = [
        _h_frame(
            -5.0 + 0.5 * i,
            0.0,
            frame_index=i,
            step_size=0.1 * (i + 1),
            convergence=1.0 / (i + 1),
        )
        for i in range(3)
    ]
    traj = min_trajectory_from_frames(frames)
    assert len(traj.atoms_list) == 3
    assert traj.dat_df.height == 3
    assert "energy" in traj.dat_df.columns
    assert traj.dat_df["energy"].to_list() == pytest.approx([-5.0, -4.5, -4.0])
    assert traj.final_atoms is traj.atoms_list[-1]


def test_dimer_trajectory_from_frames_metrics():
    frames = [
        _h_frame(
            0.0 + i,
            0.0,
            frame_index=i,
            step_size=0.05,
            delta_e=float(i),
            convergence=0.1,
            eigenvalue=-1.0,
            torque=0.0,
            angle=0.0,
            rotations=0,
        )
        for i in range(3)
    ]
    # rotations may need int via set_scalar_metadata
    traj = dimer_trajectory_from_frames(frames)
    assert len(traj.atoms_list) == 3
    assert traj.dat_df.height == 3
    assert "delta_e" in traj.dat_df.columns or "energy" in traj.dat_df.columns


def test_atoms_list_from_frames_preserves_energy_info():
    frames = [_h_frame(2.5, 0.0, frame_index=0)]
    atoms = atoms_list_from_frames(frames)
    assert atoms[0].info["energy"] == pytest.approx(2.5)
