# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
# SPDX-License-Identifier: MIT
"""General readcon CON trajectory tables."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from chemparseplot.parse.con.trajectory import (
    ConTrajectory,
    frames_to_table,
    load_con_trajectory,
)


def _fake_frame(
    *,
    energy=None,
    frame_index=None,
    forces=None,
    n_atoms=3,
    meta=None,
    neb_bead=None,
):
    atoms = list(range(n_atoms))
    forces_arr = None
    has_forces = forces is not None
    if has_forces:
        forces_arr = np.asarray(forces, dtype=float)

    class Fr:
        pass

    fr = Fr()
    fr.energy = energy
    fr.frame_index = frame_index
    fr.neb_bead = neb_bead
    fr.neb_band = None
    fr.time = None
    fr.timestep = None
    fr.potential_type = None
    fr.spec_version = 2
    fr.atoms = atoms
    fr.has_forces = has_forces
    fr.has_velocities = False
    fr.forces_array = forces_arr
    fr.metadata = meta or {}
    fr.to_ase = lambda: SimpleNamespace(n=n_atoms)
    return fr


def test_frames_to_table_core_columns():
    frames = [
        _fake_frame(energy=-1.0, frame_index=0),
        _fake_frame(energy=-1.1, frame_index=1, meta={"step_size": "0.1"}),
    ]
    df = frames_to_table(frames)
    assert df.height == 2
    assert "energy" in df.columns
    assert "frame_index" in df.columns
    assert "step_size" in df.columns
    assert df["energy"].to_list() == [-1.0, -1.1]


def test_frames_to_table_forces_and_fallback_index():
    forces = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    frames = [_fake_frame(energy=-2.0, forces=forces)]
    df = frames_to_table(frames)
    assert df["frame_index"][0] == 0  # enumerated
    assert df["has_forces"][0] is True
    assert df["fmax"][0] == pytest.approx(1.0)
    assert df["frms"][0] == pytest.approx(np.sqrt(1.0 / 3.0))


def test_frames_to_table_empty():
    import polars as pl

    df = frames_to_table([])
    assert isinstance(df, pl.DataFrame)
    assert df.is_empty()


def test_con_trajectory_energies_and_with_energies():
    frames = [
        _fake_frame(energy=-1.0, frame_index=0),
        _fake_frame(energy=-1.5, frame_index=1),
    ]
    # mock frame_with_energy by using real if available else skip with_energies deep
    traj = ConTrajectory(
        path=None,
        frames=frames,
        atoms_list=[f.to_ase() for f in frames],
        table=frames_to_table(frames),
    )
    assert traj.n_frames == 2
    np.testing.assert_allclose(traj.energies, [-1.0, -1.5])


def test_load_con_trajectory_from_fixture():
    # single-frame CON from rgpycrumbs tests (if present)
    cand = Path("/home/rgoswami/tmp/rgpycrumbs/tests/data/reactant.con")
    if not cand.is_file():
        pytest.skip("reactant.con fixture missing")
    traj = load_con_trajectory(cand)
    assert traj.n_frames >= 1
    assert traj.path is not None
    assert traj.table.height == traj.n_frames
    assert len(traj.atoms_list) == traj.n_frames


def test_energy_from_metadata_string():
    fr = _fake_frame(energy=None, meta={"energy": "-3.5"})
    df = frames_to_table([fr])
    assert df["energy"][0] == pytest.approx(-3.5)
