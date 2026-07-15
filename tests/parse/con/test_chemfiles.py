# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
# SPDX-License-Identifier: MIT
"""Chemfiles bridge and unified load_trajectory."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from chemparseplot.parse.con.chemfiles import chemfiles_available
from chemparseplot.parse.con.trajectory import load_trajectory


def test_chemfiles_available_bool():
    assert isinstance(chemfiles_available(), bool)


def test_load_trajectory_con_path():
    cand = Path("/home/rgoswami/tmp/rgpycrumbs/tests/data/reactant.con")
    if not cand.is_file():
        pytest.skip("fixture missing")
    traj = load_trajectory(cand)
    assert traj.n_frames >= 1
    assert traj.source == "readcon"


def test_load_trajectory_chemfiles_unavailable(tmp_path, monkeypatch):
    xyz = tmp_path / "h2.xyz"
    xyz.write_text("2\n\nH 0 0 0\nH 0 0 0.74\n")
    monkeypatch.setattr(
        "chemparseplot.parse.con.chemfiles.chemfiles_available", lambda: False
    )
    with pytest.raises(ImportError, match="chemfiles"):
        load_trajectory(xyz)


def test_load_chemfiles_mocked(tmp_path, monkeypatch):
    from chemparseplot.parse.con import chemfiles as ch
    from chemparseplot.parse.con.trajectory import frames_to_table

    class Fr:
        energy = -1.0
        frame_index = 0
        neb_bead = None
        neb_band = None
        time = None
        timestep = None
        potential_type = None
        spec_version = 2
        atoms = [0]
        has_forces = False
        has_velocities = False
        forces_array = None
        metadata = {}

        def to_ase(self):
            return SimpleNamespace(n=1)

    monkeypatch.setattr(ch, "chemfiles_available", lambda: True)
    monkeypatch.setattr(ch, "read_chemfiles_frames", lambda p: [Fr()])
    # bypass _require_chemfiles by patching load path
    monkeypatch.setattr(
        "chemparseplot.parse.con.chemfiles.read_chemfiles_frames",
        lambda p: [Fr()],
    )
    monkeypatch.setattr(ch, "chemfiles_available", lambda: True)

    def fake_require():
        return SimpleNamespace(read_chemfiles=lambda p: [Fr()])

    monkeypatch.setattr(ch, "_require_chemfiles", fake_require)
    monkeypatch.setattr(
        ch,
        "read_chemfiles_frames",
        lambda path: list(fake_require().read_chemfiles(str(path))),
    )
    traj = ch.load_chemfiles_trajectory(tmp_path / "x.xyz")
    assert traj.source == "chemfiles"
    assert traj.n_frames == 1
