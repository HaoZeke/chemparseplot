# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""Real-path tests for chemparseplot.parse.eon.con_io (readcon boundary)."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

readcon = pytest.importorskip("readcon")
pytest.importorskip("ase")

from ase import Atoms

from chemparseplot.parse.eon import con_io
from chemparseplot.parse.eon import con_io as con_io_mod
from chemparseplot.parse.eon.con_io import (
    CON_SUFFIXES,
    frame_with_energy,
    is_con_path,
    read_con_as_ase,
    read_con_frames,
    read_first_atoms,
    write_atoms_as_con,
    write_con_frames,
)
from tests.parse._con_fixtures import write_band


def test_is_con_path_suffixes():
    assert is_con_path("neb.con")
    assert is_con_path(Path("x.CONVEL"))
    assert is_con_path("movie.convel")
    assert not is_con_path("neb.traj")
    assert not is_con_path("neb.xyz")
    assert CON_SUFFIXES == (".con", ".convel")


def test_readcon_missing_raises_install_hint(monkeypatch):
    real_import = __import__

    def _fake_import(name, *args, **kwargs):
        if name == "readcon" or name.startswith("readcon."):
            raise ImportError("simulated missing readcon")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _fake_import)
    monkeypatch.delitem(sys.modules, "readcon", raising=False)
    with pytest.raises(ImportError, match="chemparseplot\\[neb\\]"):
        con_io_mod._readcon()


def test_frame_with_energy_roundtrip(tmp_path: Path):
    src = write_band(tmp_path / "in.con", [(1.25, 0.0)])
    frame = read_con_frames(src)[0]
    assert frame.energy == pytest.approx(1.25)

    updated = frame_with_energy(frame, -3.5)
    assert updated.energy == pytest.approx(-3.5)
    assert frame.energy == pytest.approx(1.25)

    out = tmp_path / "nested" / "out.con"
    written = write_con_frames(out, [updated])
    assert written == out
    assert out.is_file()
    back = readcon.read_con(str(out))[0]
    assert back.energy == pytest.approx(-3.5)
    assert back.to_ase().get_positions()[0, 2] == pytest.approx(0.0)


def test_frame_with_energy_scalar_metadata_fallback(tmp_path: Path):
    src = write_band(tmp_path / "in.con", [(0.5, 0.1)])
    frame = read_con_frames(src)[0]

    class _NoSetEnergy:
        def __init__(self, inner):
            self.cell = inner.cell
            self.angles = inner.angles
            self.atoms = inner.atoms
            self.prebox_header = inner.prebox_header
            self.postbox_header = inner.postbox_header
            self.metadata = inner.metadata
            self._energy = None

        def set_scalar_metadata(self, key, value):
            assert key == "energy"
            self._energy = float(value)

        @property
        def energy(self):
            return self._energy

    proxy = _NoSetEnergy(frame)
    created = []

    class _RecordingConFrame:
        def __init__(self, cell, angles, atoms, prebox_header, postbox_header, metadata):
            self.cell = cell
            self.angles = angles
            self.atoms = atoms
            self.prebox_header = prebox_header
            self.postbox_header = postbox_header
            self.metadata = metadata
            self._energy = None
            created.append(self)

        def set_scalar_metadata(self, key, value):
            assert key == "energy"
            self._energy = float(value)

        @property
        def energy(self):
            return self._energy

    fake_readcon = SimpleNamespace(ConFrame=_RecordingConFrame)
    monkey_mod = SimpleNamespace(
        _readcon=lambda: fake_readcon,
        frame_with_energy=con_io_mod.frame_with_energy,
    )
    # Call implementation with patched _readcon
    original = con_io_mod._readcon
    con_io_mod._readcon = lambda: fake_readcon
    try:
        out = frame_with_energy(proxy, 9.25)
    finally:
        con_io_mod._readcon = original
    assert created and out.energy == pytest.approx(9.25)


def test_frame_with_energy_string_metadata_reconstruct(tmp_path: Path):
    src = write_band(tmp_path / "in.con", [(0.25, 0.2)])
    frame = read_con_frames(src)[0]
    original = con_io_mod._readcon

    class _LegacyFrame:
        def __init__(self, cell, angles, atoms, prebox_header, postbox_header, metadata):
            self.cell = cell
            self.angles = angles
            self.atoms = atoms
            self.prebox_header = prebox_header
            self.postbox_header = postbox_header
            self.metadata = metadata

        @property
        def energy(self):
            raw = self.metadata.get("energy")
            return float(raw) if raw is not None else None

    class _LegacyReadcon:
        ConFrame = _LegacyFrame

        @staticmethod
        def read_con(path):
            return original().read_con(path)

    con_io_mod._readcon = lambda: _LegacyReadcon
    try:
        legacy_src = _LegacyFrame(
            frame.cell,
            frame.angles,
            frame.atoms,
            frame.prebox_header,
            frame.postbox_header,
            {str(k): v for k, v in frame.metadata.items()},
        )
        updated = frame_with_energy(legacy_src, 6.5)
    finally:
        con_io_mod._readcon = original
    assert updated.energy == pytest.approx(6.5)
    assert updated.metadata["energy"] == "6.5"


def test_write_atoms_as_con_preserves_energy(tmp_path: Path):
    atoms = Atoms("H", positions=[[0.0, 0.0, 1.5]], cell=[10, 10, 10], pbc=True)
    out = tmp_path / "atoms.con"
    write_atoms_as_con(out, [atoms], energies=[4.2])
    frames = read_con_frames(out)
    assert len(frames) == 1
    assert frames[0].energy == pytest.approx(4.2)
    ase_frames = read_con_as_ase(out)
    assert len(ase_frames) == 1
    assert ase_frames[0].get_positions()[0, 2] == pytest.approx(1.5)
    assert read_first_atoms(out).get_chemical_symbols() == ["H"]


def test_write_atoms_as_con_multi_frame_partial_energies(tmp_path: Path):
    src = write_band(tmp_path / "src.con", [(0.0, 0.0), (0.0, 1.0), (0.0, 2.0)])
    base_frames = read_con_frames(src)
    out_frames = [
        frame_with_energy(base_frames[0], 1.0),
        base_frames[1],
        frame_with_energy(base_frames[2], 3.0),
    ]
    out = tmp_path / "multi.con"
    write_con_frames(out, out_frames)
    assert is_con_path(out)
    assert is_con_path(tmp_path / "mark.convel")
    frames = read_con_frames(out)
    assert len(frames) == 3
    assert frames[0].energy == pytest.approx(1.0)
    assert frames[1].energy == pytest.approx(0.0)
    assert frames[2].energy == pytest.approx(3.0)
    zs = [f.to_ase().get_positions()[0, 2] for f in frames]
    assert zs == pytest.approx([0.0, 1.0, 2.0])

    single = tmp_path / "from_ase.con"
    write_atoms_as_con(
        single,
        [Atoms("H", positions=[[0.0, 0.0, 0.5]], cell=[10, 10, 10], pbc=True)],
        energies=[2.25],
    )
    assert read_con_frames(single)[0].energy == pytest.approx(2.25)


def test_write_atoms_as_con_without_energies(tmp_path: Path):
    atoms = Atoms("He", positions=[[1.0, 0.0, 0.0]], cell=[8, 8, 8], pbc=True)
    out = tmp_path / "plain.con"
    write_atoms_as_con(out, [atoms])
    frame = read_con_frames(out)[0]
    assert frame.energy is None
    assert frame.to_ase().get_chemical_symbols() == ["He"]


def test_read_con_frames_missing_file_raises(tmp_path: Path):
    missing = tmp_path / "nope.con"
    with pytest.raises(Exception):
        read_con_frames(missing)


def test_con_io_exported_from_parse_eon():
    assert con_io.read_con_frames is read_con_frames
    assert con_io.write_con_frames is write_con_frames
    assert hasattr(con_io, "frame_with_energy")
