# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""Real-path tests for chemparseplot.parse.eon.con_io (readcon boundary)."""

from __future__ import annotations

import struct
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

readcon = pytest.importorskip("readcon")
pytest.importorskip("ase")

from ase import Atoms

from chemparseplot.parse.eon import con_io
from chemparseplot.parse.eon import con_io as con_io_mod
from chemparseplot.parse.eon.con_io import (
    CON_SUFFIXES,
    frame_with_energy,
    frame_with_positions,
    is_con_path,
    read_con_as_ase,
    read_con_frames,
    read_first_atoms,
    read_first_frame,
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


def test_frame_with_energy_requires_set_energy(tmp_path: Path):
    src = write_band(tmp_path / "in.con", [(0.5, 0.1)])
    frame = read_con_frames(src)[0]

    class _NoSetEnergy:
        def __init__(self, cell, angles, atoms, prebox_header, postbox_header, metadata):
            self.cell = cell
            self.angles = angles
            self.atoms = atoms
            self.prebox_header = prebox_header
            self.postbox_header = postbox_header
            self.metadata = metadata

    fake_readcon = SimpleNamespace(ConFrame=_NoSetEnergy)
    original = con_io_mod._readcon
    con_io_mod._readcon = lambda: fake_readcon
    try:
        with pytest.raises(ImportError, match=r"readcon>=0\.17\.1"):
            frame_with_energy(frame, 9.25)
    finally:
        con_io_mod._readcon = original


def test_frame_with_energy_rejects_non_finite(tmp_path: Path):
    src = write_band(tmp_path / "in.con", [(0.25, 0.2)])
    frame = read_con_frames(src)[0]
    with pytest.raises(Exception, match="finite"):
        frame_with_energy(frame, float("nan"))
    assert frame.energy == pytest.approx(0.25)
    assert "nan" not in str(frame.metadata.get("energy", "")).lower()


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
    assert hasattr(con_io, "frame_with_positions")
    assert hasattr(con_io, "read_first_frame")


_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
# RoundTrip bytes from ConFrameWriter on readcon-core 694e808 (version 0.17.1).
# TODO: replace these with the published restart_exact golden when v0.17.1 is tagged.
RESTART_EXACT = _FIXTURES / "restart_exact.con"
RESTART_EXACT_VEL = _FIXTURES / "restart_exact_vel.con"
_SUBNORMAL = struct.unpack("<d", (1).to_bytes(8, "little"))[0]


def _f64_bits(value) -> bytes | None:
    if value is None:
        return None
    return struct.pack("<d", float(value))


def _array_bits(array) -> bytes:
    return np.ascontiguousarray(array, dtype=np.float64).tobytes()


def _section_bits(frame, names: tuple[str, ...]) -> list[tuple[bytes | None, ...]]:
    rows = []
    for atom in frame.atoms:
        rows.append(tuple(_f64_bits(getattr(atom, name)) for name in names))
    return rows


def test_write_con_omits_precision_and_canonical(tmp_path: Path, monkeypatch):
    src = write_band(tmp_path / "in.con", [(1.0, 0.0)])
    frame = read_con_frames(src)[0]
    calls = []
    real = readcon.write_con

    def _wrapped(*args, **kwargs):
        calls.append((args, kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(readcon, "write_con", _wrapped)
    write_con_frames(tmp_path / "out.con", [frame])
    assert calls
    _args, kwargs = calls[0]
    assert kwargs.get("precision", None) is None
    assert kwargs.get("canonical", False) is False


def test_restart_exact_rewrite_is_byte_identical(tmp_path: Path):
    for golden in (RESTART_EXACT, RESTART_EXACT_VEL):
        assert golden.is_file()
        text = golden.read_bytes()
        assert text.startswith(b"Generated by eOn\n")
        frame = read_first_frame(golden)
        out = tmp_path / golden.name
        write_con_frames(out, [frame])
        assert out.read_bytes() == text


def test_restart_exact_xyz_matches_ase_bitwise():
    frame = read_first_frame(RESTART_EXACT)
    atoms = read_con_as_ase(RESTART_EXACT)[0]
    assert _array_bits(atoms.positions) == _array_bits(frame.xyz)
    first = read_first_atoms(RESTART_EXACT)
    assert _array_bits(first.positions) == _array_bits(frame.xyz)
    coords = np.asarray(frame.xyz, dtype=np.float64).ravel()
    assert any(_f64_bits(value) == _f64_bits(-0.0) for value in coords)
    assert any(_f64_bits(value) == struct.pack("<d", _SUBNORMAL) for value in coords)
    assert frame.frc is not None
    assert not all(angle == pytest.approx(90.0) for angle in frame.angles)
    fixed = [tuple(atom.fixed) for atom in frame.atoms]
    assert any(mask != (False, False, False) for mask in fixed)


def test_restart_exact_vel_carries_natural_time():
    frame = read_first_frame(RESTART_EXACT_VEL)
    units = frame.metadata["units"]
    assert units["time"] == "(amu*angstrom^2/eV)^0.5"
    assert frame.vel is not None
    assert _array_bits(frame.xyz) == _array_bits(read_first_frame(RESTART_EXACT).xyz)


def test_frame_with_energy_keeps_xyz_bits(tmp_path: Path):
    frame = read_first_frame(RESTART_EXACT)
    for index, atom in enumerate(frame.atoms):
        atom.dx = float(index)
        atom.dy = -0.0
        atom.dz = _SUBNORMAL
    before_xyz = _array_bits(frame.xyz)
    before_disp = _array_bits(frame.disp)
    updated = frame_with_energy(frame, -4.25)
    assert _array_bits(updated.xyz) == before_xyz
    assert _array_bits(updated.disp) == before_disp
    assert updated.energy == pytest.approx(-4.25)
    out = tmp_path / "shifted.con"
    write_con_frames(out, [updated])
    assert out.read_bytes() != RESTART_EXACT.read_bytes()


def test_frame_with_positions_keeps_other_sections():
    frame = read_first_frame(RESTART_EXACT)
    for index, atom in enumerate(frame.atoms):
        atom.dx = float(index) + 0.125
        atom.dy = -0.0
        atom.dz = _SUBNORMAL
        atom.sx = 0.25 + index
        atom.sy = -0.5
        atom.sz = 1.5
    positions = np.array(frame.xyz, dtype=np.float64, copy=True)
    positions[0, 0] = positions[0, 0] + 0.5
    original_x = frame.atoms[0].x
    updated = frame_with_positions(frame, positions)
    assert updated.atoms[0] is not frame.atoms[0]
    assert _f64_bits(frame.atoms[0].x) == _f64_bits(original_x)
    assert _array_bits(np.asarray(updated.xyz)[1:]) == _array_bits(positions[1:])
    moved = _array_bits(np.asarray(updated.xyz)[0])
    original = _array_bits(np.asarray(frame.xyz)[0])
    assert moved != original
    names = ("dx", "dy", "dz", "fx", "fy", "fz", "sx", "sy", "sz")
    assert _section_bits(updated, names) == _section_bits(frame, names)
    for atom, src in zip(updated.atoms, frame.atoms, strict=True):
        assert tuple(atom.fixed) == tuple(src.fixed)
        assert atom.atom_id == src.atom_id
        assert _f64_bits(atom.mass) == _f64_bits(src.mass)
        assert atom.symbol == src.symbol


def test_read_first_atoms_stops_at_frame_zero(tmp_path: Path, monkeypatch):
    expected = _array_bits(read_first_atoms(RESTART_EXACT).positions)
    bad = tmp_path / "two.con"
    bad.write_bytes(RESTART_EXACT.read_bytes() + b"\nthis second frame is not legal\n")
    calls = {"first": 0, "all": 0}
    real_first = readcon.read_first_frame

    def _first(path):
        calls["first"] += 1
        return real_first(path)

    def _all(*_args, **_kwargs):
        calls["all"] += 1
        raise AssertionError("read_con must not parse the rest of the movie")

    monkeypatch.setattr(readcon, "read_first_frame", _first)
    monkeypatch.setattr(readcon, "read_con", _all)
    atoms = read_first_atoms(bad)
    assert calls == {"first": 1, "all": 0}
    assert _array_bits(atoms.positions) == expected
