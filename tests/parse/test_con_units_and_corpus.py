"""Plot inputs retain their physical values through CON and corpus readers."""

import numpy as np
import pytest
import readcon
import readcon_db
from ase import units
from readcon_db import ConCorpus

from chemparseplot.parse.eon.con_io import (
    frame_energy_ev,
    frame_positions_angstrom,
    frame_with_energy,
    iter_con_frames,
    iter_corpus_frames,
    read_con_as_ase,
    read_con_frames,
    read_first_atoms,
    read_first_frame,
    write_con_frames,
)
from chemparseplot.parse.eon.frame_series import (
    atoms_list_from_frames,
    energies_from_frames,
)
from chemparseplot.parse.surrogate.gpr_optim import read_atom_types


def _frame(energy, *, atomic_id=41):
    return readcon.ConFrame(
        cell=[10, 10, 10],
        angles=[90, 90, 90],
        atoms=[readcon.Atom(symbol="H", x=1, y=2, z=3, atom_id=atomic_id, mass=2.014)],
        metadata={
            "energy": energy,
            "units": {"length": "bohr", "energy": "hartree"},
            "pbc": [False, False, False],
        },
    )


def test_energy_and_ase_metadata_use_ev():
    frame = _frame(-0.5)
    expected = -0.5 * units.Hartree
    assert frame_energy_ev(frame) == pytest.approx(expected)
    np.testing.assert_allclose(energies_from_frames([frame]), [expected])
    atoms = atoms_list_from_frames([frame])[0]
    assert atoms.info["energy"] == pytest.approx(expected)
    assert frame.energy == -0.5


def test_energy_replacement_retains_the_frames_units(tmp_path):
    original = _frame(-0.5)
    replaced = frame_with_energy(original, 2 * units.Hartree)
    assert replaced.energy == pytest.approx(2)
    assert replaced.metadata["units"] == original.metadata["units"]
    assert original.energy == -0.5
    path = tmp_path / "updated.con"
    readcon.write_con(str(path), [replaced])
    assert frame_energy_ev(readcon.read_first_frame(str(path))) == pytest.approx(
        2 * units.Hartree
    )


def test_first_frame_reader_does_not_parse_trailing_frames(tmp_path):
    path = tmp_path / "band.con"
    path.write_text(readcon.write_con_string([_frame(0)]) + "not a CON frame\n")
    atoms = read_first_atoms(path)
    assert atoms.get_chemical_symbols() == ["H"]
    np.testing.assert_allclose(atoms.positions, np.array([[1, 2, 3]]) * units.Bohr)
    np.testing.assert_array_equal(read_atom_types(path), [1])
    iterator = iter_con_frames(path)
    assert next(iterator).energy == 0
    with pytest.raises(Exception):
        next(iterator)


def test_corpus_selects_one_trajectory_in_frame_order(tmp_path, monkeypatch):
    path = tmp_path / "corpus"
    corpus = ConCorpus(str(path))
    corpus.append_trajectory_frames(9, [_frame(1), _frame(2)])
    corpus.append_trajectory_frames(3, [_frame(8, atomic_id=7)])
    del corpus
    opened = []

    def open_readonly(corpus_path, *, readonly):
        assert readonly is True
        opened.append(corpus_path)
        return ConCorpus(corpus_path, readonly=readonly)

    monkeypatch.setattr(readcon_db, "ConCorpus", open_readonly)
    frames = list(iter_corpus_frames(path, trajectory=9, require_energy=True))
    assert [frame.energy for frame in frames] == [1, 2]
    assert [frame.atoms[0].atom_id for frame in frames] == [41, 41]
    assert [frame.energy for frame in iter_corpus_frames(path)] == [8, 1, 2]
    assert opened == [str(path), str(path)]


def test_saddle_positions_use_declared_length_without_ase_conversion(
    tmp_path, monkeypatch
):
    from chemparseplot.parse.surrogate.gpr_optim import _read_saddle

    path = tmp_path / "saddle.con"
    readcon.write_con(str(path), [_frame(-0.5)])

    def no_ase(*args, **kwargs):
        raise AssertionError("saddle coordinates do not require an ASE conversion")

    monkeypatch.setattr("chemparseplot.parse.eon.con_io.read_first_atoms", no_ase)
    np.testing.assert_allclose(_read_saddle(path), np.array([1, 2, 3]) * units.Bohr)


def test_ordinary_con_operations_do_not_open_a_corpus(tmp_path, monkeypatch):
    def unexpected_corpus(*args, **kwargs):
        raise AssertionError("ordinary CON operations must not open a corpus")

    monkeypatch.setattr(readcon_db, "ConCorpus", unexpected_corpus)
    path = tmp_path / "structure.con"
    frame = _frame(-0.5)
    write_con_frames(path, [frame])
    assert len(read_con_frames(path)) == 1
    assert len(list(iter_con_frames(path))) == 1
    assert read_first_frame(path).energy == -0.5
    expected = np.array([[1, 2, 3]]) * units.Bohr
    np.testing.assert_allclose(read_first_atoms(path).positions, expected)
    np.testing.assert_allclose(read_con_as_ase(path)[0].positions, expected)
    np.testing.assert_allclose(frame_positions_angstrom(frame), expected)
    assert frame_energy_ev(frame) == pytest.approx(-0.5 * units.Hartree)
    assert frame_with_energy(frame, 2 * units.Hartree).energy == pytest.approx(2)
