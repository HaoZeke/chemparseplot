# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""Tests for the HDF5 NEB reader (ChemGP output).

All HDF5 files are synthetic, created via h5py in fixtures -- no external
test data files needed.
"""
import numpy as np
import pytest

from tests.conftest import skip_if_not_env

skip_if_not_env("neb")

import h5py  # noqa: E402

from chemparseplot.parse.trajectory.hdf5 import (  # noqa: E402
    _read_path_group,
    _reconstruct_atoms,
    history_to_profile_dats,
    load_neb_history,
    load_neb_result,
    result_to_atoms_list,
    result_to_profile_dat,
)

pytestmark = pytest.mark.neb

# --- Helpers / Constants ---
N_IMAGES = 5
N_ATOMS = 3
NDOF = N_ATOMS * 3


def _make_path_arrays(n_images=N_IMAGES, n_atoms=N_ATOMS):
    """Generate consistent synthetic path data."""
    rng = np.random.default_rng(42)
    ndof = n_atoms * 3
    images = rng.random((n_images, ndof))
    energies = np.linspace(0.0, 1.0, n_images)
    gradients = rng.random((n_images, ndof)) * 0.1
    f_para = rng.random(n_images) * 0.5
    rxn_coord = np.linspace(0.0, 3.0, n_images)
    return images, energies, gradients, f_para, rxn_coord


def _write_path_group(grp, images, energies, gradients, f_para, rxn_coord):
    """Write path arrays into an HDF5 group."""
    grp.create_dataset("images", data=images)
    grp.create_dataset("energies", data=energies)
    grp.create_dataset("gradients", data=gradients)
    grp.create_dataset("f_para", data=f_para)
    grp.create_dataset("rxn_coord", data=rxn_coord)


# --- Fixtures ---


@pytest.fixture()
def result_h5(tmp_path):
    """Create a synthetic neb_result.h5 file."""
    fpath = tmp_path / "neb_result.h5"
    images, energies, gradients, f_para, rxn_coord = _make_path_arrays()

    with h5py.File(fpath, "w") as f:
        path_grp = f.create_group("path")
        _write_path_group(
            path_grp, images, energies, gradients, f_para, rxn_coord
        )

        conv_grp = f.create_group("convergence")
        conv_grp.create_dataset("max_force", data=np.array([0.5, 0.1, 0.01]))
        conv_grp.create_dataset("oracle_calls", data=np.array([10, 20, 30]))

        meta_grp = f.create_group("metadata")
        meta_grp.create_dataset("converged", data=True)
        meta_grp.create_dataset("oracle_calls", data=30)
        meta_grp.create_dataset(
            "atomic_numbers", data=np.array([1, 6, 8])
        )

    return str(fpath)


@pytest.fixture()
def result_h5_with_cell(tmp_path):
    """Create a result file that includes a cell."""
    fpath = tmp_path / "neb_result_cell.h5"
    images, energies, gradients, f_para, rxn_coord = _make_path_arrays()

    with h5py.File(fpath, "w") as f:
        path_grp = f.create_group("path")
        _write_path_group(
            path_grp, images, energies, gradients, f_para, rxn_coord
        )
        meta_grp = f.create_group("metadata")
        meta_grp.create_dataset(
            "atomic_numbers", data=np.array([1, 6, 8])
        )
        cell = np.eye(3).ravel() * 10.0
        meta_grp.create_dataset("cell", data=cell)

    return str(fpath)


@pytest.fixture()
def history_h5(tmp_path):
    """Create a synthetic neb_history.h5 file with 3 steps."""
    fpath = tmp_path / "neb_history.h5"
    n_steps = 3

    with h5py.File(fpath, "w") as f:
        steps_grp = f.create_group("steps")
        for i in range(n_steps):
            images, energies, gradients, f_para, rxn_coord = (
                _make_path_arrays()
            )
            # Shift energies per step so they are distinguishable
            energies = energies + i * 0.1
            step_grp = steps_grp.create_group(str(i))
            _write_path_group(
                step_grp, images, energies, gradients, f_para, rxn_coord
            )

        meta_grp = f.create_group("metadata")
        meta_grp.create_dataset(
            "atomic_numbers", data=np.array([1, 6, 8])
        )

    return str(fpath)


# --- Tests ---


class TestReadPathGroup:
    def test_returns_all_keys(self, result_h5):
        with h5py.File(result_h5, "r") as f:
            data = _read_path_group(f["path"])
        assert set(data.keys()) == {
            "images",
            "energies",
            "gradients",
            "f_para",
            "rxn_coord",
        }

    def test_array_shapes(self, result_h5):
        with h5py.File(result_h5, "r") as f:
            data = _read_path_group(f["path"])
        assert data["images"].shape == (N_IMAGES, NDOF)
        assert data["energies"].shape == (N_IMAGES,)
        assert data["gradients"].shape == (N_IMAGES, NDOF)
        assert data["f_para"].shape == (N_IMAGES,)
        assert data["rxn_coord"].shape == (N_IMAGES,)

    def test_values_are_numpy(self, result_h5):
        with h5py.File(result_h5, "r") as f:
            data = _read_path_group(f["path"])
        for v in data.values():
            assert isinstance(v, np.ndarray)


class TestReconstructAtoms:
    def test_correct_atom_count(self):
        images, energies, gradients, _, _ = _make_path_arrays()
        atoms_list = _reconstruct_atoms(
            images, np.array([1, 6, 8]), None, gradients, energies
        )
        assert len(atoms_list) == N_IMAGES
        for atoms in atoms_list:
            assert len(atoms) == N_ATOMS

    def test_positions_match(self):
        images, energies, gradients, _, _ = _make_path_arrays()
        atoms_list = _reconstruct_atoms(
            images, np.array([1, 6, 8]), None, gradients, energies
        )
        for i, atoms in enumerate(atoms_list):
            expected_pos = images[i].reshape(N_ATOMS, 3)
            np.testing.assert_array_almost_equal(
                atoms.positions, expected_pos
            )

    def test_energy_attached(self):
        images, energies, gradients, _, _ = _make_path_arrays()
        atoms_list = _reconstruct_atoms(
            images, np.array([1, 6, 8]), None, gradients, energies
        )
        for i, atoms in enumerate(atoms_list):
            assert atoms.get_potential_energy() == pytest.approx(
                energies[i]
            )

    def test_forces_are_negative_gradients(self):
        images, energies, gradients, _, _ = _make_path_arrays()
        atoms_list = _reconstruct_atoms(
            images, np.array([1, 6, 8]), None, gradients, energies
        )
        for i, atoms in enumerate(atoms_list):
            expected_forces = -gradients[i].reshape(N_ATOMS, 3)
            np.testing.assert_array_almost_equal(
                atoms.get_forces(), expected_forces
            )

    def test_default_atomic_numbers(self):
        images, energies, gradients, _, _ = _make_path_arrays()
        atoms_list = _reconstruct_atoms(
            images, None, None, gradients, energies
        )
        # Default is all hydrogen
        for atoms in atoms_list:
            assert all(z == 1 for z in atoms.numbers)

    def test_cell_attached(self):
        images, energies, gradients, _, _ = _make_path_arrays()
        cell = np.eye(3).ravel() * 10.0
        atoms_list = _reconstruct_atoms(
            images, np.array([1, 6, 8]), cell, gradients, energies
        )
        for atoms in atoms_list:
            np.testing.assert_array_almost_equal(
                atoms.cell[:], np.eye(3) * 10.0
            )
            assert all(atoms.pbc)


class TestResultToProfileDat:
    def test_shape(self, result_h5):
        data = result_to_profile_dat(result_h5)
        assert data.shape == (5, N_IMAGES)

    def test_index_row(self, result_h5):
        data = result_to_profile_dat(result_h5)
        np.testing.assert_array_equal(
            data[0], np.arange(N_IMAGES, dtype=float)
        )

    def test_rxn_coord_matches_input(self, result_h5):
        _, _, _, _, rxn_coord = _make_path_arrays()
        data = result_to_profile_dat(result_h5)
        np.testing.assert_array_almost_equal(data[1], rxn_coord)

    def test_energies_match_input(self, result_h5):
        _, energies, _, _, _ = _make_path_arrays()
        data = result_to_profile_dat(result_h5)
        np.testing.assert_array_almost_equal(data[2], energies)

    def test_f_para_matches_input(self, result_h5):
        _, _, _, f_para, _ = _make_path_arrays()
        data = result_to_profile_dat(result_h5)
        np.testing.assert_array_almost_equal(data[3], f_para)

    def test_last_row_is_zeros(self, result_h5):
        data = result_to_profile_dat(result_h5)
        np.testing.assert_array_equal(data[4], np.zeros(N_IMAGES))


class TestResultToAtomsList:
    def test_returns_correct_count(self, result_h5):
        atoms_list = result_to_atoms_list(result_h5)
        assert len(atoms_list) == N_IMAGES

    def test_atoms_have_calculator(self, result_h5):
        atoms_list = result_to_atoms_list(result_h5)
        for atoms in atoms_list:
            assert atoms.calc is not None

    def test_atomic_numbers_from_metadata(self, result_h5):
        atoms_list = result_to_atoms_list(result_h5)
        for atoms in atoms_list:
            np.testing.assert_array_equal(
                atoms.numbers, [1, 6, 8]
            )

    def test_cell_from_metadata(self, result_h5_with_cell):
        atoms_list = result_to_atoms_list(result_h5_with_cell)
        for atoms in atoms_list:
            np.testing.assert_array_almost_equal(
                atoms.cell[:], np.eye(3) * 10.0
            )


class TestLoadNebResult:
    def test_has_expected_keys(self, result_h5):
        result = load_neb_result(result_h5)
        assert "path" in result
        assert "convergence" in result
        assert "metadata" in result

    def test_convergence_data(self, result_h5):
        result = load_neb_result(result_h5)
        assert "max_force" in result["convergence"]
        assert len(result["convergence"]["max_force"]) == 3

    def test_metadata_scalars(self, result_h5):
        result = load_neb_result(result_h5)
        assert result["metadata"]["converged"] == True  # noqa: E712
        assert result["metadata"]["oracle_calls"] == 30


class TestLoadNebHistory:
    def test_returns_sorted_steps(self, history_h5):
        steps = load_neb_history(history_h5)
        assert len(steps) == 3

    def test_step_data_structure(self, history_h5):
        steps = load_neb_history(history_h5)
        for step in steps:
            assert set(step.keys()) == {
                "images",
                "energies",
                "gradients",
                "f_para",
                "rxn_coord",
            }

    def test_energies_differ_across_steps(self, history_h5):
        steps = load_neb_history(history_h5)
        # We shifted energies by step_idx * 0.1
        assert not np.allclose(
            steps[0]["energies"], steps[2]["energies"]
        )


class TestHistoryToProfileDats:
    def test_one_array_per_step(self, history_h5):
        dats = history_to_profile_dats(history_h5)
        assert len(dats) == 3

    def test_array_shape(self, history_h5):
        dats = history_to_profile_dats(history_h5)
        for dat in dats:
            assert dat.shape == (5, N_IMAGES)

    def test_last_row_zeros(self, history_h5):
        dats = history_to_profile_dats(history_h5)
        for dat in dats:
            np.testing.assert_array_equal(
                dat[4], np.zeros(N_IMAGES)
            )
