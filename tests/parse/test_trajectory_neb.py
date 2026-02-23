# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""Tests for the trajectory NEB parser.

The physics computations (cumulative distance, tangent forces) are validated
against hand-calculated reference values.
"""
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tests.conftest import skip_if_not_env

skip_if_not_env("neb")

from ase import Atoms  # noqa: E402

from chemparseplot.parse.trajectory.neb import (  # noqa: E402
    _get_energy,
    compute_cumulative_distance,
    compute_tangent_force,
    extract_profile_data,
    load_trajectory,
    trajectory_to_profile_dat,
)

pytestmark = pytest.mark.neb


def _make_atoms(positions, energy=0.0, forces=None):
    """Create an ASE Atoms object with attached energy and optional forces.

    Uses a SinglePointCalculator when forces are provided so that
    atoms.get_forces() and atoms.get_potential_energy() work correctly.
    """
    atoms = Atoms("H" * len(positions), positions=positions)
    if forces is not None:
        from ase.calculators.singlepoint import SinglePointCalculator

        calc = SinglePointCalculator(atoms, energy=energy, forces=forces)
        atoms.calc = calc
    else:
        atoms.info["energy"] = energy
    return atoms


class TestGetEnergy:
    def test_energy_from_info(self):
        atoms = _make_atoms([[0, 0, 0]], energy=-1.5)
        assert _get_energy(atoms) == -1.5

    def test_energy_from_calc(self):
        atoms = _make_atoms(
            [[0, 0, 0]], energy=-2.0, forces=[[0, 0, 0]]
        )
        assert _get_energy(atoms) == -2.0

    def test_energy_fallback_zero(self):
        atoms = Atoms("H", positions=[[0, 0, 0]])
        assert _get_energy(atoms) == 0.0


class TestComputeCumulativeDistance:
    def test_single_image(self):
        atoms_list = [_make_atoms([[0, 0, 0]])]
        dists = compute_cumulative_distance(atoms_list)
        assert len(dists) == 1
        assert dists[0] == 0.0

    def test_linear_path(self):
        # Three atoms along x-axis, 1 angstrom apart
        atoms_list = [
            _make_atoms([[0, 0, 0]]),
            _make_atoms([[1, 0, 0]]),
            _make_atoms([[2, 0, 0]]),
        ]
        dists = compute_cumulative_distance(atoms_list)
        np.testing.assert_array_almost_equal(dists, [0.0, 1.0, 2.0])

    def test_cumulative_property(self):
        atoms_list = [
            _make_atoms([[0, 0, 0]]),
            _make_atoms([[3, 4, 0]]),  # distance = 5
            _make_atoms([[6, 8, 0]]),  # another 5
        ]
        dists = compute_cumulative_distance(atoms_list)
        np.testing.assert_array_almost_equal(dists, [0.0, 5.0, 10.0])

    def test_monotonically_increasing(self):
        rng = np.random.default_rng(42)
        atoms_list = [_make_atoms(rng.random((3, 3))) for _ in range(10)]
        dists = compute_cumulative_distance(atoms_list)
        assert all(dists[i] <= dists[i + 1] for i in range(len(dists) - 1))


class TestComputeTangentForce:
    def test_endpoints_are_zero(self):
        atoms_list = [
            _make_atoms([[0, 0, 0]], energy=0.0, forces=[[1, 0, 0]]),
            _make_atoms([[1, 0, 0]], energy=1.0, forces=[[0, 1, 0]]),
            _make_atoms([[2, 0, 0]], energy=0.5, forces=[[-1, 0, 0]]),
        ]
        energies = np.array([0.0, 1.0, 0.5])
        f_para = compute_tangent_force(atoms_list, energies)
        assert f_para[0] == 0.0
        assert f_para[-1] == 0.0

    def test_monotonic_uphill_uses_tau_plus(self):
        """When e_next > e_curr > e_prev, tangent = tau_plus."""
        # Linear path along x: 0 -> 1 -> 2
        # Energies increasing: 0, 1, 2
        # Force at image 1: [1, 0, 0] (along path)
        atoms_list = [
            _make_atoms([[0, 0, 0]], energy=0.0, forces=[[0, 0, 0]]),
            _make_atoms([[1, 0, 0]], energy=1.0, forces=[[1, 0, 0]]),
            _make_atoms([[2, 0, 0]], energy=2.0, forces=[[0, 0, 0]]),
        ]
        energies = np.array([0.0, 1.0, 2.0])
        f_para = compute_tangent_force(atoms_list, energies)
        # tau_plus = [2,0,0]-[1,0,0] = [1,0,0], normalized = [1,0,0]
        # f_para[1] = dot([1,0,0], [1,0,0]) = 1.0
        assert f_para[1] == pytest.approx(1.0)

    def test_monotonic_downhill_uses_tau_minus(self):
        """When e_next < e_curr < e_prev, tangent = tau_minus."""
        atoms_list = [
            _make_atoms([[0, 0, 0]], energy=2.0, forces=[[0, 0, 0]]),
            _make_atoms([[1, 0, 0]], energy=1.0, forces=[[1, 0, 0]]),
            _make_atoms([[2, 0, 0]], energy=0.0, forces=[[0, 0, 0]]),
        ]
        energies = np.array([2.0, 1.0, 0.0])
        f_para = compute_tangent_force(atoms_list, energies)
        # tau_minus = [1,0,0]-[0,0,0] = [1,0,0], normalized = [1,0,0]
        # f_para[1] = dot([1,0,0], [1,0,0]) = 1.0
        assert f_para[1] == pytest.approx(1.0)

    def test_extremum_uses_bisection(self):
        """At a maximum, energy-weighted bisection is used."""
        # Image 1 is at a maximum: e_prev=0, e_curr=2, e_next=1
        atoms_list = [
            _make_atoms([[0, 0, 0]], energy=0.0, forces=[[0, 0, 0]]),
            _make_atoms([[1, 0, 0]], energy=2.0, forces=[[0.5, 0, 0]]),
            _make_atoms([[2, 0, 0]], energy=1.0, forces=[[0, 0, 0]]),
        ]
        energies = np.array([0.0, 2.0, 1.0])
        f_para = compute_tangent_force(atoms_list, energies)
        # Not monotonic: extremum case
        # de_plus = 1-2 = -1, de_minus = 2-0 = 2
        # de_max = 2, de_min = 1
        # e_next(1) > e_prev(0), so tau = tau_plus*de_max + tau_minus*de_min
        # tau_plus = [1,0,0], tau_minus = [1,0,0]
        # tau = [3, 0, 0], normalized = [1, 0, 0]
        # f_para[1] = dot([0.5, 0, 0], [1, 0, 0]) = 0.5
        assert f_para[1] == pytest.approx(0.5)

    def test_no_calc_uses_zero_forces(self):
        atoms_list = [
            _make_atoms([[0, 0, 0]], energy=0.0),
            _make_atoms([[1, 0, 0]], energy=1.0),
            _make_atoms([[2, 0, 0]], energy=0.5),
        ]
        energies = np.array([0.0, 1.0, 0.5])
        f_para = compute_tangent_force(atoms_list, energies)
        np.testing.assert_array_equal(f_para, np.zeros(3))

    def test_output_length(self):
        n = 6
        atoms_list = [
            _make_atoms(
                [[i, 0, 0]], energy=float(i), forces=[[0.1, 0, 0]]
            )
            for i in range(n)
        ]
        energies = np.arange(n, dtype=float)
        f_para = compute_tangent_force(atoms_list, energies)
        assert len(f_para) == n


class TestExtractProfileData:
    def test_returns_four_arrays(self):
        atoms_list = [
            _make_atoms([[0, 0, 0]], energy=0.0),
            _make_atoms([[1, 0, 0]], energy=1.0),
            _make_atoms([[2, 0, 0]], energy=0.5),
        ]
        result = extract_profile_data(atoms_list)
        assert len(result) == 4
        indices, distances, energies, f_para = result
        assert len(indices) == 3
        assert len(distances) == 3
        assert len(energies) == 3
        assert len(f_para) == 3

    def test_indices_are_sequential(self):
        atoms_list = [_make_atoms([[i, 0, 0]], energy=0.0) for i in range(5)]
        indices, _, _, _ = extract_profile_data(atoms_list)
        np.testing.assert_array_equal(indices, [0, 1, 2, 3, 4])

    def test_energies_match(self):
        expected = [0.0, -1.5, 0.3]
        atoms_list = [
            _make_atoms([[i, 0, 0]], energy=e)
            for i, e in enumerate(expected)
        ]
        _, _, energies, _ = extract_profile_data(atoms_list)
        np.testing.assert_array_almost_equal(energies, expected)


class TestTrajectoryToProfileDat:
    def test_shape_is_5_by_n(self):
        n = 4
        atoms_list = [
            _make_atoms([[i, 0, 0]], energy=float(i)) for i in range(n)
        ]
        data = trajectory_to_profile_dat(atoms_list)
        assert data.shape == (5, n)

    def test_last_row_is_zeros(self):
        atoms_list = [
            _make_atoms([[i, 0, 0]], energy=float(i)) for i in range(3)
        ]
        data = trajectory_to_profile_dat(atoms_list)
        np.testing.assert_array_equal(data[4], np.zeros(3))

    def test_index_row_is_sequential(self):
        atoms_list = [
            _make_atoms([[i, 0, 0]], energy=float(i)) for i in range(5)
        ]
        data = trajectory_to_profile_dat(atoms_list)
        np.testing.assert_array_equal(data[0], [0, 1, 2, 3, 4])


class TestLoadTrajectory:
    def test_single_atoms_wrapped_in_list(self):
        single_atoms = Atoms("H", positions=[[0, 0, 0]])
        with patch(
            "chemparseplot.parse.trajectory.neb.ase_read",
            return_value=single_atoms,
        ):
            result = load_trajectory("dummy.xyz")
            assert isinstance(result, list)
            assert len(result) == 1

    def test_list_returned_as_is(self):
        atoms_list = [
            Atoms("H", positions=[[i, 0, 0]]) for i in range(3)
        ]
        with patch(
            "chemparseplot.parse.trajectory.neb.ase_read",
            return_value=atoms_list,
        ):
            result = load_trajectory("dummy.xyz")
            assert len(result) == 3
