# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""Unit tests for the (s, d) projection module."""
import numpy as np
import pytest

from chemparseplot.parse.projection import (
    ProjectionBasis,
    compute_projection_basis,
    inverse_sd_to_ab,
    project_to_sd,
)


class TestComputeProjectionBasis:
    """Tests for compute_projection_basis."""

    def test_axis_aligned_path(self):
        """Path along the a-axis: u_a=1, u_b=0."""
        rmsd_a = np.array([0.0, 1.0, 2.0, 3.0])
        rmsd_b = np.array([0.0, 0.0, 0.0, 0.0])
        basis = compute_projection_basis(rmsd_a, rmsd_b)
        assert basis.a_start == 0.0
        assert basis.b_start == 0.0
        np.testing.assert_allclose(basis.u_a, 1.0)
        np.testing.assert_allclose(basis.u_b, 0.0)
        np.testing.assert_allclose(basis.v_a, 0.0)
        np.testing.assert_allclose(basis.v_b, 1.0)
        np.testing.assert_allclose(basis.path_norm, 3.0)

    def test_diagonal_path(self):
        """45-degree path: u = (1/sqrt2, 1/sqrt2)."""
        rmsd_a = np.array([0.0, 1.0, 2.0])
        rmsd_b = np.array([0.0, 1.0, 2.0])
        basis = compute_projection_basis(rmsd_a, rmsd_b)
        inv_sqrt2 = 1.0 / np.sqrt(2.0)
        np.testing.assert_allclose(basis.u_a, inv_sqrt2)
        np.testing.assert_allclose(basis.u_b, inv_sqrt2)
        np.testing.assert_allclose(basis.v_a, -inv_sqrt2)
        np.testing.assert_allclose(basis.v_b, inv_sqrt2)
        np.testing.assert_allclose(basis.path_norm, 2.0 * np.sqrt(2.0))

    def test_neb_like_path(self):
        """Typical NEB: rmsd_a increases, rmsd_b decreases."""
        rmsd_a = np.array([0.0, 1.0, 2.0, 3.0])
        rmsd_b = np.array([3.0, 2.0, 1.0, 0.0])
        basis = compute_projection_basis(rmsd_a, rmsd_b)
        np.testing.assert_allclose(basis.path_norm, np.hypot(3.0, -3.0))
        # Orthonormality check
        dot = basis.u_a * basis.v_a + basis.u_b * basis.v_b
        np.testing.assert_allclose(dot, 0.0, atol=1e-15)

    def test_nonzero_origin(self):
        """Path not starting at (0,0)."""
        rmsd_a = np.array([1.0, 2.0, 4.0])
        rmsd_b = np.array([5.0, 4.0, 2.0])
        basis = compute_projection_basis(rmsd_a, rmsd_b)
        assert basis.a_start == 1.0
        assert basis.b_start == 5.0

    def test_zero_length_raises(self):
        """Degenerate path (single point) should raise ValueError."""
        rmsd_a = np.array([1.0, 1.0, 1.0])
        rmsd_b = np.array([2.0, 2.0, 2.0])
        with pytest.raises(ValueError, match="zero length"):
            compute_projection_basis(rmsd_a, rmsd_b)

    def test_frozen_dataclass(self):
        """Basis should be immutable."""
        rmsd_a = np.array([0.0, 1.0])
        rmsd_b = np.array([0.0, 1.0])
        basis = compute_projection_basis(rmsd_a, rmsd_b)
        with pytest.raises(AttributeError):
            basis.u_a = 0.5


class TestProjectToSd:
    """Tests for project_to_sd."""

    def test_endpoints_project_correctly(self):
        """First point -> s=0, last point -> s=path_norm; both d near 0."""
        rmsd_a = np.array([0.0, 1.0, 2.0, 3.0])
        rmsd_b = np.array([3.0, 2.0, 1.0, 0.0])
        basis = compute_projection_basis(rmsd_a, rmsd_b)
        s, d = project_to_sd(rmsd_a, rmsd_b, basis)
        np.testing.assert_allclose(s[0], 0.0, atol=1e-15)
        np.testing.assert_allclose(s[-1], basis.path_norm, atol=1e-12)
        np.testing.assert_allclose(d[0], 0.0, atol=1e-15)
        np.testing.assert_allclose(d[-1], 0.0, atol=1e-12)

    def test_on_path_points_have_zero_deviation(self):
        """Points exactly on the line from start to end have d=0."""
        rmsd_a = np.linspace(0, 3, 7)
        rmsd_b = np.linspace(3, 0, 7)
        basis = compute_projection_basis(rmsd_a, rmsd_b)
        s, d = project_to_sd(rmsd_a, rmsd_b, basis)
        np.testing.assert_allclose(d, 0.0, atol=1e-14)
        # s should be monotonically increasing
        assert np.all(np.diff(s) > 0)

    def test_off_path_point_has_nonzero_deviation(self):
        """A point displaced from the path should have nonzero d."""
        rmsd_a = np.array([0.0, 2.0])
        rmsd_b = np.array([0.0, 0.0])
        basis = compute_projection_basis(rmsd_a, rmsd_b)
        # Point at (1, 1) is above the path (path is along a-axis)
        s, d = project_to_sd(np.array([1.0]), np.array([1.0]), basis)
        np.testing.assert_allclose(s[0], 1.0)
        np.testing.assert_allclose(d[0], 1.0)

    def test_shape_preserved(self):
        n = 15
        rmsd_a = np.linspace(0, 5, n)
        rmsd_b = np.linspace(5, 0, n)
        basis = compute_projection_basis(rmsd_a, rmsd_b)
        s, d = project_to_sd(rmsd_a, rmsd_b, basis)
        assert s.shape == (n,)
        assert d.shape == (n,)


class TestInverseSdToAb:
    """Tests for inverse_sd_to_ab (round-trip consistency)."""

    def test_roundtrip(self):
        """project then inverse should recover the original coordinates."""
        rng = np.random.default_rng(42)
        rmsd_a = np.sort(rng.uniform(0, 5, 20))
        rmsd_b = 5.0 - rmsd_a + rng.normal(0, 0.2, 20)
        basis = compute_projection_basis(rmsd_a, rmsd_b)
        s, d = project_to_sd(rmsd_a, rmsd_b, basis)
        a_rec, b_rec = inverse_sd_to_ab(s, d, basis)
        np.testing.assert_allclose(a_rec, rmsd_a, atol=1e-12)
        np.testing.assert_allclose(b_rec, rmsd_b, atol=1e-12)

    def test_grid_roundtrip(self):
        """Meshgrid coordinates should survive the round-trip."""
        rmsd_a = np.array([0.0, 3.0])
        rmsd_b = np.array([4.0, 0.0])
        basis = compute_projection_basis(rmsd_a, rmsd_b)
        sg = np.linspace(0, basis.path_norm, 10)
        dg = np.linspace(-2, 2, 10)
        ss, dd = np.meshgrid(sg, dg)
        a, b = inverse_sd_to_ab(ss.ravel(), dd.ravel(), basis)
        s_back, d_back = project_to_sd(a, b, basis)
        np.testing.assert_allclose(s_back, ss.ravel(), atol=1e-12)
        np.testing.assert_allclose(d_back, dd.ravel(), atol=1e-12)
