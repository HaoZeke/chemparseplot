# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""Regression tests ensuring the projection refactor is numerically identical.

Compares the old inline implementation (copied here as reference) against
the new ``chemparseplot.parse.projection`` module.
"""
import numpy as np

from chemparseplot.parse.projection import (
    compute_projection_basis,
    inverse_sd_to_ab,
    project_to_sd,
)


def _old_inline_project(rmsd_r, rmsd_p):
    """Reference implementation: the old inline code from plot/neb.py."""
    r_start, p_start = rmsd_r[0], rmsd_p[0]
    r_end, p_end = rmsd_r[-1], rmsd_p[-1]

    vec_r, vec_p = r_end - r_start, p_end - p_start
    path_norm = np.hypot(vec_r, vec_p)
    u_r, u_p = vec_r / path_norm, vec_p / path_norm
    v_r, v_p = -u_p, u_r

    s_data = (rmsd_r - r_start) * u_r + (rmsd_p - p_start) * u_p
    d_data = (rmsd_r - r_start) * v_r + (rmsd_p - p_start) * v_p
    return s_data, d_data, r_start, p_start, u_r, u_p, v_r, v_p


def _old_inline_inverse(s, d, r_start, p_start, u_r, u_p, v_r, v_p):
    """Reference: old inverse for grid evaluation."""
    eval_r = r_start + s * u_r + d * v_r
    eval_p = p_start + s * u_p + d * v_p
    return eval_r, eval_p


class TestProjectionRefactorRegression:
    """Ensure new module matches old inline code exactly."""

    def _make_neb_data(self, n=11, seed=123):
        rng = np.random.default_rng(seed)
        rmsd_r = np.linspace(0.1, 3.5, n) + rng.normal(0, 0.05, n)
        rmsd_p = np.linspace(3.5, 0.1, n) + rng.normal(0, 0.05, n)
        return rmsd_r, rmsd_p

    def test_forward_projection_matches(self):
        rmsd_r, rmsd_p = self._make_neb_data()
        # Old
        s_old, d_old, *_ = _old_inline_project(rmsd_r, rmsd_p)
        # New
        basis = compute_projection_basis(rmsd_r, rmsd_p)
        s_new, d_new = project_to_sd(rmsd_r, rmsd_p, basis)
        np.testing.assert_allclose(s_new, s_old, atol=1e-14)
        np.testing.assert_allclose(d_new, d_old, atol=1e-14)

    def test_inverse_projection_matches(self):
        rmsd_r, rmsd_p = self._make_neb_data()
        s_old, d_old, r_start, p_start, u_r, u_p, v_r, v_p = _old_inline_project(
            rmsd_r, rmsd_p
        )
        # Old inverse
        eval_r_old, eval_p_old = _old_inline_inverse(
            s_old, d_old, r_start, p_start, u_r, u_p, v_r, v_p
        )
        # New inverse
        basis = compute_projection_basis(rmsd_r, rmsd_p)
        s_new, d_new = project_to_sd(rmsd_r, rmsd_p, basis)
        eval_r_new, eval_p_new = inverse_sd_to_ab(s_new, d_new, basis)
        np.testing.assert_allclose(eval_r_new, eval_r_old, atol=1e-14)
        np.testing.assert_allclose(eval_p_new, eval_p_old, atol=1e-14)

    def test_grid_evaluation_matches(self):
        """Simulate the grid evaluation path from plot_landscape_surface."""
        rmsd_r, rmsd_p = self._make_neb_data()
        s_old, d_old, r_start, p_start, u_r, u_p, v_r, v_p = _old_inline_project(
            rmsd_r, rmsd_p
        )
        # Build a grid in (s, d) space
        sg = np.linspace(s_old.min(), s_old.max(), 50)
        dg = np.linspace(d_old.min() - 1, d_old.max() + 1, 50)
        xg, yg = np.meshgrid(sg, dg)
        # Old: grid_pts_eval
        old_a = r_start + xg.ravel() * u_r + yg.ravel() * v_r
        old_b = p_start + xg.ravel() * u_p + yg.ravel() * v_p
        # New
        basis = compute_projection_basis(rmsd_r, rmsd_p)
        new_a, new_b = inverse_sd_to_ab(xg.ravel(), yg.ravel(), basis)
        np.testing.assert_allclose(new_a, old_a, atol=1e-13)
        np.testing.assert_allclose(new_b, old_b, atol=1e-13)

    def test_extra_points_projection_matches(self):
        """Simulate the extra_points projection path."""
        rmsd_r, rmsd_p = self._make_neb_data()
        _, _, r_start, p_start, u_r, u_p, v_r, v_p = _old_inline_project(
            rmsd_r, rmsd_p
        )
        # Extra points
        rng = np.random.default_rng(99)
        extra = rng.uniform(0, 4, (5, 2))
        # Old
        old_s = (extra[:, 0] - r_start) * u_r + (extra[:, 1] - p_start) * u_p
        old_d = (extra[:, 0] - r_start) * v_r + (extra[:, 1] - p_start) * v_p
        # New
        basis = compute_projection_basis(rmsd_r, rmsd_p)
        new_s, new_d = project_to_sd(extra[:, 0], extra[:, 1], basis)
        np.testing.assert_allclose(new_s, old_s, atol=1e-14)
        np.testing.assert_allclose(new_d, old_d, atol=1e-14)

    def test_multiple_seeds(self):
        """Run with multiple random seeds for robustness."""
        for seed in range(10):
            rmsd_r, rmsd_p = self._make_neb_data(n=7 + seed, seed=seed * 37)
            s_old, d_old, *_ = _old_inline_project(rmsd_r, rmsd_p)
            basis = compute_projection_basis(rmsd_r, rmsd_p)
            s_new, d_new = project_to_sd(rmsd_r, rmsd_p, basis)
            np.testing.assert_allclose(s_new, s_old, atol=1e-14)
            np.testing.assert_allclose(d_new, d_old, atol=1e-14)
