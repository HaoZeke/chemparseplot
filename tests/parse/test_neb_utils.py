# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
import numpy as np
import pytest

from tests.conftest import skip_if_not_env

skip_if_not_env("neb")

from chemparseplot.parse.neb_utils import (  # noqa: E402
    compute_synthetic_gradients,
    create_landscape_dataframe,
)

pytestmark = pytest.mark.neb


class TestComputeSyntheticGradients:
    """Tests for the synthetic 2D gradient projection."""

    def test_zero_force_gives_zero_gradients(self):
        rmsd_r = np.array([0.0, 1.0, 2.0, 3.0])
        rmsd_p = np.array([3.0, 2.0, 1.0, 0.0])
        f_para = np.zeros(4)
        grad_r, grad_p = compute_synthetic_gradients(rmsd_r, rmsd_p, f_para)
        np.testing.assert_array_equal(grad_r, np.zeros(4))
        np.testing.assert_array_equal(grad_p, np.zeros(4))

    def test_output_shape_matches_input(self):
        n = 7
        rmsd_r = np.linspace(0, 3, n)
        rmsd_p = np.linspace(3, 0, n)
        f_para = np.random.default_rng(42).standard_normal(n)
        grad_r, grad_p = compute_synthetic_gradients(rmsd_r, rmsd_p, f_para)
        assert grad_r.shape == (n,)
        assert grad_p.shape == (n,)

    def test_sign_convention(self):
        """Positive f_para should yield negative gradients along the path."""
        rmsd_r = np.array([0.0, 1.0, 2.0])
        rmsd_p = np.array([2.0, 1.0, 0.0])
        f_para = np.array([0.0, 1.0, 0.0])
        grad_r, grad_p = compute_synthetic_gradients(rmsd_r, rmsd_p, f_para)
        # Interior point: positive force -> negative gradient component
        assert grad_r[1] < 0
        assert grad_p[1] > 0  # rmsd_p decreases, so dp < 0, so -f*t_p > 0

    def test_constant_rmsd_no_division_by_zero(self):
        """When all RMSD values are identical, norm_ds=0 is handled."""
        rmsd_r = np.ones(5)
        rmsd_p = np.ones(5)
        f_para = np.ones(5)
        grad_r, grad_p = compute_synthetic_gradients(rmsd_r, rmsd_p, f_para)
        assert not np.any(np.isnan(grad_r))
        assert not np.any(np.isnan(grad_p))


class TestCreateLandscapeDataframe:
    """Tests for the landscape DataFrame factory."""

    def test_schema(self):
        n = 5
        df = create_landscape_dataframe(
            rmsd_r=np.zeros(n),
            rmsd_p=np.zeros(n),
            grad_r=np.zeros(n),
            grad_p=np.zeros(n),
            z=np.zeros(n),
            step=0,
        )
        assert set(df.columns) == {"r", "p", "grad_r", "grad_p", "z", "step"}
        assert df.height == n

    def test_values_roundtrip(self):
        r = np.array([0.0, 1.0, 2.0])
        p = np.array([3.0, 2.0, 1.0])
        gr = np.array([0.1, 0.2, 0.3])
        gp = np.array([-0.1, -0.2, -0.3])
        z = np.array([-1.0, 0.5, -0.8])
        step = 3

        df = create_landscape_dataframe(r, p, gr, gp, z, step)
        np.testing.assert_array_almost_equal(df["r"].to_numpy(), r)
        np.testing.assert_array_almost_equal(df["p"].to_numpy(), p)
        np.testing.assert_array_almost_equal(df["z"].to_numpy(), z)
        assert df["step"][0] == step

    def test_step_is_constant(self):
        n = 4
        df = create_landscape_dataframe(
            np.zeros(n),
            np.zeros(n),
            np.zeros(n),
            np.zeros(n),
            np.zeros(n),
            step=7,
        )
        assert all(s == 7 for s in df["step"].to_list())
