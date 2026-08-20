# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""Finite IMQ representer matches the Lean Gram identities."""

from __future__ import annotations

import numpy as np
import pytest

from chemparseplot.plot.representer import (
    adj_k2,
    apply_k2,
    coalesce_sites,
    det_k2,
    farthest_indices,
    imq_gram,
    solve_representer,
    training_residual,
)

pytestmark = pytest.mark.pure


def test_gram_adjugate_matches_lean() -> None:
    a, b, c, y1, y2 = 4, 1, 5, 3, -2
    got = apply_k2(a, b, c, *adj_k2(a, b, c, y1, y2))
    det = det_k2(a, b, c)
    assert got == (det * y1, det * y2)
    assert det == 19


def test_kernel2_det_pos() -> None:
    assert det_k2(4, 1, 5) > 0


def test_imq_gram_interpolates_two_sites() -> None:
    xy = np.array([[0.0, 0.0], [1.0, 0.0]])
    z = np.array([2.0, -1.0])
    resid = training_residual(xy, z, eps=0.5)
    np.testing.assert_allclose(resid, 0.0, atol=1e-10)


def test_imq_gram_symmetric_diag_one() -> None:
    xy = np.array([[0.0, 0.0], [0.5, 0.2], [1.0, 1.0]])
    K = imq_gram(xy, 0.8)
    np.testing.assert_allclose(K, K.T)
    np.testing.assert_allclose(np.diag(K), 1.0)


def test_solve_representer_rejects_duplicate_sites() -> None:
    xy = np.array([[0.0, 0.0], [0.0, 0.0]])
    z = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="stable representer"):
        solve_representer(xy, z, eps=1.0, nugget=0.0)


def test_coalesce_and_farthest() -> None:
    s1 = np.array([0.0, 0.0, 1.0])
    s2 = np.array([0.0, 0.0, 1.0])
    z = np.array([1.0, 3.0, 0.0])
    c1, c2, _, _, cz, _ = coalesce_sites(s1, s2, z)
    assert c1.size == 2
    assert cz[0] == pytest.approx(2.0)
    idx = farthest_indices(np.column_stack([c1, c2]), 2)
    assert sorted(idx.tolist()) == [0, 1]
