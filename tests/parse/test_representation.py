# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""Energy representation: metric plane + E, not occupancy invert."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from chemparseplot.parse.representation import (
    ENERGY_SCHEMA,
    from_path_forces,
    load_energy_table,
    rotate_to_progress,
)
from chemparseplot.parse.types import LANDFOLD_FES_SCHEMA, EnergyRepresentation

pytestmark = pytest.mark.pure


def test_energy_schema_rejects_occupancy_fes() -> None:
    with pytest.raises(ValueError, match="energy"):
        EnergyRepresentation.from_mapping(
            {
                "schema": LANDFOLD_FES_SCHEMA,
                "x": np.array([0.0, 1.0]),
                "y": np.array([0.0, 1.0]),
                "energy": np.array([0.0, 1.0]),
            }
        )


def test_from_mapping_requires_matching_energy() -> None:
    with pytest.raises(ValueError, match="same length"):
        EnergyRepresentation.from_mapping(
            {
                "schema": ENERGY_SCHEMA,
                "x": np.array([0.0, 1.0]),
                "y": np.array([0.0, 1.0]),
                "energy": np.array([0.1]),
            }
        )


def test_load_energy_table_reads_optional_force(tmp_path: Path) -> None:
    path = tmp_path / "cloud.energy.csv"
    path.write_text(
        "# chemparseplot.energy.v1\n"
        "# x y energy f_para\n"
        "0.0 2.0 -1.0 0.5\n"
        "1.0 1.0  0.2 0.1\n"
        "2.0 0.0 -0.8 0.0\n"
    )
    rep = load_energy_table(path, frame="rmsd")
    assert rep.schema == ENERGY_SCHEMA
    assert rep.frame == "rmsd"
    np.testing.assert_allclose(rep.energy, [-1.0, 0.2, -0.8])
    np.testing.assert_allclose(rep.f_para, [0.5, 0.1, 0.0])
    assert rep.grad_x is None


def test_from_path_forces_attaches_synthetic_gradients() -> None:
    r = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    p = np.array([4.0, 3.0, 2.0, 1.0, 0.0])
    e = np.array([0.0, 0.4, 0.8, 0.3, -0.2])
    f = np.array([0.1, 0.2, 0.0, -0.1, 0.0])
    rep = from_path_forces(r, p, e, f, frame="rmsd", smooth=False)
    assert rep.grad_x is not None and rep.grad_y is not None
    assert rep.grad_x.shape == r.shape
    # Interior: r increases, p decreases, f>0 -> grad_r < 0, grad_p > 0
    assert rep.grad_x[1] < 0.0
    assert rep.grad_y[1] > 0.0


def test_rotate_to_progress_is_isometry_at_endpoints() -> None:
    r = np.array([0.0, 1.0, 2.0, 3.0])
    p = np.array([3.0, 2.0, 1.0, 0.0])
    e = np.array([0.0, 0.5, 0.4, -0.2])
    gx = np.array([0.1, 0.2, 0.0, -0.1])
    gy = np.array([-0.1, -0.2, 0.0, 0.1])
    rep = EnergyRepresentation(
        x=r,
        y=p,
        energy=e,
        grad_x=gx,
        grad_y=gy,
        frame="rmsd",
    )
    sd = rotate_to_progress(rep)
    assert sd.frame == "progress"
    np.testing.assert_allclose(sd.x[0], 0.0, atol=1e-12)
    np.testing.assert_allclose(sd.y[0], 0.0, atol=1e-12)
    np.testing.assert_allclose(sd.y[-1], 0.0, atol=1e-12)
    da = r - r[0]
    db = p - p[0]
    np.testing.assert_allclose(sd.x**2 + sd.y**2, da**2 + db**2, atol=1e-12)
    # Rotation preserves gradient Euclidean norm.
    assert sd.grad_x is not None and sd.grad_y is not None
    np.testing.assert_allclose(
        sd.grad_x**2 + sd.grad_y**2,
        gx**2 + gy**2,
        atol=1e-12,
    )
    np.testing.assert_array_equal(sd.energy, e)
