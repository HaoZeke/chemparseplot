# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""Load landfold FES CSV and landfold.fes.v1 mappings."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from chemparseplot.parse.landfold import load_fes_csv, load_fes_result
from chemparseplot.parse.types import LANDFOLD_FES_SCHEMA, LandfoldFesResult

pytestmark = pytest.mark.pure


def _write_grid(path: Path) -> None:
    xs = np.array([0.0, 1.0, 2.0])
    ys = np.array([10.0, 20.0])
    lines = ["# x y F rho"]
    for y in ys:
        for x in xs:
            f = x + 0.1 * y
            lines.append(f"{x:.8f} {y:.8f} {f:.8f} 1.00000000e0")
        lines.append("")
    path.write_text("\n".join(lines) + "\n")


def test_load_fes_csv_regular_grid(tmp_path: Path) -> None:
    csv = tmp_path / "fes.csv"
    _write_grid(csv)
    result = load_fes_csv(csv, kt=0.168)
    assert isinstance(result, LandfoldFesResult)
    assert result.schema == LANDFOLD_FES_SCHEMA
    np.testing.assert_allclose(result.x, [0.0, 1.0, 2.0])
    np.testing.assert_allclose(result.y, [10.0, 20.0])
    assert result.free_energy.shape == (2, 3)
    assert result.density.shape == (2, 3)
    assert result.kt == pytest.approx(0.168)
    assert result.free_energy[0, 1] == pytest.approx(2.0)
    assert result.metadata["source"] == str(csv)


def test_load_fes_result_from_binding_dict() -> None:
    payload = {
        "schema": LANDFOLD_FES_SCHEMA,
        "x": [0.0, 1.0],
        "y": [0.0, 1.0],
        "free_energy": [[0.0, 1.0], [2.0, 3.0]],
        "density": [[1.0, 0.5], [0.25, 0.1]],
        "kt": 1.0,
        "metadata": {"provenance": "test"},
    }
    result = load_fes_result(payload)
    assert result["schema"] == LANDFOLD_FES_SCHEMA
    np.testing.assert_allclose(result["free_energy"], [[0.0, 1.0], [2.0, 3.0]])
    assert result.metadata["provenance"] == "test"


def test_rejects_wrong_schema() -> None:
    with pytest.raises(ValueError, match="expected schema"):
        load_fes_result(
            {
                "schema": "landfold.fes.v0",
                "x": [0.0],
                "y": [0.0],
                "free_energy": [[0.0]],
            }
        )


def test_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape"):
        load_fes_result(
            {
                "schema": LANDFOLD_FES_SCHEMA,
                "x": [0.0, 1.0],
                "y": [0.0],
                "free_energy": [[0.0]],
            }
        )


def test_rejects_irregular_csv(tmp_path: Path) -> None:
    csv = tmp_path / "jagged.csv"
    csv.write_text("# x y F\n0 0 1\n1 0 1\n0 1 1\n")
    with pytest.raises(ValueError, match="regular grid"):
        load_fes_csv(csv)
