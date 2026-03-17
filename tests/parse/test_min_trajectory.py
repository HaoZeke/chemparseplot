# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""Tests for minimization trajectory parser."""
import numpy as np
import polars as pl
import pytest

from chemparseplot.parse.eon.min_trajectory import (
    MinTrajectoryData,
    parse_min_dat,
)


@pytest.fixture
def synthetic_min_dat(tmp_path):
    """Create a synthetic minimization .dat TSV file."""
    dat = tmp_path / "min.dat"
    lines = [
        "iteration\tstep_size\tconvergence\tenergy",
        "0\t0.00000e+00\t5.00000e-01\t-12.345678",
        "1\t1.20000e-01\t2.50000e-01\t-12.456789",
        "2\t8.00000e-02\t5.00000e-02\t-12.567890",
        "3\t3.00000e-02\t8.00000e-03\t-12.578901",
    ]
    dat.write_text("\n".join(lines) + "\n")
    return dat


class TestParseMinDat:
    def test_schema(self, synthetic_min_dat):
        df = parse_min_dat(synthetic_min_dat)
        expected_cols = {"iteration", "step_size", "convergence", "energy"}
        assert set(df.columns) == expected_cols

    def test_row_count(self, synthetic_min_dat):
        df = parse_min_dat(synthetic_min_dat)
        assert df.height == 4

    def test_values(self, synthetic_min_dat):
        df = parse_min_dat(synthetic_min_dat)
        assert df["iteration"][0] == 0
        assert df["iteration"][3] == 3
        np.testing.assert_allclose(df["energy"][0], -12.345678, atol=1e-6)
        np.testing.assert_allclose(df["convergence"][3], 0.008, atol=1e-4)

    def test_energy_decreasing(self, synthetic_min_dat):
        df = parse_min_dat(synthetic_min_dat)
        energies = df["energy"].to_numpy()
        assert np.all(np.diff(energies) <= 0)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(Exception):
            parse_min_dat(tmp_path / "nonexistent.dat")


class TestMinTrajectoryData:
    def test_dataclass_fields(self):
        from dataclasses import fields

        names = {f.name for f in fields(MinTrajectoryData)}
        assert names == {
            "atoms_list",
            "dat_df",
            "initial_atoms",
            "final_atoms",
        }
