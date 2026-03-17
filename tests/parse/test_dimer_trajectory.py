# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""Tests for dimer trajectory parser."""
import numpy as np
import polars as pl
import pytest

from chemparseplot.parse.eon.dimer_trajectory import (
    DimerTrajectoryData,
    parse_climb_dat,
)


@pytest.fixture
def synthetic_climb_dat(tmp_path):
    """Create a synthetic climb.dat TSV file."""
    dat = tmp_path / "climb.dat"
    lines = [
        "iteration\tstep_size\tdelta_e\tconvergence\teigenvalue\ttorque\tangle\trotations",
        "1\t1.0000000e-01\t0.012345\t5.00000e-02\t-0.123400\t0.050000\t12.3456\t5",
        "2\t8.5000000e-02\t0.023456\t3.20000e-02\t-0.234500\t0.030000\t8.1234\t3",
        "3\t6.0000000e-02\t0.034000\t1.10000e-02\t-0.345600\t0.010000\t3.4567\t2",
    ]
    dat.write_text("\n".join(lines) + "\n")
    return dat


class TestParseClimbDat:
    def test_schema(self, synthetic_climb_dat):
        df = parse_climb_dat(synthetic_climb_dat)
        expected_cols = {
            "iteration",
            "step_size",
            "delta_e",
            "convergence",
            "eigenvalue",
            "torque",
            "angle",
            "rotations",
        }
        assert set(df.columns) == expected_cols

    def test_row_count(self, synthetic_climb_dat):
        df = parse_climb_dat(synthetic_climb_dat)
        assert df.height == 3

    def test_values(self, synthetic_climb_dat):
        df = parse_climb_dat(synthetic_climb_dat)
        assert df["iteration"][0] == 1
        assert df["iteration"][2] == 3
        np.testing.assert_allclose(df["eigenvalue"][0], -0.1234, atol=1e-4)
        np.testing.assert_allclose(df["convergence"][2], 0.011, atol=1e-4)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(Exception):
            parse_climb_dat(tmp_path / "nonexistent.dat")


class TestDimerTrajectoryData:
    def test_dataclass_fields(self):
        """Verify dataclass has expected fields."""
        from dataclasses import fields

        names = {f.name for f in fields(DimerTrajectoryData)}
        assert names == {
            "atoms_list",
            "dat_df",
            "initial_atoms",
            "saddle_atoms",
            "mode_vector",
        }
