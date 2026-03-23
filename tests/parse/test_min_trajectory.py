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
        with pytest.raises((FileNotFoundError, OSError)):
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


class TestLoadMinTrajectory:
    """Tests for load_min_trajectory with synthetic eOn output."""

    def _make_job_dir(self, tmp_path, prefix="min"):
        from ase.build import molecule
        from ase.io import write

        h2o = molecule("H2O")
        frames = [h2o.copy() for _ in range(4)]
        for i, f in enumerate(frames):
            f.positions[0, 0] += 0.05 * i
        write(str(tmp_path / prefix), frames, format="eon")

        dat = tmp_path / f"{prefix}.dat"
        lines = [
            "iteration\tstep_size\tconvergence\tenergy",
            "0\t0.0\t0.5\t-12.3",
            "1\t0.12\t0.25\t-12.4",
            "2\t0.08\t0.05\t-12.5",
            "3\t0.03\t0.008\t-12.55",
        ]
        dat.write_text("\n".join(lines) + "\n")

        # min.con (final structure)
        write(str(tmp_path / "min.con"), frames[-1], format="eon")
        return tmp_path

    def test_load_full(self, tmp_path):
        from chemparseplot.parse.eon.min_trajectory import load_min_trajectory

        job = self._make_job_dir(tmp_path)
        traj = load_min_trajectory(job)
        assert len(traj.atoms_list) == 4
        assert traj.dat_df.height == 4
        assert traj.initial_atoms is not None
        assert traj.final_atoms is not None

    def test_missing_movie_raises(self, tmp_path):
        from chemparseplot.parse.eon.min_trajectory import load_min_trajectory

        (tmp_path / "min.dat").write_text("iteration\tstep_size\n")
        with pytest.raises(FileNotFoundError, match="movie file"):
            load_min_trajectory(tmp_path)

    def test_missing_dat_raises(self, tmp_path):
        from ase.build import molecule
        from ase.io import write

        from chemparseplot.parse.eon.min_trajectory import load_min_trajectory

        write(str(tmp_path / "min"), molecule("H2O"), format="eon")
        with pytest.raises(FileNotFoundError, match=r"\.dat"):
            load_min_trajectory(tmp_path)

    def test_custom_prefix(self, tmp_path):
        from chemparseplot.parse.eon.min_trajectory import load_min_trajectory

        self._make_job_dir(tmp_path, prefix="react_neb")
        traj = load_min_trajectory(tmp_path, prefix="react_neb")
        assert len(traj.atoms_list) == 4

    def test_no_min_con_uses_last_frame(self, tmp_path):
        from ase.build import molecule
        from ase.io import write

        from chemparseplot.parse.eon.min_trajectory import load_min_trajectory

        h2o = molecule("H2O")
        write(str(tmp_path / "min"), [h2o, h2o], format="eon")
        (tmp_path / "min.dat").write_text(
            "iteration\tstep_size\tconvergence\tenergy\n"
            "0\t0.0\t0.5\t-10.0\n"
            "1\t0.1\t0.01\t-10.5\n"
        )
        traj = load_min_trajectory(tmp_path)
        assert traj.final_atoms is not None  # fell back to last frame
