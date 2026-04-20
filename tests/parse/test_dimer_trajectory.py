# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""Tests for dimer trajectory parser."""

import numpy as np
import polars as pl
import pytest
from ase.build import molecule

from chemparseplot.parse.eon.dimer_trajectory import (
    DimerTrajectoryData,
    parse_climb_dat,
    table_from_dimer_metadata,
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
        with pytest.raises((FileNotFoundError, OSError)):
            parse_climb_dat(tmp_path / "nonexistent.dat")


class DummyDimerFrame:
    def __init__(self, iteration=None, **metadata):
        self.frame_index = iteration
        self.energy = metadata.pop("energy", None)
        self.metadata = metadata

    def to_ase(self):
        return molecule("H2O")


class TestMetadataDimerFallback:
    def test_table_from_dimer_metadata_skips_initial_frame(self):
        df = table_from_dimer_metadata(
            [
                DummyDimerFrame(iteration=0),
                DummyDimerFrame(
                    iteration=1,
                    step_size=0.1,
                    delta_e=0.01,
                    convergence=0.05,
                    eigenvalue=-0.12,
                    torque=0.05,
                    angle=12.3,
                    rotations=5,
                ),
            ]
        )
        assert df.columns == [
            "iteration",
            "step_size",
            "delta_e",
            "convergence",
            "eigenvalue",
            "torque",
            "angle",
            "rotations",
        ]
        assert df["iteration"].to_list() == [1]


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


class TestLoadDimerTrajectory:
    """Tests for load_dimer_trajectory with synthetic eOn output."""

    def _write_con(self, path, atoms):
        """Write a .con file via ASE."""
        from ase.io import write

        write(str(path), atoms, format="eon")

    def _make_job_dir(self, tmp_path):
        """Create a synthetic saddle search job directory."""
        from ase.build import molecule

        h2o = molecule("H2O")

        # reactant.con
        self._write_con(tmp_path / "reactant.con", h2o)

        # saddle.con (slightly perturbed)
        saddle = h2o.copy()
        saddle.positions[0, 0] += 0.3
        self._write_con(tmp_path / "saddle.con", saddle)

        # climb movie (3 frames: initial + 2 iterations)
        from ase.io import write

        frames = [h2o.copy() for _ in range(3)]
        for i, f in enumerate(frames):
            f.positions[0, 0] += 0.1 * i
        write(str(tmp_path / "climb"), frames, format="eon")

        # climb.dat
        dat = tmp_path / "climb.dat"
        lines = [
            "iteration\tstep_size\tdelta_e\tconvergence\teigenvalue\ttorque\tangle\trotations",
            "1\t1.0e-01\t0.01\t5.0e-02\t-0.12\t0.05\t12.3\t5",
            "2\t8.5e-02\t0.02\t3.2e-02\t-0.23\t0.03\t8.1\t3",
        ]
        dat.write_text("\n".join(lines) + "\n")

        # mode.dat
        mode = tmp_path / "mode.dat"
        mode.write_text("1.0 0.0 0.0\n0.0 1.0 0.0\n0.0 0.0 1.0\n")

        return tmp_path

    def test_load_full(self, tmp_path):
        from chemparseplot.parse.eon.dimer_trajectory import load_dimer_trajectory

        job = self._make_job_dir(tmp_path)
        traj = load_dimer_trajectory(job)
        assert len(traj.atoms_list) == 3
        assert traj.dat_df.height == 2
        assert traj.initial_atoms is not None
        assert traj.saddle_atoms is not None
        assert traj.mode_vector is not None
        assert traj.mode_vector.shape == (3, 3)

    def test_missing_climb_raises(self, tmp_path):
        from chemparseplot.parse.eon.dimer_trajectory import load_dimer_trajectory

        (tmp_path / "climb.dat").write_text("iteration\tstep_size\n")
        with pytest.raises(FileNotFoundError, match="climb movie"):
            load_dimer_trajectory(tmp_path)

    def test_missing_dat_raises(self, tmp_path):
        from ase.build import molecule
        from ase.io import write

        from chemparseplot.parse.eon.dimer_trajectory import load_dimer_trajectory

        write(str(tmp_path / "climb"), molecule("H2O"), format="eon")
        with pytest.raises(FileNotFoundError, match=r"climb\.dat"):
            load_dimer_trajectory(tmp_path)

    def test_no_reactant_uses_first_frame(self, tmp_path):
        from ase.build import molecule
        from ase.io import write

        from chemparseplot.parse.eon.dimer_trajectory import load_dimer_trajectory

        h2o = molecule("H2O")
        write(str(tmp_path / "climb"), [h2o, h2o], format="eon")
        (tmp_path / "climb.dat").write_text(
            "iteration\tstep_size\tdelta_e\tconvergence\teigenvalue\ttorque\tangle\trotations\n"
            "1\t0.1\t0.01\t0.05\t-0.1\t0.05\t10\t3\n"
        )
        traj = load_dimer_trajectory(tmp_path)
        assert traj.initial_atoms is not None  # fell back to first frame
        assert traj.saddle_atoms is None
        assert traj.mode_vector is None

    def test_missing_dat_uses_frame_metadata(self, tmp_path, monkeypatch):
        from chemparseplot.parse.eon.dimer_trajectory import load_dimer_trajectory

        (tmp_path / "climb").write_text("dummy movie")

        frames = [
            DummyDimerFrame(iteration=0),
            DummyDimerFrame(
                iteration=1,
                step_size=0.1,
                delta_e=0.01,
                convergence=0.05,
                eigenvalue=-0.12,
                torque=0.05,
                angle=12.3,
                rotations=5,
            ),
            DummyDimerFrame(
                iteration=2,
                step_size=0.08,
                delta_e=0.02,
                convergence=0.03,
                eigenvalue=-0.2,
                torque=0.03,
                angle=8.1,
                rotations=3,
            ),
        ]
        monkeypatch.setattr(
            "chemparseplot.parse.eon._trajectory_common.readcon.read_con",
            lambda _: frames,
        )

        traj = load_dimer_trajectory(tmp_path)
        assert len(traj.atoms_list) == 3
        assert traj.dat_df["iteration"].to_list() == [1, 2]
        assert traj.dat_df["rotations"].to_list() == [5, 3]
