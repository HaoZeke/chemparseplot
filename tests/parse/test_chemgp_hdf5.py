# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT

"""Tests for ChemGP HDF5 I/O functions."""

from pathlib import Path

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
h5py = pytest.importorskip("h5py")

pytestmark = pytest.mark.pure


class TestHDF5IO:
    """Test HDF5 I/O functions."""

    @pytest.fixture
    def sample_h5_file(self, tmp_path: Path) -> Path:
        """Create a sample HDF5 file for testing."""
        h5_path = tmp_path / "test.h5"

        with h5py.File(h5_path, "w") as f:
            table_grp = f.create_group("table")
            table_grp.create_dataset("step", data=np.array([1, 2, 3, 4, 5]))
            table_grp.create_dataset("energy", data=np.array([0.1, 0.2, 0.3, 0.4, 0.5]))
            table_grp.create_dataset("force", data=np.array([0.5, 0.4, 0.3, 0.2, 0.1]))

            grid_grp = f.create_group("grids")
            energy_ds = grid_grp.create_dataset("energy", data=np.random.rand(10, 10))
            energy_ds.attrs["x_range"] = [0.0, 1.0]
            energy_ds.attrs["y_range"] = [0.0, 1.0]
            energy_ds.attrs["x_length"] = 10
            energy_ds.attrs["y_length"] = 10

            path_grp = f.create_group("paths")
            path1_grp = path_grp.create_group("path1")
            path1_grp.create_dataset("path1_x", data=np.array([0.1, 0.2, 0.3]))
            path1_grp.create_dataset("path1_y", data=np.array([0.4, 0.5, 0.6]))

            points_grp = f.create_group("points")
            train_grp = points_grp.create_group("train")
            train_grp.create_dataset("train_x", data=np.array([0.2, 0.4, 0.6]))
            train_grp.create_dataset("train_y", data=np.array([0.3, 0.5, 0.7]))

            f.attrs["conv_tol"] = 0.01
            f.attrs["n_steps"] = 100

        return h5_path

    def test_read_h5_table(self, sample_h5_file: Path) -> None:
        """Test reading table from HDF5."""
        from chemparseplot.parse.chemgp_hdf5 import read_h5_table

        with h5py.File(sample_h5_file, "r") as f:
            df = read_h5_table(f, "table")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        assert "step" in df.columns
        assert "energy" in df.columns
        assert "force" in df.columns
        assert df["step"].iloc[0] == 1
        assert df["energy"].iloc[-1] == 0.5

    def test_read_h5_grid_with_coords(self, sample_h5_file: Path) -> None:
        """Test reading grid with axis coordinates."""
        from chemparseplot.parse.chemgp_hdf5 import read_h5_grid

        with h5py.File(sample_h5_file, "r") as f:
            data, x_coords, y_coords = read_h5_grid(f, "energy")

        assert data.shape == (10, 10)
        assert x_coords is not None
        assert y_coords is not None
        assert len(x_coords) == 10
        assert len(y_coords) == 10
        assert x_coords[0] == 0.0
        assert x_coords[-1] == 1.0

    def test_read_h5_path(self, sample_h5_file: Path) -> None:
        """Test reading path from HDF5."""
        from collections.abc import Mapping

        from chemparseplot.parse.chemgp_hdf5 import ArrayGroup, read_h5_path

        with h5py.File(sample_h5_file, "r") as f:
            path_data = read_h5_path(f, "path1")

        assert isinstance(path_data, Mapping)
        assert isinstance(path_data, ArrayGroup)
        assert "path1_x" in path_data
        assert "path1_y" in path_data
        assert len(path_data["path1_x"]) == 3

    def test_read_h5_points(self, sample_h5_file: Path) -> None:
        """Test reading points from HDF5."""
        from collections.abc import Mapping

        from chemparseplot.parse.chemgp_hdf5 import ArrayGroup, read_h5_points

        with h5py.File(sample_h5_file, "r") as f:
            points_data = read_h5_points(f, "train")

        assert isinstance(points_data, Mapping)
        assert isinstance(points_data, ArrayGroup)
        assert "train_x" in points_data
        assert "train_y" in points_data
        assert len(points_data["train_x"]) == 3

    def test_read_h5_metadata(self, sample_h5_file: Path) -> None:
        """Test reading metadata from HDF5."""
        from collections.abc import Mapping

        from chemparseplot.parse.chemgp_hdf5 import MetadataAttrs, read_h5_metadata

        with h5py.File(sample_h5_file, "r") as f:
            metadata = read_h5_metadata(f)

        assert isinstance(metadata, Mapping)
        assert isinstance(metadata, MetadataAttrs)
        assert "conv_tol" in metadata
        assert "n_steps" in metadata
        assert metadata["conv_tol"] == 0.01
        assert metadata["n_steps"] == 100
