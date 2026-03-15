# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT

"""Tests for PLUMED FES calculation and minima finding."""

import numpy as np
import pytest

pytestmark = pytest.mark.pure


class TestCalculateFES1D:
    """Test 1D FES calculation from synthetic HILLS data."""

    def test_single_hill_1d(self) -> None:
        """Test FES from a single 1D Gaussian hill."""
        from chemparseplot.parse.plumed import calculate_fes_from_hills

        # 1D HILLS: time, cv1, sigma1, height, bias_factor
        hills_data = np.array([[0.0, 0.0, 0.1, 1.0, 0.0]])
        hills = {
            "hillsfile": hills_data,
            "per": [False],
            "pcv1": None,
            "pcv2": None,
        }

        result = calculate_fes_from_hills(hills, npoints=64)

        assert result["dimension"] == 1
        assert result["fes"].shape == (64,)
        assert result["rows"] == 64
        # FES should be most negative near cv1=0 (center of the hill)
        center_idx = np.argmin(result["fes"])
        assert abs(result["x"][center_idx]) < 0.1

    def test_multiple_hills_1d(self) -> None:
        """Test FES from multiple 1D Gaussian hills."""
        from chemparseplot.parse.plumed import calculate_fes_from_hills

        hills_data = np.array([
            [0.0, -1.0, 0.2, 2.0, 0.0],
            [1.0, 1.0, 0.2, 2.0, 0.0],
        ])
        hills = {
            "hillsfile": hills_data,
            "per": [False],
            "pcv1": None,
            "pcv2": None,
        }

        result = calculate_fes_from_hills(hills, npoints=128)

        assert result["dimension"] == 1
        assert len(result["x"]) == 128
        # Should have two wells
        fes = result["fes"]
        assert np.min(fes) < 0


class TestCalculateFES2D:
    """Test 2D FES calculation from synthetic HILLS data."""

    def test_single_hill_2d(self) -> None:
        """Test FES from a single 2D Gaussian hill."""
        from chemparseplot.parse.plumed import calculate_fes_from_hills

        # 2D HILLS: time, cv1, cv2, sigma1, sigma2, height, bias_factor
        hills_data = np.array([[0.0, 0.0, 0.0, 0.1, 0.1, 1.0, 0.0]])
        hills = {
            "hillsfile": hills_data,
            "per": [False, False],
            "pcv1": None,
            "pcv2": None,
        }

        result = calculate_fes_from_hills(hills, npoints=32)

        assert result["dimension"] == 2
        assert result["fes"].shape == (32, 32)
        assert len(result["x"]) == 32
        assert len(result["y"]) == 32


class TestFindFESMinima:
    """Test FES minima finding."""

    def test_find_minima_1d(self) -> None:
        """Test finding minima on a 1D FES with a clear well."""
        pytest.importorskip("pandas")
        from chemparseplot.parse.plumed import (
            calculate_fes_from_hills,
            find_fes_minima,
        )

        # Create a 1D FES with a single deep well at cv1=0
        hills_data = np.array([[0.0, 0.0, 0.3, 5.0, 0.0]])
        hills = {
            "hillsfile": hills_data,
            "per": [False],
            "pcv1": None,
            "pcv2": None,
        }

        result = calculate_fes_from_hills(hills, npoints=64)
        result["fes"] -= np.min(result["fes"])

        minima_result = find_fes_minima(result, nbins=8)

        assert minima_result is not None
        assert "minima" in minima_result
        assert len(minima_result["minima"]) >= 1
        assert "letter" in minima_result["minima"].columns
        assert "CV1" in minima_result["minima"].columns
        assert "free_energy" in minima_result["minima"].columns

    def test_find_minima_2d(self) -> None:
        """Test finding minima on a 2D FES."""
        pytest.importorskip("pandas")
        from chemparseplot.parse.plumed import (
            calculate_fes_from_hills,
            find_fes_minima,
        )

        # Create a 2D FES with a single deep well
        hills_data = np.array([[0.0, 0.0, 0.0, 0.3, 0.3, 5.0, 0.0]])
        hills = {
            "hillsfile": hills_data,
            "per": [False, False],
            "pcv1": None,
            "pcv2": None,
        }

        result = calculate_fes_from_hills(hills, npoints=32)
        result["fes"] -= np.min(result["fes"])

        minima_result = find_fes_minima(result, nbins=8)

        assert minima_result is not None
        assert "minima" in minima_result
        assert len(minima_result["minima"]) >= 1
        assert "CV1" in minima_result["minima"].columns
        assert "CV2" in minima_result["minima"].columns

    def test_nbins_validation(self) -> None:
        """Test that invalid nbins raises ValueError."""
        pytest.importorskip("pandas")
        from chemparseplot.parse.plumed import (
            calculate_fes_from_hills,
            find_fes_minima,
        )

        hills_data = np.array([[0.0, 0.0, 0.3, 5.0, 0.0]])
        hills = {
            "hillsfile": hills_data,
            "per": [False],
            "pcv1": None,
            "pcv2": None,
        }

        result = calculate_fes_from_hills(hills, npoints=64)

        with pytest.raises(ValueError, match="integer multiple"):
            find_fes_minima(result, nbins=7)
