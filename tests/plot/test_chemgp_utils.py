# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT

"""Tests for ChemGP plotting utility functions."""

from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
pytestmark = pytest.mark.pure


class TestPlottingFunctions:
    """Test plotting utility functions."""

    def test_detect_clamp_mb(self) -> None:
        """Test clamp detection for Muller-Brown."""
        from chemparseplot.plot.chemgp import detect_clamp

        clamp_lo, clamp_hi, step = detect_clamp("mb_surface.h5")

        assert clamp_lo == -200.0
        assert clamp_hi == 50.0
        assert step == 25.0

    def test_detect_clamp_leps(self) -> None:
        """Test clamp detection for LEPS."""
        from chemparseplot.plot.chemgp import detect_clamp

        clamp_lo, clamp_hi, step = detect_clamp("leps_potential.h5")

        assert clamp_lo == -5.0
        assert clamp_hi == 5.0
        assert step == 0.5

    def test_detect_clamp_unknown(self) -> None:
        """Test clamp detection for unknown filename."""
        from chemparseplot.plot.chemgp import detect_clamp

        clamp_lo, clamp_hi, step = detect_clamp("unknown.h5")

        assert clamp_lo is None
        assert clamp_hi is None
        assert step is None

    def test_save_plot_matplotlib(self, tmp_path: Path) -> None:
        """Test saving matplotlib figure."""
        plt = pytest.importorskip("matplotlib.pyplot")
        from chemparseplot.plot.chemgp import save_plot

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])

        output_path = tmp_path / "test.pdf"
        save_plot(fig, output_path, dpi=300)

        assert output_path.exists()
        assert output_path.stat().st_size > 0
        plt.close(fig)
