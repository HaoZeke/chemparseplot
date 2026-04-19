# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""Tests for structure strip rendering, spacing, and dividers."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from ase.build import molecule

from chemparseplot.plot.neb import (
    _render_atoms,
    plot_structure_inset,
    plot_structure_strip,
    render_structure_to_image,
)
from chemparseplot.plot.structs import StructurePlacement


@pytest.fixture
def h2o():
    return molecule("H2O")


@pytest.fixture
def small_molecules():
    return [molecule(m) for m in ["H2O", "CH4", "NH3"]]


class TestRenderStructureToImage:
    def test_returns_rgba_array(self, h2o):
        img = render_structure_to_image(h2o, zoom=0.3, rotation="0x,90y,0z")
        assert img.ndim == 3
        assert img.shape[2] in (3, 4)  # RGB or RGBA
        assert img.dtype in (np.float32, np.float64)

    def test_different_rotations(self, h2o):
        img1 = render_structure_to_image(h2o, 0.3, "0x,0y,0z")
        img2 = render_structure_to_image(h2o, 0.3, "90x,0y,0z")
        # Different rotations produce valid images (may differ in size)
        assert img1.ndim == 3
        assert img2.ndim == 3


class TestRenderAtoms:
    def test_ase_backend(self, h2o):
        img = _render_atoms(h2o, "ase", 0.3, "0x,90y,0z")
        assert img.ndim == 3

    def test_unknown_backend_falls_to_ase(self, h2o):
        # "ase" is the else branch -- any unknown value goes there
        img = _render_atoms(h2o, "unknown_renderer", 0.3, "0x,90y,0z")
        assert img.ndim == 3

    def test_xyzrender_not_installed(self, h2o):
        """xyzrender raises RuntimeError if binary not on PATH."""
        import shutil

        if shutil.which("xyzrender") is not None:
            pytest.skip("xyzrender is installed")
        with pytest.raises(RuntimeError, match="xyzrender"):
            _render_atoms(h2o, "xyzrender", 0.3, "0x,90y,0z")

    def test_solvis_not_installed(self, h2o):
        """solvis raises RuntimeError if not importable."""
        try:
            import solvis

            pytest.skip("solvis is installed")
        except ImportError:
            pass
        with pytest.raises(RuntimeError, match="solvis"):
            _render_atoms(h2o, "solvis", 0.3, "0x,90y,0z")

    def test_ovito_not_installed(self, h2o):
        """ovito raises RuntimeError if not importable."""
        try:
            import ovito

            pytest.skip("ovito is installed")
        except ImportError:
            pass
        with pytest.raises(RuntimeError, match="ovito"):
            _render_atoms(h2o, "ovito", 0.3, "0x,90y,0z")


class TestPlotStructureStrip:
    def test_basic_strip(self, small_molecules):
        fig, ax = plt.subplots(figsize=(6, 2))
        plot_structure_strip(ax, small_molecules, ["A", "B", "C"])
        # Axes should be turned off
        assert not ax.axison
        plt.close(fig)

    def test_col_spacing(self, small_molecules):
        fig, ax = plt.subplots(figsize=(8, 2))
        plot_structure_strip(ax, small_molecules, ["A", "B", "C"], col_spacing=2.5)
        xlim = ax.get_xlim()
        # With 3 items, spacing 2.5: x goes from -0.5 to 2*2.5+0.5=5.5
        assert xlim[1] == pytest.approx(5.5)
        plt.close(fig)

    def test_default_spacing(self, small_molecules):
        fig, ax = plt.subplots(figsize=(6, 2))
        plot_structure_strip(ax, small_molecules, ["A", "B", "C"])
        xlim = ax.get_xlim()
        # Default col_spacing=1.5: x goes from -0.5 to 2*1.5+0.5=3.5
        assert xlim[1] == pytest.approx(3.5)
        plt.close(fig)

    def test_dividers(self, small_molecules):
        fig, ax = plt.subplots(figsize=(8, 2))
        plot_structure_strip(
            ax,
            small_molecules,
            ["A", "B", "C"],
            show_dividers=True,
            divider_color="red",
            divider_style="-",
        )
        # Should have 2 divider lines (between 3 items)
        [
            ln
            for ln in ax.lines
            if len(ln.get_xdata()) == 2 and ln.get_xdata()[0] == ln.get_xdata()[1]
        ]
        # axvline creates Line2D objects
        assert len(ax.lines) >= 2
        plt.close(fig)

    def test_no_dividers_by_default(self, small_molecules):
        fig, ax = plt.subplots(figsize=(6, 2))
        plot_structure_strip(ax, small_molecules, ["A", "B", "C"])
        # No divider lines
        assert len(ax.lines) == 0
        plt.close(fig)

    def test_empty_labels(self, small_molecules):
        fig, ax = plt.subplots(figsize=(6, 2))
        plot_structure_strip(ax, small_molecules, [])
        plt.close(fig)

    def test_single_structure(self, h2o):
        fig, ax = plt.subplots(figsize=(3, 2))
        plot_structure_strip(ax, [h2o], ["Mol"])
        plt.close(fig)

    def test_typed_structure_entries(self, small_molecules):
        fig, ax = plt.subplots(figsize=(6, 2))
        entries = [
            StructurePlacement(atoms=atoms, x=float(i), label=label)
            for i, (atoms, label) in enumerate(zip(small_molecules, ["A", "B", "C"]))
        ]
        plot_structure_strip(ax, entries)
        assert not ax.axison
        plt.close(fig)

    def test_max_cols_wrapping(self):
        mols = [molecule("H2O") for _ in range(8)]
        fig, ax = plt.subplots(figsize=(8, 4))
        plot_structure_strip(ax, mols, [str(i) for i in range(8)], max_cols=4)
        plt.close(fig)


class TestPlotStructureInset:
    def test_basic_inset(self, h2o):
        fig, ax = plt.subplots()
        ax.plot([0, 1, 2], [0, 1, 0])
        plot_structure_inset(ax, h2o, x=1, y=1, xybox=(40, 40), rad=0.3)
        assert len(ax.artists) > 0
        plt.close(fig)

    def test_custom_arrow(self, h2o):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        plot_structure_inset(
            ax,
            h2o,
            x=0.5,
            y=0.5,
            xybox=(30, 30),
            rad=0.2,
            arrow_props={"color": "red", "alpha": 0.5},
        )
        plt.close(fig)
