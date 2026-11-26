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
    _crop_transparent_rgba,
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


class TestCropContent:
    def test_crops_opaque_matte_background(self):
        img = np.zeros((100, 120, 4), dtype=np.float32)
        img[..., 3] = 1.0
        img[30:70, 40:80, :3] = (1.0, 0.2, 0.2)

        cropped = _crop_transparent_rgba(img)

        assert cropped.shape[0] < img.shape[0]
        assert cropped.shape[1] < img.shape[1]
        assert np.any(cropped[..., 0] > 0.5)


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
        assert len(ax.images) == 1
        plt.close(fig)

    def test_col_spacing(self, small_molecules):
        fig, ax = plt.subplots(figsize=(8, 2))
        plot_structure_strip(ax, small_molecules, ["A", "B", "C"], col_spacing=2.5)
        assert len(ax.images) == 1
        plt.close(fig)

    def test_default_spacing(self, small_molecules):
        fig, ax = plt.subplots(figsize=(6, 2))
        plot_structure_strip(ax, small_molecules, ["A", "B", "C"])
        assert len(ax.images) == 1
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
        assert len(ax.images) == 1
        plt.close(fig)

    def test_typed_structure_entries(self, small_molecules):
        fig, ax = plt.subplots(figsize=(6, 2))
        entries = [
            StructurePlacement(atoms=atoms, x=float(i), label=label)
            for i, (atoms, label) in enumerate(
                zip(small_molecules, ["A", "B", "C"], strict=False)
            )
        ]
        plot_structure_strip(ax, entries)
        assert not ax.axison
        plt.close(fig)

    def test_max_cols_wrapping(self):
        mols = [molecule("H2O") for _ in range(8)]
        fig, ax = plt.subplots(figsize=(8, 4))
        plot_structure_strip(ax, mols, [str(i) for i in range(8)], max_cols=4)
        assert len(ax.images) == 1
        plt.close(fig)

    def test_custom_label_fontsize(self, small_molecules):
        fig, ax = plt.subplots(figsize=(6, 2))
        plot_structure_strip(
            ax,
            small_molecules,
            ["A", "B", "C"],
            label_fontsize=18,
            label_padding_px=12,
        )
        assert all(text.get_fontsize() == 18 for text in ax.texts)
        plt.close(fig)

    @pytest.mark.parametrize(
        ("keyword", "value"),
        [("label_fontsize", 0), ("label_padding_px", -1)],
    )
    def test_invalid_label_layout_value(self, small_molecules, keyword, value):
        fig, ax = plt.subplots(figsize=(6, 2))
        with pytest.raises(ValueError):
            plot_structure_strip(
                ax,
                small_molecules,
                ["A", "B", "C"],
                **{keyword: value},
            )
        plt.close(fig)

    def test_composes_single_canvas_not_annotation_boxes(
        self, monkeypatch, small_molecules
    ):
        fig, ax = plt.subplots(figsize=(6, 2))

        def _fake_render(*_args, **_kwargs):
            img = np.zeros((60, 80, 4), dtype=np.float32)
            img[10:50, 20:60, :3] = 1.0
            img[10:50, 20:60, 3] = 1.0
            return img

        monkeypatch.setattr("chemparseplot.plot.neb._render_atoms", _fake_render)
        plot_structure_strip(ax, small_molecules, ["A", "B", "C"])
        assert len(ax.images) == 1
        assert len(ax.artists) == 0
        plt.close(fig)

    def test_separate_label_axis_keeps_text_out_of_image_band(
        self, monkeypatch, small_molecules
    ):
        fig = plt.figure(figsize=(6, 2.4))
        grid = fig.add_gridspec(2, 1, height_ratios=(4, 1), hspace=0.08)
        image_ax = fig.add_subplot(grid[0])
        label_ax = fig.add_subplot(grid[1])

        def _fake_render(*_args, **_kwargs):
            img = np.zeros((60, 80, 4), dtype=np.float32)
            img[10:50, 20:60, :3] = 1.0
            img[10:50, 20:60, 3] = 1.0
            return img

        labels = ["Cage\n0.00 eV", "Prism\n0.10 eV", "Book\n0.20 eV"]
        monkeypatch.setattr("chemparseplot.plot.neb._render_atoms", _fake_render)
        plot_structure_strip(
            image_ax,
            small_molecules,
            labels,
            label_ax=label_ax,
        )

        fig.canvas.draw()
        mpl_renderer = fig.canvas.get_renderer()
        image_bbox = image_ax.get_window_extent(mpl_renderer)
        label_bbox = label_ax.get_window_extent(mpl_renderer)

        assert len(image_ax.texts) == 0
        assert [text.get_text() for text in label_ax.texts] == labels
        for text in label_ax.texts:
            text_bbox = text.get_window_extent(mpl_renderer)
            assert label_bbox.contains(text_bbox.x0, text_bbox.y0)
            assert label_bbox.contains(text_bbox.x1, text_bbox.y1)
            assert text_bbox.y1 <= image_bbox.y0
        plt.close(fig)

    def test_row_geometry_keeps_pitch_and_bottom_inset(self, monkeypatch):
        """Row pitch is the axes height minus the 6 px pads.

        Captions sit ``label_padding_px`` above the slot bottom. The label
        band reserves space inside each row and is not removed before the
        rows are split.
        """
        mols = [molecule("H2O") for _ in range(4)]
        labels = ["A", "B\n2", "C", "D"]
        label_fontsize = 12.0
        label_padding_px = 7.0
        src_px = 80
        fig, ax = plt.subplots(figsize=(8, 4), dpi=100)

        def _fake_render(*_args, **_kwargs):
            return np.ones((src_px, src_px, 4), dtype=np.float32)

        def _keep(img, *_args, **_kwargs):
            return img

        def _exact_size(_img, target_h, target_w):
            return np.ones((target_h, target_w, 4), dtype=np.float32)

        blits = []

        def _record(_canvas, img_data, x0, y0):
            blits.append((x0, y0, img_data.shape[1], img_data.shape[0]))

        monkeypatch.setattr("chemparseplot.plot.neb._render_atoms", _fake_render)
        monkeypatch.setattr("chemparseplot.plot.neb._crop_transparent_rgba", _keep)
        monkeypatch.setattr("chemparseplot.plot.neb._resize_rgba_image", _exact_size)
        monkeypatch.setattr("chemparseplot.plot.neb._alpha_blit_rgba", _record)
        plot_structure_strip(
            ax,
            mols,
            labels,
            max_cols=2,
            prefer_single_row=False,
            label_fontsize=label_fontsize,
            label_padding_px=label_padding_px,
        )

        _x0, ax_w = ax.get_xlim()
        _y0, ax_h = ax.get_ylim()
        n_plot = len(mols)
        max_cols = 2
        effective_max_cols = max(1, int(max_cols))
        n_cols = min(n_plot, effective_max_cols)
        n_rows = (n_plot + effective_max_cols - 1) // effective_max_cols
        max_label_lines = 2
        label_band_px = label_fontsize * fig.dpi / 72 * (0.35 + 1.15 * max_label_lines)
        top_padding_px = 6.0
        bottom_padding_px = 6.0
        row_gap_px = 10.0
        row_pitch_px = max(
            24.0,
            (ax_h - top_padding_px - bottom_padding_px) / max(n_rows, 1),
        )
        slot_width_px = ax_w / max(n_cols, 1)
        col_gap_px = max(6.0, (1.5 - 1.0) * slot_width_px * 0.22)
        target_width_px = max(20, round((slot_width_px - col_gap_px) * 0.82))
        available_height_px = max(20.0, row_pitch_px - row_gap_px - label_band_px)
        target_height_px = max(20, round(available_height_px))
        scale = min(target_width_px / src_px, target_height_px / src_px)
        scaled_h = max(1, round(src_px * scale))

        assert n_rows == 2
        assert row_pitch_px == pytest.approx((ax_h - 12.0) / n_rows)
        assert row_pitch_px > label_band_px
        assert len(ax.texts) == n_plot
        assert len(blits) == n_plot
        for i, text in enumerate(ax.texts):
            col = i % effective_max_cols
            row = i // effective_max_cols
            slot_center_x = (col + 0.5) * slot_width_px
            slot_top_y = ax_h - top_padding_px - row * row_pitch_px
            slot_bottom_y = slot_top_y - row_pitch_px
            img_top_y = (
                slot_top_y
                - (available_height_px - target_height_px) / 2
                - target_height_px
            )
            image_y = round(img_top_y) + (target_height_px - scaled_h) // 2
            label_x, label_y = text.get_position()
            _blit_x, blit_y, _blit_w, blit_h = blits[i]
            assert label_x == pytest.approx(slot_center_x)
            assert label_y == pytest.approx(max(2.0, slot_bottom_y + label_padding_px))
            assert blit_y == image_y
            assert blit_h == scaled_h
        assert ax.texts[0].get_position()[1] - ax.texts[2].get_position()[1] == (
            pytest.approx(row_pitch_px)
        )
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
