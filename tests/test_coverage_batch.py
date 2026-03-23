# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""Batch coverage tests for under-covered modules."""

import matplotlib

matplotlib.use("Agg")

import importlib

import matplotlib.pyplot as plt
import numpy as np
import pytest


# ============================================================
# chemparseplot.parse.file_
# ============================================================
class TestFindFilePaths:
    def test_glob_matching(self, tmp_path):
        from chemparseplot.parse.file_ import find_file_paths

        for i in range(3):
            (tmp_path / f"neb_{i:03d}.dat").write_text("data")
        (tmp_path / "other.txt").write_text("other")
        result = find_file_paths(str(tmp_path / "neb_*.dat"))
        assert len(result) == 3

    def test_no_matches(self, tmp_path):
        from chemparseplot.parse.file_ import find_file_paths

        result = find_file_paths(str(tmp_path / "*.nonexistent"))
        assert result == []

    def test_sorted_order(self, tmp_path):
        from chemparseplot.parse.file_ import find_file_paths

        for name in ["c.dat", "a.dat", "b.dat"]:
            (tmp_path / name).write_text("")
        result = find_file_paths(str(tmp_path / "*.dat"))
        names = [p.name for p in result]
        assert names == ["a.dat", "b.dat", "c.dat"]


# ============================================================
# chemparseplot.util (parse_target_coords)
# ============================================================
class TestParseTargetCoords:
    def test_valid_coords(self):
        from chemparseplot.util import parse_target_coords

        text = "1.0 2.0 3.0\n4.0 5.0 6.0\n"
        coords = parse_target_coords(text)
        assert coords.shape == (2, 3)
        np.testing.assert_allclose(coords[0], [1.0, 2.0, 3.0])

    def test_empty_text(self):
        from chemparseplot.util import parse_target_coords

        coords = parse_target_coords("")
        assert coords.size == 0

    def test_invalid_lines_skipped(self):
        from chemparseplot.util import parse_target_coords

        text = "1.0 2.0 3.0\nbad line\n4.0 5.0 6.0\n"
        coords = parse_target_coords(text)
        assert coords.shape == (2, 3)

    def test_wrong_column_count(self):
        from chemparseplot.util import parse_target_coords

        text = "1.0 2.0\n3.0 4.0 5.0\n"
        coords = parse_target_coords(text)
        assert coords.shape == (1, 3)

    def test_non_numeric(self):
        from chemparseplot.util import parse_target_coords

        text = "abc def ghi\n1.0 2.0 3.0\n"
        coords = parse_target_coords(text)
        assert coords.shape == (1, 3)


# ============================================================
# chemparseplot.plot.theme
# ============================================================
class TestPlotTheme:
    def test_get_ruhi_theme(self):
        from chemparseplot.plot.theme import get_theme

        t = get_theme("ruhi")
        assert t.name == "ruhi"
        assert t.font_family == "Atkinson Hyperlegible"

    def test_get_batlow_theme(self):
        from chemparseplot.plot.theme import get_theme

        t = get_theme("cmc.batlow")
        assert t.name == "cmc.batlow"

    def test_get_unknown_falls_to_ruhi(self):
        from chemparseplot.plot.theme import get_theme

        t = get_theme("nonexistent_theme")
        assert t.name == "ruhi"

    def test_overrides(self):
        from chemparseplot.plot.theme import get_theme

        t = get_theme("ruhi", font_size=20, facecolor="black")
        assert t.font_size == 20
        assert t.facecolor == "black"

    def test_none_overrides_ignored(self):
        from chemparseplot.plot.theme import get_theme

        t = get_theme("ruhi", font_size=None)
        assert t.font_size == 12  # Default

    def test_setup_global_theme(self):
        from chemparseplot.plot.theme import get_theme, setup_global_theme

        t = get_theme("ruhi")
        setup_global_theme(t)
        assert plt.rcParams["text.color"] == t.textcolor

    def test_apply_axis_theme(self):
        from chemparseplot.plot.theme import apply_axis_theme, get_theme

        fig, ax = plt.subplots()
        t = get_theme("ruhi")
        apply_axis_theme(ax, t)
        assert ax.get_facecolor() == matplotlib.colors.to_rgba(t.facecolor)
        plt.close(fig)

    def test_setup_publication_theme(self):
        from chemparseplot.plot.theme import get_theme, setup_publication_theme

        t = get_theme("ruhi")
        setup_publication_theme(t)
        assert plt.rcParams["axes.spines.top"] is False
        assert plt.rcParams["axes.spines.right"] is False

    def test_build_cmap(self):
        from chemparseplot.plot.theme import build_cmap

        cmap = build_cmap(["#FF0000", "#00FF00", "#0000FF"], "test_cmap_123")
        assert cmap is not None
        assert cmap.name == "test_cmap_123"

    def test_build_cmap_reregister(self):
        from chemparseplot.plot.theme import build_cmap

        # Register twice -- should not raise
        build_cmap(["#FF0000", "#0000FF"], "test_cmap_dup")
        build_cmap(["#FF0000", "#0000FF"], "test_cmap_dup")

    def test_ruhi_colors(self):
        from chemparseplot.plot.theme import RUHI_COLORS

        assert "coral" in RUHI_COLORS
        assert "teal" in RUHI_COLORS


# ============================================================
# chemparseplot.plot.__init__ (lazy imports)
# ============================================================
class TestPlotLazyImports:
    def test_getattr_unknown_raises(self):
        from chemparseplot import plot

        with pytest.raises(AttributeError, match="no attribute"):
            _ = plot.nonexistent_submodule

    def test_exported_names(self):
        from chemparseplot.plot import (
            RUHI_COLORS,
            RUHI_THEME,
            PlotTheme,
            get_theme,
            setup_global_theme,
            setup_publication_theme,
        )

        assert PlotTheme is not None
        assert callable(get_theme)


# ============================================================
# chemparseplot.parse.neb_utils (calculate_landscape_coords)
# ============================================================
class TestCalculateLandscapeCoords:
    def test_with_mock_ira(self):
        """Test with a mock IRA that returns identity RMSD."""
        from unittest.mock import MagicMock, patch

        from ase.build import molecule

        atoms_list = [molecule("H2O") for _ in range(5)]
        for i, a in enumerate(atoms_list):
            a.positions[0, 0] += 0.1 * i

        mock_ira = MagicMock()

        def fake_rmsd(atoms, ira, ref_atom, ira_kmax):
            return np.array(
                [
                    float(np.sqrt(((a.positions - ref_atom.positions) ** 2).mean()))
                    for a in atoms
                ]
            )

        with patch(
            "rgpycrumbs.geom.api.alignment.calculate_rmsd_from_ref",
            side_effect=fake_rmsd,
        ):
            from chemparseplot.parse.neb_utils import calculate_landscape_coords

            rmsd_a, rmsd_b = calculate_landscape_coords(
                atoms_list, mock_ira, ira_kmax=1.8
            )
            assert len(rmsd_a) == 5
            assert len(rmsd_b) == 5
            assert rmsd_a[0] == pytest.approx(0.0, abs=1e-10)

    def test_explicit_refs(self):
        from unittest.mock import patch

        from ase.build import molecule

        atoms_list = [molecule("H2O") for _ in range(3)]
        ref_a = molecule("H2O")
        ref_b = molecule("NH3")

        captured_refs = []

        def fake_rmsd(atoms, ira, ref_atom, ira_kmax):
            captured_refs.append(ref_atom)
            return np.zeros(len(atoms))

        with patch(
            "rgpycrumbs.geom.api.alignment.calculate_rmsd_from_ref",
            side_effect=fake_rmsd,
        ):
            from chemparseplot.parse.neb_utils import calculate_landscape_coords

            calculate_landscape_coords(atoms_list, None, 1.8, ref_a=ref_a, ref_b=ref_b)
            assert len(captured_refs) == 2
            assert captured_refs[0] is ref_a
            assert captured_refs[1] is ref_b

    def test_defaults_to_first_last(self):
        from unittest.mock import patch

        from ase.build import molecule

        atoms_list = [molecule("H2O"), molecule("NH3"), molecule("CH4")]
        captured_refs = []

        def fake_rmsd(atoms, ira, ref_atom, ira_kmax):
            captured_refs.append(ref_atom)
            return np.zeros(len(atoms))

        with patch(
            "rgpycrumbs.geom.api.alignment.calculate_rmsd_from_ref",
            side_effect=fake_rmsd,
        ):
            from chemparseplot.parse.neb_utils import calculate_landscape_coords

            calculate_landscape_coords(atoms_list, None, 1.8)
            assert captured_refs[0] is atoms_list[0]
            assert captured_refs[1] is atoms_list[-1]


# ============================================================
# chemparseplot.plot.neb (rendering and plotting functions)
# ============================================================
class TestNebPlotFunctions:
    def test_plot_energy_path_hermite(self):
        from chemparseplot.plot.neb import plot_energy_path

        fig, ax = plt.subplots()
        rc = np.linspace(0, 3, 11)
        energy = np.sin(rc)
        f_para = np.cos(rc)
        plot_energy_path(ax, rc, energy, f_para, "blue", 1.0, 10, method="hermite")
        assert len(ax.lines) >= 1
        plt.close(fig)

    def test_plot_energy_path_spline(self):
        from chemparseplot.plot.neb import plot_energy_path

        fig, ax = plt.subplots()
        rc = np.linspace(0, 3, 11)
        energy = np.sin(rc)
        f_para = np.cos(rc)
        plot_energy_path(ax, rc, energy, f_para, "red", 0.8, 5, method="spline")
        plt.close(fig)

    def test_plot_eigenvalue_path(self):
        from chemparseplot.plot.neb import plot_eigenvalue_path

        fig, ax = plt.subplots()
        rc = np.linspace(0, 3, 11)
        eigenval = np.linspace(-0.5, 0.1, 11)
        plot_eigenvalue_path(ax, rc, eigenval, "green", 1.0, 10)
        assert len(ax.lines) >= 2  # data + zero line
        plt.close(fig)

    def test_augment_minima_points(self):
        from chemparseplot.plot.neb import _augment_minima_points

        r = np.array([0.0, 1.0, 2.0])
        p = np.array([2.0, 1.0, 0.0])
        z = np.array([-1.0, 0.0, -0.8])
        ar, ap, az = _augment_minima_points(r, p, z)
        # Should have original + collar points for 2 endpoints
        assert len(ar) > len(r)

    def test_augment_with_gradients(self):
        from chemparseplot.plot.neb import _augment_with_gradients

        r = np.array([0.0, 1.0, 2.0])
        p = np.array([2.0, 1.0, 0.0])
        z = np.array([-1.0, 0.0, -0.8])
        gr = np.array([0.1, 0.2, 0.1])
        gp = np.array([-0.1, -0.2, -0.1])
        ar, ap, az = _augment_with_gradients(r, p, z, gr, gp)
        # 5x points (original + 4 helpers per point)
        assert len(ar) == 5 * len(r)

    def test_augment_with_gradients_none(self):
        from chemparseplot.plot.neb import _augment_with_gradients

        r = np.array([1.0, 2.0])
        p = np.array([2.0, 1.0])
        z = np.array([0.0, 1.0])
        ar, ap, az = _augment_with_gradients(r, p, z, None, None)
        np.testing.assert_array_equal(ar, r)

    def test_plot_mmf_peaks_empty(self):
        from chemparseplot.plot.neb import plot_mmf_peaks_overlay

        fig, ax = plt.subplots()
        plot_mmf_peaks_overlay(ax, [], [], [])
        plt.close(fig)

    def test_plot_mmf_peaks_with_path(self):
        from chemparseplot.plot.neb import plot_mmf_peaks_overlay

        fig, ax = plt.subplots()
        path_r = np.linspace(0, 3, 11)
        path_p = np.linspace(3, 0, 11)
        peak_r = np.array([1.5])
        peak_p = np.array([1.5])
        peak_e = np.array([0.5])
        plot_mmf_peaks_overlay(
            ax,
            peak_r,
            peak_p,
            peak_e,
            project_path=True,
            path_rmsd_r=path_r,
            path_rmsd_p=path_p,
        )
        plt.close(fig)

    def test_plot_mmf_peaks_no_projection(self):
        from chemparseplot.plot.neb import plot_mmf_peaks_overlay

        fig, ax = plt.subplots()
        plot_mmf_peaks_overlay(
            ax,
            np.array([1.0]),
            np.array([2.0]),
            np.array([0.1]),
            project_path=False,
        )
        plt.close(fig)

    def test_plot_neb_evolution_empty(self):
        from chemparseplot.plot.neb import plot_neb_evolution

        fig, ax = plt.subplots()
        plot_neb_evolution(ax, [], [])
        plt.close(fig)

    def test_plot_neb_evolution(self):
        from chemparseplot.plot.neb import plot_neb_evolution

        fig, ax = plt.subplots()
        steps_r = [np.linspace(0, 3, 5) + 0.1 * i for i in range(3)]
        steps_p = [np.linspace(3, 0, 5) + 0.1 * i for i in range(3)]
        plot_neb_evolution(ax, steps_r, steps_p, project_path=True)
        assert len(ax.lines) == 3
        plt.close(fig)

    def test_plot_neb_evolution_no_projection(self):
        from chemparseplot.plot.neb import plot_neb_evolution

        fig, ax = plt.subplots()
        steps_r = [np.linspace(0, 3, 5)]
        steps_p = [np.linspace(3, 0, 5)]
        plot_neb_evolution(ax, steps_r, steps_p, project_path=False)
        plt.close(fig)


# ============================================================
# chemparseplot.parse.orca.neb.opi_parser
# ============================================================
class TestOpiParser:
    def test_calculate_rmsd(self):
        from ase.build import molecule

        from chemparseplot.parse.orca.neb.opi_parser import _calculate_rmsd

        h2o = molecule("H2O")
        rmsd = _calculate_rmsd(h2o, h2o)
        assert rmsd == pytest.approx(0.0)

    def test_calculate_rmsd_different(self):
        from ase.build import molecule

        from chemparseplot.parse.orca.neb.opi_parser import _calculate_rmsd

        h2o1 = molecule("H2O")
        h2o2 = molecule("H2O")
        h2o2.positions += 1.0
        rmsd = _calculate_rmsd(h2o1, h2o2)
        assert rmsd > 0

    def test_compute_synthetic_gradients(self):
        from chemparseplot.parse.orca.neb.opi_parser import _compute_synthetic_gradients

        rmsd_r = np.array([0.0, 1.0, 2.0, 3.0])
        rmsd_p = np.array([3.0, 2.0, 1.0, 0.0])
        forces = [np.ones((3, 3)) for _ in range(4)]
        from ase.build import molecule

        atoms_list = [molecule("H2O") for _ in range(4)]
        gr, gp = _compute_synthetic_gradients(rmsd_r, rmsd_p, forces, atoms_list)
        assert len(gr) == 4
        assert len(gp) == 4

    def test_compute_synthetic_gradients_none_forces(self):
        from chemparseplot.parse.orca.neb.opi_parser import _compute_synthetic_gradients

        rmsd_r = np.array([0.0, 1.0])
        rmsd_p = np.array([1.0, 0.0])
        forces = [None, None]
        from ase.build import molecule

        atoms_list = [molecule("H2O"), molecule("H2O")]
        gr, gp = _compute_synthetic_gradients(rmsd_r, rmsd_p, forces, atoms_list)
        np.testing.assert_array_equal(gr, np.zeros(2))

    def test_parse_fallback_missing_file(self, tmp_path):
        from chemparseplot.parse.orca.neb.opi_parser import parse_orca_neb_fallback

        result = parse_orca_neb_fallback("nonexistent", working_dir=tmp_path)
        assert result is None
