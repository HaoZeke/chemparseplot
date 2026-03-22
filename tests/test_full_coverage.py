# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""Full coverage tests for all under-covered modules."""

import importlib.util
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    pd = None
    _HAS_PANDAS = False

try:
    import cmcrameri
    _HAS_CMCRAMERI = True
except ImportError:
    _HAS_CMCRAMERI = False


# ============================================================
# chemparseplot.parse.chemgp_hdf5
# ============================================================
@pytest.mark.skipif(not _HAS_PANDAS, reason="pandas required")
class TestChemGPHdf5:
    """Tests for HDF5 reader functions using mock h5py-like objects."""

    def _make_mock_dataset(self, data, attrs=None):
        ds = MagicMock()
        ds.__getitem__ = MagicMock(return_value=data)
        ds.attrs = attrs or {}
        return ds

    def _make_mock_group(self, datasets):
        g = MagicMock()
        g.keys.return_value = list(datasets.keys())
        g.__getitem__ = lambda self_, k: datasets[k]
        g.__contains__ = lambda self_, k: k in datasets
        return g

    def test_read_h5_table(self):
        from chemparseplot.parse.chemgp_hdf5 import read_h5_table

        col_a = MagicMock()
        col_a.__getitem__ = MagicMock(return_value=np.array([1.0, 2.0, 3.0]))
        col_a.__getitem__.return_value.dtype = np.dtype("f8")
        col_b = MagicMock()
        col_b.__getitem__ = MagicMock(return_value=np.array([4.0, 5.0, 6.0]))
        col_b.__getitem__.return_value.dtype = np.dtype("f8")

        table_group = MagicMock()
        table_group.keys.return_value = ["a", "b"]
        table_group.__getitem__ = lambda s, k: {"a": col_a, "b": col_b}[k]

        f = MagicMock()
        f.__getitem__ = MagicMock(return_value=table_group)

        df = read_h5_table(f, "table")
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["a", "b"]
        assert len(df) == 3

    def test_read_h5_table_string_data(self):
        from chemparseplot.parse.chemgp_hdf5 import read_h5_table

        col_s = MagicMock()
        col_s.__getitem__ = MagicMock(return_value=np.array([b"foo", b"bar"]))
        col_s.__getitem__.return_value.dtype = np.dtype("S3")

        table_group = MagicMock()
        table_group.keys.return_value = ["label"]
        table_group.__getitem__ = lambda s, k: {"label": col_s}[k]

        f = MagicMock()
        f.__getitem__ = MagicMock(return_value=table_group)

        df = read_h5_table(f, "table")
        assert df["label"].tolist() == ["foo", "bar"]

    def test_read_h5_grid_with_ranges(self):
        from chemparseplot.parse.chemgp_hdf5 import read_h5_grid

        data = np.ones((10, 10))
        ds = MagicMock()
        ds.__getitem__ = MagicMock(return_value=data)
        ds.attrs = {
            "x_range": (0.0, 1.0),
            "x_length": 10,
            "y_range": (-1.0, 1.0),
            "y_length": 10,
        }

        f = MagicMock()
        f.__getitem__ = MagicMock(return_value=ds)

        grid_data, x_coords, y_coords = read_h5_grid(f, "test_grid")
        np.testing.assert_array_equal(grid_data, data)
        assert x_coords is not None
        assert y_coords is not None
        assert len(x_coords) == 10
        assert len(y_coords) == 10

    def test_read_h5_grid_no_ranges(self):
        from chemparseplot.parse.chemgp_hdf5 import read_h5_grid

        data = np.ones((5, 5))
        ds = MagicMock()
        ds.__getitem__ = MagicMock(return_value=data)
        ds.attrs = {}

        f = MagicMock()
        f.__getitem__ = MagicMock(return_value=ds)

        grid_data, x_coords, y_coords = read_h5_grid(f, "test_grid")
        np.testing.assert_array_equal(grid_data, data)
        assert x_coords is None
        assert y_coords is None

    def test_read_h5_path(self):
        from chemparseplot.parse.chemgp_hdf5 import read_h5_path

        x_ds = MagicMock()
        x_ds.__getitem__ = MagicMock(return_value=np.array([0.0, 1.0, 2.0]))
        y_ds = MagicMock()
        y_ds.__getitem__ = MagicMock(return_value=np.array([3.0, 4.0, 5.0]))

        path_group = MagicMock()
        path_group.keys.return_value = ["x", "y"]
        path_group.__getitem__ = lambda s, k: {"x": x_ds, "y": y_ds}[k]

        f = MagicMock()
        f.__getitem__ = MagicMock(return_value=path_group)

        result = read_h5_path(f, "mep")
        assert "x" in result
        assert "y" in result

    def test_read_h5_points(self):
        from chemparseplot.parse.chemgp_hdf5 import read_h5_points

        pc1_ds = MagicMock()
        pc1_ds.__getitem__ = MagicMock(return_value=np.array([1.0, 2.0]))
        pc2_ds = MagicMock()
        pc2_ds.__getitem__ = MagicMock(return_value=np.array([3.0, 4.0]))

        pts_group = MagicMock()
        pts_group.keys.return_value = ["pc1", "pc2"]
        pts_group.__getitem__ = lambda s, k: {"pc1": pc1_ds, "pc2": pc2_ds}[k]

        f = MagicMock()
        f.__getitem__ = MagicMock(return_value=pts_group)

        result = read_h5_points(f, "fps")
        assert "pc1" in result
        assert "pc2" in result

    def test_read_h5_metadata(self):
        from chemparseplot.parse.chemgp_hdf5 import read_h5_metadata

        f = MagicMock()
        f.attrs = MagicMock()
        f.attrs.keys.return_value = ["surface", "version"]
        f.attrs.__getitem__ = lambda s, k: {"surface": "mb", "version": 2}[k]

        meta = read_h5_metadata(f)
        assert meta["surface"] == "mb"
        assert meta["version"] == 2

    def test_validate_hdf5_structure_ok(self):
        from chemparseplot.parse.chemgp_hdf5 import validate_hdf5_structure

        f = MagicMock()
        f.__contains__ = lambda s, k: k in {"grids", "table"}

        missing = validate_hdf5_structure(f)
        assert missing == []

    def test_validate_hdf5_structure_missing(self):
        from chemparseplot.parse.chemgp_hdf5 import validate_hdf5_structure

        f = MagicMock()
        f.__contains__ = lambda s, k: k in {"grids"}

        with pytest.raises(ValueError, match="Missing groups"):
            validate_hdf5_structure(f)

    def test_validate_hdf5_structure_custom_groups(self):
        from chemparseplot.parse.chemgp_hdf5 import validate_hdf5_structure

        f = MagicMock()
        f.__contains__ = lambda s, k: k in {"paths", "metadata"}

        missing = validate_hdf5_structure(f, required_groups=["paths", "metadata"])
        assert missing == []


# ============================================================
# chemparseplot.parse.chemgp_jsonl
# ============================================================
@pytest.mark.skipif(not _HAS_PANDAS, reason="pandas required")
class TestChemGPJsonl:
    def test_parse_comparison_jsonl(self, tmp_path):
        from chemparseplot.parse.chemgp_jsonl import parse_comparison_jsonl

        lines = [
            json.dumps(
                {"method": "gp_minimize", "step": 0, "oracle_calls": 1, "energy": -0.5}
            ),
            json.dumps(
                {
                    "method": "gp_minimize",
                    "step": 1,
                    "oracle_calls": 2,
                    "energy": -1.0,
                    "force": 0.1,
                }
            ),
            json.dumps({"method": "neb", "oracle_calls": 5, "max_force": 0.3}),
            json.dumps({"summary": True, "total": 10}),
        ]
        f = tmp_path / "comp.jsonl"
        f.write_text("\n".join(lines))

        data = parse_comparison_jsonl(f)
        assert "gp_minimize" in data.traces
        assert "neb" in data.traces
        assert data.summary is not None
        assert data.summary["total"] == 10

        gp_trace = data.traces["gp_minimize"]
        assert gp_trace.steps == [0, 1]
        assert gp_trace.oracle_calls == [1, 2]
        assert gp_trace.energies == [-0.5, -1.0]
        assert gp_trace.forces == [0.1]

        neb_trace = data.traces["neb"]
        assert neb_trace.forces == [0.3]
        assert neb_trace.energies is None

    def test_parse_rff_quality_jsonl(self, tmp_path):
        from chemparseplot.parse.chemgp_jsonl import parse_rff_quality_jsonl

        lines = [
            json.dumps(
                {"type": "exact_gp", "energy_mae": 0.01, "gradient_mae": 0.02}
            ),
            json.dumps(
                {
                    "type": "rff",
                    "d_rff": 50,
                    "energy_mae_vs_true": 0.5,
                    "gradient_mae_vs_true": 1.0,
                    "energy_mae_vs_gp": 0.3,
                    "gradient_mae_vs_gp": 0.8,
                }
            ),
            json.dumps(
                {
                    "type": "rff",
                    "d_rff": 100,
                    "energy_mae_vs_true": 0.2,
                    "gradient_mae_vs_true": 0.4,
                    "energy_mae_vs_gp": 0.1,
                    "gradient_mae_vs_gp": 0.3,
                }
            ),
        ]
        f = tmp_path / "rff.jsonl"
        f.write_text("\n".join(lines))

        data = parse_rff_quality_jsonl(f)
        assert data.exact_energy_mae == pytest.approx(0.01)
        assert data.exact_gradient_mae == pytest.approx(0.02)
        assert data.d_rff_values == [50, 100]
        assert len(data.energy_mae_vs_true) == 2

    def test_parse_gp_quality_jsonl(self, tmp_path):
        from chemparseplot.parse.chemgp_jsonl import parse_gp_quality_jsonl

        lines = [
            json.dumps(
                {
                    "type": "grid_meta",
                    "nx": 2,
                    "ny": 2,
                    "x_min": -2,
                    "x_max": 2,
                    "y_min": -2,
                    "y_max": 2,
                }
            ),
            json.dumps(
                {"type": "minimum", "id": 0, "x": -0.5, "y": 1.0, "energy": -150.0}
            ),
            json.dumps(
                {"type": "saddle", "id": 0, "x": 0.0, "y": 0.5, "energy": -10.0}
            ),
            json.dumps(
                {
                    "type": "train_point",
                    "n_train": 5,
                    "x": 0.0,
                    "y": 0.0,
                    "energy": -100.0,
                }
            ),
            json.dumps(
                {
                    "type": "grid",
                    "n_train": 5,
                    "ix": 0,
                    "iy": 0,
                    "x": -2.0,
                    "y": -2.0,
                    "true_e": -50.0,
                    "gp_e": -48.0,
                    "gp_var": 0.5,
                }
            ),
            json.dumps(
                {
                    "type": "grid",
                    "n_train": 5,
                    "ix": 1,
                    "iy": 0,
                    "x": 2.0,
                    "y": -2.0,
                    "true_e": -30.0,
                    "gp_e": -28.0,
                    "gp_var": 1.0,
                }
            ),
            json.dumps(
                {
                    "type": "grid",
                    "n_train": 5,
                    "ix": 0,
                    "iy": 1,
                    "x": -2.0,
                    "y": 2.0,
                    "true_e": -40.0,
                    "gp_e": -38.0,
                    "gp_var": 0.7,
                }
            ),
            json.dumps(
                {
                    "type": "grid",
                    "n_train": 5,
                    "ix": 1,
                    "iy": 1,
                    "x": 2.0,
                    "y": 2.0,
                    "true_e": -20.0,
                    "gp_e": -18.0,
                    "gp_var": 1.5,
                }
            ),
        ]
        f = tmp_path / "gp_quality.jsonl"
        f.write_text("\n".join(lines))

        data = parse_gp_quality_jsonl(f)
        assert data.meta["nx"] == 2
        assert len(data.stationary) == 2
        assert data.stationary[0].kind == "minimum"
        assert data.stationary[1].kind == "saddle"
        assert 5 in data.grids
        grid = data.grids[5]
        assert grid.n_train == 5
        assert grid.nx == 2
        assert grid.ny == 2
        assert grid.train_x == [0.0]
        assert grid.gp_e[0][0] == pytest.approx(-48.0)

    def test_optimizer_trace_defaults(self):
        from chemparseplot.parse.chemgp_jsonl import OptimizerTrace

        t = OptimizerTrace(method="test")
        assert t.steps == []
        assert t.energies is None
        assert t.forces is None

    def test_comparison_data_defaults(self):
        from chemparseplot.parse.chemgp_jsonl import ComparisonData

        d = ComparisonData()
        assert d.traces == {}
        assert d.summary is None


# ============================================================
# chemparseplot.plot.chemgp
# ============================================================
@pytest.mark.skipif(not _HAS_PANDAS, reason="pandas required")
class TestChemGPPlot:
    """Smoke tests for ChemGP plot functions."""

    def test_plot_convergence_curve_basic(self):
        from chemparseplot.plot.chemgp import plot_convergence_curve

        df = pd.DataFrame(
            {
                "oracle_calls": [1, 2, 3, 4, 5, 6],
                "max_fatom": [1.0, 0.5, 0.3, 0.8, 0.4, 0.2],
                "method": ["GP"] * 3 + ["NEB"] * 3,
            }
        )
        p = plot_convergence_curve(df, conv_tol=0.25)
        assert p is not None

    def test_plot_convergence_curve_dict_tol(self):
        from chemparseplot.plot.chemgp import plot_convergence_curve

        df = pd.DataFrame(
            {
                "oracle_calls": [1, 2, 3],
                "max_fatom": [1.0, 0.5, 0.1],
                "method": ["GP"] * 3,
            }
        )
        p = plot_convergence_curve(df, conv_tol={"GP": 0.2}, log_y=False)
        assert p is not None

    def test_plot_rff_quality(self):
        from chemparseplot.plot.chemgp import plot_rff_quality

        df = pd.DataFrame(
            {
                "d_rff": [50, 100, 200],
                "energy_mae": [0.5, 0.2, 0.1],
                "gradient_mae": [1.0, 0.4, 0.2],
            }
        )
        p = plot_rff_quality(df, exact_e_mae=0.05, exact_g_mae=0.1)
        assert p is not None

    def test_plot_hyperparameter_sensitivity(self):
        from chemparseplot.plot.chemgp import plot_hyperparameter_sensitivity

        x = np.linspace(-2, 2, 20)
        y_true = np.sin(x) * 50
        panels = {}
        for j in range(1, 4):
            for i in range(1, 4):
                panels[f"gp_ls{j}_sv{i}"] = {
                    "E_pred": np.sin(x) * 50 + np.random.randn(20),
                    "E_std": np.full(20, 10.0),
                }
        fig = plot_hyperparameter_sensitivity(x, y_true, panels)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_hyperparameter_sensitivity_missing_panel(self):
        from chemparseplot.plot.chemgp import plot_hyperparameter_sensitivity

        x = np.linspace(-2, 2, 20)
        y_true = np.sin(x) * 50
        panels = {
            "gp_ls1_sv1": {
                "E_pred": np.sin(x) * 50,
                "E_std": np.full(20, 10.0),
            }
        }
        fig = plot_hyperparameter_sensitivity(x, y_true, panels)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_trust_region(self):
        from chemparseplot.plot.chemgp import plot_trust_region

        x = np.linspace(-2, 2, 100)
        e_true = np.sin(x) * 50
        e_pred = np.sin(x) * 45
        e_std = np.full(100, 5.0)
        in_trust = (np.abs(x) < 1).astype(float)
        train_x = np.array([-1.5, -0.5, 0.5, 1.5])

        fig = plot_trust_region(x, e_true, e_pred, e_std, in_trust, train_x)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_trust_region_no_train(self):
        from chemparseplot.plot.chemgp import plot_trust_region

        x = np.linspace(-2, 2, 100)
        e_true = np.sin(x) * 50
        e_pred = np.sin(x) * 45
        e_std = np.full(100, 5.0)
        in_trust = np.ones(100)

        fig = plot_trust_region(x, e_true, e_pred, e_std, in_trust)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_fps_projection(self):
        from chemparseplot.plot.chemgp import plot_fps_projection

        p = plot_fps_projection([0, 1, 2], [0, 1, 0], [0.5, 1.5], [0.3, 0.8])
        assert p is not None

    def test_plot_energy_profile(self):
        from chemparseplot.plot.chemgp import plot_energy_profile

        df = pd.DataFrame(
            {
                "image": [0, 1, 2, 3, 4],
                "energy": [0.0, 0.3, 0.5, 0.3, 0.0],
                "method": ["NEB"] * 5,
            }
        )
        p = plot_energy_profile(df)
        assert p is not None

    def test_plot_surface_contour_basic(self):
        from chemparseplot.plot.chemgp import plot_surface_contour

        x = np.linspace(-2, 2, 30)
        X, Y = np.meshgrid(x, x)
        Z = np.sin(X) * np.cos(Y)
        fig = plot_surface_contour(X, Y, Z, clamp_lo=-1, clamp_hi=1)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_surface_contour_with_paths_and_points(self):
        from chemparseplot.plot.chemgp import plot_surface_contour

        x = np.linspace(-2, 2, 30)
        X, Y = np.meshgrid(x, x)
        Z = np.sin(X) * np.cos(Y)
        paths = {"mep": (np.array([-1, 0, 1]), np.array([0, 1, 0]))}
        points = {
            "minima": (np.array([-1.0]), np.array([0.0])),
            "saddles": (np.array([0.0]), np.array([1.0])),
            "endpoints": (np.array([-2.0, 2.0]), np.array([0.0, 0.0])),
        }
        fig = plot_surface_contour(X, Y, Z, paths=paths, points=points)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_gp_progression(self):
        from chemparseplot.plot.chemgp import plot_gp_progression

        x = np.linspace(-2, 2, 20)
        X, Y = np.meshgrid(x, x)
        gp_mean = np.sin(X) * np.cos(Y) * 50
        grids = {
            5: {"gp_mean": gp_mean, "train_x": [0, 1], "train_y": [0, 1]},
            10: {"gp_mean": gp_mean * 0.9, "train_x": [0, 1, -1], "train_y": [0, 1, -1]},
        }
        fig = plot_gp_progression(grids, gp_mean, x, x)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_nll_landscape(self):
        from chemparseplot.plot.chemgp import plot_nll_landscape

        x = np.linspace(-3, 3, 25)
        X, Y = np.meshgrid(x, x)
        nll = (X - 0.5) ** 2 + (Y + 1) ** 2
        grad_norm = np.sqrt(2 * (X - 0.5) ** 2 + 2 * (Y + 1) ** 2)
        fig = plot_nll_landscape(X, Y, nll, grid_grad_norm=grad_norm, optimum=(0.5, -1.0))
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_nll_landscape_no_extras(self):
        from chemparseplot.plot.chemgp import plot_nll_landscape

        x = np.linspace(-3, 3, 25)
        X, Y = np.meshgrid(x, x)
        nll = (X - 0.5) ** 2 + (Y + 1) ** 2
        fig = plot_nll_landscape(X, Y, nll)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_variance_overlay(self):
        from chemparseplot.plot.chemgp import plot_variance_overlay

        x = np.linspace(-2, 2, 20)
        X, Y = np.meshgrid(x, x)
        E = np.sin(X) * np.cos(Y) * 100
        V = np.exp(-X**2 - Y**2) + 0.01

        stationary = {"min0": (0.0, 0.0), "saddle0": (1.0, 1.0)}
        train_pts = ([0.0, 1.0, -1.0], [0.0, 1.0, -1.0])

        fig = plot_variance_overlay(
            X, Y, E, V, train_points=train_pts, stationary=stationary
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_detect_clamp(self):
        from chemparseplot.plot.chemgp import detect_clamp

        lo, hi, step = detect_clamp("mb_surface.h5")
        assert lo == pytest.approx(-200.0)
        assert hi == pytest.approx(50.0)

        lo2, hi2, step2 = detect_clamp("leps_surface.h5")
        assert lo2 == pytest.approx(-5.0)

        lo3, hi3, step3 = detect_clamp("unknown.h5")
        assert lo3 is None

    def test_safe_plot_decorator(self):
        from chemparseplot.plot.chemgp import safe_plot

        @safe_plot
        def good_func():
            return 42

        assert good_func() == 42

        @safe_plot
        def bad_func():
            raise KeyError("missing_key")

        with pytest.raises(KeyError):
            bad_func()

    def test_save_plot_matplotlib(self, tmp_path):
        from chemparseplot.plot.chemgp import save_plot

        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        output = tmp_path / "test.png"
        save_plot(fig, output)
        assert output.exists()

    @pytest.mark.skipif(
        not importlib.util.find_spec("plotnine"), reason="plotnine required"
    )
    def test_save_plot_plotnine(self, tmp_path):
        from chemparseplot.plot.chemgp import save_plot

        # Create a mock ggplot with a save method
        mock_gg = MagicMock()
        mock_gg.__class__ = type("ggplot", (), {})
        # Not a plt.Figure, so it goes to the else branch
        output = tmp_path / "test_gg.png"
        save_plot(mock_gg, output)
        mock_gg.save.assert_called_once()


# ============================================================
# chemparseplot.plot.plumed
# ============================================================
@pytest.mark.skipif(not _HAS_PANDAS, reason="pandas required")
class TestPlumedPlot:
    def test_plot_fes_2d(self):
        from chemparseplot.plot.plumed import plot_fes_2d

        x = np.linspace(-3, 3, 30)
        y = np.linspace(-3, 3, 30)
        X, Y = np.meshgrid(x, y)
        fes = np.sin(X) * np.cos(Y)

        fes_result = {"fes": fes, "x": x, "y": y, "dimension": 2}
        fig = plot_fes_2d(fes_result)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_fes_2d_with_minima(self):
        from chemparseplot.plot.plumed import plot_fes_2d

        x = np.linspace(-3, 3, 30)
        y = np.linspace(-3, 3, 30)
        X, Y = np.meshgrid(x, y)
        fes = np.sin(X) * np.cos(Y)

        fes_result = {"fes": fes, "x": x, "y": y, "dimension": 2}
        minima_df = pd.DataFrame(
            {"CV1": [0.0, -1.5], "CV2": [0.0, 1.5], "letter": ["A", "B"]}
        )
        minima_result = {"minima": minima_df}

        fig = plot_fes_2d(fes_result, minima_result=minima_result)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_fes_1d(self):
        from chemparseplot.plot.plumed import plot_fes_1d

        x = np.linspace(-3, 3, 100)
        fes = np.sin(x) * 10

        fes_result = {"fes": fes, "x": x, "dimension": 1}
        fig = plot_fes_1d(fes_result)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_fes_1d_with_minima(self):
        from chemparseplot.plot.plumed import plot_fes_1d

        x = np.linspace(-3, 3, 100)
        fes = np.sin(x) * 10

        fes_result = {"fes": fes, "x": x, "dimension": 1}
        minima_df = pd.DataFrame(
            {"CV1": [-1.57, 1.57], "free_energy": [-10.0, -10.0], "letter": ["A", "B"]}
        )
        minima_result = {"minima": minima_df}

        fig = plot_fes_1d(fes_result, minima_result=minima_result)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# ============================================================
# chemparseplot.plot.structs
# ============================================================
@pytest.mark.skipif(not _HAS_CMCRAMERI, reason="cmcrameri required")
class TestPlotStructs:
    def test_energy_path_namedtuple(self):
        from chemparseplot.plot.structs import EnergyPath

        ep = EnergyPath(label="test", distance=[1, 2], energy=[3, 4])
        assert ep.label == "test"

    def test_xy_data_namedtuple(self):
        from chemparseplot.plot.structs import XYData

        xy = XYData(label="a", x=[1], y=[2])
        assert xy.label == "a"

    def test_base_plotter(self):
        from chemparseplot.plot.structs import BasePlotter

        bp = BasePlotter()
        assert bp.fig is not None
        assert bp.ax is not None
        bp.add_title("Test Title")
        plt.close(bp.fig)

    def test_two_dim_plot_init(self):
        from chemparseplot.plot.structs import TwoDimPlot

        tdp = TwoDimPlot()
        assert tdp.x_unit == "dimensionless"
        assert tdp.y_unit == "dimensionless"
        assert repr(tdp) == "TwoDimPlot with 0 datasets"
        plt.close(tdp.fig)

    def test_two_dim_plot_set_labels(self):
        from chemparseplot.plot.structs import TwoDimPlot

        tdp = TwoDimPlot()
        tdp.set_labels("Distance", "Energy")
        assert tdp.x_label == "Distance"
        assert tdp.y_label == "Energy"
        plt.close(tdp.fig)


# ============================================================
# chemparseplot.plot.geomscan
# ============================================================
import sys
_PY310 = sys.version_info < (3, 11)


@pytest.mark.skipif(not _HAS_CMCRAMERI or _PY310, reason="cmcrameri required, py3.10 has recursion issue")
class TestGeomscanPlot:
    def test_plot_energy_paths(self):
        """Test that plot_energy_paths creates a figure (mocking rgpycrumbs)."""
        from unittest.mock import MagicMock

        from chemparseplot.plot.structs import BasePlotter

        mock_path = MagicMock()
        mock_path.label = "path1"
        mock_dist = MagicMock()
        mock_dist.to.return_value = MagicMock(m=np.linspace(0, 3, 10))
        mock_energy = MagicMock()
        mock_energy.to.return_value = MagicMock(m=np.sin(np.linspace(0, 3, 10)))
        mock_path.distance = mock_dist
        mock_path.energy = mock_energy

        units = {"distance": "angstrom", "energy": "eV"}

        with patch(
            "chemparseplot.plot.geomscan.spline_interp",
            return_value=(np.linspace(0, 3, 100), np.sin(np.linspace(0, 3, 100))),
        ):
            from chemparseplot.plot.geomscan import plot_energy_paths

            plotter = plot_energy_paths([mock_path], units)
            assert plotter is not None
            plt.close(plotter.fig)


# ============================================================
# chemparseplot.plot.__init__ (lazy imports coverage)
# ============================================================
@pytest.mark.skipif(not _HAS_CMCRAMERI, reason="cmcrameri required")
class TestPlotInitLazy:
    def test_lazy_geomscan(self):
        from chemparseplot.plot import geomscan

        assert hasattr(geomscan, "plot_energy_paths")

    def test_lazy_structs(self):
        from chemparseplot.plot import structs

        assert hasattr(structs, "BasePlotter")

    @pytest.mark.skipif(
        not _HAS_PANDAS or not importlib.util.find_spec("plotnine"),
        reason="chemgp needs pandas + plotnine",
    )
    def test_lazy_chemgp(self):
        from chemparseplot.plot import chemgp

        assert hasattr(chemgp, "plot_convergence_curve")

    def test_lazy_ureg(self):
        from chemparseplot.plot import ureg

        assert ureg is not None


# ============================================================
# chemparseplot.util (module-level code that uses ASE)
# ============================================================
class TestUtilModuleLevel:
    def test_parse_target_coords_multiline(self):
        from chemparseplot.util import parse_target_coords

        text = """
        1.0 2.0 3.0
        4.0 5.0 6.0
        7.0 8.0 9.0
        """
        coords = parse_target_coords(text)
        assert coords.shape == (3, 3)

    def test_parse_target_coords_non_numeric_value(self):
        from chemparseplot.util import parse_target_coords

        text = "1.0 abc 3.0\n"
        coords = parse_target_coords(text)
        assert coords.size == 0


# ============================================================
# chemparseplot.parse.orca.neb.opi_parser
# ============================================================
class TestOpiParserFull:
    def test_parse_orca_neb_with_mock(self):
        """Full parse_orca_neb with mocked OPI Output class."""
        from chemparseplot.parse.orca.neb.opi_parser import parse_orca_neb

        mock_output_cls = MagicMock()
        mock_output_inst = MagicMock()
        mock_output_cls.return_value = mock_output_inst
        mock_output_inst.parse.return_value = None
        mock_output_inst.terminated_normally.return_value = True
        mock_output_inst.num_results_gbw = 3

        # Energies in Hartree
        mock_output_inst.get_final_energy.side_effect = [
            -100.0, -99.9, -100.0
        ]

        # Geometry mock
        mock_geom = MagicMock()
        mock_geom.coordinates.cartesians = np.zeros((3, 3))
        mock_atom = MagicMock()
        mock_atom.atomic_number = 1
        mock_geom.atoms = [mock_atom, mock_atom, mock_atom]
        mock_output_inst.get_geometry.return_value = mock_geom

        # Gradient mock
        mock_output_inst.get_gradient.return_value = np.ones((3, 3)) * 0.01

        with patch(
            "chemparseplot.parse.orca.neb.opi_parser._get_opi_output",
            return_value=mock_output_cls,
        ):
            result = parse_orca_neb("test_job", working_dir=Path("/tmp"))

        assert result["n_images"] == 3
        assert result["converged"] is True
        assert result["source"] == "opi"
        assert len(result["energies"]) == 3
        assert result["barrier_forward"] is not None

    def test_parse_orca_neb_no_geometries(self):
        """Test parse_orca_neb when geometry extraction fails."""
        from chemparseplot.parse.orca.neb.opi_parser import parse_orca_neb

        mock_output_cls = MagicMock()
        mock_output_inst = MagicMock()
        mock_output_cls.return_value = mock_output_inst
        mock_output_inst.parse.return_value = None
        mock_output_inst.terminated_normally.return_value = False
        mock_output_inst.num_results_gbw = 2
        mock_output_inst.get_final_energy.side_effect = [-100.0, -99.5]
        mock_output_inst.get_geometry.side_effect = AttributeError("no geom")
        mock_output_inst.get_gradient.side_effect = AttributeError("no grad")

        with patch(
            "chemparseplot.parse.orca.neb.opi_parser._get_opi_output",
            return_value=mock_output_cls,
        ):
            result = parse_orca_neb("test_job", working_dir=Path("/tmp"))

        assert result["n_images"] == 2
        assert result["converged"] is False
        assert result["rmsd_r"] is None
        assert result["rmsd_p"] is None

    def test_parse_orca_neb_single_image(self):
        """Test with a single image (no barrier computation)."""
        from chemparseplot.parse.orca.neb.opi_parser import parse_orca_neb

        mock_output_cls = MagicMock()
        mock_output_inst = MagicMock()
        mock_output_cls.return_value = mock_output_inst
        mock_output_inst.parse.return_value = None
        mock_output_inst.terminated_normally.return_value = True
        mock_output_inst.num_results_gbw = 1
        mock_output_inst.get_final_energy.return_value = -100.0
        mock_output_inst.get_geometry.side_effect = AttributeError
        mock_output_inst.get_gradient.side_effect = AttributeError

        with patch(
            "chemparseplot.parse.orca.neb.opi_parser._get_opi_output",
            return_value=mock_output_cls,
        ):
            result = parse_orca_neb("test_job", working_dir=Path("/tmp"))

        assert result["barrier_forward"] is None


# ============================================================
# chemparseplot.parse.plumed
# ============================================================
@pytest.mark.skipif(not _HAS_PANDAS, reason="pandas required")
class TestPlumedParse:
    def test_calculate_fes_2d(self):
        from chemparseplot.parse.plumed import calculate_fes_from_hills

        hills_data = np.array(
            [
                [0.0, 0.0, 0.0, 0.5, 0.5, 1.0, 0.0],
                [1.0, 0.5, 0.5, 0.5, 0.5, 1.0, 0.0],
                [2.0, -0.5, -0.5, 0.5, 0.5, 0.5, 0.0],
            ]
        )
        hills = {"hillsfile": hills_data, "per": [False, False]}
        result = calculate_fes_from_hills(hills, npoints=16)
        assert result["dimension"] == 2
        assert result["fes"].shape == (16, 16)
        assert len(result["x"]) == 16
        assert len(result["y"]) == 16

    def test_calculate_fes_1d(self):
        from chemparseplot.parse.plumed import calculate_fes_from_hills

        hills_data = np.array(
            [
                [0.0, 0.0, 0.5, 1.0, 0.0],
                [1.0, 0.5, 0.5, 1.0, 0.0],
                [2.0, -0.5, 0.5, 0.5, 0.0],
            ]
        )
        hills = {"hillsfile": hills_data, "per": [False]}
        result = calculate_fes_from_hills(hills, npoints=16)
        assert result["dimension"] == 1
        assert len(result["fes"]) == 16

    def test_calculate_fes_periodic(self):
        from chemparseplot.parse.plumed import calculate_fes_from_hills

        hills_data = np.array(
            [
                [0.0, 1.0, 1.0, 0.3, 0.3, 1.0, 0.0],
                [1.0, 2.0, 2.0, 0.3, 0.3, 1.0, 0.0],
            ]
        )
        hills = {
            "hillsfile": hills_data,
            "per": [True, True],
            "pcv1": [0, 2 * np.pi],
            "pcv2": [0, 2 * np.pi],
        }
        result = calculate_fes_from_hills(hills, npoints=16)
        assert result["dimension"] == 2

    def test_calculate_fes_imax_zero(self):
        from chemparseplot.parse.plumed import calculate_fes_from_hills

        hills_data = np.array(
            [
                [0.0, 0.0, 0.0, 0.5, 0.5, 1.0, 0.0],
            ]
        )
        hills = {"hillsfile": hills_data, "per": [False, False]}
        result = calculate_fes_from_hills(hills, imax=0, npoints=16)
        assert np.all(result["fes"] == 0)

    def test_calculate_fes_imax_warning(self):
        from chemparseplot.parse.plumed import calculate_fes_from_hills

        hills_data = np.array(
            [[0.0, 0.0, 0.0, 0.5, 0.5, 1.0, 0.0]]
        )
        hills = {"hillsfile": hills_data, "per": [False, False]}
        with pytest.warns(UserWarning, match="only 1 hills are available"):
            result = calculate_fes_from_hills(hills, imax=100, npoints=16)

    def test_calculate_fes_imin_gt_imax(self):
        from chemparseplot.parse.plumed import calculate_fes_from_hills

        hills_data = np.array(
            [[0.0, 0.0, 0.0, 0.5, 0.5, 1.0, 0.0]] * 5
        )
        hills = {"hillsfile": hills_data, "per": [False, False]}
        with pytest.raises(ValueError, match="imax cannot be lower than imin"):
            calculate_fes_from_hills(hills, imin=5, imax=2, npoints=16)

    def test_calculate_fes_unsupported_cols(self):
        from chemparseplot.parse.plumed import calculate_fes_from_hills

        hills_data = np.array([[0.0, 1.0, 2.0]])
        hills = {"hillsfile": hills_data, "per": [False]}
        with pytest.raises(ValueError, match="Unsupported number of columns"):
            calculate_fes_from_hills(hills, npoints=16)

    def test_calculate_fes_1d_periodic(self):
        from chemparseplot.parse.plumed import calculate_fes_from_hills

        hills_data = np.array(
            [
                [0.0, 1.0, 0.3, 1.0, 0.0],
            ]
        )
        hills = {
            "hillsfile": hills_data,
            "per": [True],
            "pcv1": [0, 2 * np.pi],
        }
        result = calculate_fes_from_hills(hills, npoints=16)
        assert result["dimension"] == 1

    def test_calculate_fes_1d_imax_zero(self):
        from chemparseplot.parse.plumed import calculate_fes_from_hills

        hills_data = np.array(
            [[0.0, 0.0, 0.5, 1.0, 0.0]]
        )
        hills = {"hillsfile": hills_data, "per": [False]}
        result = calculate_fes_from_hills(hills, imax=0, npoints=16)
        assert np.all(result["fes"] == 0)

    def test_calculate_fes_xlim_ylim(self):
        from chemparseplot.parse.plumed import calculate_fes_from_hills

        hills_data = np.array(
            [
                [0.0, 0.0, 0.0, 0.5, 0.5, 1.0, 0.0],
            ]
        )
        hills = {"hillsfile": hills_data, "per": [False, False]}
        result = calculate_fes_from_hills(
            hills, xlim=[-5, 5], ylim=[-5, 5], npoints=16
        )
        assert result["x"][0] == pytest.approx(-5.0)
        assert result["y"][0] == pytest.approx(-5.0)

    def test_find_fes_minima_2d(self):
        from chemparseplot.parse.plumed import calculate_fes_from_hills, find_fes_minima

        # Create hills that produce a surface with at least one minimum
        hills_data = np.array(
            [
                [0.0, 0.0, 0.0, 0.3, 0.3, 2.0, 0.0],
                [1.0, 1.5, 1.5, 0.3, 0.3, 1.0, 0.0],
            ]
        )
        hills = {"hillsfile": hills_data, "per": [False, False]}
        fes_result = calculate_fes_from_hills(hills, npoints=16)
        minima = find_fes_minima(fes_result, nbins=4)
        # May or may not find minima depending on the surface, but should not crash
        if minima is not None:
            assert "minima" in minima
            assert "letter" in minima["minima"].columns

    def test_find_fes_minima_1d(self):
        from chemparseplot.parse.plumed import calculate_fes_from_hills, find_fes_minima

        hills_data = np.array(
            [
                [0.0, 0.0, 0.3, 2.0, 0.0],
                [1.0, 1.5, 0.3, 1.0, 0.0],
            ]
        )
        hills = {"hillsfile": hills_data, "per": [False]}
        fes_result = calculate_fes_from_hills(hills, npoints=16)
        result = find_fes_minima(fes_result, nbins=4)
        # May or may not find minima
        if result is not None:
            assert "minima" in result

    def test_find_fes_minima_bad_nbins(self):
        from chemparseplot.parse.plumed import calculate_fes_from_hills, find_fes_minima

        hills_data = np.array(
            [[0.0, 0.0, 0.5, 1.0, 0.0]]
        )
        hills = {"hillsfile": hills_data, "per": [False]}
        fes_result = calculate_fes_from_hills(hills, npoints=16)
        with pytest.raises(ValueError, match="integer multiple"):
            find_fes_minima(fes_result, nbins=7)

    def test_find_fes_minima_too_many_bins(self):
        from chemparseplot.parse.plumed import calculate_fes_from_hills, find_fes_minima

        hills_data = np.array(
            [[0.0, 0.0, 0.5, 1.0, 0.0]]
        )
        hills = {"hillsfile": hills_data, "per": [False]}
        fes_result = calculate_fes_from_hills(hills, npoints=16)
        with pytest.raises(ValueError, match="nbins is too high"):
            find_fes_minima(fes_result, nbins=16)


# ============================================================
# chemparseplot.plot.neb (surface fitting and ORCA profile)
# ============================================================
class TestNebPlotSurface:
    def test_plot_landscape_path_overlay(self):
        from chemparseplot.plot.neb import plot_landscape_path_overlay

        fig, ax = plt.subplots()
        r = np.linspace(0, 3, 11)
        p = np.linspace(3, 0, 11)
        z = np.sin(np.linspace(0, np.pi, 11))
        cb = plot_landscape_path_overlay(ax, r, p, z, "viridis", "E (eV)", project_path=True)
        assert cb is not None
        plt.close(fig)

    def test_plot_landscape_path_overlay_no_projection(self):
        from chemparseplot.plot.neb import plot_landscape_path_overlay

        fig, ax = plt.subplots()
        r = np.linspace(0, 3, 11)
        p = np.linspace(3, 0, 11)
        z = np.sin(np.linspace(0, np.pi, 11))
        cb = plot_landscape_path_overlay(ax, r, p, z, "invalid_cmap", "E", project_path=False)
        assert cb is not None
        plt.close(fig)

    def test_smoothing_params(self):
        from chemparseplot.plot.neb import SmoothingParams

        sp = SmoothingParams()
        assert sp.window_length == 5
        assert sp.polyorder == 2

        sp2 = SmoothingParams(window_length=7, polyorder=3)
        assert sp2.window_length == 7

    def test_inset_image_pos(self):
        from chemparseplot.plot.neb import InsetImagePos

        pos = InsetImagePos(x=1.0, y=2.0, rad=0.3)
        assert pos.x == 1.0

    def test_plot_orca_neb_profile(self, tmp_path):
        from chemparseplot.plot.neb import plot_orca_neb_profile

        neb_data = {
            "energies": [0.0, 0.5, 1.0, 0.5, 0.0],
            "n_images": 5,
            "barrier_forward": 1.0,
            "barrier_reverse": 1.0,
        }
        output = tmp_path / "profile.png"
        plot_orca_neb_profile(neb_data, output)
        assert output.exists()

    def test_plot_orca_neb_profile_no_energies(self, tmp_path):
        from chemparseplot.plot.neb import plot_orca_neb_profile

        with pytest.raises(ValueError, match="No energy data"):
            plot_orca_neb_profile({"energies": []}, tmp_path / "empty.png")

    def test_plot_orca_neb_energy_profile_with_rmsd(self, tmp_path):
        from chemparseplot.plot.neb import plot_orca_neb_energy_profile

        neb_data = {
            "energies": np.array([0.0, 0.3, 0.8, 0.3, 0.0]),
            "rmsd_r": np.linspace(0, 3, 5),
            "rmsd_p": np.linspace(3, 0, 5),
            "grad_r": np.array([0.0, 0.1, 0.0, -0.1, 0.0]),
            "grad_p": np.array([0.0, -0.1, 0.0, 0.1, 0.0]),
            "n_images": 5,
            "barrier_forward": 0.8,
            "barrier_reverse": 0.8,
        }
        output = tmp_path / "orca_profile.png"
        plot_orca_neb_energy_profile(neb_data, output)
        assert output.exists()

    def test_plot_orca_neb_energy_profile_no_rmsd(self, tmp_path):
        from chemparseplot.plot.neb import plot_orca_neb_energy_profile

        neb_data = {
            "energies": np.array([0.0, 0.5, 0.0]),
            "rmsd_r": None,
            "rmsd_p": None,
            "grad_r": None,
            "grad_p": None,
            "n_images": 3,
            "barrier_forward": 0.5,
            "barrier_reverse": 0.5,
        }
        output = tmp_path / "orca_profile2.png"
        plot_orca_neb_energy_profile(neb_data, output)
        assert output.exists()

    def test_plot_orca_neb_energy_profile_empty(self, tmp_path):
        from chemparseplot.plot.neb import plot_orca_neb_energy_profile

        with pytest.raises(ValueError, match="No energy data"):
            plot_orca_neb_energy_profile(
                {"energies": np.array([])}, tmp_path / "empty.png"
            )

    def test_render_structure_to_image(self):
        from chemparseplot.plot.neb import render_structure_to_image

        from ase.build import molecule

        atoms = molecule("H2O")
        img = render_structure_to_image(atoms, zoom=0.3, rotation="0x,0y,0z")
        assert img.ndim >= 2

    def test_check_xyzrender_missing(self):
        from chemparseplot.plot.neb import _check_xyzrender

        with patch("shutil.which", return_value=None):
            with pytest.raises(RuntimeError, match="xyzrender"):
                _check_xyzrender()

    def test_render_atoms_ase(self):
        from chemparseplot.plot.neb import _render_atoms

        from ase.build import molecule

        atoms = molecule("H2O")
        img = _render_atoms(atoms, "ase", 0.3, "0x,0y,0z")
        assert img.ndim >= 2

    def test_render_atoms_xyzrender_fallback(self):
        """xyzrender falls back to ASE when not installed."""
        from chemparseplot.plot.neb import _render_atoms

        from ase.build import molecule

        atoms = molecule("H2O")
        with patch("shutil.which", return_value=None):
            # Should fall back to ASE, not raise
            img = _render_atoms(atoms, "xyzrender", 0.3, "0x,0y,0z")
            assert img.ndim == 3


# ============================================================
# chemparseplot.parse.trajectory.hdf5 (lines 253-294)
# ============================================================
class TestTrajectoryHdf5:
    def _make_h5_result(self, tmp_path, n_images=5, n_atoms=3):
        """Create a minimal neb_result.h5 file."""
        import h5py

        ndof = n_atoms * 3
        h5_path = str(tmp_path / "neb_result.h5")
        with h5py.File(h5_path, "w") as f:
            path_grp = f.create_group("path")
            path_grp.create_dataset(
                "images", data=np.random.randn(n_images, ndof)
            )
            path_grp.create_dataset(
                "energies", data=np.linspace(0, 1, n_images)
            )
            path_grp.create_dataset(
                "gradients", data=np.random.randn(n_images, ndof) * 0.01
            )
            path_grp.create_dataset(
                "f_para", data=np.random.randn(n_images) * 0.1
            )
            path_grp.create_dataset(
                "rxn_coord", data=np.linspace(0, 3, n_images)
            )
            conv_grp = f.create_group("convergence")
            conv_grp.create_dataset("max_force", data=np.array([0.1, 0.05]))
            meta_grp = f.create_group("metadata")
            meta_grp.create_dataset(
                "atomic_numbers", data=np.ones(n_atoms, dtype=int)
            )
            meta_grp.create_dataset("n_atoms", data=n_atoms)
        return h5_path

    def _make_h5_history(self, tmp_path, n_steps=3, n_images=5, n_atoms=3):
        """Create a minimal neb_history.h5 file."""
        import h5py

        ndof = n_atoms * 3
        h5_path = str(tmp_path / "neb_history.h5")
        with h5py.File(h5_path, "w") as f:
            steps_grp = f.create_group("steps")
            meta_grp = f.create_group("metadata")
            meta_grp.create_dataset(
                "atomic_numbers", data=np.ones(n_atoms, dtype=int)
            )
            for s in range(n_steps):
                sg = steps_grp.create_group(str(s))
                sg.create_dataset(
                    "images", data=np.random.randn(n_images, ndof)
                )
                sg.create_dataset(
                    "energies", data=np.linspace(0, 1, n_images) + 0.01 * s
                )
                sg.create_dataset(
                    "gradients", data=np.random.randn(n_images, ndof) * 0.01
                )
                sg.create_dataset(
                    "f_para", data=np.random.randn(n_images) * 0.1
                )
                sg.create_dataset(
                    "rxn_coord", data=np.linspace(0, 3, n_images)
                )
        return h5_path

    def test_load_neb_result(self, tmp_path):
        from chemparseplot.parse.trajectory.hdf5 import load_neb_result

        h5_path = self._make_h5_result(tmp_path)
        result = load_neb_result(h5_path)
        assert "path" in result
        assert "convergence" in result
        assert "metadata" in result
        assert result["path"]["energies"].shape == (5,)
        assert "max_force" in result["convergence"]

    def test_load_neb_history(self, tmp_path):
        from chemparseplot.parse.trajectory.hdf5 import load_neb_history

        h5_path = self._make_h5_history(tmp_path)
        steps = load_neb_history(h5_path)
        assert len(steps) == 3
        assert "energies" in steps[0]

    def test_result_to_profile_dat(self, tmp_path):
        from chemparseplot.parse.trajectory.hdf5 import result_to_profile_dat

        h5_path = self._make_h5_result(tmp_path)
        dat = result_to_profile_dat(h5_path)
        assert dat.shape == (5, 5)

    def test_result_to_atoms_list(self, tmp_path):
        from chemparseplot.parse.trajectory.hdf5 import result_to_atoms_list

        h5_path = self._make_h5_result(tmp_path)
        atoms_list = result_to_atoms_list(h5_path)
        assert len(atoms_list) == 5
        assert atoms_list[0].calc is not None

    def test_history_to_profile_dats(self, tmp_path):
        from chemparseplot.parse.trajectory.hdf5 import history_to_profile_dats

        h5_path = self._make_h5_history(tmp_path)
        dats = history_to_profile_dats(h5_path)
        assert len(dats) == 3
        assert dats[0].shape == (5, 5)

    def test_reconstruct_atoms_no_cell(self):
        from chemparseplot.parse.trajectory.hdf5 import _reconstruct_atoms

        images = np.random.randn(3, 6)
        gradients = np.random.randn(3, 6) * 0.01
        energies = np.array([0.0, 0.5, 0.0])
        atoms_list = _reconstruct_atoms(images, None, None, gradients, energies)
        assert len(atoms_list) == 3
        assert atoms_list[0].get_chemical_symbols() == ["H", "H"]

    def test_reconstruct_atoms_with_cell(self):
        from chemparseplot.parse.trajectory.hdf5 import _reconstruct_atoms

        images = np.random.randn(2, 9)
        gradients = np.random.randn(2, 9) * 0.01
        energies = np.array([0.0, 1.0])
        atomic_numbers = np.array([8, 1, 1])
        cell = np.eye(3).ravel() * 10.0

        atoms_list = _reconstruct_atoms(
            images, atomic_numbers, cell, gradients, energies
        )
        assert len(atoms_list) == 2
        assert atoms_list[0].pbc.all()
        assert atoms_list[0].get_chemical_symbols() == ["O", "H", "H"]

    def test_history_to_landscape_df(self, tmp_path):
        """Test history_to_landscape_df with mocked IRA and RMSD functions."""
        from chemparseplot.parse.trajectory.hdf5 import history_to_landscape_df

        h5_path = self._make_h5_history(tmp_path, n_steps=2, n_images=4, n_atoms=2)

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
            df = history_to_landscape_df(h5_path)

        assert "r" in df.columns
        assert "p" in df.columns
        assert "z" in df.columns
        assert "step" in df.columns
        assert len(df) == 8  # 2 steps * 4 images


# ============================================================
# chemparseplot.parse.trajectory.neb (fallback paths)
# ============================================================
class TestTrajectoryNeb:
    def test_get_energy_from_calc(self):
        from chemparseplot.parse.trajectory.neb import _get_energy

        from ase import Atoms
        from ase.calculators.singlepoint import SinglePointCalculator

        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
        calc = SinglePointCalculator(atoms, energy=-1.5)
        atoms.calc = calc
        assert _get_energy(atoms) == pytest.approx(-1.5)

    def test_get_energy_from_info(self):
        from chemparseplot.parse.trajectory.neb import _get_energy

        from ase import Atoms

        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
        atoms.info["energy"] = -2.0
        assert _get_energy(atoms) == pytest.approx(-2.0)

    def test_get_energy_default(self):
        from chemparseplot.parse.trajectory.neb import _get_energy

        from ase import Atoms

        atoms = Atoms("H")
        assert _get_energy(atoms) == pytest.approx(0.0)

    def test_compute_cumulative_distance(self):
        from chemparseplot.parse.trajectory.neb import compute_cumulative_distance

        from ase import Atoms

        atoms_list = []
        for i in range(4):
            a = Atoms("H", positions=[[float(i), 0, 0]])
            atoms_list.append(a)

        dists = compute_cumulative_distance(atoms_list)
        np.testing.assert_allclose(dists, [0.0, 1.0, 2.0, 3.0])

    def test_compute_tangent_force(self):
        from chemparseplot.parse.trajectory.neb import compute_tangent_force

        from ase import Atoms
        from ase.calculators.singlepoint import SinglePointCalculator

        atoms_list = []
        energies = np.array([0.0, 0.5, 1.0, 0.5, 0.0])
        for i, e in enumerate(energies):
            a = Atoms("H", positions=[[float(i), 0, 0]])
            forces = np.array([[0.1 * (2 - i), 0, 0]])
            calc = SinglePointCalculator(a, energy=e, forces=forces)
            a.calc = calc
            atoms_list.append(a)

        f_para = compute_tangent_force(atoms_list, energies)
        assert len(f_para) == 5
        assert f_para[0] == 0.0
        assert f_para[-1] == 0.0

    def test_compute_tangent_force_extremum(self):
        """Test the extremum bisection branch."""
        from chemparseplot.parse.trajectory.neb import compute_tangent_force

        from ase import Atoms
        from ase.calculators.singlepoint import SinglePointCalculator

        # Energy pattern: 0, 1, 0.5, 1, 0 (extremum at index 2)
        energies = np.array([0.0, 1.0, 0.5, 1.0, 0.0])
        atoms_list = []
        for i, e in enumerate(energies):
            a = Atoms("H", positions=[[float(i), 0, 0]])
            forces = np.array([[0.1, 0, 0]])
            calc = SinglePointCalculator(a, energy=e, forces=forces)
            a.calc = calc
            atoms_list.append(a)

        f_para = compute_tangent_force(atoms_list, energies)
        assert len(f_para) == 5

    def test_extract_profile_data(self):
        from chemparseplot.parse.trajectory.neb import extract_profile_data

        from ase import Atoms
        from ase.calculators.singlepoint import SinglePointCalculator

        atoms_list = []
        for i in range(3):
            a = Atoms("H", positions=[[float(i), 0, 0]])
            forces = np.array([[0.0, 0, 0]])
            calc = SinglePointCalculator(a, energy=float(i), forces=forces)
            a.calc = calc
            atoms_list.append(a)

        idx, dist, energy, f_para = extract_profile_data(atoms_list)
        assert len(idx) == 3
        assert len(dist) == 3
        assert len(energy) == 3

    def test_trajectory_to_profile_dat(self):
        from chemparseplot.parse.trajectory.neb import trajectory_to_profile_dat

        from ase import Atoms
        from ase.calculators.singlepoint import SinglePointCalculator

        atoms_list = []
        for i in range(3):
            a = Atoms("H", positions=[[float(i), 0, 0]])
            forces = np.array([[0.0, 0, 0]])
            calc = SinglePointCalculator(a, energy=float(i), forces=forces)
            a.calc = calc
            atoms_list.append(a)

        dat = trajectory_to_profile_dat(atoms_list)
        assert dat.shape == (5, 3)

    def test_load_trajectory(self, tmp_path):
        from chemparseplot.parse.trajectory.neb import load_trajectory

        from ase import Atoms
        from ase.io import write as ase_write

        atoms_list = [Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74 + 0.01 * i]]) for i in range(3)]
        traj_file = str(tmp_path / "test.extxyz")
        ase_write(traj_file, atoms_list, format="extxyz")

        loaded = load_trajectory(traj_file)
        assert len(loaded) == 3

    def test_trajectory_to_landscape_df(self):
        """Test trajectory_to_landscape_df with mocked IRA."""
        from chemparseplot.parse.trajectory.neb import trajectory_to_landscape_df

        from ase import Atoms
        from ase.calculators.singlepoint import SinglePointCalculator

        atoms_list = []
        for i in range(5):
            a = Atoms("H2", positions=[[float(i), 0, 0], [float(i), 0, 0.74]])
            forces = np.zeros((2, 3))
            calc = SinglePointCalculator(a, energy=float(i) * 0.1, forces=forces)
            a.calc = calc
            atoms_list.append(a)

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
            df = trajectory_to_landscape_df(atoms_list, step=0)

        assert "r" in df.columns
        assert "p" in df.columns
        assert len(df) == 5


# ============================================================
# Additional coverage: structs.py TwoDimPlot methods
# ============================================================
@pytest.mark.skipif(not _HAS_CMCRAMERI, reason="cmcrameri required")
class TestTwoDimPlotMethods:
    def test_set_labels(self):
        from chemparseplot.plot.structs import TwoDimPlot

        p = TwoDimPlot()
        p.set_labels("Distance", "Energy")
        assert p.x_label == "Distance"
        assert p.y_label == "Energy"
        plt.close(p.fig)

    def test_set_units(self):
        from unittest.mock import MagicMock
        from chemparseplot.plot.structs import TwoDimPlot

        p = TwoDimPlot()
        # Monkey-patch redraw to avoid pint
        p.redraw_plot = MagicMock()
        p.set_units("eV", "Angstrom")
        assert p.x_unit == "eV"
        p.redraw_plot.assert_called_once()
        plt.close(p.fig)

    def test_repr(self):
        from chemparseplot.plot.structs import TwoDimPlot

        p = TwoDimPlot()
        assert "0 datasets" in repr(p)
        plt.close(p.fig)

    def test_add_data_and_rmdat(self):
        from unittest.mock import MagicMock
        from chemparseplot.plot.structs import TwoDimPlot, XYData

        p = TwoDimPlot()
        p.redraw_plot = MagicMock()
        mock_x = MagicMock()
        mock_y = MagicMock()
        data = XYData(label="test", x=mock_x, y=mock_y)
        p.add_data(data)
        assert len(p.data) == 1
        p.rmdat(["test"])
        assert len(p.data) == 0
        plt.close(p.fig)


# ============================================================
# Additional coverage: neb.py ORCA landscape + profile helpers
# ============================================================
class TestOrcaNebHighLevel:
    def test_plot_orca_neb_profile(self, tmp_path):
        from chemparseplot.plot.neb import plot_orca_neb_profile

        data = {
            "energies": [0.0, 0.5, 1.0, 0.8, 0.2],
            "n_images": 5,
            "barrier_forward": 1.0,
            "barrier_reverse": 0.8,
        }
        output = tmp_path / "profile.png"
        plot_orca_neb_profile(data, output)
        assert output.exists()

    def test_plot_orca_neb_profile_no_barrier(self, tmp_path):
        from chemparseplot.plot.neb import plot_orca_neb_profile

        data = {"energies": [0.0, 0.5], "n_images": 2}
        output = tmp_path / "profile2.png"
        plot_orca_neb_profile(data, output)
        assert output.exists()

    def test_plot_orca_neb_energy_profile(self, tmp_path):
        from chemparseplot.plot.neb import plot_orca_neb_energy_profile

        data = {
            "energies": np.array([0.0, 0.3, 1.0, 0.7, 0.1]),
            "n_images": 5,
            "rmsd_r": np.linspace(0, 3, 5),
            "rmsd_p": np.linspace(3, 0, 5),
            "grad_r": np.zeros(5),
            "grad_p": np.zeros(5),
            "barrier_forward": 1.0,
            "barrier_reverse": 0.9,
        }
        output = tmp_path / "eprofile.png"
        plot_orca_neb_energy_profile(data, output)
        assert output.exists()

    def test_plot_orca_neb_energy_profile_no_rmsd(self, tmp_path):
        from chemparseplot.plot.neb import plot_orca_neb_energy_profile

        data = {"energies": np.array([0.0, 0.5, 0.1]), "n_images": 3}
        output = tmp_path / "eprofile2.png"
        plot_orca_neb_energy_profile(data, output)
        assert output.exists()
