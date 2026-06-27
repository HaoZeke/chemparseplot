# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""Smoke tests for optimization plot functions."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pytest
from matplotlib.axes import Axes

from chemparseplot.plot.optimization import (
    _LABELS,
    OVERLAY_COLORS,
    annotate_endpoint,
    create_landscape_axes,
    default_strip_zoom,
    enforce_strip_clearance,
    plot_convergence_panel,
    plot_dimer_mode_evolution,
    plot_optimization_landscape,
    plot_optimization_profile,
    plot_single_ended_convergence,
    plot_single_ended_profile,
    project_landscape_path,
    render_endpoint_strip,
    save_landscape_figure,
)


@pytest.fixture
def synth_landscape_data():
    """Synthetic RMSD + energy data for landscape tests."""
    rng = np.random.default_rng(42)
    n = 25
    rmsd_a = np.linspace(0.1, 3.0, n) + rng.normal(0, 0.05, n)
    rmsd_b = np.linspace(3.0, 0.1, n) + rng.normal(0, 0.05, n)
    energies = -np.cos(np.linspace(0, np.pi, n)) + rng.normal(0, 0.1, n)
    grad_a = np.gradient(energies) * 0.5
    grad_b = -np.gradient(energies) * 0.5
    return rmsd_a, rmsd_b, grad_a, grad_b, energies


@pytest.fixture
def synth_convergence_df():
    """Synthetic iteration DataFrame."""
    return pl.DataFrame(
        {
            "iteration": list(range(10)),
            "step_size": [0.1 * (0.9**i) for i in range(10)],
            "convergence": [1.0 * (0.7**i) for i in range(10)],
            "energy": [-10.0 - 0.5 * i for i in range(10)],
        }
    )


class TestPlotOptimizationProfile:
    def test_basic_energy(self):
        fig, ax = plt.subplots()
        iters = np.arange(10)
        energies = np.random.default_rng(42).standard_normal(10)
        plot_optimization_profile(ax, iters, energies)
        assert len(ax.lines) == 1
        assert ax.get_xlabel() == "Iteration"
        assert ax.get_ylabel() == "Energy (eV)"
        plt.close(fig)

    def test_energy_unit_conversion(self):
        fig, ax = plt.subplots()
        iters = np.arange(3)
        energies = np.array([0.0, 1.0, 2.0])
        plot_optimization_profile(ax, iters, energies, energy_unit="kcal/mol")
        assert ax.get_ylabel() == "Energy (kcal/mol)"
        np.testing.assert_allclose(
            ax.lines[0].get_ydata(),
            np.array([0.0, 23.06054783, 46.12109566]),
            rtol=1e-6,
        )
        plt.close(fig)

    def test_with_eigenvalues(self):
        fig, (ax, ax_ev) = plt.subplots(1, 2)
        iters = np.arange(8)
        energies = np.linspace(-1, 0.5, 8)
        eigenvals = np.linspace(-0.5, -0.01, 8)
        plot_optimization_profile(
            ax, iters, energies, eigenvalues=eigenvals, ax_eigen=ax_ev
        )
        assert len(ax.lines) == 1
        assert len(ax_ev.lines) >= 2  # data + hline
        plt.close(fig)

    def test_eigenvalue_unit_conversion(self):
        fig, (ax, ax_ev) = plt.subplots(1, 2)
        iters = np.arange(2)
        plot_optimization_profile(
            ax,
            iters,
            np.array([0.0, 1.0]),
            eigenvalues=np.array([0.0, 1.0]),
            ax_eigen=ax_ev,
            energy_unit="kJ/mol",
        )
        assert ax_ev.get_ylabel() == "Eigenvalue (kJ/mol/$\\AA^2$)"
        np.testing.assert_allclose(
            ax_ev.lines[0].get_ydata(),
            np.array([0.0, 96.48533212]),
            rtol=1e-6,
        )
        plt.close(fig)

    def test_eigenvalues_without_ax_ignored(self):
        fig, ax = plt.subplots()
        iters = np.arange(5)
        plot_optimization_profile(
            ax, iters, np.zeros(5), eigenvalues=np.zeros(5), ax_eigen=None
        )
        assert len(ax.lines) == 1
        plt.close(fig)

    def test_custom_colors(self):
        fig, (ax, ax_ev) = plt.subplots(1, 2)
        iters = np.arange(5)
        plot_optimization_profile(
            ax,
            iters,
            np.zeros(5),
            eigenvalues=np.zeros(5),
            ax_eigen=ax_ev,
            color="red",
            eigen_color="blue",
        )
        plt.close(fig)


try:
    import jax

    _HAS_JAX = True
except ImportError:
    _HAS_JAX = False


@pytest.mark.skipif(not _HAS_JAX, reason="jax required for surface models")
class TestPlotOptimizationLandscape:
    def test_projected(self, synth_landscape_data):
        fig, ax = plt.subplots()
        rmsd_a, rmsd_b, grad_a, grad_b, z = synth_landscape_data
        plot_optimization_landscape(
            ax,
            rmsd_a,
            rmsd_b,
            grad_a,
            grad_b,
            z,
            project_path=True,
            label_mode="optimization",
        )
        assert "Optimization" in ax.get_xlabel()
        assert "Lateral" in ax.get_ylabel()
        plt.close(fig)

    def test_unprojected(self, synth_landscape_data):
        fig, ax = plt.subplots()
        rmsd_a, rmsd_b, grad_a, grad_b, z = synth_landscape_data
        plot_optimization_landscape(
            ax,
            rmsd_a,
            rmsd_b,
            grad_a,
            grad_b,
            z,
            project_path=False,
        )
        assert "ref A" in ax.get_xlabel()
        plt.close(fig)

    def test_reaction_labels(self, synth_landscape_data):
        fig, ax = plt.subplots()
        rmsd_a, rmsd_b, grad_a, grad_b, z = synth_landscape_data
        plot_optimization_landscape(
            ax,
            rmsd_a,
            rmsd_b,
            grad_a,
            grad_b,
            z,
            label_mode="reaction",
        )
        assert "Reaction" in ax.get_xlabel()
        plt.close(fig)

    def test_surface_contourf_uses_full_finite_range(self, monkeypatch):
        fig, ax = plt.subplots()
        rmsd_a = np.array([0.0, 1.0, 2.0, 3.0])
        rmsd_b = np.array([3.0, 2.0, 1.0, 0.0])
        grad_a = np.zeros_like(rmsd_a)
        grad_b = np.zeros_like(rmsd_b)
        z = np.array([0.0, 1.0, 2.0, 3.0])
        contourf_calls = []

        original_contourf = Axes.contourf

        def recording_contourf(self, *args, **kwargs):
            contourf_calls.append((args, kwargs))
            return original_contourf(self, *args, **kwargs)

        monkeypatch.setattr(Axes, "contourf", recording_contourf)

        plot_optimization_landscape(
            ax,
            rmsd_a,
            rmsd_b,
            grad_a,
            grad_b,
            z,
            project_path=True,
            method="rbf",
        )

        surface_call = next(
            kwargs for _, kwargs in contourf_calls if kwargs.get("zorder") == 10
        )
        levels = np.asarray(surface_call["levels"])

        assert surface_call["extend"] == "both"
        assert np.isfinite(levels).all()
        assert levels[0] <= np.nanmin(z)
        assert levels[-1] >= np.nanmax(z)
        plt.close(fig)


class TestPlotConvergencePanel:
    def test_basic(self, synth_convergence_df):
        fig, (ax_f, ax_s) = plt.subplots(1, 2)
        plot_convergence_panel(ax_f, ax_s, synth_convergence_df)
        assert ax_f.get_ylabel() == "Convergence"
        assert ax_s.get_ylabel() == "Step size"
        plt.close(fig)

    def test_custom_columns(self):
        df = pl.DataFrame(
            {
                "it": [0, 1, 2],
                "frc": [1.0, 0.5, 0.1],
                "stp": [0.1, 0.08, 0.05],
            }
        )
        fig, (ax_f, ax_s) = plt.subplots(1, 2)
        plot_convergence_panel(
            ax_f,
            ax_s,
            df,
            force_col="frc",
            step_col="stp",
            iter_col="it",
        )
        plt.close(fig)


class TestSingleEndedHelpers:
    def test_project_landscape_path_reuses_basis(self, monkeypatch):
        calls = []

        def _fake_compute_projection_basis(*_args):
            calls.append("compute")
            return "basis"

        def _fake_project_to_sd(_a, _b, basis):
            calls.append(("project", basis))
            return [1.0, 2.0], [3.0, 4.0]

        monkeypatch.setattr(
            "chemparseplot.plot.optimization.compute_projection_basis",
            _fake_compute_projection_basis,
        )
        monkeypatch.setattr(
            "chemparseplot.plot.optimization.project_to_sd",
            _fake_project_to_sd,
        )

        x, y, basis = project_landscape_path([0.0, 1.0], [1.0, 0.0], project_path=True)
        assert basis == "basis"
        assert calls == ["compute", ("project", "basis")]

        calls.clear()
        x2, y2, basis2 = project_landscape_path(
            [0.0, 1.0], [1.0, 0.0], project_path=True, basis="reused"
        )
        assert basis2 == "reused"
        assert calls == [("project", "reused")]
        assert x == x2 and y == y2

    def test_save_landscape_figure_skips_tight_layout_with_strip(
        self, monkeypatch, tmp_path
    ):
        class _FakeFigure:
            def __init__(self):
                self.tight_layout_calls = 0
                self.saved = []

            def tight_layout(self):
                self.tight_layout_calls += 1

            def savefig(self, *args, **kwargs):
                self.saved.append((args, kwargs))

        fake_fig = _FakeFigure()
        monkeypatch.setattr(
            "chemparseplot.plot.optimization.plt.close", lambda _fig: None
        )

        save_landscape_figure(fake_fig, tmp_path / "strip.pdf", dpi=100, has_strip=True)
        assert fake_fig.tight_layout_calls == 0
        assert "bbox_inches" not in fake_fig.saved[0][1]

        fake_fig = _FakeFigure()
        monkeypatch.setattr(
            "chemparseplot.plot.optimization.plt.close", lambda _fig: None
        )
        save_landscape_figure(fake_fig, tmp_path / "plain.pdf", dpi=100, has_strip=False)
        assert fake_fig.tight_layout_calls == 1
        assert fake_fig.saved[0][1]["bbox_inches"] == "tight"

    def test_plot_single_ended_profile_handles_optional_eigen_column(
        self, monkeypatch, tmp_path
    ):
        class _Column:
            def __init__(self, values):
                self._values = np.asarray(values)

            def to_numpy(self):
                return self._values

        class _Frame:
            def __init__(self, columns):
                self._columns = {key: _Column(val) for key, val in columns.items()}

            @property
            def columns(self):
                return list(self._columns)

            def __getitem__(self, key):
                return self._columns[key]

        trajs = [
            type(
                "Traj",
                (),
                {"dat_df": _Frame({"iteration": [0, 1], "delta_e": [0.0, 1.0]})},
            )(),
            type(
                "Traj",
                (),
                {
                    "dat_df": _Frame(
                        {
                            "iteration": [0, 1],
                            "delta_e": [0.0, 1.0],
                            "eigenvalue": [-1.0, -0.5],
                        }
                    )
                },
            )(),
        ]

        called = {}

        def _fake_save(fig, output, *, dpi):
            called["axes"] = len(fig.axes)

        monkeypatch.setattr(
            "chemparseplot.plot.optimization.save_standard_figure", _fake_save
        )
        plot_single_ended_profile(
            trajs,
            ["a", "b"],
            tmp_path / "profile.pdf",
            100,
            energy_unit="eV",
            energy_column="delta_e",
            title="Energy vs Iteration",
            eigen_column="eigenvalue",
        )
        assert called["axes"] == 2

    def test_plot_single_ended_convergence_adds_overlay_legend(
        self, monkeypatch, tmp_path
    ):
        calls = []

        def _fake_panel(ax_force, ax_step, dat_df, *, color):
            calls.append(color)

        monkeypatch.setattr(
            "chemparseplot.plot.optimization.plot_convergence_panel", _fake_panel
        )
        monkeypatch.setattr(
            "chemparseplot.plot.optimization.save_standard_figure",
            lambda fig, output, *, dpi: None,
        )

        trajs = [
            type("Traj", (), {"dat_df": object()})(),
            type("Traj", (), {"dat_df": object()})(),
        ]
        plot_single_ended_convergence(trajs, ["one", "two"], tmp_path / "conv.pdf", 100)
        assert len(calls) == 2

    def test_create_landscape_axes_returns_optional_strip_axis(self):
        fig, ax, ax_strip = create_landscape_axes(dpi=100, has_strip=True, theme=None)
        assert ax is not None
        assert ax_strip is not None
        assert fig.get_size_inches()[1] == pytest.approx(5.37 + 1.20)
        plt.close(fig)

    def test_enforce_strip_clearance_uses_measured_text_extents(self):
        fig, ax, ax_strip = create_landscape_axes(dpi=100, has_strip=True, theme=None)
        ax.set_xlabel("A deliberately oversized x label", fontsize=26)
        ax.set_xticks([0.0, 1.0, 2.0])
        ax.set_xticklabels(["0.00", "1.00", "2.00"], fontsize=18)
        fig.canvas.draw()

        before = ax_strip.get_window_extent(fig.canvas.get_renderer()).y1
        enforce_strip_clearance(fig, ax, ax_strip, min_clearance_px=48.0)
        fig.canvas.draw()

        renderer = fig.canvas.get_renderer()
        strip_top = ax_strip.get_window_extent(renderer).y1
        label_bottom = ax.xaxis.label.get_window_extent(renderer).y0
        tick_bottom = min(
            tick.get_window_extent(renderer).y0
            for tick in ax.get_xticklabels()
            if tick.get_text()
        )
        assert strip_top < before
        assert min(label_bottom, tick_bottom) - strip_top >= 47.5
        plt.close(fig)

    def test_annotate_endpoint_adds_text(self):
        fig, ax = plt.subplots()
        annotate_endpoint(ax, 0.0, 1.0, "X", boxed=True)
        assert ax.texts[0].get_text() == "X"
        plt.close(fig)

    def test_default_strip_zoom_has_floor(self):
        class _FakeAtoms:
            def __len__(self):
                return 500

        assert default_strip_zoom([_FakeAtoms()]) >= 0.14

    def test_default_strip_zoom_is_gentler_for_small_structures(self):
        class _FakeAtoms:
            def __len__(self):
                return 20

        assert default_strip_zoom([_FakeAtoms()]) == pytest.approx(0.38)

    def test_overlay_palette_is_nonempty(self):
        assert OVERLAY_COLORS


class TestPlotDimerModeEvolution:
    def test_basic(self):
        rng = np.random.default_rng(42)
        modes = [rng.standard_normal((5, 3)) for _ in range(10)]
        # Last mode is the reference
        fig, ax = plt.subplots()
        plot_dimer_mode_evolution(ax, modes)
        assert len(ax.lines) >= 2  # data + hline
        assert ax.get_ylim()[1] <= 1.1
        plt.close(fig)

    def test_single_mode_skips(self):
        fig, ax = plt.subplots()
        plot_dimer_mode_evolution(ax, [np.ones((3, 3))])
        assert len(ax.lines) == 0  # skipped
        plt.close(fig)

    def test_zero_norm_handled(self):
        fig, ax = plt.subplots()
        modes = [np.zeros((3, 3)), np.ones((3, 3))]
        plot_dimer_mode_evolution(ax, modes)
        plt.close(fig)

    def test_aligned_modes(self):
        """Identical modes should give cos=1 for all."""
        mode = np.array([[1.0, 0, 0], [0, 1, 0]])
        fig, ax = plt.subplots()
        plot_dimer_mode_evolution(ax, [mode, mode, mode])
        plt.close(fig)


class TestLabelsDict:
    def test_keys(self):
        assert "reaction" in _LABELS
        assert "optimization" in _LABELS
        for k in _LABELS:
            assert "x" in _LABELS[k]
            assert "y" in _LABELS[k]
