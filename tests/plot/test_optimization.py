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

from chemparseplot.plot.optimization import (
    _LABELS,
    plot_convergence_panel,
    plot_dimer_mode_evolution,
    plot_optimization_landscape,
    plot_optimization_profile,
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
    return pl.DataFrame({
        "iteration": list(range(10)),
        "step_size": [0.1 * (0.9 ** i) for i in range(10)],
        "convergence": [1.0 * (0.7 ** i) for i in range(10)],
        "energy": [-10.0 - 0.5 * i for i in range(10)],
    })


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
            ax, iters, np.zeros(5),
            eigenvalues=np.zeros(5), ax_eigen=ax_ev,
            color="red", eigen_color="blue",
        )
        plt.close(fig)


try:
    import jax  # noqa: F401
    _HAS_JAX = True
except ImportError:
    _HAS_JAX = False


@pytest.mark.skipif(not _HAS_JAX, reason="jax required for surface models")
class TestPlotOptimizationLandscape:
    def test_projected(self, synth_landscape_data):
        fig, ax = plt.subplots()
        rmsd_a, rmsd_b, grad_a, grad_b, z = synth_landscape_data
        cb = plot_optimization_landscape(
            ax, rmsd_a, rmsd_b, grad_a, grad_b, z,
            project_path=True, label_mode="optimization",
        )
        assert "Optimization" in ax.get_xlabel()
        assert "Lateral" in ax.get_ylabel()
        plt.close(fig)

    def test_unprojected(self, synth_landscape_data):
        fig, ax = plt.subplots()
        rmsd_a, rmsd_b, grad_a, grad_b, z = synth_landscape_data
        cb = plot_optimization_landscape(
            ax, rmsd_a, rmsd_b, grad_a, grad_b, z,
            project_path=False,
        )
        assert "ref A" in ax.get_xlabel()
        plt.close(fig)

    def test_reaction_labels(self, synth_landscape_data):
        fig, ax = plt.subplots()
        rmsd_a, rmsd_b, grad_a, grad_b, z = synth_landscape_data
        plot_optimization_landscape(
            ax, rmsd_a, rmsd_b, grad_a, grad_b, z,
            label_mode="reaction",
        )
        assert "Reaction" in ax.get_xlabel()
        plt.close(fig)


class TestPlotConvergencePanel:
    def test_basic(self, synth_convergence_df):
        fig, (ax_f, ax_s) = plt.subplots(1, 2)
        plot_convergence_panel(ax_f, ax_s, synth_convergence_df)
        assert ax_f.get_ylabel() == "Convergence"
        assert ax_s.get_ylabel() == "Step size"
        plt.close(fig)

    def test_custom_columns(self):
        df = pl.DataFrame({
            "it": [0, 1, 2],
            "frc": [1.0, 0.5, 0.1],
            "stp": [0.1, 0.08, 0.05],
        })
        fig, (ax_f, ax_s) = plt.subplots(1, 2)
        plot_convergence_panel(
            ax_f, ax_s, df,
            force_col="frc", step_col="stp", iter_col="it",
        )
        plt.close(fig)


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
