"""Surrogate-search figures: axes contents on fixture cells and synthetic tables."""

from pathlib import Path

import matplotlib as mpl
import numpy as np
import pytest

mpl.use("Agg")
import matplotlib.pyplot as plt

pytest.importorskip("h5py")

from chemparseplot.parse.surrogate import (
    BandHistory,
    BandSnapshot,
    CampaignTable,
    CellRecord,
    ScalingTable,
)
from chemparseplot.parse.surrogate.gpr_optim import (
    parse_gpr_optim_campaign,
    parse_gpr_optim_cell,
)
from chemparseplot.plot import surrogate as surr

REC = Path(__file__).parent.parent / "fixtures" / "surrogate" / "record"


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def baker():
    return parse_gpr_optim_cell(REC / "baker" / "25_hcnh2")


@pytest.fixture(scope="module")
def oxirane():
    return parse_gpr_optim_cell(REC / "birkholz" / "16_oxirane")


def _labels(ax):
    return [t.get_text() for t in ax.get_legend().get_texts()]


def test_band_profile_encodes_mean_truth_and_climbing_image(baker):
    fig = surr.plot_band_profile(baker.band)
    ax = fig.axes[0]
    labels = _labels(ax)
    assert "surrogate mean" in labels and "climbing image" in labels
    assert "true energy at image" in labels
    assert ax.get_xlabel().startswith("path coordinate")
    assert "eV" in ax.get_ylabel()
    # energies are relative to the reactant image
    assert ax.lines[0].get_ydata()[0] == pytest.approx(0.0)


def test_band_profile_ribbon_and_reference_only_when_recorded():
    x = np.linspace(0, 3, 5)
    e = np.array([0.0, 0.8, 1.5, 0.7, -0.2])
    snap = BandSnapshot(x, e, sigma=np.full(5, 0.1))
    ref = BandSnapshot(x, e + 0.05)
    ax = surr.plot_band_profile(BandHistory(final=snap, reference=ref)).axes[0]
    assert len(ax.collections) == 1  # ribbon
    assert any("true potential" in t for t in _labels(ax))
    ax2 = surr.plot_band_profile(BandHistory(final=BandSnapshot(x, e))).axes[0]
    assert not ax2.collections


def test_band_profile_energy_unit(baker):
    ax = surr.plot_band_profile(baker.band, energy_unit="kcal/mol").axes[0]
    assert "kcal/mol" in ax.get_ylabel()


def test_search_convergence_log_force_tolerance_and_rug(baker):
    fig = surr.plot_search_convergence(baker.search)
    ax = fig.axes[0]
    assert ax.get_yscale() == "log"
    assert ax.get_ylabel().startswith("max atomic force")
    tol = [
        ln
        for ln in ax.lines
        if ln.get_linestyle() == "--" and len(set(ln.get_ydata())) == 1
    ]
    assert tol and tol[0].get_ydata()[0] == pytest.approx(0.0514221)
    assert len(fig.axes) == 2  # training-size panel
    legend_texts = [t.get_text() for t in fig.legends[0].get_texts()]
    assert "true force, evaluated images" in legend_texts
    assert any("acquisition" in t for t in legend_texts)


def test_search_convergence_flags_calls_outside_the_counter(baker):
    baker.search.meta["total_calls"] = 10_000
    try:
        ax = surr.plot_search_convergence(baker.search).axes[0]
        assert any("reports 10000 calls" in t.get_text() for t in ax.texts)
    finally:
        baker.search.meta["total_calls"] = baker.cell.search_calls


def test_comparison_has_ledger_bars(baker):
    fig = surr.plot_search_comparison([baker, baker], ["a", "b"])
    ledger = fig.axes[2]
    assert len(ledger.patches) >= 2
    assert ledger.get_xlabel().startswith("oracle calls")


def test_model_diagnostics_panels(baker):
    fig = surr.plot_model_diagnostics(baker.search)
    assert len(fig.axes) == 4
    assert fig.axes[0].get_yscale() == "log"


def test_band_evolution_falls_back_to_acquisition_record(baker):
    fig = surr.plot_band_evolution(baker.band)
    assert fig.axes[0].get_ylabel().startswith("image the oracle")


def test_band_evolution_snapshots():
    snaps = [
        BandSnapshot(
            np.linspace(0, 1, 4),
            np.array([0, 1, 2, 0.0]) * (1 + k),
            outer=k,
            oracle_calls=10 * k,
        )
        for k in range(6)
    ]
    fig = surr.plot_band_evolution(BandHistory(final=snaps[-1], snapshots=snaps))
    visible = [a for a in fig.axes if a.get_visible()]
    assert len(visible) == 6


def test_single_ended(oxirane):
    fig = surr.plot_single_ended(oxirane.single_ended)
    top, bottom = fig.axes
    assert bottom.get_yscale() == "log"
    assert "measured curvature" in _labels(top)
    assert any(ln.get_linestyle() == ":" for ln in top.lines) or top.collections


def _table():
    cells = [
        CellRecord(
            "a",
            search_calls=40,
            converged=True,
            passed=True,
            index=1,
            energy_delta=0.01,
            wall={"band": 3.0},
        ),
        CellRecord(
            "b",
            search_calls=90,
            converged=True,
            passed=False,
            index=2,
            energy_delta=0.2,
            wall={"band": 5.0, "dimer": 20.0},
        ),
        CellRecord("c", search_calls=15, converged=False, passed=False),
    ]
    return CampaignTable(
        "ours",
        cells,
        baselines={"ASE ML-NEB": {"a": 120, "b": 300}, "DFT-NEB": {"a": 400}},
    )


def test_campaign_dumbbell_sorted_log_and_failures():
    fig = surr.plot_campaign_calls(_table())
    ax = fig.axes[0]
    assert ax.get_xscale() == "log"
    assert [t.get_text() for t in ax.get_yticklabels()] == ["c", "a", "b"]
    labels = _labels(ax)
    assert {"ours", "ASE ML-NEB", "DFT-NEB", "did not pass"} <= set(labels)


def test_campaign_matrix_and_walls():
    fig = surr.plot_campaign_matrix(_table())
    assert len(fig.axes[0].lines) == 12
    walls = surr.plot_campaign_walls(_table())
    assert walls.axes[0].get_xlabel() == "wall time (s)"
    assert [t.get_text() for t in walls.axes[0].get_yticklabels()][-1] == "b"


def test_campaign_from_fixture_record():
    t = parse_gpr_optim_campaign(REC)
    fig = surr.plot_campaign_calls(t)
    assert len(fig.axes[0].get_yticklabels()) == 2


def test_scaling_and_efficiency():
    t = ScalingTable(
        {
            "ours": ([1, 2, 4, 8], [100.0, 55.0, 30.0, 18.0]),
            "other": ([1, 2, 4], [100.0, 80.0, 70.0]),
        },
        reference="ours",
    )
    ax = surr.plot_scaling(t).axes[0]
    assert ax.get_xscale() == "log" and ax.get_yscale() == "log"
    assert _labels(ax)[0] == "ideal"
    fig = surr.plot_efficiency_table(t)
    texts = [x.get_text() for x in fig.axes[0].texts]
    assert "1.00" in texts and len(texts) == 7
    w, sp = t.speedup("ours")
    assert sp[-1] == pytest.approx(100 / 18)


def test_plots_accept_mlneb_snapshots():
    from chemparseplot.parse.surrogate import read_search

    pytest.importorskip("ase")
    s = read_search(REC.parent / "mlneb", "ml-neb")
    fig = surr.plot_band_evolution(s.band)
    assert len([a for a in fig.axes if a.get_visible()]) == 3
    ax = surr.plot_band_profile(s.band).axes[0]
    assert ax.get_ylabel().startswith("energy above reactant")
    assert surr.plot_search_convergence(s.search).axes[0].get_yscale() == "log"


def test_reduced_landscape_projects_band_and_observations(baker):
    s, d = surr.reduced_coordinates(baker.band)[0]
    assert s[0] == pytest.approx(0.0) and s[-1] > 0 and d[0] == pytest.approx(0.0)
    fig = surr.plot_reduced_landscape(baker.band)
    ax = fig.axes[0]
    assert ax.get_xlabel().startswith("progress")
    assert len(fig.axes) == 2  # colorbar
    first = [
        c
        for c in ax.collections
        if hasattr(c, "get_offsets") and len(c.get_offsets()) == 13
    ]
    assert first, "one marker per retained observation"


def test_reduced_landscape_needs_geometries():
    snap = BandSnapshot(np.arange(3.0), np.zeros(3))
    with pytest.raises(ValueError, match="no geometries"):
        surr.reduced_coordinates(BandHistory(final=snap))


def test_speedup_defaults_to_each_series_own_baseline():
    t = ScalingTable({"a": ([2, 4], [10.0, 6.0]), "b": ([1, 2], [3.0, 2.0])})
    assert t.speedup("a")[1].tolist() == pytest.approx([1.0, 10 / 6])
    assert t.speedup("b")[1].tolist() == pytest.approx([1.0, 1.5])
    shared = ScalingTable(t.series, reference="a")
    assert shared.speedup("b")[1].tolist() == pytest.approx([10 / 3, 5.0])
    fig = surr.plot_efficiency_table(t)
    assert "1.00" in [x.get_text() for x in fig.axes[0].texts]
