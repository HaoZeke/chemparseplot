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


def test_scaling_bars_capped_markers_and_calls():
    t = ScalingTable(
        {"c": ([1, 2, 4], [10.0, 6.0, 4.0])},
        time_min={"c": np.array([9.0, 5.5, 3.5])},
        time_max={"c": np.array([11.0, 6.5, 4.5])},
        calls={"c": np.array([100.0, 100.0, 140.0])},
        capped={"c": np.array([False, False, True])},
    )
    lo, hi = t.speedup_range("c")
    assert lo[0] < 1.0 < hi[0]
    ax = surr.plot_scaling(t).axes[0]
    assert "c (capped)" in _labels(ax)
    assert not ax.texts  # call counts live in their own panel


SCALING = REC.parent / "scaling"


def test_strong_scaling_panels_from_csv():
    from chemparseplot.parse.surrogate.gpr_optim import parse_scaling_csv

    t = parse_scaling_csv(SCALING / "wall.csv", SCALING / "counts.csv")
    assert t.layouts["c1"] == ["1x1", "2x1", "1x2", "4x1", "2x2", "4x2"]
    assert t.calls["c1"].tolist() == [50, 50, 50, 52, 52, 90]
    assert t.capped["c1"].tolist() == [False] * 5 + [True]
    fig = surr.plot_strong_scaling(t, "c1")
    wall, speed, calls, per_call = fig.axes
    assert speed.get_xlabel().startswith("cores")  # straight ideal line on cores
    ideal = next(ln for ln in speed.lines if ln.get_linestyle() == ":")
    assert list(ideal.get_xdata()) == sorted(ideal.get_xdata())
    assert wall.get_yscale() == "log" and calls.get_ylabel().startswith("oracle calls")
    assert per_call.get_ylabel() == "seconds per call"
    assert [x.get_text() for x in calls.get_xticklabels()] == t.layouts["c1"]
    # seconds per call is wall over the search calls, so the call jump shows
    y = per_call.lines[0].get_ydata()
    assert y[-1] == pytest.approx(t.series["c1"][1][-1] / 90)


def test_every_cell_keeps_its_own_layouts(tmp_path):
    from chemparseplot.parse.surrogate.gpr_optim import parse_scaling_csv

    f = tmp_path / "w.csv"
    f.write_text(
        "cell,set,ranks,threads,repetition,stage,seconds,calls,partition\n"
        "a,s,1,1,1,pipeline,10,,p\na,s,2,2,1,pipeline,5,,p\n"
        "b,s,1,1,1,pipeline,10,,p\nb,s,4,1,1,pipeline,5,,p\n"
    )
    t = parse_scaling_csv(f)
    assert t.layouts == {"a": ["1x1", "2x2"], "b": ["1x1", "4x1"]}


def test_efficiency_table_has_a_row_per_layout():
    from chemparseplot.parse.surrogate.gpr_optim import parse_scaling_csv

    t = parse_scaling_csv(SCALING / "wall.csv", SCALING / "counts.csv")
    ax = surr.plot_efficiency_table(t).axes[0]
    rows = [x.get_text() for x in ax.get_yticklabels()]
    assert sorted(rows) == sorted(t.layouts["c1"]) and len(rows) == 6
    assert "1.00" in [x.get_text() for x in ax.texts]
    # the two 2-core layouts get different efficiencies
    sp = surr.plot_scaling(t).axes[0]
    assert "c1 (capped)" in _labels(sp)


def test_pop_efficiencies():
    fig = surr.plot_pop_efficiencies(
        ["1x1", "2x2"], {"load balance": [1.0, 0.8], "communication": [1.0, 0.9]}
    )
    ax = fig.axes[0]
    assert len(ax.lines) == 3 and ax.get_ylim()[0] == 0


def _embedded(pdf):
    import re

    names = re.findall(rb"/FontName\s*/(?:[A-Z]{6}\+)?([^\s/>\[\]]+)", pdf.read_bytes())
    return sorted({n.decode() for n in names})


def test_font_family_is_registered_and_embedded_in_pdf(tmp_path):
    import shutil

    import matplotlib.font_manager as fm

    face = Path(fm.findfont("DejaVu Serif"))
    fonts = tmp_path / "fonts"
    fonts.mkdir()
    shutil.copy(face, fonts / "DejaVuSerif.ttf")
    surr.set_font("DejaVu Serif", [fonts])
    try:
        fig = surr.plot_pop_efficiencies(["1x1", "2x2"], {r"load $\sigma$": [1.0, 0.8]})
        assert fig.axes[0].xaxis.label.get_fontfamily() == ["DejaVu Serif"]
        out = tmp_path / "f.pdf"
        from chemparseplot.plot.provenance import save_with_provenance

        save_with_provenance(fig, out, {}, sidecar=False)
        names = _embedded(out)
        assert names and all(n.startswith("DejaVuSerif") for n in names), names
    finally:
        surr.set_font(None)


def test_missing_font_is_an_error_not_a_fallback():
    with pytest.raises(ValueError, match="No Such Family 123.*searched.*fonts"):
        surr.set_font("No Such Family 123", ["/nonexistent/dir"])
    assert surr._FONT[0] is None
