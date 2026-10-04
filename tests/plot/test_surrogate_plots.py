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
    legend = ax.get_legend() or ax.figure.legends[0]
    return [t.get_text() for t in legend.get_texts()]


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
    assert ax.get_ylabel().startswith("energy relative to reactant")
    assert surr.plot_search_convergence(s.search).axes[0].get_yscale() == "log"


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


def _unknown_font(tmp_path, family="ZetaTestFace"):
    """A copy of DejaVu Sans renamed so matplotlib does not know the family."""
    import matplotlib.font_manager as fm
    from fontTools.ttLib import TTFont

    font = TTFont(fm.findfont("DejaVu Sans"))
    for rec in font["name"].names:
        if rec.nameID in {1, 4, 16}:
            rec.string = family
        elif rec.nameID == 6:
            rec.string = family + "-Regular"
    fonts = tmp_path / "fonts"
    fonts.mkdir()
    font.save(fonts / f"{family}-Regular.ttf")
    return fonts


def test_font_family_is_registered_and_embedded_in_pdf(tmp_path):
    fonts = _unknown_font(tmp_path)
    assert "ZetaTestFace" not in {f.name for f in surr.font_manager.fontManager.ttflist}
    surr.set_font("ZetaTestFace", [fonts])
    try:
        fig = surr.plot_pop_efficiencies(["1x1", "2x2"], {r"load $\sigma$": [1.0, 0.8]})
        assert fig.axes[0].xaxis.label.get_fontfamily() == ["ZetaTestFace"]
        out = tmp_path / "f.pdf"
        from chemparseplot.plot.provenance import save_with_provenance

        save_with_provenance(fig, out, {}, sidecar=False)
        names = _embedded(out)
        assert names and all(n.startswith("ZetaTestFace") for n in names), names
    finally:
        surr.set_font(None)


def test_missing_font_is_an_error_not_a_fallback():
    with pytest.raises(ValueError, match="No Such Family 123.*searched.*fonts"):
        surr.set_font("No Such Family 123", ["/nonexistent/dir"])
    assert surr._FONT[0] is None


def test_time_breakdown_stacks_components_and_remainder():
    from chemparseplot.parse.surrogate.gpr_optim import parse_breakdown_csv

    out = parse_breakdown_csv(SCALING / "breakdown.csv", ["band_oracle", "band_refit"])
    layouts, parts, total = out["c1"]
    assert layouts == ["1x1", "2x1", "4x2"]
    assert parts["band_oracle"].tolist() == pytest.approx([20.15, 12.15, 8.15])
    fig = surr.plot_time_breakdown(layouts, parts, total, title="c1")
    ax = fig.axes[0]
    assert [t.get_text() for t in ax.get_legend().get_texts()] == [
        "band_oracle",
        "band_refit",
        "other",
    ]
    heights = [p.get_height() for p in ax.patches]
    assert sum(heights[:3]) == pytest.approx(sum(parts["band_oracle"]))
    # components plus remainder reach the total at every layout
    tops = [sum(heights[i::3]) for i in range(3)]
    assert tops == pytest.approx(list(total))
    assert ax.get_ylabel() == "wall time (s)"


CASES = SCALING.parent / "cases"
_BANNED = ("retained", "force point", "upper bound", "force threshold only")


def _texts(fig):
    out = [t.get_text() for t in fig.findobj(mpl.text.Text) if t.get_text()]
    for ax in fig.axes:
        out += [t.get_text() for t in ax.get_yticklabels()]
    return out


def test_cases_calls_stacked_bars_titles_and_legend():
    from chemparseplot.parse.surrogate import attach_baselines, parse_cases_csv

    boards = parse_cases_csv(CASES / "cases.csv")
    attach_baselines(boards, CASES / "baselines.csv")
    assert [b.name for b in boards] == ["Baker-Chan", "Birkholz-Schlegel"]
    fig = surr.plot_cases_calls(boards)
    a, b = fig.axes[:2]
    assert a.get_title(loc="left") == "Baker-Chan: 3/3 first-order saddles"
    assert b.get_title(loc="left") == "Birkholz-Schlegel: 1/2 first-order saddles"
    assert [t.get_text() for t in a.get_yticklabels()] == ["HCCH", "HCN", "H2CO"]
    assert a.get_xlabel() == "Oracle evaluations" and a.get_xlim()[0] == 0
    assert a.get_xscale() == "linear"
    legend = [t.get_text() for t in fig.legends[0].get_texts()]
    assert legend == [
        "search",
        "independent validation",
        "not certified",
        "ASE ML-NEB",
        "FSM + TS",
    ]
    assert "65" in [t.get_text() for t in a.texts]  # 39 + 26 printed at the bar end
    # the unconverged baseline marker is open
    open_markers = [ln for ln in a.lines if ln.get_markerfacecolor() == "white"]
    assert len(open_markers) == 1
    assert a.get_ylim()[0] > a.get_ylim()[1]  # file order, top to bottom
    assert not [w for w in _BANNED if any(w in t for t in _texts(fig))]


def test_cases_wall_log_axis_reference_tick_and_labels():
    from chemparseplot.parse.surrogate import parse_cases_csv

    boards = parse_cases_csv(CASES / "cases.csv")
    fig = surr.plot_cases_wall(boards, reference_label="earlier record")
    a, b = fig.axes[:2]
    assert a.get_xscale() == "log" and a.get_xlabel() == "Wall time (s)"
    assert len(a.collections) == 2  # two reference ticks on Baker rows with a value
    assert [t.get_text() for t in fig.legends[0].get_texts()] == [
        "earlier record",
        "not certified",
    ]
    # the value is printed inside the bar end, never beyond the reference tick
    h2co = next(x for x in a.texts if x.get_text() == "26")
    assert h2co.xy[0] == pytest.approx(26.0) and h2co.get_ha() == "right"
    assert h2co.get_color() == "white"
    assert not [w for w in _BANNED if any(w in t for t in _texts(fig))]


def test_cases_csv_errors(tmp_path):
    from chemparseplot.parse.surrogate import attach_baselines, parse_cases_csv

    bad = tmp_path / "b.csv"
    bad.write_text("board,case\nx,y\n")
    with pytest.raises(ValueError, match="lacks column"):
        parse_cases_csv(bad)
    boards = parse_cases_csv(CASES / "cases.csv")
    base = tmp_path / "base.csv"
    base.write_text("board,case,method,calls,converged\nBaker-Chan,99_nope,M,1,true\n")
    with pytest.raises(ValueError, match="unknown case"):
        attach_baselines(boards, base)


def test_board_panels_are_sized_by_rows_and_top_aligned():
    from chemparseplot.parse.surrogate import parse_cases_csv

    boards = parse_cases_csv(CASES / "cases.csv")  # 3 and 2 rows
    for fig in (surr.plot_cases_calls(boards), surr.plot_cases_wall(boards)):
        a, b = fig.axes[:2]
        ha, hb = a.get_position().height, b.get_position().height
        assert ha / hb == pytest.approx(3 / 2)
        assert a.get_position().y1 == pytest.approx(b.get_position().y1)  # top-aligned
        # same row pitch: bar thickness per row is identical in both panels
        for ax in (a, b):
            lo, hi = ax.get_ylim()
            assert lo - hi == pytest.approx(len(ax.get_yticklabels()))


def test_wall_axis_range_and_minor_decade_labels():
    from chemparseplot.parse.surrogate import parse_cases_csv

    boards = parse_cases_csv(CASES / "cases.csv")  # 6.0 s .. 1200 s (tick)
    ax = surr.plot_cases_wall(boards).axes[0]
    lo, hi = ax.get_xlim()
    assert lo == pytest.approx(6.0 * 10**-0.5) and hi == pytest.approx(
        1200 * 10 ** (1 / 3)
    )
    ax.figure.canvas.draw()
    shown = {
        t.get_text()
        for t, v in zip(
            ax.get_xticklabels(which="minor"), ax.get_xticks(minor=True), strict=True
        )
        if t.get_text() and lo <= v <= hi
    }
    assert shown == {"3", "30", "300"}


def test_wall_value_follows_the_bar_without_a_reference():
    from chemparseplot.parse.surrogate import parse_cases_csv

    boards = parse_cases_csv(CASES / "cases.csv")
    ax = surr.plot_cases_wall(boards).axes[0]
    hcn = next(t for t in ax.texts if t.get_text() == "6.0")  # HCN has no reference
    assert hcn.get_ha() == "left" and hcn.get_color() == "black"
    h2co = next(t for t in ax.texts if t.get_text() == "26")
    assert h2co.get_ha() == "right"


def test_calls_prints_search_counts_only_where_they_fit():
    from chemparseplot.parse.surrogate import CaseBoard, CaseRow

    rows = [
        CaseRow("a", "Wide", 400, 20, 5.0, True),
        CaseRow("b", "Narrow", 4, 20, 5.0, True),
    ]
    a = surr.plot_cases_calls([CaseBoard("B", rows)]).axes[0]
    inside = {t.get_text() for t in a.texts if t.get_color() == "white"}
    assert inside == {"400"}  # the search segment of 4 cannot hold its number


def test_profile_legend_says_measured_and_acquisition_legend_clears_the_axis(baker):
    ax = surr.plot_band_profile(baker.band).axes[0]
    assert any(t.startswith("oracle evaluations (") for t in _labels(ax))
    assert not any("retained" in t for t in _labels(ax))
    fig = surr.plot_band_evolution(baker.band)
    assert fig.legends and fig.axes[0].get_legend() is None


def test_profile_observation_modes_state_distance_honestly(baker):
    pts = baker.band.points
    assert pts.distance.shape == pts.energy.shape and (pts.distance >= 0).all()
    fade = surr.plot_band_profile(baker.band, observations="fade")
    labels = _labels(fade.axes[0])
    assert any(
        t.startswith("oracle evaluations (")
        and t.endswith("projected onto the final path")
        for t in labels
    )
    assert len(fade.axes) == 2  # colourbar for the distance
    assert "distance from the final path" in fade.axes[1].get_ylabel()
    near = surr.plot_band_profile(
        baker.band, observations="near", observation_distance=0.05
    )
    text = next(t for t in _labels(near.axes[0]) if t.startswith("oracle evaluations"))
    assert "farther than 0.05 \u00c5 omitted" in text and len(near.axes) == 1
    none = surr.plot_band_profile(baker.band, observations="none")
    assert not any(t.startswith("oracle evaluations") for t in _labels(none.axes[0]))
    assert none.axes[0].get_ylabel() == "energy relative to reactant (eV)"


def _jax_surfaces():
    pytest.importorskip("jax")
    pytest.importorskip("rgpycrumbs.surfaces")


def test_landscape_uses_the_shared_neb_functions_and_labels(baker, monkeypatch):
    _jax_surfaces()
    from chemparseplot.plot import neb as neb_plot

    calls = []
    for name in (
        "plot_landscape_surface",
        "plot_landscape_path_overlay",
        "mark_saddle_point",
    ):
        orig = getattr(neb_plot, name)

        def wrap(*a, _o=orig, _n=name, **k):
            calls.append(_n)
            return _o(*a, **k)

        monkeypatch.setattr(neb_plot, name, wrap)
    fig = surr.plot_reduced_landscape(baker.band)
    assert calls == [
        "plot_landscape_surface",
        "plot_landscape_path_overlay",
        "mark_saddle_point",
    ]
    ax = fig.axes[0]
    # the same axis labels and equal-metric square window as plt-neb
    assert ax.get_xlabel() == r"Reaction progress $s$ ($\AA$)"
    assert ax.get_ylabel() == r"Orthogonal deviation $d$ ($\AA$)"
    xs, ys = ax.get_xlim(), ax.get_ylim()
    assert xs[1] - xs[0] == pytest.approx(ys[1] - ys[0])
    assert [t.get_text() for t in fig.legends[0].get_texts()] == [
        "energy surface: GP fitted afresh to the oracle energies and "
        "in-plane gradients, not the search's model",
        "relative variance contours (0 at the data, 1 far from it)",
        "faded: relative variance above 0.95, no oracle evaluation nearby",
        "final path (coloured by surrogate energy)",
        "oracle evaluations (fill: true energy, as the colourbar)",
        "climbing image",
        "certified saddle",
    ]
    assert fig.axes[1].get_ylabel() == "energy relative to reactant (eV)"


def test_landscape_has_no_unexplained_marks(baker):
    _jax_surfaces()
    fig = surr.plot_reduced_landscape(baker.band)
    # without label_every the only text on the axes is the variance contour labels
    assert all(t.get_text().startswith("relative variance ") for t in fig.axes[0].texts)
    numbered = surr.plot_reduced_landscape(baker.band, label_every=3)
    texts = [
        t.get_text()
        for t in numbered.axes[0].texts
        if not t.get_text().startswith("relative variance ")
    ]
    assert texts and all(t.isdigit() for t in texts)
    entry = numbered.legends[0].get_texts()[4].get_text()
    assert "numbers: order of evaluation" in entry


def test_landscape_colouring_surface_and_saddle_options(baker):
    _jax_surfaces()
    by_iter = surr.plot_reduced_landscape(baker.band, color_by="iteration")
    assert by_iter.axes[-1].get_ylabel() == "order of evaluation"
    entry = by_iter.legends[0].get_texts()[4].get_text()
    assert "shade: order of evaluation" in entry
    flat = surr.plot_reduced_landscape(baker.band, surface=None)
    assert not any("surface" in t.get_text() for t in flat.legends[0].get_texts())
    saved = baker.band.saddle_certified
    baker.band.saddle_certified = False
    try:
        un = surr.plot_reduced_landscape(baker.band, surface=None)
        names = [t.get_text() for t in un.legends[0].get_texts()]
        assert "reported saddle (not certified)" in names
    finally:
        baker.band.saddle_certified = saved


def test_landscape_rmsd_gradients_recover_a_known_plane():
    ref_a, ref_b = np.zeros(6), np.array([2.0, 0, 0, 0, 0, 0])
    x = np.array([[1.0, 0.5, 0, 0, 0, 0], [0.5, 1.0, 0, 0, 0, 0]])
    root = np.sqrt(2)  # E = 3 a + b through the chain rule: the fit returns (3, 1)
    g = []
    for xi in x:
        da, db = xi - ref_a, xi - ref_b
        g.append(3 * da / (root * np.linalg.norm(da)) + db / (root * np.linalg.norm(db)))
    ea, eb = surr.rmsd_gradients(x, np.array(g), ref_a, ref_b)
    assert ea == pytest.approx([3, 3]) and eb == pytest.approx([1, 1])


def test_landscape_pdf_embeds_only_the_requested_family(baker, tmp_path):
    _jax_surfaces()
    from chemparseplot.plot.provenance import save_with_provenance

    fonts = _unknown_font(tmp_path, "ZetaLandscape")
    surr.set_font("ZetaLandscape", [fonts])
    try:
        fig = surr.plot_reduced_landscape(baker.band, title="HCN", label_every=4)
        out = tmp_path / "land.pdf"
        save_with_provenance(fig, out, {}, sidecar=False)
    finally:
        surr.set_font(None)
    names = _embedded(out)
    assert names and all(n.startswith("ZetaLandscape") for n in names), names


def test_landscape_fades_where_the_model_has_no_data(baker):
    _jax_surfaces()
    faded = surr.plot_reduced_landscape(baker.band)
    plain = surr.plot_reduced_landscape(baker.band, fade_variance=None)
    names = [t.get_text() for t in plain.legends[0].get_texts()]
    assert not any(n.startswith("faded") for n in names)
    # the fade is one more filled contour set, drawn white over the surface
    assert len(faded.axes[0].collections) > len(plain.axes[0].collections)
    white = [
        c
        for c in faded.axes[0].collections
        if getattr(c, "get_alpha", lambda: None)() == pytest.approx(0.78)
    ]
    assert white and tuple(white[0].get_facecolor()[0][:3]) == (1.0, 1.0, 1.0)
