"""gpr_optim record parser against cells cut from a real campaign."""

from pathlib import Path

import numpy as np
import pytest

from chemparseplot.parse.surrogate import CampaignTable, SurrogateSearch
from chemparseplot.parse.surrogate.gpr_optim import (
    parse_gpr_optim_campaign,
    parse_gpr_optim_cell,
    parse_gprn_log,
    read_mlflow_metrics,
)

h5py = pytest.importorskip("h5py")

REC = Path(__file__).parent.parent / "fixtures" / "surrogate" / "record"
BAKER = REC / "baker" / "25_hcnh2"
OXIRANE = REC / "birkholz" / "16_oxirane"


def test_log_groups_tags_and_survives_interleaving():
    log = parse_gprn_log(BAKER / "band" / "gprn.log")
    assert {"gp_oie", "acq-pick", "ceiling"} <= set(log)
    outers = [d["outer"] for d in log["gp_oie"] if "max_gp_F" in d]
    assert outers[:3] == [0, 1, 2]
    last = log["acq-pick"][-1]
    assert last["outer"] == 5 and last["worst"] == 9  # cut at the interleaved tag


def test_mlflow_metrics_columns():
    run = next((REC / "mlruns").glob("*/*"))
    m = read_mlflow_metrics(run)
    assert set(m["dimer.force"]) == {"time_ms", "value", "step"}
    assert len(m["dimer.force"]["value"]) == len(m["dimer.oracle_calls"]["value"])


def test_two_ended_cell():
    s = parse_gpr_optim_cell(BAKER)
    assert isinstance(s, SurrogateSearch) and s.producer == "gpr_optim"
    assert s.cell.label == "25_hcnh2" and s.cell.search_calls == 58
    assert s.cell.index == 1 and s.cell.converged
    assert s.cell.ledger["initial"] == 18 and s.cell.ledger["total"] == 58
    h = s.search
    assert h.force_tolerance == pytest.approx(0.0514221)
    assert np.all(np.diff(h["outer"]) > 0)
    assert {"true_force", "surrogate_force", "n_train"} <= set(h.series)
    assert h.retained_cap == 128
    assert any(e.kind == "ceiling" for e in h.events)
    assert s.provenance["result.json"] and len(s.provenance["band/band.h5"]) == 64


def test_band_from_h5_marks_evaluated_images():
    b = parse_gpr_optim_cell(BAKER).band
    f = b.final
    assert f.n_images == 6 and f.climbing == 3
    assert f.coordinate[0] == 0 and np.all(np.diff(f.coordinate) > 0)
    assert f.evaluated.sum() == 3  # endpoints and one image coincide with rows
    assert np.all(np.isfinite(f.true_force[f.evaluated]))
    p = b.points
    assert len(p) == 13 and p.positions.shape == (13, 15)
    assert np.nanmin(p.coordinate) >= 0 and np.nanmax(p.coordinate) <= f.coordinate[-1]


def test_dimer_cell_from_mlflow():
    s = parse_gpr_optim_cell(OXIRANE)
    d = s.single_ended
    assert s.cell.method == "gprd" and s.cell.ledger["dimer"] == 846
    assert s.cell.ledger["total"] == 894
    assert d.oracle_calls[0] == 1 and np.all(np.diff(d.oracle_calls[:35]) > 0)
    assert d.curvature[1] < 0
    assert np.isfinite(d.curvature_measured).sum() == 2
    assert any(e.kind == "spectrum" for e in d.escape)


def test_campaign_table():
    t = parse_gpr_optim_campaign(REC)
    assert isinstance(t, CampaignTable)
    assert sorted(c.label for c in t.cells) == ["16_oxirane", "25_hcnh2"]
    assert t.tolerance == pytest.approx(0.0514221)
    assert t.to_frame().shape[0] == 2
