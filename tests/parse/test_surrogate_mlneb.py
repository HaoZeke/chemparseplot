"""ML-NEB output fills the same records as the gpr_optim parser."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("ase")

from chemparseplot.parse.surrogate import SurrogateSearch, read_search
from chemparseplot.parse.surrogate.mlneb import read_pipe_table

RUN = Path(__file__).parent.parent / "fixtures" / "surrogate" / "mlneb"


def test_pipe_table_keeps_nan_and_dates():
    t = read_pipe_table(RUN / "ml_summary.txt")
    assert t["Step"].tolist() == [1, 2, 3, 4, 5]
    assert np.isnan(t["Uncertainty/[eV]"][0])
    assert t["Date"][0].startswith("12 ")


def test_band_snapshots_and_acquisitions():
    s = read_search(RUN, "ml-neb", force_tolerance=0.05)
    assert isinstance(s, SurrogateSearch) and s.producer == "ml-neb"
    b = s.band
    assert len(b.snapshots) == 3 and b.final.n_images == 8
    first = b.snapshots[0]
    assert first.oracle_calls == 3 and first.sigma is not None
    assert first.evaluated.sum() >= 2  # endpoints sit on images
    assert np.all(np.diff(first.coordinate) > 0)
    assert [e.outer for e in b.events] == [0, 1, 2]
    assert all(e.image >= 0 for e in b.events)  # each pick lies on a band image
    assert len(b.points) == 7


def test_search_series_and_ledger():
    s = read_search(RUN, "ml-neb", force_tolerance=0.05)
    h = s.search
    assert h["oracle_calls"].tolist() == [3, 4, 5, 6, 7]
    assert {"true_force", "energy", "time_refit", "n_train"} <= set(h.series)
    assert h.converged is False
    assert s.cell.ledger == {"endpoints": 2, "outer": 5, "total": 7}
    assert len(s.provenance["predicted.traj"]) == 64


def test_unknown_producer():
    with pytest.raises(ValueError, match="unknown producer"):
        read_search(RUN, "nope")
