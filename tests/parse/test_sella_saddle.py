from pathlib import Path

import pytest

from chemparseplot.parse.sella.saddle_search import (
    LogEnd,
    LogStart,
    SellaLog,
    _sella_loglist,
    parse_log_line,
    parse_sella_saddle,
)

try:
    from rgpycrumbs.basetypes import SpinID
except ImportError:  # pragma: no cover - test env should provide rgpycrumbs
    SpinID = None


pytestmark = pytest.mark.neb


def test_parse_log_line_returns_typed_records():
    start = parse_log_line("INFO START 00:00:01 -10.5 foo bar baz qux", "start")
    end = parse_log_line("INFO 12 00:00:03 -9.1 0.2 foo bar baz", "end")

    assert isinstance(start, LogStart)
    assert start == LogStart(tstart="00:00:01", init_energy=-10.5)
    assert isinstance(end, LogEnd)
    assert end == LogEnd(
        iter_steps=12,
        tend="00:00:03",
        saddle_energy=-9.1,
        saddle_fmax=0.2,
    )


def test_sella_loglist_returns_typed_entries(tmp_path: Path):
    log_path = tmp_path / "geom.log"
    log_path.write_text(
        "date step time energy fmax cmax rtrust rho trj_id\n"
        "0 1 00:00:01 -10.0 0.4 0.3 0.2 0.1 4\n"
    )

    result = _sella_loglist(log_path)

    assert result == [
        SellaLog(
            step_id=1,
            time_float=result[0].time_float,
            energy=-10.0,
            fmax=0.4,
            cmax=0.3,
            rtrust=0.2,
            rho=0.1,
            trj_id=4,
        )
    ]


@pytest.mark.skipif(SpinID is None, reason="rgpycrumbs not available")
def test_parse_sella_saddle_uses_npes_fallback(tmp_path: Path):
    (tmp_path / "npes.txt").write_text("npes 7\n")
    (tmp_path / "run.log").write_text(
        "header\n"
        "INFO START 00:00:01 -10.5 foo bar baz qux\n"
        "INFO 12 00:00:03 -9.1 0.2 foo bar baz\n"
    )

    result = parse_sella_saddle(tmp_path, SpinID(mol_id=8, spin="singlet"))

    assert result.success is True
    assert result.method == "Sella"
    assert result.pes_calls == 7
    assert result.iter_steps == 12
    assert result.init_energy == -10.5
    assert result.saddle_energy == -9.1
    assert result.barrier == pytest.approx(1.4)
    assert result.mol_id == 8
    assert result.spin == "singlet"
