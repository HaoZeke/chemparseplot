# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT

"""Focused tests for eOn saddle ``results.dat`` parsing."""

from pathlib import Path

import pytest

pytestmark = pytest.mark.pure


def test_read_results_dat_success(tmp_path: Path):
    from chemparseplot.parse.eon.saddle_search import _read_results_dat

    (tmp_path / "results.dat").write_text(
        "0 termination_reason\n"
        "100 total_force_calls\n"
        "50 iterations\n"
        "-123.450000 potential_energy_saddle\n"
    )

    result = _read_results_dat(tmp_path)

    assert result is not None
    assert result.termination_status == "GOOD"
    assert result.pes_calls == 100
    assert result.iter_steps == 50
    assert result.saddle_energy == pytest.approx(-123.45)
    assert result.terminal_only is False


def test_read_results_dat_failure_status_only(tmp_path: Path):
    from chemparseplot.parse.eon.saddle_search import _read_results_dat

    (tmp_path / "results.dat").write_text("5 termination_reason\n")

    result = _read_results_dat(tmp_path)

    assert result is not None
    assert result.termination_status == "BAD_MAX_ITERATIONS"
    assert result.terminal_only is True
    assert result.pes_calls is None
