# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""Tests for eOn minimization result parsing."""

from chemparseplot.parse.eon.minimization import min_e_result


def test_min_e_result_reads_good_run(tmp_path):
    results = tmp_path / "results.dat"
    results.write_text("GOOD termination_reason\n-12.345 potential_energy\n")
    assert min_e_result(tmp_path) == -12.345


def test_min_e_result_returns_none_for_missing_file(tmp_path):
    assert min_e_result(tmp_path) is None


def test_min_e_result_returns_none_for_failed_run(tmp_path):
    results = tmp_path / "results.dat"
    results.write_text("BAD termination_reason\n-12.345 potential_energy\n")
    assert min_e_result(tmp_path) is None
