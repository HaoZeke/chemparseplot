# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""ORCA text grammar track tests."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("parsimonious")

from chemparseplot.api import parse_orca_final_energy
from chemparseplot.parse.grammar.orca_text import (
    extract_cartesian_angstrom_blocks,
    extract_final_energies_hartree,
    parse_orca_text_summary,
)
from chemparseplot.parse.orca import parse_orca_text_summary as orca_export

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"
SNIPPET = (FIXTURES / "orca_energy_coords_snippet.out").read_text()


def test_extract_final_energies_from_fixture():
    energies = extract_final_energies_hartree(SNIPPET)
    assert len(energies) == 6
    assert energies[0] == pytest.approx(-38.900957131258)
    assert energies[-1] == pytest.approx(-38.911679795433)


def test_cartesian_block_last_geometry():
    blocks = extract_cartesian_angstrom_blocks(SNIPPET)
    assert len(blocks) == 1
    assert [a.symbol for a in blocks[0]] == ["C", "H", "H"]
    assert blocks[0][0].x == pytest.approx(0.253633)


def test_parse_orca_text_summary():
    summary = parse_orca_text_summary(SNIPPET)
    assert summary.n_atoms == 3
    assert summary.final_energy_hartree == pytest.approx(-38.911679795433)
    assert summary.final_energy.m == pytest.approx(-38.911679795433)
    assert str(summary.final_energy.units) in ("hartree", "Eh")
    assert len(summary.last_geometry) == 3


def test_api_parse_orca_final_energy():
    e = parse_orca_final_energy(SNIPPET)
    assert float(e.m) == pytest.approx(-38.911679795433)


def test_public_orca_export_alias():
    assert orca_export is parse_orca_text_summary or callable(orca_export)
    s = orca_export(SNIPPET)
    assert s.final_energy_hartree is not None
