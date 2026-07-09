# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""Drive shipped chemparseplot.api entry points (parse → typed result)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from chemparseplot.api import (
    convert_energy_magnitude,
    extract_orca_geomscan_energy,
    normalize_energy_unit,
    suite_pins,
)

# Inline fixture (same content as tests/fixtures_geomscan_snippet.txt)
_GEOMSCAN_SNIPPET = """
The Calculated Surface using the 'Actual Energy'
   7.55890395  -0.74239862
   7.32930292  -0.74349939
   7.09970189  -0.74467446

The Calculated Surface using the SCF energy
   7.55890395  -0.74239862
   7.32930292  -0.74349939
   7.09970189  -0.74467446
"""

FIXTURE_FILE = Path(__file__).parent / "fixtures_geomscan_snippet.txt"


def test_normalize_and_convert_energy_library_path():
    assert normalize_energy_unit("ev") == "eV"
    kcal = convert_energy_magnitude(1.0, "kcal/mol")
    val = float(np.asarray(kcal).reshape(-1)[0])
    assert 20.0 < val < 25.0
    back = float(np.asarray(convert_energy_magnitude(val, "eV", source_unit="kcal/mol")))
    assert abs(back - 1.0) < 1e-6


def test_extract_orca_geomscan_energy_typed_nonempty():
    """Public parse path: ORCA geomscan text → pint Quantities with data."""
    dist, energy = extract_orca_geomscan_energy(_GEOMSCAN_SNIPPET, "Actual Energy")
    assert dist.size == 3
    assert energy.size == 3
    assert str(dist.units) == "bohr"
    assert str(energy.units) == "hartree"
    np.testing.assert_allclose(
        dist.magnitude,
        [7.55890395, 7.32930292, 7.09970189],
    )
    np.testing.assert_allclose(
        energy.magnitude,
        [-0.74239862, -0.74349939, -0.74467446],
    )


def test_extract_orca_geomscan_energy_from_fixture_file():
    text = FIXTURE_FILE.read_text(encoding="utf-8")
    dist, energy = extract_orca_geomscan_energy(text, "SCF energy")
    assert dist.size >= 1
    assert energy.size == dist.size
    assert dist.size == 3


def test_suite_pins_returns_dict():
    pins = suite_pins()
    assert isinstance(pins, dict)
