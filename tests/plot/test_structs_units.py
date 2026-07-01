# SPDX-FileCopyrightText: 2023-present Rohit Goswami <rog32@hi.is>
#
# SPDX-License-Identifier: MIT
"""Pint-backed energy conversion and registry helpers."""

from __future__ import annotations

import numpy as np
import pytest

pint = pytest.importorskip("pint")

from chemparseplot.plot.structs import convert_energy
from chemparseplot.units import (
    ENERGY_UNITS,
    as_energy_quantity,
    convert_energy_magnitude,
    normalize_energy_unit,
)


def test_normalize_energy_unit_aliases():
    assert normalize_energy_unit("ev") == "eV"
    assert normalize_energy_unit("kcal_mol") == "kcal/mol"
    assert normalize_energy_unit("kJ/mol") == "kJ/mol"
    with pytest.raises(ValueError):
        normalize_energy_unit("hartree")


def test_convert_energy_magnitude_matches_chemical_factors():
    eV = np.array([1.0, 2.0])
    kcal = convert_energy_magnitude(eV, "kcal/mol")
    assert np.allclose(kcal, eV * 23.06054783061903)
    kj = convert_energy_magnitude(eV, "kJ/mol")
    assert np.allclose(kj, eV * 96.48533212331002)
    # Round trip
    back = convert_energy_magnitude(kcal, "eV", source_unit="kcal/mol")
    assert np.allclose(back, eV)


def test_convert_energy_uses_pint_quantity_path():
    q = as_energy_quantity([1.0], "eV")
    out = convert_energy(q, "kcal/mol")
    assert np.allclose(out, [23.06054783061903])


def test_energy_units_tuple():
    assert "eV" in ENERGY_UNITS
